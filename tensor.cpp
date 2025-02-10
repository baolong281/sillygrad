#include "tensor.h"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <array>

using namespace std;

Buffer* Buffer::to(string new_device) {
    // Check if GPU is available when requesting GPU device
    if (new_device == "gpu") {
        #ifndef CUDA_ENABLED
            throw GPUNotAvailableError();
        #endif
    }

    auto shape = this->shape;
    size_t size = shape[0] * shape[1];  // Remove sizeof(float)
    
    if (new_device == "cpu") {
        auto op = new CPUOperation();
        float* new_data = op->move_data(data, size);
        delete op;
        
        // Clean up old data and deleter
        deleter->free(data);
        deleter = std::make_unique<CPUDeleter>();
        data = new_data;
    } else {
        #ifdef CUDA_ENABLED
            auto op = new GPUOperation();
            float* new_data = op->move_data(data, size);
            delete op;
            
            // Clean up old data and deleter
            deleter->free(data);
            deleter = std::make_unique<GPUDeleter>();
            data = new_data;
        #endif
    }
    
    device = new_device;
    return this;
}

Tensor::Tensor(Buffer* data, const string& device, bool requires_grad) {
  this->device = device;
  this->data = data;
  auto shape = data->shape;
  this->_backward = []() {};
  this->prev = {};
  this->requires_grad = requires_grad;

  if (requires_grad) {
    size_t size = shape[0] * shape[1];
    float* grad_data = new float[size]();  // Initialize to zero
    this->grad = new Buffer(grad_data, device, shape);
  }

  if (device == "cpu") {
    this->op_type = new CPUOperation{};
  } else {
    this->op_type = new GPUOperation{};
  }
}

Tensor::Tensor(vector<vector<float>>* data, const string& device, bool requires_grad) {
  auto shape = std::array<size_t, 2>{data->size(), data->at(0).size()};
  size_t size = shape[0] * shape[1];
  
  // Convert 2D vector to contiguous array
  float* data_contiguous = new float[size];
  for (size_t i = 0; i < shape[0]; i++) {
    for (size_t j = 0; j < shape[1]; j++) {
      data_contiguous[i * shape[1] + j] = (*data)[i][j];
    }
  }
  
  auto buffer = new Buffer(data_contiguous, device, shape);
  this->device = device;
  this->data = buffer;
  this->_backward = []() {};
  this->prev = {};
  this->requires_grad = requires_grad;

  if (requires_grad) {
    float* grad_data = new float[size]();  // Initialize to zero
    this->grad = new Buffer(grad_data, device, shape);
  }

  if (device == "cpu") {
    this->op_type = new CPUOperation{};
  } else {
    this->op_type = new GPUOperation{};
  }
}

// Factory methods
shared_ptr<Tensor> Tensor::create(Buffer* data, const string& device, bool requires_grad) {
    return shared_ptr<Tensor>(new Tensor(data, device, requires_grad));
}

shared_ptr<Tensor> Tensor::create(vector<vector<float>>* data, const string& device, bool requires_grad) {
    return shared_ptr<Tensor>(new Tensor(data, device, requires_grad));
}

shared_ptr<Tensor> Tensor::to(string device) {
    this->data = this->data->to(device);
    if (requires_grad) {
        this->grad = this->grad->to(device);
    }
    this->device = device;
    if (device == "cpu") {
        this->op_type = new CPUOperation{};
    } else {
        this->op_type = new GPUOperation{};
    }
    return shared_from_this();
}

shared_ptr<Tensor> Tensor::operator*(float c) {
    auto out_data = op_type->scalar_mul(data, c);
    auto out = Tensor::create(out_data, device, requires_grad);
    out->requires_grad = this->requires_grad;
    if (requires_grad) {
        out->_backward = [self=shared_from_this(), c, out]() {
            if (self->grad) {
                auto temp = self->op_type->scalar_mul(out->grad, c);
                self->grad = self->op_type->add(self->grad, temp);
                delete temp;
            }
        };
        out->prev = {shared_from_this()};
    }
    return out;
}

shared_ptr<Tensor> Tensor::operator+(shared_ptr<Tensor> other) {
    auto out_data = op_type->add(data, other->data);
    auto out = Tensor::create(out_data, device, requires_grad);
    out->requires_grad = this->requires_grad || other->requires_grad;
    if (out->requires_grad) {
        out->_backward = [self=shared_from_this(), other, out]() {
            if (self->grad) {
                self->grad = self->op_type->add(self->grad, out->grad);
            }
            if (other->grad) {
                other->grad = other->op_type->add(other->grad, out->grad);
            }
        };
        out->prev = {shared_from_this(), other};
    }
    return out;
}
shared_ptr<Tensor> Tensor::operator-(shared_ptr<Tensor> other) {
    auto out_data = op_type->subtract(data, other->data);
    auto out = Tensor::create(out_data, device, requires_grad);
    out->requires_grad = this->requires_grad || other->requires_grad;
    if (out->requires_grad) {
        out->_backward = [self=shared_from_this(), other, out]() {
            if (self->grad) {
                self->grad = self->op_type->add(self->grad, out->grad);
            }
            if (other->grad) {
                auto neg_grad = self->op_type->negate(out->grad);
                other->grad = other->op_type->add(other->grad, neg_grad);
                delete neg_grad;
            }
        };
        out->prev = {shared_from_this(), other};
    }
    return out;
}

shared_ptr<Tensor> Tensor::operator-() {
    auto out_data = op_type->negate(data);
    auto out = Tensor::create(out_data, device, requires_grad);
    out->requires_grad = this->requires_grad;
    if (requires_grad) {
        out->_backward = [self=shared_from_this(), out]() {
            if (self->grad) {
                auto neg_grad = self->op_type->negate(out->grad);
                self->grad = self->op_type->add(self->grad, neg_grad);
                delete neg_grad;
            }
        };
        out->prev = {shared_from_this()};
    }
    return out;
}

shared_ptr<Tensor> Tensor::pow(float exp) {
    auto out_data = op_type->pow(data, exp);
    auto out = Tensor::create(out_data, device, requires_grad);
    out->requires_grad = this->requires_grad;
    if (requires_grad) {
        out->_backward = [self=shared_from_this(), exp, out]() {
            if (self->grad) {
                // Power rule: d/dx(x^n) = nx^(n-1)
                auto temp = self->op_type->pow(self->data, exp - 1);
                auto temp2 = self->op_type->scalar_mul(temp, exp);
                auto temp3 = self->op_type->mul(temp2, out->grad);
                self->grad = self->op_type->add(self->grad, temp3);
                delete temp;
                delete temp2;
                delete temp3;
            }
        };
        out->prev = {shared_from_this()};
    }
    return out;
}

void print_tensor(float* data, std::array<size_t, 2> shape, string device) {
  auto out = string("Tensor(");
  for (size_t i = 0; i < shape[0]; i++) {
    if (i == 0) {
      out += "[";
    } else {
      out += "       [";
    }
    for (size_t j = 0; j < shape[1]; j++) {
      out += to_string(data[i * shape[1] + j]);
      if (j != shape[1] - 1) {
        out += ", ";
      }
    }
    out += "]";
    if (i != shape[0] - 1) {
      out += ",\n";
    }
  }
  cout << out << ")" << endl;
}

void Tensor::print_data() const { op_type->print_buffer(data); }

void Tensor::print_grad() const { op_type->print_buffer(grad); }
