#include "tensor.h"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

Buffer::Buffer(vector<float> *data, string device, vector<size_t> shape) {
  this->device = device;
  this -> shape = shape;
  if(device=="cpu"){
	this->data = data;
  }
  else {
	// auto op = new GPUOperation();
	// this -> data = op -> move_data(data);
  }
  
}

Buffer* Buffer::Buffer::to(string device) {
  if (device == "cpu") {
		auto op = new CPUOperation();
		this->data = op->move_data(data);
  } else {
		// auto op = new GPUOperation();
		// this->data = op->move_data(data);
  }
	this->device = device;
	return this;
}

Tensor::Tensor(Buffer *data, string device, bool requires_grad) {
  this->device = device;
  this->data = data;
  auto shape = data->shape;
  this->_backward = []() {};
  this->prev = {};
  this->requires_grad = requires_grad;

  Buffer *grad;

  if (requires_grad) {
	grad = new Buffer(new vector<float>(shape[0] * shape[1], 0), device, shape);
  }

  if (device == "cpu") {
	this->op_type = new CPUOperation{};
	this->grad = grad;
  } else {
	// this->op_type = new GPUOperation{};
	// this->data = this->data->to(device);
	// this->grad = this->grad->to(device);
  }
}

Tensor::Tensor(vector<vector<float>>* data, string device, bool requires_grad) {
  auto shape = vector<size_t>{data->size(), data->at(0).size()};
  auto data_contiguous = new vector<float>(shape[0] * shape[1]);
  for (size_t i = 0; i < shape[0]; i++) {
	for (size_t j = 0; j < shape[1]; j++) {
	  data_contiguous->at(i * shape[1] + j) = data->at(i)[j];
	}
  }
  auto buffer = new Buffer(data_contiguous, device, shape);
  this->device = device;
  this->data = buffer;
  this->_backward = []() {};
  this->prev = {};
  this->requires_grad = requires_grad;

  Buffer* grad;

  if (requires_grad) {
    grad = new Buffer(new vector<float>(shape[0] * shape[1], 0), device, shape);
  }

  if (device == "cpu") {
    this->op_type = new CPUOperation{};
    this->grad = grad;
  } else {
	// this->op_type = new GPUOperation{};
    // this->data = this -> data -> to(device);
    // this->grad = this -> grad -> to(device);
  }
}

Tensor *Tensor::to(string device) {
  this->data = this -> data -> to (device);
  if(requires_grad) {
	  this->grad = this -> grad -> to(device);
  }
  this->device = device;
  return this;
}

Tensor Tensor::operator*(float c) {
	auto out = Tensor(op_type->scalar_mul(data, c));
	// out . _backward = [this, out, c] {
	// 	this -> prev.insert(&out);
	// 	out . _backward();
	// };
	// out._backward = [this, out, c] { this->grad += c * out.grad; };
	return out;
}

Tensor Tensor::operator+(Tensor &other) {
  auto prev = set<Tensor *>{this, &other};

  auto out = Tensor(op_type->add(data, other.data));
  // out . _backward = [this, out, other] {
  // 	this -> prev.insert(&out);
  // 	this -> prev.insert(&other);
  // 	out . _backward();
  // };

//   out._backward = [this, out, other] {
//     this->grad += 1.0 * out.grad;
//     other->grad += 1.0 * out.grad;
//   };

  return out;
}

Tensor Tensor::operator-(Tensor &other) {
  auto prev = set<Tensor *>{this, &other};

  auto out = Tensor(op_type->subtract(data, other.data));
  // out . _backward = [this, out, other] {
  // 	this -> prev.insert(&out);
  // 	this -> prev.insert(&other);
  // 	out . _backward();
  // };

//   out._backward = [this, out, other] {
//     this->grad += 1.0 * out.grad;
//     other->grad += 1.0 * out.grad;
//   };

  return out;
}

Tensor Tensor::operator-() {
  auto prev = set<Tensor *>{this};

  auto out = Tensor(op_type->negate(data));
  // out . _backward = [this, out, other] {
  // 	this -> prev.insert(&out);
  // 	this -> prev.insert(&other);
  // 	out . _backward();
  // };

//   out._backward = [this, out, other] {
//     this->grad += 1.0 * out.grad;
//     other->grad += 1.0 * out.grad;
//   };

  return out;
}

void print_tensor(vector<float>* data, vector<std::size_t> shape, string device) {
  auto out = string("Tensor(");
  for (size_t i = 0; i < shape.at(0); i++) {
	if (i == 0) {
	  out += "[";
	} else {
	  out += "       [";
	}
	for (int j = 0; j < shape[1]; j++) {
	  out += to_string(data->at(i * shape[1] + j));
	  if (j != shape[1] - 1) {
		out += ", ";
	  }
	}
	out += "]";
	if (i != shape[0] - 1) {
	  out += ",\n";
	}
  }
  cout << out << endl;
}

void Tensor::print_data() const { print_tensor(data -> data, data -> shape, device); }

void Tensor::print_grad() const { print_tensor(data -> data, data->shape, device); }
