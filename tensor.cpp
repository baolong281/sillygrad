#include "tensor.h"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

Tensor::Tensor(vector<vector<float>> *data, string device, bool requires_grad) {
  this->device = device;
  this->data = data;
  this->_backward = []() {};
  this->shape = {data->size(), data->at(0).size()};
  this->prev = {};

  Mat *grad;

  if (requires_grad) {
    grad = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
  }

  if (device == "cpu") {
    this->op_type = new CPUOperation{};
    this->data = data;
    this->grad = grad;
  } else {
    this->op_type = new GPUOperation();
    this->data = op_type->move_data(data);
    this->grad = op_type->move_data(grad);
  }
}

Tensor *Tensor::to(string device) {
  if (device == "cpu") {
    this->op_type = new CPUOperation{};
    this->data = op_type->move_data(data);
  } else {
    // this -> op_type = new GPUOperation();
    this->data = op_type->move_data(data);
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

void print_tensor(Mat *data, string device) {
  auto out = string("Tensor(");
  auto shape = vector<size_t>{data->size(), data->at(0).size()};
  for (size_t i = 0; i < shape.at(0); i++) {
    if (i == 0) {
      out += "[";
    } else {
      out += "       [";
    }
    for (int j = 0; j < shape[1]; j++) {
      out += to_string(data->at(i)[j]);
      if (j != shape[1] - 1) {
        out += ", ";
      }
    }
    out += "]";
    if (i != shape[0] - 1) {
      out += ",\n";
    }
  }
  out += ", device=" + device + ")";
  cout << out << endl;
}

void Tensor::print_data() const { print_tensor(data, device); }

void Tensor::print_grad() const { print_tensor(grad, device); }
