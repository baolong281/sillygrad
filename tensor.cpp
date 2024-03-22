#include "tensor.h"
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

Tensor::Tensor(vector<vector<float>>* data, string device, bool requires_grad) {
	this -> device = device;
	this -> data = data;
	this -> _backward = []() {};
	this -> shape = {data->size(), data->at(0).size()};
	this -> prev = {};

	Mat* grad;

	if(requires_grad) {
		grad = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	}

	if(device == "cpu") {
		this -> op_type = new CPUOperation{};
		this -> data = data;
		this -> grad = grad;
	} else {
		// this->op_type = GPUOperation();
		this->data = op_type -> move_data(data);
		this -> grad = op_type -> move_data(grad);
	}
}

Tensor::operator std::string() const {
	auto out = string("Tensor(");
	for(auto i = 0; i < shape.size(); i++) {
		out += to_string(shape[i]);
		if(i != shape.size() - 1) {
			out += ", ";
		}
	}
	out += ")";
	return out;
}

void print_tens(Mat* data, string device) {
	auto out = string("Tensor(");
	auto shape = vector<size_t>{data->size(), data->at(0).size()};
	for(size_t i=0; i<shape.at(0); i++) {
		if(i == 0) {
			out += "[";
		} else {
			out += "       [";
		}
		for(int j=0; j<shape[1]; j++) {
			out += to_string(data->at(i)[j]);
			if(j != shape[1] - 1) {
				out += ", ";
			}
		}
		out += "]";
		if(i != shape[0] - 1) {
			out += ",\n";
		}

	}
	out += ", device="+device+")";
	cout << out << endl;
}

void Tensor::print_data() const {
	print_tens(data, device);
}

void Tensor::print_grad() const {
	print_tens(grad, device);
}

Tensor* Tensor::to(string device) {
	if(device == "cpu") {
		this -> op_type = new CPUOperation{};
		this -> data = op_type -> move_data(data);
	} else {
		// this -> op_type = new GPUOperation();
		this -> data = op_type -> move_data(data);
	}
	this -> device = device;
	return this;
}

Tensor Tensor::operator+ (Tensor& other) {
	auto out = Tensor(op_type -> add(data, other.data));
	// out . _backward = [this, out, other] {
	// 	this -> prev.insert(&out);
	// 	this -> prev.insert(&other);
	// 	out . _backward();
	// };
	return out;
}

Mat* CPUOperation::move_data(Mat* data) {
	auto shape = vector<size_t>{data->size(), data->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = data->at(i)[j];
		}
	}
	return out;
}

Mat* CPUOperation::add(Mat* A, Mat* B) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = A->at(i)[j] + B->at(i)[j];
		}
	}
	return out;
}

Mat* CPUOperation::negate(Mat* A) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = -A->at(i)[j];
		}
	}
	return out;
}

Mat* CPUOperation::subtract(Mat* A, Mat* B) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = A->at(i)[j] - B->at(i)[j];
		}
	}
	return out;
}

Mat* CPUOperation::scalar_mul(Mat* A, float c) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = A->at(i)[j] * c;
		}
	}
	return out;
}

Mat* CPUOperation::mul(Mat* A, Mat* B) {
	auto shape = vector<size_t>{A->size(), B->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = 0;
			for(auto k = 0; k < A->at(0).size(); k++) {
				(*out)[i][j] += A->at(i)[k] * B->at(k)[j];
			}
		}
	}
	return out;
}

Mat* CPUOperation::pow(Mat* A, float exp) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[0], vector<float>(shape[1]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[i][j] = std::pow(A->at(i)[j], exp);
		}
	}
	return out;
}


Mat* CPUOperation::transpose(Mat* A) {
	auto shape = vector<size_t>{A->size(), A->at(0).size()};
	auto out = new vector<vector<float>>(shape[1], vector<float>(shape[0]));
	for(auto i = 0; i < shape[0]; i++) {
		for(auto j = 0; j < shape[1]; j++) {
			(*out)[j][i] = A->at(i)[j];
		}
	}
	return out;
}