#include "tensor.h"
#include <cmath>
#include <vector>
#include <iostream>

// todo : add support for broadcasting
// todo : error handling

void CPUOperation::free_memory(vector<float>* data) {
    delete data;
}

vector<float>* CPUOperation::move_data(vector<float>* data) {
	auto new_data = new vector<float>(*data);
	return new_data;
}

Buffer* CPUOperation::add(Buffer* A, Buffer* B) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < size; i++) {
		(*out)[i] = (*A->data)[i] + (*B->data)[i];
	}

	return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::negate(Buffer* A) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < size; i++) {
		(*out)[i] = -(*A->data)[i];
	}

	return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::subtract(Buffer* A, Buffer* B) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < size; i++) {
		(*out)[i] = (*A->data)[i] - (*B->data)[i];
	}

	return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::scalar_mul(Buffer* A, float c) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < size; i++) {
		(*out)[i] = c * (*A->data)[i];
	}

	return new Buffer(out, "cpu", shape);
}

//matmul 
Buffer* CPUOperation::mul(Buffer* A, Buffer* B) {
	auto shape_A = A -> shape;
	auto shape_B = B -> shape;
	auto out = new vector<float>(shape_A[0] * shape_B[1], 0);

	for (int i = 0; i < shape_A[0]; i++) {
		for (int j = 0; j < shape_B[1]; j++) {
			for (int k = 0; k < shape_A[1]; k++) {
				(*out)[i * shape_B[1] + j] += (*A->data)[i * shape_A[1] + k] * (*B->data)[k * shape_B[1] + j];
			}
		}
	}

	return new Buffer(out, "cpu", {shape_A[0], shape_B[1]});
}

Buffer* CPUOperation::pow(Buffer* A, float exp) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < size; i++) {
		auto val = std::pow((*A->data)[i], exp);
		(*out)[i] = val;
	}

	return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::transpose(Buffer* A) {
	auto shape = A -> shape;
	auto size = shape[0] * shape[1];
	auto out = new vector<float>(size);

	for (int i = 0; i < shape[0]; i++) {
		for (int j = 0; j < shape[1]; j++) {
			(*out)[j * shape[0] + i] = (*A->data)[i * shape[1] + j];
		}
	}

	return new Buffer(out, "cpu", {shape[1], shape[0]});
}

void CPUOperation::print_buffer(Buffer* buff) {
  auto out = string("Tensor(");
  for (size_t i = 0; i < buff -> shape.at(0); i++) {
	if (i == 0) {
	  out += "[";
	} else {
	  out += "       [";
	}
	for (int j = 0; j < buff -> shape[1]; j++) {
	  out += to_string(buff -> data->at(i * buff -> shape[1] + j));
	  if (j != buff -> shape[1] - 1) {
		out += ", ";
	  }
	}
	out += "]";
	if (i != buff -> shape[0] - 1) {
	  out += ",\n";
	}
  }
  cout << out << endl;
}
