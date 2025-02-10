#include "tensor.h"
#include <cmath>
#include <iostream>
#include <ostream>

// todo : add support for broadcasting
// todo : error handling

void CPUOperation::free_memory(float* data) { 
    delete[] data; 
}

float* CPUOperation::move_data(float* data, size_t size) {
    float* new_data = new float[size];
    std::copy(data, data + size, new_data);
    return new_data;
}

Buffer* CPUOperation::add(Buffer* A, Buffer* B) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < size; i++) {
        out[i] = A->data[i] + B->data[i];
    }

    return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::negate(Buffer* A) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < size; i++) {
        out[i] = -A->data[i];
    }

    return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::subtract(Buffer* A, Buffer* B) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < size; i++) {
        out[i] = A->data[i] - B->data[i];
    }

    return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::scalar_mul(Buffer* A, float c) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < size; i++) {
        out[i] = c * A->data[i];
    }

    return new Buffer(out, "cpu", shape);
}

// matmul
Buffer* CPUOperation::mul(Buffer* A, Buffer* B) {
    auto shape_A = A->shape;
    auto shape_B = B->shape;
    float* out = new float[shape_A[0] * shape_B[1]]();  // Initialize to zero

    for (size_t i = 0; i < shape_A[0]; i++) {
        for (size_t j = 0; j < shape_B[1]; j++) {
            for (size_t k = 0; k < shape_A[1]; k++) {
                out[i * shape_B[1] + j] += A->data[i * shape_A[1] + k] *
                                         B->data[k * shape_B[1] + j];
            }
        }
    }

    return new Buffer(out, "cpu", {shape_A[0], shape_B[1]});
}

Buffer* CPUOperation::pow(Buffer* A, float exp) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < size; i++) {
        out[i] = std::pow(A->data[i], exp);
    }

    return new Buffer(out, "cpu", shape);
}

Buffer* CPUOperation::transpose(Buffer* A) {
    auto shape = A->shape;
    size_t size = shape[0] * shape[1];
    float* out = new float[size];

    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            out[j * shape[0] + i] = A->data[i * shape[1] + j];
        }
    }

    return new Buffer(out, "cpu", {shape[1], shape[0]});
}

void CPUOperation::print_buffer(Buffer* buff) {
    auto out = string("Tensor(");
    for (size_t i = 0; i < buff->shape[0]; i++) {
        if (i == 0) {
            out += "[";
        } else {
            out += "       [";
        }
        for (size_t j = 0; j < buff->shape[1]; j++) {
            out += to_string(buff->data[i * buff->shape[1] + j]);
            if (j != buff->shape[1] - 1) {
                out += ", ";
            }
        }
        out += "]";
        if (i != buff->shape[0] - 1) {
            out += ",\n";
        }
    }
    cout << out << ")" << endl;
}

// GPU stubs
void throw_gpu_error() {
    std::cerr << "GPU operations not supported. Compile with CUDA flags set."
              << std::endl;
}

float* __attribute__((weak))
GPUOperation::move_data(float* data, size_t size) {
    throw_gpu_error();
    return nullptr;
}

void __attribute__((weak)) GPUOperation::free_memory(float* data) {
    throw_gpu_error();
}

Buffer* __attribute__((weak)) GPUOperation::mul(Buffer* A, Buffer* B) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::scalar_mul(Buffer* A, float c) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::add(Buffer* A, Buffer* B) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::subtract(Buffer* A, Buffer* B) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::negate(Buffer* A) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::pow(Buffer* A, float exp) {
    throw_gpu_error();
    return nullptr;
}

Buffer* __attribute__((weak)) GPUOperation::transpose(Buffer* data) {
    throw_gpu_error();
    return nullptr;
}

void __attribute__((weak)) GPUOperation::print_buffer(Buffer* data) {
    throw_gpu_error();
}
