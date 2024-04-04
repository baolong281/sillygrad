#include "tensor.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

using namespace std;

extern "C" {
void GPUOperation::free_memory(vector<float> *data) {
    if (data != nullptr) {
        cudaFree(data);
        data = nullptr;
    }
}

vector<float> *GPUOperation::move_data(vector<float> *data) {
    vector<float> *out = nullptr;
    if (data != nullptr) {
        auto size = data->size() * sizeof(float);
        cudaMemcpy(out, data, size, cudaMemcpyHostToDevice);
    }

    return out;
}

void GPUOperation::print_buffer(Buffer *buff) {
    auto out = string("Tensor(");
    for (size_t i = 0; i < buff->shape.at(0); i++) {
        if (i == 0) {
            out += "[";
        } else {
            out += "       [";
        }
        for (int j = 0; j < buff->shape[1]; j++) {
            out += to_string(buff->data->at(i * buff->shape[1] + j));
            if (j != buff->shape[1] - 1) {
                out += ", ";
            }
        }
        out += "]";
        if (i != buff->shape[0] - 1) {
            out += ",\n";
        }
    }
    cout << out << endl;
}

Buffer *GPUOperation::mul(Buffer *A, Buffer *B) { return nullptr; }

Buffer *GPUOperation::scalar_mul(Buffer *A, float c) { return nullptr; }

Buffer *GPUOperation::add(Buffer *A, Buffer *B) { return nullptr; }

Buffer *GPUOperation::subtract(Buffer *A, Buffer *B) { return nullptr; }

Buffer *GPUOperation::negate(Buffer *A) { return nullptr; }

Buffer *GPUOperation::pow(Buffer *A, float exp) { return nullptr; }

Buffer *GPUOperation::transpose(Buffer *data) { return nullptr; }

void BANANA() { cout << "BANANA" << endl; }
}