#include "tensor.h"
#include <cuda_runtime.h>

// Helper function to check CUDA errors
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// CUDA kernel for element-wise addition
__global__ void add_kernel(float *A, float *B, float *C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] + B[idx];
  }
}

// Implementation of GPU Operations
float* GPUOperation::move_data(float* cpu_data, size_t size) {
    float* gpu_data;
    CHECK_CUDA(cudaMalloc(&gpu_data, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu_data, cpu_data, size * sizeof(float), cudaMemcpyHostToDevice));
    return gpu_data;
}

void GPUOperation::free_memory(float* gpu_data) {
    if (gpu_data) {
        CHECK_CUDA(cudaFree(gpu_data));
    }
}

Buffer* GPUOperation::add(Buffer* A, Buffer* B) {
    size_t size = A->shape[0] * A->shape[1];
    float* output;
    CHECK_CUDA(cudaMalloc(&output, size * sizeof(float)));

    // Configure CUDA kernel
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Launch kernel
    add_kernel<<<num_blocks, block_size>>>(A->data, B->data, output, size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    return new Buffer(output, "gpu", A->shape);
}
