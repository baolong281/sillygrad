#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>

using namespace std;

typedef vector<vector<float>> Mat;

class Tensor;

class BufferDeleter {
public:
    virtual ~BufferDeleter() = default;
    virtual void free(float* data) = 0;
};

class CPUDeleter : public BufferDeleter {
public:
    void free(float* data) override {
        delete[] data;
    }
};

#ifdef CUDA_ENABLED
class GPUDeleter : public BufferDeleter {
public:
    void free(float* data) override {
        cudaFree(data);
    }
};
#endif

class GPUNotAvailableError : public std::runtime_error {
public:
    GPUNotAvailableError() : std::runtime_error("GPU operations not available. Compile with CUDA enabled.") {}
};

class Buffer {
public:
    float* data;
    std::string device;
    std::array<size_t, 2> shape;
    std::unique_ptr<BufferDeleter> deleter;

    Buffer(float* data, std::string device, std::array<size_t, 2> shape) 
        : data(data), device(device), shape(shape) {
        if (device == "cpu") {
            deleter = std::make_unique<CPUDeleter>();
        }
        else if (device == "gpu") {
            #ifdef CUDA_ENABLED
                deleter = std::make_unique<GPUDeleter>();
            #else
                throw GPUNotAvailableError();
            #endif
        }
    }

    Buffer(const std::vector<float>* vec, std::string device, std::array<size_t, 2> shape) {
        if (device == "gpu") {
            #ifndef CUDA_ENABLED
                throw GPUNotAvailableError();
            #endif
        }
        
        this->device = device;
        this->shape = shape;
        
        size_t size = shape[0] * shape[1];
        float* raw_data = new float[size];
        std::copy(vec->begin(), vec->end(), raw_data);
        
        if (device == "cpu") {
            this->data = raw_data;
            deleter = std::make_unique<CPUDeleter>();
        } else {
            #ifdef CUDA_ENABLED
                auto op = new GPUOperation();
                this->data = op->move_data(raw_data, size);
                delete[] raw_data;
                deleter = std::make_unique<GPUDeleter>();
            #endif
        }
    }

    Buffer* to(string new_device);

    ~Buffer() {
        if (data) {
            deleter->free(data);
        }
    }
};

class Operations {
public:
    virtual Buffer* mul(Buffer* A, Buffer* B) = 0;
    virtual Buffer* scalar_mul(Buffer* A, float c) = 0;
    virtual Buffer* add(Buffer* A, Buffer* B) = 0;
    virtual Buffer* subtract(Buffer* A, Buffer* B) = 0;
    virtual Buffer* negate(Buffer* A) = 0;
    virtual Buffer* pow(Buffer* A, float exp) = 0;
    virtual float* move_data(float* data, size_t size) = 0;
    virtual Buffer* transpose(Buffer* A) = 0;
    virtual void free_memory(float* data) = 0;
    virtual void print_buffer(Buffer* data) = 0;
    virtual ~Operations() = default;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
  string device;
  function<void()> _backward;
  set<shared_ptr<Tensor>> prev;
  Operations *op_type;
  bool requires_grad;
  Tensor(Buffer* data, const std::string& device, bool requires_grad);  // Private constructor
  Tensor(std::vector<std::vector<float>>* data, const std::string& device, bool requires_grad);

public:
  Buffer *data;
  Buffer *grad;
  static std::shared_ptr<Tensor> create(Buffer* data, const std::string& device = "cpu", bool requires_grad = true);
  static std::shared_ptr<Tensor> create(std::vector<std::vector<float>>* data, const std::string& device = "cpu", bool requires_grad = true);
  shared_ptr<Tensor> self;
  ~Tensor() {
    delete data;
    delete grad;
    delete op_type;
  }
  shared_ptr<Tensor> operator+(shared_ptr<Tensor> other);
  shared_ptr<Tensor> operator-(shared_ptr<Tensor> other);
  shared_ptr<Tensor> operator-();
  shared_ptr<Tensor> operator*(float c);

  friend shared_ptr<Tensor> operator*(float c, shared_ptr<Tensor> A) { return A->operator*(c); }

  friend shared_ptr<Tensor> operator+(const shared_ptr<Tensor>& a, const shared_ptr<Tensor>& b) {
    return (*a) + b;
  }
  
  friend shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& a, const shared_ptr<Tensor>& b) {
    return (*a) - b;
  }

  friend shared_ptr<Tensor> operator-(const shared_ptr<Tensor>& a) {
    return -(*a);
  }

  shared_ptr<Tensor> pow(float exp);
  shared_ptr<Tensor> to(string device);

  void print_data() const;
  void print_grad() const;
  // add more operations later

  void backward() {
    vector<shared_ptr<Tensor>> topo;
    set<shared_ptr<Tensor>> visited;
    
    function<void(shared_ptr<Tensor>)> build_topo = [&](shared_ptr<Tensor> v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    build_topo(self);
    
    this->grad = new Buffer(op_type->scalar_mul(data, 0.0f)->data, device, data->shape);
    this->grad->data[0] = 1.0f;
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
  }
};

class CPUOperation : public Operations {
public:
  Buffer *mul(Buffer *A, Buffer *B) override;
  Buffer *scalar_mul(Buffer *A, float c) override;
  Buffer *add(Buffer *A, Buffer *B) override;
  Buffer *subtract(Buffer *A, Buffer *B) override;
  Buffer *negate(Buffer *A) override;
  Buffer *pow(Buffer *A, float exp) override;
  Buffer *transpose(Buffer *data) override;
  float* move_data(float* data, size_t size) override;
  void free_memory(float* data) override;
  void print_buffer(Buffer *data) override;
};

class GPUOperation : public Operations {
public:
  float* move_data(float* data, size_t size) override;
  void free_memory(float* data) override;
  Buffer *mul(Buffer *A, Buffer *B) override;
  Buffer *scalar_mul(Buffer *A, float c) override;
  Buffer *add(Buffer *A, Buffer *B) override;
  Buffer *subtract(Buffer *A, Buffer *B) override;
  Buffer *negate(Buffer *A) override;
  Buffer *pow(Buffer *A, float exp) override;
  Buffer *transpose(Buffer *data) override;
  void print_buffer(Buffer *data) override;
};

#endif
