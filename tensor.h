#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace std;

typedef vector<vector<float>> Mat;

class Tensor;

class Buffer {
public:
  vector<float>* data;
  string device;
  vector<size_t> shape;
  Buffer(vector<float>* data, string device, vector<size_t> shape);
  ~Buffer() {
	delete data;
  }
  Buffer* to(string device);
};

class Operations {
public:
  virtual Buffer* mul(Buffer* A, Buffer* B) = 0;
  virtual Buffer* scalar_mul(Buffer* A, float c) = 0;
  virtual Buffer* add(Buffer* A, Buffer* B) = 0;
  virtual Buffer* subtract(Buffer* A, Buffer* B) = 0;
  virtual Buffer* negate(Buffer* A) = 0;
  virtual Buffer* pow(Buffer* A, float exp) = 0;
  virtual vector<float>* move_data(vector<float>* data) = 0;
  virtual Buffer* transpose(Buffer* A) = 0;
  virtual void free_memory(vector<float>* data) = 0;
  virtual void print_buffer(Buffer* data) = 0;
  virtual ~Operations() = default;
};

class Tensor {
private:
  string device;
  function<void()> _backward;
  set<Tensor *> prev;
  Operations *op_type;
  bool requires_grad;

public:
  Buffer* data;
  Buffer* grad;
  Tensor(Buffer* data, string device = "cpu",
         bool requires_grad = true);
  Tensor(vector<vector<float>>* data, string device = "cpu",
         bool requires_grad = true);
  ~Tensor() {
	delete data;
	delete grad;
    delete op_type;
  }
  Tensor operator+(Tensor &other);
  Tensor operator-(Tensor &other);
  Tensor operator-();
  // scalar mul
  Tensor operator*(float c);

  friend Tensor operator*(float c, Tensor &A) {
    return A * c;
  }

  Tensor pow(float exp);
  Tensor *to(string device);

  void print_data() const;
  void print_grad() const;
  // add more operations later
};

class CPUOperation : public Operations {
	public:
		Buffer* mul(Buffer* A, Buffer* B) override;
		Buffer* scalar_mul(Buffer* A, float c) override;
		Buffer* add(Buffer* A, Buffer* B) override;
		Buffer* subtract(Buffer* A, Buffer* B) override;
		Buffer* negate(Buffer* A) override;
		Buffer* pow(Buffer* A, float exp) override;
		Buffer* transpose(Buffer* data) override;
		vector<float>* move_data(vector<float>* data) override;
		void free_memory(vector<float>* data) override;
		void print_buffer(Buffer* data) override;
};

class GPUOperation : public Operations {
	public:
		vector<float>* move_data(vector<float>* data) override;
		void free_memory(vector<float>* data) override;
		Buffer* mul(Buffer* A, Buffer* B) override;
		Buffer* scalar_mul(Buffer* A, float c) override;
		Buffer* add(Buffer* A, Buffer* B) override;
		Buffer* subtract(Buffer* A, Buffer* B) override;
		Buffer* negate(Buffer* A) override;
		Buffer* pow(Buffer* A, float exp) override;
		Buffer* transpose(Buffer* data) override;
		void print_buffer(Buffer* data) override;
};

#endif
