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

class Operations {
public:
  virtual Mat *mul(Mat *A, Mat *B) = 0;
  virtual Mat *scalar_mul(Mat *A, float c) = 0;
  virtual Mat *add(Mat *A, Mat *B) = 0;
  virtual Mat *subtract(Mat *A, Mat *B) = 0;
  virtual Mat *negate(Mat *A) = 0;
  virtual Mat *pow(Mat *A, float exp) = 0;
  virtual Mat *move_data(Mat *data) = 0;
  virtual Mat *transpose(Mat *A) = 0;
  virtual void free_memory(Mat *data) = 0;
  virtual ~Operations() = default;
};

class Tensor {
private:
  string device;
  function<void()> _backward;
  set<Tensor *> prev;
  vector<size_t> shape;
  Operations *op_type;
  bool requires_grad;

public:
  Mat *data;
  Mat *grad;
  Tensor(vector<vector<float>> *data, string device = "cpu",
         bool requires_grad = true);
  ~Tensor() {
    op_type->free_memory(data);
    op_type->free_memory(grad);
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
  vector<size_t> const size() { return shape; }
  // add more operations later
};

class CPUOperation : public Operations {
	Mat *mul(Mat *A, Mat *B) override;
  	Mat *scalar_mul(Mat *A, float c) override;
  	Mat *add(Mat *A, Mat *B) override;
  	Mat *subtract(Mat *A, Mat *B) override;
  	Mat *negate(Mat *A) override;
  	Mat *pow(Mat *A, float exp) override;
  	Mat *move_data(Mat *data) override;
  	Mat *transpose(Mat *data) override;
  	void free_memory(Mat *data) override;
};

class GPUOperation : public Operations {
  	Mat *mul(Mat *A, Mat *B) override;
  	Mat *scalar_mul(Mat *A, float c) override;
  	Mat *add(Mat *A, Mat *B) override;
  	Mat *subtract(Mat *A, Mat *B) override;
  	Mat *negate(Mat *A) override;
  	Mat *pow(Mat *A, float exp) override;
  	Mat *transpose(Mat *data) override;
  	Mat *move_data(Mat *data) override;
  	void free_memory(Mat *data) override;
};

#endif
