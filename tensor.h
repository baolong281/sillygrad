#ifndef TENSOR_H
#define TENSOR_H

#include <functional>
#include <set>
#include <vector>
using namespace std;

enum Device {
	CPU,
	GPU
};

class Tensor {
	private:
		Device device;
		vector<vector<float>> data;
		function<void()> _backward;
		set<Tensor*> prev;
		vector<int> shape;

	public:
		Tensor* to(Device device);
		Tensor mul(const Tensor& other);
		Tensor scalar_mul(float other);
		Tensor operator+(const Tensor& other);
		Tensor operator-(const Tensor& other);
		Tensor operator-();
		Tensor pow(float exp);

		// add more operations later
};

class Operation {
	virtual Tensor mul(const Tensor& A, const Tensor& B);
	virtual Tensor scalar_mul(const Tensor& A, float other);
	virtual Tensor add(const Tensor& A, const Tensor& B);
	virtual Tensor subtract(const Tensor& A, const Tensor& B);
	virtual Tensor negate(const Tensor& A);
	virtual Tensor pow(float exp);
};

class CPUOperation : public Operation {
	Tensor mul(const Tensor& A, const Tensor& B);
	Tensor scalar_mul(const Tensor& A, float other);
	Tensor add(const Tensor& A, const Tensor& B);
	Tensor subtract(const Tensor& A, const Tensor& B);
	Tensor negate(const Tensor& A);
	Tensor pow(float exp);
};

class GPUOperation : public Operation {
	Tensor mul(const Tensor& A, const Tensor& B);
	Tensor scalar_mul(const Tensor& A, float other);
	Tensor add(const Tensor& A, const Tensor& B);
	Tensor subtract(const Tensor& A, const Tensor& B);
	Tensor negate(const Tensor& A);
	Tensor pow(float exp);
};

#endif