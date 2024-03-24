#include "tensor.h"
#include <cmath>
#include <vector>

void CPUOperation::free_memory(Mat* data) {
    delete data;
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
