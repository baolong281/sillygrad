#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
	auto A = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}});
	auto B = Tensor(new vector<vector<float>>{{5, 6}, {7, 8}});

	auto C = 4 * A;

	A.print_data();
	C.print_data();

	auto D = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}}, "gpu");
	auto E = Tensor(new vector<vector<float>>{{5, 6}, {7, 8}}, "gpu");

	D.print_data();

}