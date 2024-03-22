#include "tensor.h"

using namespace std;

int main() {
	auto A = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}});
	auto B = Tensor(new vector<vector<float>>{{5, 6}, {7, 8}});

	A.print_data();
	A.print_grad();

	auto C = A + B;

}