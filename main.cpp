#include "tensor.h"

using namespace std;

int main() {
	auto A = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}});
	auto B = Tensor(new vector<vector<float>>{{5, 6}, {7, 8}});

    // auto C = A * 4.f;

	A.print_data();
    // C.print_data();

}