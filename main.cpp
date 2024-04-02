#include "tensor.h"
#include <vector>

using namespace std;

int main() {
	auto buff = new Buffer(new vector<float>{1, 2, 3, 4}, "cpu", {4});
	auto A = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}});


	A.print_data();
	(4 * A).print_data();
	(A + A).print_data();
	(A - A).print_data();
	(-A).print_data();
	// E.print_data();

	// auto D = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}}, "gpu");
	// auto E = Tensor(new vector<vector<float>>{{5, 6}, {7, 8}}, "gpu");

	// D.print_data();

}