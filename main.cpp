#include "tensor.h"
#include <sys/types.h>
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

    auto B = Tensor(new vector<vector<float>>{{1, 2}, {3, 4}}, "gpu");
    // B.print_data();

    // D.print_data();
}