#include "tensor.h"
#include <sys/types.h>
#include <vector>

using namespace std;

int main() {
  auto A = Tensor::create(new vector<vector<float>>{{1, 2}, {3, 4}});

  A->print_data();
  
  auto B = 4.0 * A;
  B->print_data();
  
  auto C = A + B;
  C->print_data();
  
  auto D = A - B;
  D->print_data();
  
  auto E = -A;
  E->print_data();
  
  auto F = A->pow(2);
  F->print_data();

  auto G = A->to("gpu");
  // G->print_data();
}
