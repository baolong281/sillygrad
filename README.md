# sillygrad

torch-like tensor library written in c++ with support for acceleration \
inspired by [micrograd](https://github.com/karpathy/micrograd) and 
[c++ micrograd](https://github.com/10-zin/cpp-micrograd).

## running main
```
// compile with cuda
make main --CUDA_ENABLED=1 
``` 

# todo:
- rewriting to support tensors
- tensor broadcasting and shapes
- writing cuda kernels
- fixing memory leaks
- write nn module 
