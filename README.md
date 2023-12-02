# sillygrad

torch-like deep learning framework written in c++ and accelerated with cuda based off of [micrograd](https://github.com/karpathy/micrograd) and
[c++ micrograd](https://github.com/10-zin/cpp-micrograd).

### requirements

- libcurl
- gzip

### running mnist

```
make mnist
./mnist --cuda
```

to build and run mnist with cuda

### running tests

make sure to have gtest installed

```
make test
```

https://curl.se/docs/faq.html#Link_errors_when_building_libcur
