CXX=clang++
CXXFLAGS=-g -std=c++17
NVCC=nvcc

# Set the default value for the CUDA flag
CUDA_ENABLED ?= 0

# Set the NVCC and CUDA flags based on the CUDA_ENABLED value
ifeq ($(CUDA_ENABLED),1)
    NVCC = nvcc
    CUDA_FLAGS = -Xcompiler -fPIC
    CUDA_LIB = kernels.so
else
    NVCC =
    CUDA_FLAGS =
    CUDA_LIB =
endif

main: tensor.o main.o ops.o $(CUDA_LIB)
	$(CXX) $(CXXFLAGS) main.o tensor.o $(CUDA_LIB) ops.o -o main

kernels.so: ops.cu tensor.h
	$(NVCC) -shared -w -std=c++17 $(CUDA_FLAGS) ops.cu -o kernels.so

tensor.o: tensor.cpp tensor.h
	$(CXX) $(CXXFLAGS) -c tensor.cpp 

ops.o: ops.cpp tensor.h
	$(CXX) $(CXXFLAGS) -c ops.cpp 

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp 

cuda: nn.cu engine.o nn.o
	$(NVCXX) -w -std=c++17 engine.o nn.o nn.cu -o main
	./main

test: engine.o test.o nn.o
	$(CXX) $(CXXFLAGS) -lgtest nn.o test.o engine.o -o testing
	./testing

mnist: engine.o nn.o mnist.o
	$(CXX) $(CXXFLAGS) nn.o engine.o mnist.o -o mnist -lcurl

mnist.o: mnist.cpp
	$(CXX) $(CXXFLAGS) -c mnist.cpp

test.o: ./test/test.cpp
	$(CXX) $(CXXFLAGS) -c -lgtest ./test/test.cpp

engine.o: engine.cpp engine.h
	$(CXX) $(CXXFLAGS) -c engine.cpp 

nn.o: nn.cpp nn.h
	$(CXX) $(CXXFLAGS) -c nn.cpp

clean:
	rm ./*.o

