CC=g++
CFLAGS=-g -std=c++17
NVCC=nvcc

ifeq ($(OS),Windows_NT)
    CFLAGS += -DCURL_STATICLIB -IC:\msys64\curl\include\curl -LC:\msys64\curl\lib\libcurl.dll.a -lcurl
else
    CFLAGS += -lcurl
endif

main: tensor.o main.o
	$(CC) $(CFLAGS) main.o tensor.o -o main 
	./main

tensor.o: tensor.cpp tensor.h
	$(CC) $(CFLAGS) -c tensor.cpp 

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp 

cuda: nn.cu engine.o nn.o
	$(NVCC) -w -std=c++17 engine.o nn.o nn.cu -o main
	./main

test: engine.o test.o nn.o
	$(CC) $(CFLAGS) -lgtest nn.o test.o engine.o -o testing
	./testing

mnist: engine.o nn.o mnist.o
	$(CC) $(CFLAGS) nn.o engine.o mnist.o -o mnist -lcurl

mnist.o: mnist.cpp
	$(CC) $(CFLAGS) -c mnist.cpp

test.o: ./test/test.cpp
	$(CC) $(CFLAGS) -c -lgtest ./test/test.cpp

engine.o: engine.cpp engine.h
	$(CC) $(CFLAGS) -c engine.cpp 

nn.o: nn.cpp nn.h
	$(CC) $(CFLAGS) -c nn.cpp

clean:
	rm ./*.o

