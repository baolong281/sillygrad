cc=clang++
CFLAGS=-g -std=c++17

main: engine.o main.o nn.o
	clang++ $(CFLAGS) nn.o main.o engine.o -o main
	./main

test: engine.o test.o nn.o
	clang++ $(CFLAGS) -lgtest nn.o test.o engine.o -o testing
	./testing

mnist: engine.o nn.o mnist.o
	clang++ $(CFLAGS) nn.o engine.o mnist.o -o mnist

mnist.o: mnist.cpp
	clang++ $(CFLAGS) -c mnist.cpp

test.o: ./test/test.cpp
	clang++ $(CFLAGS) -c -lgtest ./test/test.cpp

main.o: ./test/main.cpp
	clang++ $(CFLAGS) -c ./test/main.cpp

engine.o: engine.cpp engine.h
	clang++ $(CFLAGS) -c engine.cpp 

nn.o: nn.cpp nn.h
	clang++ $(CFLAGS) -c nn.cpp

clean:
	rm *.o main

