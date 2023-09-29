CC=g++
CFLAGS=-g -std=c++17

main: engine.o main.o nn.o
	$(CC) $(CFLAGS) nn.o main.o engine.o -o main
	./main

test: engine.o test.o nn.o
	$(CC) $(CFLAGS) -lgtest nn.o test.o engine.o -o testing
	./testing

mnist: engine.o nn.o mnist.o
	$(CC) $(CFLAGS) -lcurl nn.o engine.o mnist.o -o mnist

mnist.o: mnist.cpp
	$(CC) $(CFLAGS) -c mnist.cpp

test.o: ./test/test.cpp
	$(CC) $(CFLAGS) -c -lgtest ./test/test.cpp

main.o: ./test/main.cpp
	$(CC) $(CFLAGS) -c ./test/main.cpp

engine.o: engine.cpp engine.h
	$(CC) $(CFLAGS) -c engine.cpp 

nn.o: nn.cpp nn.h
	$(CC) $(CFLAGS) -c nn.cpp

clean:
	rm *.o main

