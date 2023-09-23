cc=clang++
CFLAGS=-g -std=c++17

main: engine.o main.o
	clang++ $(CFLAGS) main.o engine.o -o main
	./main

test: engine.o  test.o
	clang++ $(CFLAGS) -lgtest test.o engine.o -o testing
	./testing

test.o: ./test/test.cpp
	clang++ $(CFLAGS) -c -lgtest ./test/test.cpp

main.o: ./test/main.cpp
	clang++ $(CFLAGS) -c ./test/main.cpp

engine.o: engine.cpp engine.h
	clang++ $(CFLAGS) -c engine.cpp 

clean:
	rm *.o main

