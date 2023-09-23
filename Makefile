cc=clang++
CFLAGS=-g -std=c++17

main: engine.o main.o
	clang++ $(CFLAGS) main.o engine.o -o main
	./main

test: engine.o ./test/test.cpp
	clang++ $(CFLAGS) -lgtest engine.o ./test/test.cpp -o testing
	./testing

main.o: ./test/main.cpp
	clang++ $(CFLAGS) -c ./test/main.cpp

engine.o: engine.cpp engine.h
	clang++ $(CFLAGS) -c engine.cpp 

clean:
	rm *.o main

