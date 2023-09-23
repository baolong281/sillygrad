cc=clang++
CFLAGS=-g -std=c++17

# main: engine.o main.o
# 	$(CC) $(CFLAGS) main.o engine.o -o main
#
# main.o: test/main.cpp
# 	$(CC) $(CFLAGS) -c test/main.cpp
#
# engine.o: engine.cpp engine.h
# 	$(CC) $(CFLAGS) -c engine.cpp 
#
# clean:
# 	rm *.o main

main:
	clang++ $(CFLAGS) ./test/main.cpp -o main
	./main
