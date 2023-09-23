#include "../engine.h"
#include <iostream>
#include <functional>
#include <memory>

using namespace std;

int main() {
    auto a = make_shared<Value>(75);
    auto b = make_shared<Value>(32);
    auto c = a + b;
    c -> print();
}



