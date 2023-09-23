#include <string>
#include <unordered_set>
#include <memory>
#include <functional>

#include "engine.h"

Value::Value(float data, std::unordered_set<std::shared_ptr<Value> > prev, std::string op) {
    this -> data = data;
    this -> _prev = std::move(prev);
    this -> op = std::move(op);
    this -> grad = 0;

    // set backward to lambda expression that calls backward on each of its children`
    this -> _backward = [this] {
        for(const auto& child : this -> _prev) {
            child -> _backward();
        }
    };
};

float Value::get_data() {
    return this -> data;
};

void Value::set_data(float data) {
    this -> data = data;
};

float Value::get_grad() {
    return this -> grad;
};

void Value::set_grad(float grad) {
    this -> grad = grad;
};


std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    printf("BOOBS");
};





