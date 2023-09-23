#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include <iostream>
#include "engine.h"

Value::Value(float data, std::unordered_set<std::shared_ptr<Value> > prev, std::string op) {
    this -> data = data;
    this -> prev = std::move(prev);
    this -> op = std::move(op);
    this -> grad = 0;

    // set backward to lambda expression that calls backward on each of its children`
    this -> _backward = [this] {
        for(const auto& child : this -> prev) {
            child -> _backward();
        }
    };
};

void Value::print() const {
    std::cout << "grad: " << this -> grad << "\n" << "data: " <<  this -> data;
}

float Value::get_data() {
    return this -> data;
};

void Value::set_data(float data) {
    this -> data = data;
};

float Value::get_grad() const {
    return this -> grad;
};

void Value::set_grad(float grad) {
    this -> grad = grad;
};

// defining addition
// all operations done with pointers to values
// turn pointer to new value
// auto a = std::make_shared<Value>(7);
// auto b = std::make_shared<Value>(3);
// auto c = *a + b
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {

    auto _prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(data + other -> data, _prev, "+");

    //defining backward function of new value 
    out -> _backward = [this, out, other] {
        this -> grad += 1.0 * out -> grad;
        other -> grad += 1.0 * out -> grad;
    };

    return out;
}

//defines addition between pointers instead of having to dereference them
// auto a = std::make_shared<Value>(7);
// auto b = std::make_shared<Value>(3);
// auto c = a + b
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}


std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {

    auto _prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(data * other -> data, _prev, "*");

    out -> _backward = [this, out, other] {
        this -> grad += this -> data * out -> grad;
        other -> grad += other -> data *  out -> grad;
    };

    return out;
};

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {

    auto _prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(data - other -> data, _prev, "-");

    out -> _backward = [this, out, other] {
        this -> grad += 1.0 * out -> grad;
        other -> grad += 1.0 * out -> grad;
    };

    return out;
};

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) - rhs;
}

std::shared_ptr<Value> Value::operator-() {
    return shared_from_this() * std::make_shared<Value>(-1);
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    return *this * std::make_shared<Value>(1.0 / other -> data);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) / rhs;
}

