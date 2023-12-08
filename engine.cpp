#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include <iostream>
#include <cmath>
#include <vector>
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
    std::cout << "grad: " << this -> grad << " data: " <<  this -> data << std::endl;
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
        other -> grad += other -> data * out -> grad;
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

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs) {
    return -(*lhs);
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

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
    auto _prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(std::pow(data, other -> data), _prev, "pow");

    out -> _backward = [this, out, other] {
        this -> grad += other -> data * std::pow(this -> data, other -> data - 1) * out -> grad;
        other -> grad += std::pow(this -> data, other -> data) * std::log(this -> data) * out -> grad;
    };

    return out;
}

std::unordered_set<std::shared_ptr<Value>> Value::get_prev() const {
    return this -> prev;
}

void build_topo(std::shared_ptr<Value> v, std::unordered_set<std::shared_ptr<Value>>& visited, std::vector<std::shared_ptr<Value>>& topo) {
   if(visited.find(v) == visited.end()) {
        visited.insert(v);
        for(const auto& child : v -> get_prev()) {
            build_topo(child, visited, topo);
        }
        topo.push_back(v);
   }
}

void Value::backward() {
    auto topo = std::vector<std::shared_ptr<Value>>{};
    auto visited = std::unordered_set<std::shared_ptr<Value>>{};
	topo.reserve(4000);

    build_topo(shared_from_this(), visited, topo);

    this -> grad = 1.0;
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& v = *it;
        v->_backward();
    }
}


std::shared_ptr<Value> Value::log() {
    auto out = std::make_shared<Value>(std::log(data), std::unordered_set<std::shared_ptr<Value>>{shared_from_this()}, "log");

    auto ln10 = 2.30258509299;

    out -> _backward = [this, out, ln10] {
        this -> grad += 1.0 / (this -> data * ln10) * out -> grad;
    };

    return out;
}

std::shared_ptr<Value> log(const std::shared_ptr<Value>& lhs) {
    return lhs -> log();
}

std::shared_ptr<Value> Value::relu() {
    auto new_data = std::max(0.0f, data);

    auto out = std::make_shared<Value>(new_data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this()}, "relu");

    out -> _backward = [this, out] {
        this -> grad += (this -> data > 0) * out -> grad;
    };

    return out;
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value>& lhs) {
    return lhs -> relu();
}

std::shared_ptr<Value> Value::leaky_relu() {
    auto c = .01;
    auto new_data = std::max((float)c*data, data);
    auto out = std::make_shared<Value>(new_data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this()}, "relu");

    out -> _backward = [this, out, c] {
        if(this -> data > 0) {
            this -> grad += 1 * out -> grad;
        } else {
            this -> grad += 1 * out -> grad * c;
        }
    };

    return out;
}

std::shared_ptr<Value> leaky_relu(const std::shared_ptr<Value>& lhs) {
    return lhs -> leaky_relu();
}

std::shared_ptr<Value> Value::exp() {
    auto out = std::make_shared<Value>(std::exp(data), std::unordered_set<std::shared_ptr<Value>>{shared_from_this()}, "exp");

    this -> _backward = [this, out] {
        this -> grad += out -> data * out -> grad;
    };

    return out;
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value>& lhs) {
    return lhs -> exp();
}
