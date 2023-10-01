#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <unordered_set>
#include <memory>
#include <functional>

class Value : public std::enable_shared_from_this<Value> {
private:
    float data;
    float grad;
    std::function<void()> _backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;

public:
    Value(float data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = "");

    void set_grad(float grad_value);
    float get_data();
    void set_data(float data);
    float get_grad() const;
    void print() const;
    std::unordered_set<std::shared_ptr<Value>> get_prev() const;

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-();
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> pow(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> log(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> log();
    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> leaky_relu();
    std::shared_ptr<Value> exp();

    void backward();
};

//define operations between two value pointers 
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> log(const std::shared_ptr<Value>& lhs);
std::shared_ptr<Value> relu(const std::shared_ptr<Value>& lhs);
std::shared_ptr<Value> leaky_relu(const std::shared_ptr<Value>& lhs);
std::shared_ptr<Value> exp(const std::shared_ptr<Value>& lhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs);

#endif
