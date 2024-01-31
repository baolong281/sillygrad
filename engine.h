#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <unordered_set>
#include <memory>
#include <functional>

using namespace std;

class Value : public enable_shared_from_this<Value> {
private:
    float data;
    float grad;
    function<void()> _backward;
    unordered_set<shared_ptr<Value>> prev;
    string op;

public:
    Value(float data = 0.f, unordered_set<shared_ptr<Value>> prev = {}, string op = "");
    void set_grad(float grad_value);
    float get_data();
    void set_data(float data);
    float get_grad() const;
    void print() const;
    unordered_set<shared_ptr<Value>> get_prev() const;

    shared_ptr<Value> operator+(const shared_ptr<Value>& other);
    shared_ptr<Value> operator-();
    shared_ptr<Value> operator-(const shared_ptr<Value>& other);
    shared_ptr<Value> pow(const shared_ptr<Value>& other);
    shared_ptr<Value> operator/(const shared_ptr<Value>& other);
    shared_ptr<Value> operator*(const shared_ptr<Value>& other);
    shared_ptr<Value> log(const shared_ptr<Value>& other);
    shared_ptr<Value> log();
    shared_ptr<Value> relu();
    shared_ptr<Value> leaky_relu();
    shared_ptr<Value> exp();

    void backward();
};

//define operations between two value pointers 
shared_ptr<Value> operator+(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> operator*(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> operator-(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> operator/(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> pow(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> log(const shared_ptr<Value>& lhs);
shared_ptr<Value> relu(const shared_ptr<Value>& lhs);
shared_ptr<Value> leaky_relu(const shared_ptr<Value>& lhs);
shared_ptr<Value> exp(const shared_ptr<Value>& lhs);
shared_ptr<Value> operator-(const shared_ptr<Value>& lhs);

#endif
