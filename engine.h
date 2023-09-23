#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <unordered_set>
#include <memory>
#include <functional>

struct Value : std::enable_shared_from_this<Value> {
    private:
        float data;
        float grad;
        std::function<void()> _backward;
        std::unordered_set<std::shared_ptr<Value> > _prev; 
        std::string op;

    Value(float data, std::unordered_set<std::shared_ptr<Value> > prev, std::string op);

    void backward();
    
    void set_grad(float grad_val);
    void set_data(float data_val);
    float get_data();
    float get_grad();

    // defining operations
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);

    //negative sign
    std::shared_ptr<Value> operator-();

    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);

    std::shared_ptr<Value> pow(const std::shared_ptr<Value>& other);

#endif






};