#ifndef NN_H
#define NN_H

#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include "engine.h"
#include <vector>

class Neuron {
    private:
        std::string _activation;
        std::vector<std::shared_ptr<Value>> w;
        std::shared_ptr<Value> b;

    public:
        Neuron(int nin, std::string activation="");
        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
};

class Layer {
    private:
        std::vector<Neuron> neurons;
        std::string _activation;
    public:
        Layer(int nin, int nout, std::string activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
};

class MLP {
    private:
        std::vector<Layer> layers;
    public:
        MLP(std::vector<int> sizes, std::string activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
        void step(float lr);
};

std::vector<std::shared_ptr<Value>> softmax(std::vector<std::shared_ptr<Value>>& x);
std::shared_ptr<Value> cross_entropy(std::vector<std::shared_ptr<Value>>& x, std::vector<std::shared_ptr<Value>>& y);
std::vector<std::shared_ptr<Value>> layer_norm(std::vector<std::shared_ptr<Value>>& x);

#endif
