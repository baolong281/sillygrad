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
        
};

class Layer {
    private:
        std::vector<Neuron> neurons;
        std::string _activation;
    public:
        Layer(int nin, int nout, std::string activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
};

class MLP {
    private:
        std::vector<Layer> layers;
    public:
        MLP(std::vector<int> sizes, std::string activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
};

std::vector<std::shared_ptr<Value>> softmax(std::vector<std::shared_ptr<Value>>& x);

#endif
