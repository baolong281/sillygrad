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
        Neuron(int nin, bool activation);
        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x);
};

#endif
