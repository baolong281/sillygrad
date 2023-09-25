#include "engine.h"
#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include "engine.h"
#include <vector>
#include <random>
#include "nn.h"


Neuron::Neuron(int nin, bool activation) {
    this -> _activation = activation;
    auto std = sqrt(2/nin);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for(int i = 0; i < nin; i++) {
        this -> w.push_back(std::make_shared<Value>(std * dis(gen)));
    }
    this -> b = std::make_shared<Value>(std * dis(gen));
}

std::shared_ptr<Value> Neuron::operator()(std::vector<std::shared_ptr<Value>>& x) {
    auto act = std::make_shared<Value>(0);
    for(int i = 0; i < x.size(); i++) {
        act = act + x[i] * this -> w[i];
    }
    if(this -> _activation=="relu") {
        return relu(act + this -> b);
    } else if (this -> _activation=="leaky_relu") {
        return leaky_relu(act + this -> b); 
    } else{
        return act + this -> b;
    }
}
