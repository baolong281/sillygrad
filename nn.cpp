#include "engine.h"
#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include "engine.h"
#include <vector>
#include <random>
#include "nn.h"


Neuron::Neuron(int nin, std::string activation) {
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

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    auto params = std::vector<std::shared_ptr<Value>>{};
    params.reserve(this -> w.size() + 1);
    params.push_back(b);
    for(auto& weight: w) {
        params.push_back(weight);
    }
    return params;
}


Layer::Layer(int nin, int nout, std::string activation) {
    this -> _activation = activation;
    for(int i = 0; i < nout; i++) {
        this -> neurons.push_back(Neuron(nin, activation));
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out;
    out.reserve(neurons.size()+1);
    for (auto& neuron: neurons){
        out.emplace_back(neuron(x));
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    auto params = std::vector<std::shared_ptr<Value>>{};
    params.reserve(neurons.size() * (neurons[0].parameters().size()));
    for(auto& neuron: neurons) {
        auto neuron_params = neuron.parameters();
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
}

MLP::MLP(std::vector<int> sizes, std::string activation) {
    for(int i = 0; i < sizes.size() - 1; i++) {
        if(i!=sizes.size()-2) {
            this -> layers.push_back(Layer(sizes[i], sizes[i+1], activation));
        } else {
            this -> layers.push_back(Layer(sizes[i], sizes[i+1], ""));
        }
    }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out = x;
    for(auto& layer: layers) {
        out = layer(out);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    auto params = std::vector<std::shared_ptr<Value>>{};
    for(auto& layer: layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}


