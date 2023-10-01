#include "engine.h"
#include <iostream>
#include <ostream>
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
    auto std = sqrt(2/(float)nin);
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

std::vector<std::shared_ptr<Value>> layer_norm(std::vector<std::shared_ptr<Value>>& x) {
    auto out = std::vector<std::shared_ptr<Value>>{};
    auto mean = .0f;
    auto std = 0.f;
    for(auto& val: x) {
        mean = mean + val -> get_data();
    }
    mean = mean / x.size();
    auto mean_val = std::make_shared<Value>(mean);
    for(auto& val: x) {
        auto ins = val -> get_data() - mean;
        std = std + pow(ins, 2);
    }
    std = sqrt(std / x.size());
    auto std_val = std::make_shared<Value>(std);
    for(auto& val: x) {
        out.push_back((val - mean_val) / std_val);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out = x;

    for(int i=0; i<layers.size(); i++) {
        if(i==layers.size()-1) {
            out = layers[i](out);
            break;
        }
        auto logits = layers[i](out);
        out = layer_norm(logits);
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

std::vector<std::shared_ptr<Value>> softmax(std::vector<std::shared_ptr<Value>>& x) {
    auto out = std::vector<std::shared_ptr<Value>>{};
    auto denom = std::make_shared<Value>(0);
    for(auto& val: x) {
        denom = denom + exp(val);
    }
    for(auto& val: x) {
        out.push_back(exp(val) / denom);
    }
    return out;
}

void Neuron::zero_grad() {
    for(auto& weight: w) {
        weight -> set_grad(0);
    }
    b -> set_grad(0);
}

void Layer::zero_grad() {
    for(auto& neuron: neurons) {
        neuron.zero_grad();
    }
}

void MLP::zero_grad() {
    for(auto& layer: layers) {
        layer.zero_grad();
    }
}

void MLP::step(float lr) {
    for(auto& val: this -> parameters()) {
        val -> set_data(val -> get_data() - lr * val -> get_grad());
    }
}

std::shared_ptr<Value> cross_entropy(std::vector<std::shared_ptr<Value>>& x, std::vector<std::shared_ptr<Value>>& y) {
    auto out = std::make_shared<Value>(0);
    for(int i = 0; i < x.size(); i++) {
        out = out + y[i] * log(x[i]);
    }
    return -out;
}

