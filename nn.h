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
        std::vector<std::shared_ptr<Value>> w;
        std::shared_ptr<Value> b;
        char* _activation;
		int nin;

    public:
        Neuron(int nin, char* activation={});
        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
		Neuron to_cuda();
		void set_act(char** activation);
		void set_w(std::vector<std::shared_ptr<Value>>* w);
		void set_b(std::shared_ptr<Value>* b);
		std::string get_activation();
		int get_nin();
};

class Layer {
    private:
        std::vector<Neuron> neurons;
        char* _activation;
    public:
        Layer(int nin, int nout, char* activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
		Layer to_cuda();
};

class MLP {
    private:
        std::vector<Layer> layers;
    public:
        MLP(std::vector<int> sizes, char* activation);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters();
        void zero_grad();
        void step(float lr);
		MLP to_cuda();
};

std::vector<std::shared_ptr<Value>> softmax(std::vector<std::shared_ptr<Value>>& x);
std::shared_ptr<Value> cross_entropy(std::vector<std::shared_ptr<Value>>& x, std::vector<std::shared_ptr<Value>>& y);
std::vector<std::shared_ptr<Value>> layer_norm(std::vector<std::shared_ptr<Value>>& x);

#endif
