#ifndef NN_H
#define NN_H

#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include "engine.h"
#include <vector>

using namespace std;

class Neuron {
    private:
        // vector<shared_ptr<Value>> w;
        shared_ptr<Value> b;
		shared_ptr<shared_ptr<Value>[]> w;
        string _activation;
		int nin;

    public:
        Neuron(int nin, string activation={});
        shared_ptr<Value> operator()(vector<shared_ptr<Value>>& x);
        vector<shared_ptr<Value>> parameters();
        void zero_grad();
		Neuron* to_cuda();
		void set_act(string* activation);
		void set_w(shared_ptr<shared_ptr<Value>[]> w);
		void set_b(shared_ptr<Value>* b);
		string get_activation();
		int get_nin();
};

class Layer {
    private:
        vector<Neuron> neurons;
        char* _activation;
    public:
        Layer(int nin, int nout, char* activation);
        vector<shared_ptr<Value>> operator()(vector<shared_ptr<Value>>& x);
        vector<shared_ptr<Value>> parameters();
        void zero_grad();
		Layer to_cuda();
};

class MLP {
    private:
        vector<Layer> layers;
    public:
        MLP(vector<int> sizes, char* activation);
        vector<shared_ptr<Value>> operator()(vector<shared_ptr<Value>>& x);
        vector<shared_ptr<Value>> parameters();
        void zero_grad();
        void step(float lr);
		MLP* to_cuda();
};

vector<shared_ptr<Value>> softmax(vector<shared_ptr<Value>>& x);
shared_ptr<Value> cross_entropy(vector<shared_ptr<Value>>& x, vector<std::shared_ptr<Value>>& y);
vector<shared_ptr<Value>> layer_norm(vector<shared_ptr<Value>>& x);

#endif
