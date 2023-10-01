#include <gtest/gtest.h>
#include "../engine.h"
#include <iostream>
#include <functional>
#include <memory>
#include "../nn.h"

TEST(Addition, AdditionCorrect) {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(3);
    auto c = a + b;
    EXPECT_EQ(5, c -> get_data());
}

TEST(Subtraction, SubtractionCorrect) {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(3);
    auto c = a - b;
    EXPECT_EQ(-1, c -> get_data());
}

TEST(Multiply, MultiplyCorrect1) {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(3);
    auto c = a * b;
    EXPECT_EQ(6, c -> get_data());
}

TEST(Multiply, MultiplyCorrect2) {
    auto a = std::make_shared<Value>(2);
    auto b = std::make_shared<Value>(-3);
    auto c = a * b;
    EXPECT_EQ(-6, c -> get_data());
}

TEST(Division, DivisionCorrect1) {
    auto a = std::make_shared<Value>(6);
    auto b = std::make_shared<Value>(3);
    auto c = a / b;
    EXPECT_EQ(2, c -> get_data());
}

TEST(Division, DivisionCorrect2) {
    auto a = std::make_shared<Value>(6);
    auto b = std::make_shared<Value>(-3);
    auto c = a /  b;
    EXPECT_EQ(-2, c -> get_data());
}

TEST(Exponent, ExponentCorrect) {
    auto a = std::make_shared<Value>(6);
    auto c = exp(a);
    EXPECT_FLOAT_EQ(c -> get_data(), std::exp(6));
}

TEST(Grad, Gradient1) {
    auto a = std::make_shared<Value>(6);
    auto b = std::make_shared<Value>(-3);
    auto c = a / b;
    c -> backward();
    EXPECT_EQ(c -> get_grad(), 1);
    EXPECT_EQ(b -> get_grad(), 1/-3);
    EXPECT_EQ(a -> get_grad(), 6);
}

TEST(Grad, Gradient2) {
    auto a = std::make_shared<Value>(6);
    auto b = std::make_shared<Value>(-3);
    auto c = a / b;
    c -> backward();
    EXPECT_EQ(c -> get_grad(), 1);
    EXPECT_EQ(b -> get_grad(), 1/-3);
    EXPECT_EQ(a -> get_grad(), 6);
}

TEST(Neuron, Neuron1) {
    auto n = Neuron(2, "relu");
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2)};
    auto y = n(x);
    EXPECT_EQ(0, 0);
}

TEST(Neuron, GetParams) {
    auto n = Neuron(5, "relu");
    EXPECT_EQ(n.parameters().size(), 6);
}

TEST(Layer, Parameters) {
    auto layer = Layer(2, 3, "relu");
    EXPECT_EQ(layer.parameters().size(), 9);
}

TEST(Layer, Init1) {
    auto layer = Layer(4, 5, "relu");
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2), std::make_shared<Value>(3), std::make_shared<Value>(4)};
    auto y = layer(x);
    EXPECT_EQ(y.size(), 5);
}

TEST(MLP, Parameters) {
    auto network = MLP({2, 3, 4}, "relu");
    EXPECT_EQ(network.parameters().size(), 25);
}

TEST(MLP, Init1) {
    auto network = MLP({2, 3, 4}, "relu");
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2)};
    auto y = network(x);
    EXPECT_EQ(y.size(), 4);
}

TEST(Softmax, SoftmaxSum) {
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2), std::make_shared<Value>(3)};
    auto y = softmax(x);
    float sum = 0;
    for(auto& val: y) {
        sum += val -> get_data();
    }
    EXPECT_FLOAT_EQ(sum, 1);
}

TEST(Step, ChangeData) {
    auto network = MLP({2, 3, 4}, "relu");
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2)};
    auto params = network.parameters();
    for(auto& val: params) {
        val -> set_data(1);
        val -> set_grad(1);
    }
    network.step(0.1);
    for(auto& val: params) {
        EXPECT_FLOAT_EQ(val -> get_data(), 0.9);
    }
}

TEST(Step, CorrectGrad) {
    auto network = MLP({2, 3, 4}, "relu");
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2)};
    auto params = network.parameters();
    for(auto& val: params) {
        val -> set_data(1);
        val -> set_grad(0.5);
    }
    network.step(0.1);
    for(auto& val: params) {
        EXPECT_FLOAT_EQ(val -> get_data(), 0.95);
    }
}

TEST(CrossEntropy, CorrectEntropy) {
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(0.1), std::make_shared<Value>(0.2), std::make_shared<Value>(0.7)};
    auto y = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(0), std::make_shared<Value>(0), std::make_shared<Value>(1)};
    auto loss = cross_entropy(x, y);
    EXPECT_FLOAT_EQ(loss -> get_data(), -std::log(0.7));
}


int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
