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
    auto n = Neuron(2, false);
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1), std::make_shared<Value>(2)};
    auto y = n(x);
    y -> print();
    EXPECT_EQ(0, 0);
}

int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
