#include <gtest/gtest.h>
#include "../engine.h"
#include <iostream>
#include <functional>
#include <memory>

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

int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
