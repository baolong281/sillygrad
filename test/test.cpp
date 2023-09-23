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

int main(int argc, char* argv[]){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
