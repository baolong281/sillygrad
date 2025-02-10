#include <gtest/gtest.h>
#include "../tensor.h"
#include <vector>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up code that will be called before each test
    }

    void TearDown() override {
        // Clean up code that will be called after each test
    }
};

// Test tensor creation
TEST_F(TensorTest, CreateTensor) {
    auto tensor = Tensor::create(new std::vector<std::vector<float>>{{1, 2}, {3, 4}});
    
    EXPECT_EQ(tensor->data->shape[0], 2);
    EXPECT_EQ(tensor->data->shape[1], 2);
}

// Test scalar multiplication
TEST_F(TensorTest, ScalarMultiplication) {
    auto tensor = Tensor::create(new std::vector<std::vector<float>>{{1, 2}, {3, 4}});
    auto result = 2.0f * tensor;
    
    float expected[4] = {2, 4, 6, 8};
    for (size_t i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result->data->data[i], expected[i]);
    }
}

// Test tensor addition
TEST_F(TensorTest, TensorAddition) {
    auto tensor1 = Tensor::create(new std::vector<std::vector<float>>{{1, 2}, {3, 4}});
    auto tensor2 = Tensor::create(new std::vector<std::vector<float>>{{5, 6}, {7, 8}});
    
    auto result = tensor1 + tensor2;
    
    float expected[4] = {6, 8, 10, 12};
    for (size_t i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result->data->data[i], expected[i]);
    }
}

// Test tensor subtraction
TEST_F(TensorTest, TensorSubtraction) {
    auto tensor1 = Tensor::create(new std::vector<std::vector<float>>{{5, 6}, {7, 8}});
    auto tensor2 = Tensor::create(new std::vector<std::vector<float>>{{1, 2}, {3, 4}});
    
    auto result = tensor1 - tensor2;
    
    float expected[4] = {4, 4, 4, 4};
    for (size_t i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result->data->data[i], expected[i]);
    }
}

// Test tensor negation
TEST_F(TensorTest, TensorNegation) {
    auto tensor = Tensor::create(new std::vector<std::vector<float>>{{1, 2}, {3, 4}});
    auto result = -tensor;
    
    float expected[4] = {-1, -2, -3, -4};
    for (size_t i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result->data->data[i], expected[i]);
    }
}

TEST_F(TensorTest, ScalarMultiplicationBackward) {
    auto x = Tensor::create(new std::vector<std::vector<float>>{{2.0f}});
    auto y = 3.0f * x;  // y = 3x
    y->backward();
    
    // dy/dx = 3
    EXPECT_FLOAT_EQ(x->grad->data[0], 3.0f);
}

TEST_F(TensorTest, AdditionBackward) {
    auto x = Tensor::create(new std::vector<std::vector<float>>{{2.0f}});
    auto y = Tensor::create(new std::vector<std::vector<float>>{{3.0f}});
    
    auto z = x + y;  // z = x + y
    z->backward();
    
    // dz/dx = 1, dz/dy = 1
    EXPECT_FLOAT_EQ(x->grad->data[0], 1.0f);
    EXPECT_FLOAT_EQ(y->grad->data[0], 1.0f);
}

TEST_F(TensorTest, SubtractionBackward) {
    auto x = Tensor::create(new std::vector<std::vector<float>>{{2.0f}});
    auto y = Tensor::create(new std::vector<std::vector<float>>{{3.0f}});
    
    auto z = x - y;  // z = x - y
    z->backward();
    
    // dz/dx = 1, dz/dy = -1
    EXPECT_FLOAT_EQ(x->grad->data[0], 1.0f);
    EXPECT_FLOAT_EQ(y->grad->data[0], -1.0f);
}

TEST_F(TensorTest, NegationBackward) {
    auto x = Tensor::create(new std::vector<std::vector<float>>{{2.0f}});
    auto y = -x;  // y = -x
    y->backward();
    
    // dy/dx = -1
    EXPECT_FLOAT_EQ(x->grad->data[0], -1.0f);
}

TEST_F(TensorTest, CompositeBackward) {
    auto x = Tensor::create(new std::vector<std::vector<float>>{{2.0f}});
    auto y = Tensor::create(new std::vector<std::vector<float>>{{3.0f}});
    
    // z = 2x + y
    auto z = (2.0f * x) + y;
    z->backward();
    
    // dz/dx = 2, dz/dy = 1
    EXPECT_FLOAT_EQ(x->grad->data[0], 2.0f);
    EXPECT_FLOAT_EQ(y->grad->data[0], 1.0f);
} 