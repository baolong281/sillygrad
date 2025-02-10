cmake_minimum_required(VERSION 3.18)

project(sillygrad VERSION 0.1.0
                  DESCRIPTION "c++ tensor library"
                  LANGUAGES CXX)  # Removed CUDA from LANGUAGES

# Set the C++ standard to C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_ENABLED "Enable CUDA" OFF)

# Enable testing
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_library(tensor_cpu SHARED tensor.cpp ops.cpp tensor.h)

if(CUDA_ENABLED)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_library(tensor_cuda SHARED tensor.cpp ops.cu tensor.h)
    target_compile_definitions(tensor_cuda PRIVATE CUDA_ENABLED)
endif()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensor_cpu)

if(CUDA_ENABLED)
    target_link_libraries(main PRIVATE tensor_cuda)
endif()

# Add test executable
add_executable(tensor_tests tests/tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE
    tensor_cpu
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_tests)
