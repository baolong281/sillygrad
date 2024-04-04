cmake_minimum_required(VERSION 3.18)

project(sillygrad VERSION 0.1.0
                  DESCRIPTION "c++ tensor library"
                  LANGUAGES CXX CUDA)

option(CUDA_ENABLED "Enable CUDA" OFF)

add_library(tensor_cpu SHARED tensor.cpp ops.cpp tensor.h)

if(CUDA_ENABLED)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    add_library(tensor_cuda SHARED tensor.cpp ops.cu tensor.h)
endif()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensor_cpu)

if(CUDA_ENABLED)
    target_link_libraries(main PRIVATE tensor_cuda)
endif()