cmake_minimum_required(VERSION 3.10)
project(adder_trees)

set(CMAKE_CXX_STANDARD 11)

add_executable(adder_trees main.cpp)

add_executable(dead_zone_uniform_quantization dead_zone_uniform_quantization.cpp)
add_executable(dead_zone_uniform_quantization_layer dead_zone_uniform_quantization_layer.cpp)
add_executable(pareto_condition pareto_condition.cpp)