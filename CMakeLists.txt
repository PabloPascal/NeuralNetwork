cmake_minimum_required(VERSION 3.13 FATAL_ERROR)


project(neural_network)


add_executable(neural_network 
main.cpp
activation_function.hpp 
NeuralNetwork.hpp 
NeuralNetwork.cpp
tests.hpp
logger.hpp
utils.hpp

numerical/linalg.hpp
numerical/matrix_type.hpp
numerical/solver.hpp
)
