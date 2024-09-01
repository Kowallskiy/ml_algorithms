cmake_minimum_required(VERSION 3.15)
project(my_project)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(pybind11 REQUIRED)

pybind11_add_module(my_project bindings.cpp LinearRegression.cpp DecisionTree.cpp KMeans.cpp LogisticRegression.cpp KNN.cpp)