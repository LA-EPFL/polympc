cmake_minimum_required(VERSION 3.0)
project(benchmark)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

include_directories(../src)

add_executable(qp_bench qp_bench.cpp)
target_link_libraries(qp_bench Eigen3::Eigen)
