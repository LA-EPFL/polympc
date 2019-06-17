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

add_executable(sqp_bench sqp_bench.cpp)
target_link_libraries(sqp_bench Eigen3::Eigen)

option(CASADI_AVAILABLE "indicate if CasADi is available" OFF)

if(CASADI_AVAILABLE)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    find_package(CASADI REQUIRED)
    include_directories( ${CASADI_INCLUDE_DIR} )
    if(NOT CASADI_FOUND)
        message(WARNING "CasADi not found")
    endif()

    add_executable(autodiff_bench autodiff_bench.cpp)
    target_link_libraries(autodiff_bench Eigen3::Eigen ${CASADI_LIBRARIES})
endif(CASADI_AVAILABLE)
