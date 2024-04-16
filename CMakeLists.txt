cmake_minimum_required(VERSION 3.6)

project(Lab1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(MPI MPI.cpp)

target_link_libraries(MPI PRIVATE MPI::MPI_CXX)

add_executable(OpenMP OpenMP.cpp)
