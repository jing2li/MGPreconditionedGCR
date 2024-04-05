cmake_minimum_required(VERSION 3.27)
project(MGPreconditionedGCR LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "-O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Set the path to the Eigen library
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0)

add_executable(MGPreconditionedGCR main.cpp
        GCR.h
        Fields.h
        MG.cpp
        MG.h
        utils.h
        utils.cpp
        Mesh.h
        EigenSolver.cpp
        EigenSolver.h
        Parse.cpp
        Parse.h
        Operator.h
)
