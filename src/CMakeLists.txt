cmake_minimum_required(VERSION 3.27)
project(MGPreconditionedGCR LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "-O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# Set the path to the Eigen library
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0)

add_executable(MGPreconditionedGCR main.cpp
        GCR.cpp
        GCR.h
        Fields.cpp
        Fields.h
        MG.cpp
        MG.h
        utils.h
        utils.cpp
        Mesh.cpp
        Mesh.h
        Operator.cpp
        Operator.h
        EigenSolver.cpp
        EigenSolver.h
        Parse.cpp
        Sparse.cpp
        Sparse.h
        Parse.h
)
