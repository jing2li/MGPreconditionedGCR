cmake_minimum_required(VERSION 3.27)
project(MGPreconditionedGCR)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_HOST_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
#set(CMAKE_CUDA_FLAGS "-Wall --allow-unsupported-compiler")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

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
)
