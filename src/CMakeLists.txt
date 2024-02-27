cmake_minimum_required(VERSION 3.27)
project(MGPreconditionedGCR)

set(CMAKE_CXX_STANDARD 20)

# Set the path to the Eigen library
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-3.4.0)

add_executable(MGPreconditionedGCR main.cpp
        GCR.cpp
        GCR.h
        Fields.cpp
        Fields.h
        MG.cpp
        MG.h)
