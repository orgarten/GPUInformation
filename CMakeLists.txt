cmake_minimum_required(VERSION 3.16)
project(GPUInformation CXX)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "Path to cuda compiler")

add_executable(GPUInformation GPUInformation.cu)

set_target_properties(
        GPUInformation
        PROPERTIES
        CMAKE_CXX_STANDARD 17
        CMAKE_CUDA_STANDARD 14
        CUDA_SEPARABLE_COMPILATION ON)