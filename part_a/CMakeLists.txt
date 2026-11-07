cmake_minimum_required(VERSION 3.18)
project(part_a LANGUAGES CXX)
option(PART_A_ENABLE_CUDA "Build GPU baselines" OFF)
option(PART_A_PROFILE_INFO "Include CUDA profiling information" OFF)

add_subdirectory(cpu_reference)
add_executable(attention main.cpp)
target_link_libraries(attention PRIVATE attention_reference)
target_include_directories(attention PRIVATE common validation)
set_target_properties(attention PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_definitions(attention PRIVATE NOMINMAX)

if(PART_A_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(attention PRIVATE gpu_baseline/qk.cu gpu_baseline/softmax.cu
    gpu_baseline/pv.cu gpu_baseline/matmul_tiled.cu gpu_baseline/softmax_parallel.cu
    gpu_baseline/matmul_tensorcore.cu)
  target_link_libraries(attention PRIVATE CUDA::cudart)
  string(REPLACE ";" "," PART_A_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
  target_compile_definitions(attention PRIVATE PART_A_ENABLE_CUDA
    PART_A_TOOLKIT="${CMAKE_CUDA_COMPILER_VERSION}"
    PART_A_HOST_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    PART_A_BUILD_CONFIG="$<CONFIG>"
    PART_A_ARCHITECTURES="${PART_A_ARCH_LIST}"
    PART_A_PROFILE_ENABLED="$<BOOL:${PART_A_PROFILE_INFO}>")
  # Preserve quoted compiler flags
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build-info/$<CONFIG>/build_flags.hpp"
    CONTENT "#pragma once\n#define PART_A_CXX_FLAGS R\"flags(${CMAKE_CXX_FLAGS} $<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}>)flags\"\n#define PART_A_CUDA_FLAGS R\"flags(${CMAKE_CUDA_FLAGS} $<$<CONFIG:Release>:${CMAKE_CUDA_FLAGS_RELEASE}>)flags\"\n")
  target_include_directories(attention PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/build-info/$<CONFIG>")
  set_target_properties(attention PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  if(PART_A_PROFILE_INFO)
    target_compile_options(attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
      $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
  endif()
endif()
