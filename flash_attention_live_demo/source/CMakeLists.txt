cmake_minimum_required(VERSION 3.20)
project(attention_live_demo LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED ATTENTION_REPO)
  message(FATAL_ERROR "ATTENTION_REPO is required")
endif()
add_executable(demo_runner demo_runner.cu
  ${ATTENTION_REPO}/part_a/cpu_reference/sampled_reference.cpp
  ${ATTENTION_REPO}/part_a/gpu_baseline/matmul_tiled.cu
  ${ATTENTION_REPO}/part_a/gpu_baseline/softmax_parallel.cu
  ${ATTENTION_REPO}/part_a/gpu_baseline/matmul_tensorcore.cu
  ${ATTENTION_REPO}/part_b/src/dense_flash_attention.cu
  ${ATTENTION_REPO}/part_c/src/flash_attention_2_tensor_core.cu)
target_include_directories(demo_runner PRIVATE ${ATTENTION_REPO}/part_a/common ${ATTENTION_REPO}/part_b/include ${ATTENTION_REPO}/part_c/include)
target_compile_definitions(demo_runner PRIVATE PART_A_ENABLE_CUDA NOMINMAX)
target_link_libraries(demo_runner PRIVATE CUDA::cudart)
set_target_properties(demo_runner PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD_REQUIRED ON)
