add_library(attention_gpu_baseline STATIC
  src/qk.cu src/softmax.cu src/pv.cu src/attention_forward.cu
  src/matmul_tiled.cu src/softmax_parallel.cu src/benchmark_baseline.cu)
target_include_directories(attention_gpu_baseline PUBLIC include ../common)
target_compile_features(attention_gpu_baseline PUBLIC cxx_std_17)
set_target_properties(attention_gpu_baseline PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
target_compile_definitions(attention_gpu_baseline PRIVATE NOMINMAX)
option(PART_A_PROFILE_INFO "Emit CUDA line info and compiler resource reports" OFF)
if(PART_A_PROFILE_INFO)
  target_compile_options(attention_gpu_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
add_executable(gpu_baseline_tests tests/baseline_tests.cpp)
target_link_libraries(gpu_baseline_tests PRIVATE
  attention_gpu_baseline attention_reference attention_validation)
foreach(variant naive tiled parallel optimized)
  add_test(NAME gpu_${variant} COMMAND gpu_baseline_tests all ${variant})
  set_tests_properties(gpu_${variant} PROPERTIES LABELS "GPU;${variant}" RUN_SERIAL TRUE TIMEOUT 120)
endforeach()
