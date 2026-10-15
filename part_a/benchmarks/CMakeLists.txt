add_executable(benchmark_harness_tests tests/harness_tests.cpp)
target_include_directories(benchmark_harness_tests PRIVATE . ../common ../gpu_baseline/include)
target_compile_features(benchmark_harness_tests PRIVATE cxx_std_17)
add_test(NAME benchmark_harness COMMAND benchmark_harness_tests)
if(PART_A_ENABLE_CUDA)
  find_package(Git QUIET)
  set(BENCHMARK_SOURCE_COMMIT "unknown")
  set(BENCHMARK_SOURCE_STATE "unknown")
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE BENCHMARK_SOURCE_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain -- . ":(exclude)results"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE BENCHMARK_DIRTY OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(BENCHMARK_DIRTY)
      set(BENCHMARK_SOURCE_STATE "dirty at configure; save git diff separately")
    else()
      set(BENCHMARK_SOURCE_STATE "clean at configure")
    endif()
  endif()
  set(BENCHMARK_BUILD_INFO "CMake=${CMAKE_VERSION};generator=${CMAKE_GENERATOR};CXX=${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION};CUDA=${CMAKE_CUDA_COMPILER_VERSION};arch=${CMAKE_CUDA_ARCHITECTURES};CUDA_flags=${CMAKE_CUDA_FLAGS};Release_flags=${CMAKE_CUDA_FLAGS_RELEASE};profile_info=${PART_A_PROFILE_INFO}")
  string(REPLACE "\\" "\\\\" BENCHMARK_BUILD_INFO "${BENCHMARK_BUILD_INFO}")
  string(REPLACE "\"" "\\\"" BENCHMARK_BUILD_INFO "${BENCHMARK_BUILD_INFO}")
  configure_file(benchmark_build_info.hpp.in benchmark_build_info.hpp @ONLY)
  add_executable(attention_benchmark attention_benchmark.cpp)
  target_include_directories(attention_benchmark PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(attention_benchmark PRIVATE attention_gpu_baseline attention_reference attention_validation)
  target_compile_definitions(attention_benchmark PRIVATE PART_A_BUILD_CONFIG="$<CONFIG>")
  add_test(NAME gpu_config_validation COMMAND ${CMAKE_COMMAND}
    -DBENCHMARK_EXE=$<TARGET_FILE:attention_benchmark>
    -DRESULT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/validation-results
    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gpu_validation.cmake)
  set_tests_properties(gpu_config_validation PROPERTIES LABELS "GPU;validation" RUN_SERIAL TRUE TIMEOUT 180)
endif()
