string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef RUN_ID)
execute_process(COMMAND "${BENCHMARK_EXE}" --validate-only --variants all --configs study
  --output "${RESULT_ROOT}/${RUN_ID}" RESULT_VARIABLE STATUS)
if(NOT STATUS EQUAL 0)
  message(FATAL_ERROR "GPU configuration validation failed: ${STATUS}")
endif()
