#!/usr/bin/env bash
# Run from any directory. Logs every command, its output and its exit code.
# No Git operations. Needs CUDA GPU access, CMake, Python and Matplotlib.
set -uo pipefail
# Named CLI publishes the standardized report; legacy positional workflows stay available.
case "${1:-}" in
    --fa1-pilot)
        shift
        exec python3 "$(dirname "$0")/scripts/compare_fa1.py" "$@"
        ;;
    --export)
        shift
        if [ "$#" -eq 0 ]; then printf 'Usage: run.sh --export STUDY --output BUNDLE\n' >&2; exit 2; fi
        native_study="$1"; shift
        exec python3 "$(dirname "$0")/scripts/export_results.py" --study "$native_study" "$@"
        ;;
    ""|--output|--build|--arch|--device|--grid|--dry-run|--no-profile|--supplementary-profile|--analyze-only|--help|-h)
        exec python3 "$(dirname "$0")/scripts/run_standard.py" "$@"
        ;;
esac
# Use the same entry point for the new bounded study; retain legacy positional CLI.
if [ "${1:-}" = "--complete" ]; then
    shift
    complete_output="${1:-part_c/results/complete_$(date -u +%Y%m%dT%H%M%SZ)}"
    complete_build="${2:-/tmp/part_c-evaluation-build}"
    if [ -e "$complete_output" ]; then
        printf 'Choose a fresh result directory: %s already exists.\n' "$complete_output" >&2
        exit 2
    fi
    bash "$0" "$complete_output/regression" "$complete_build" || exit $?
    PART_C_BUILD_DIR="$complete_build" bash "$(dirname "$0")/evaluate.sh" "$complete_output/main"
    exit $?
fi
if [ "${1:-}" = "--study" ]; then
    shift
    exec bash "$(dirname "$0")/evaluate.sh" "$@"
fi
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
task_run="${1:-part_c/results/run_$(date -u +%Y%m%dT%H%M%SZ)}"
task_build="${2:-$task_root/part_c/build}"
mkdir -p "$task_run/logs"
task_failed=0
record() {
    local label="$1"; shift
    printf '%q ' "$@" >> "$task_run/commands.log"
    printf '\n' >> "$task_run/commands.log"
    "$@" > "$task_run/logs/$label.log" 2>&1
    local status=$?
    printf 'exit_code=%s\n' "$status" >> "$task_run/commands.log"
    printf '%s: exit %s\n' "$label" "$status"
    if [ "$status" -ne 0 ]; then task_failed=1; fi
    return "$status"
}
legacy_tests() {
    record legacy_build nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo part_c/tests/causal_flash_attn_full_test.cu -o "$task_build/part_c_legacy_tests" || return 1
    record legacy_correctness "$task_build/part_c_legacy_tests"
}
if [ "${3:-}" = "--legacy-only" ]; then
    legacy_tests
    exit "$task_failed"
fi
record gpu nvidia-smi || exit 1
record nvcc_version nvcc --version
record host_compiler c++ --version
record configure cmake -S part_c -B "$task_build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON || exit 1
record build cmake --build "$task_build" --parallel 2 || exit 1
record source_hashes sha256sum part_c/src/flash_attention_1.cu part_c/src/flash_attention_1_tensor_core.cu part_c/src/flash_attention_2_cuda_core.cu part_c/src/flash_attention_2_tensor_core.cu part_c/include/flash_attention/*.h part_c/tests/*.cpp part_c/tests/*.h part_c/tests/kernel_bodies.json part_c/CMakeLists.txt "$task_build/part_c_runner"
record fa1_invariants "$task_build/part_c_fa1_test" --output "$task_run/fa1_invariants" || exit 1
record fa1_wmma_invariants "$task_build/part_c_fa1_test" --tensorcore --output "$task_run/fa1_wmma_invariants" || exit 1
record correctness "$task_build/part_c_runner" --suite correctness --output "$task_run/correctness" || exit 1
record benchmark "$task_build/part_c_runner" --suite benchmark --grid full --output "$task_run/benchmark" || exit 1
for task_tool in memcheck racecheck synccheck initcheck; do
    record "$task_tool" compute-sanitizer --tool "$task_tool" --error-exitcode 3 "$task_build/part_c_runner" --suite smoke --output "$task_run/$task_tool"
done
record plots python3 part_c/scripts/plot_results.py "$task_run/benchmark"
record binary_inspection python3 part_c/scripts/inspect_binary.py "$task_build/part_c_runner" "$task_run/binary_inspection"
record audit python3 part_c/scripts/audit_results.py "$task_run"
legacy_tests
exit "$task_failed"
