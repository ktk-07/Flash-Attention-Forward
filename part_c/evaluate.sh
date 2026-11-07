#!/usr/bin/env bash
# Reproducible bounded evaluation; existing studies are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
build="${PART_C_BUILD_DIR:-/tmp/part_c-evaluation-build}"
if [ "${1:-}" = "--verify" ]; then
    shift
    exec python3 part_c/scripts/verify_evaluation.py "$@" --build "$build"
fi
output="${1:-part_c/results/bounded_evaluation}"
if [ -f "$output/plan.json" ]; then
    printf 'Choose a fresh output directory: %s already contains a study.\n' "$output" >&2
    exit 2
fi
mkdir -p "$output/logs"
arch="${PART_C_CUDA_ARCHITECTURES:-86}"
printf '%q ' cmake -S part_c -B "$build" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$arch" > "$output/logs/build_commands.txt"
printf '\n' >> "$output/logs/build_commands.txt"
cmake -S part_c -B "$build" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$arch" 2>&1 | tee "$output/logs/configure.txt"
printf '%q ' cmake --build "$build" -j 2 >> "$output/logs/build_commands.txt"
printf '\n' >> "$output/logs/build_commands.txt"
cmake --build "$build" -j 2 2>&1 | tee "$output/logs/build.txt"
"$build/part_c_input_contract" | tee "$output/logs/input_contract.txt"
# Replace the shell so later edits cannot resume a stale script-file offset.
exec python3 part_c/scripts/evaluate.py --build "$build" --output "$output" --grid "${ATTENTION_GRID:-full}"
