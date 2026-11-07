#!/usr/bin/env bash
# Reproducible bounded evaluation; existing result directories are never overwritten.
set -euo pipefail
cd "$(dirname "$0")/.."
build="${PART_C_BUILD_DIR:-/tmp/part_c-evaluation-build}"
output="${1:-part_c/results/bounded_evaluation}"
cmake -S part_c -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j 2
"$build/part_c_input_contract"
python3 part_c/scripts/evaluate.py --build "$build" --output "$output"
