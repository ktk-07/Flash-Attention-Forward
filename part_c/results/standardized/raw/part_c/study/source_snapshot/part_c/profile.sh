#!/usr/bin/env bash
# Independent profiling workflow; does not overwrite benchmark results or use Git.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
exec python3 part_c/scripts/profile_kernels.py "$@"
