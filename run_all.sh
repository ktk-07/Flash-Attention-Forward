#!/usr/bin/env bash
# Invoke from any directory. No Git operations; GPU jobs run sequentially.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$project_root/scripts/run_all.py" "$@"
