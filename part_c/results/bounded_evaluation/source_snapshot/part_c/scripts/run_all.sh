#!/usr/bin/env bash
# Compatibility entry point; the workflow now lives at part_c/run.sh.
set -euo pipefail
exec bash "$(dirname "$0")/../run.sh" "$@"
