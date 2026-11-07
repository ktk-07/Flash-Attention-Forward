#!/usr/bin/env bash
# Compatibility alias for the unified runner.
set -euo pipefail
exec bash "$(dirname "$0")/run.sh" "$@"
