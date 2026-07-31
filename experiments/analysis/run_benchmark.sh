#!/bin/bash
# Compatibility wrapper for the maintained pruning benchmark entry point.
# The former generate_benchmark_table.py and its JSON manifest were removed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_benchmark_pruning_curve.sh" "$@"
