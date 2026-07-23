#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# moe4_cap05x is owned by the expert-count scan to avoid concurrent output collisions.
export CAS_SKIP_SHARED_MOE4=1
exec ./run_batch_experiments.sh --yes --cas_moe_capacity_scan "$@"
