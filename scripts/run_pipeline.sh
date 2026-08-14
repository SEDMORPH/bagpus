#!/usr/bin/env bash
# run_pipeline.sh — Run all five pipeline steps in sequence.
#
# Usage (from the repository root):
#   ./scripts/run_pipeline.sh config/run_uds_dblplaw_tauhalf.py
#
# Each step caches its outputs and skips regeneration where files exist.

set -euo pipefail

CONFIG=${1:-config/run_uds_dblplaw_tauhalf.py}
SCRIPTS="$(dirname "$0")"

echo "=== Pipeline config: $CONFIG ==="

for step in step1_prepare step2_simulate step3_train step4_infer step5_analyse; do
    echo ""
    echo "--- $step ---"
    python "$SCRIPTS/$step.py" --config "$CONFIG"
done

echo ""
echo "=== Pipeline complete ==="
