#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

N_JOBS="${N_JOBS:-8}"
N_PERM="${N_PERM:-500}"
CLASS_SET="${CLASS_SET:-both}"
EXTRA_ARGS=("$@")

export PYTHONWARNINGS="ignore:.*multi_class.*deprecated.*:FutureWarning"

for BIN_WIDTH in 2 4; do
    echo "=== Running bin_width=${BIN_WIDTH} ==="
    PYTHONPATH=src \
    /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run -n segmentation_grounded_sam --no-capture-output \
    python results/mcolon/20260407_pbx_analysis_cont/01_pairwise_all_pairs_positioning.py \
      --class-set "$CLASS_SET" \
      --bin-width "$BIN_WIDTH" \
      --n-permutations "$N_PERM" \
      --n-jobs "$N_JOBS" \
      "${EXTRA_ARGS[@]}"
done
