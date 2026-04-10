#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

N_JOBS="${N_JOBS:-8}"
N_PERM="${N_PERM:-500}"
CLASS_SET="${CLASS_SET:-both}"
RUN_TAG="${RUN_TAG:-feature4}"
EXTRA_ARGS=("$@")

export PYTHONWARNINGS="ignore:.*multi_class.*deprecated.*:FutureWarning"

run_one() {
  local class_set="$1"
  local output_dir="$2"
  PYTHONPATH=src \
  /net/trapnell/vol1/home/mdcolon/software/miniconda3/bin/conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260407_pbx_analysis_cont/01_pairwise_all_pairs_positioning.py \
    --class-set "$class_set" \
    --bin-width 2 \
    --n-permutations "$N_PERM" \
    --n-jobs "$N_JOBS" \
    --output-dir "$output_dir" \
    "${EXTRA_ARGS[@]}"
}

BASE="results/mcolon/20260407_pbx_analysis_cont/results/positioning/pairwise"

case "$CLASS_SET" in
  canonical)
    run_one canonical "$BASE/combined_pairwise_4class_bin2_perm${N_PERM}_${RUN_TAG}"
    ;;
  wik_ab)
    run_one wik_ab "$BASE/combined_pairwise_5class_bin2_perm${N_PERM}_${RUN_TAG}"
    ;;
  both)
    run_one canonical "$BASE/combined_pairwise_4class_bin2_perm${N_PERM}_${RUN_TAG}"
    run_one wik_ab "$BASE/combined_pairwise_5class_bin2_perm${N_PERM}_${RUN_TAG}"
    ;;
  *)
    echo "Unknown CLASS_SET: $CLASS_SET" >&2
    exit 1
    ;;
esac
