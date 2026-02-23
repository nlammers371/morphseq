#!/usr/bin/env bash
set -euo pipefail

# Data root containing morphseq_playground assets; override by setting DATA_ROOT env var.
DATA_ROOT_DEFAULT="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground"
DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"

# Experiments that need metadata rebuild + SAM2 refresh.
EXPERIMENTS=(
  20230525
  20230531
  20230615
)

MICROSCOPE="Keyence"

for exp in "${EXPERIMENTS[@]}"; do
  echo "=== ${exp}: Build01 metadata-only ==="
  python -m src.run_morphseq_pipeline.cli build01 \
    --data-root "${DATA_ROOT}" \
    --exp "${exp}" \
    --microscope "${MICROSCOPE}" \
    --metadata-only \
    --overwrite

  echo "=== ${exp}: SAM2 segmentation refresh ==="
  python -m src.run_morphseq_pipeline.cli sam2 \
    --data-root "${DATA_ROOT}" \
    --exp "${exp}" \
    --workers 8 \
    --confidence-threshold 0.45 \
    --iou-threshold 0.5

  echo "=== ${exp}: metadata + SAM2 refresh complete ==="
  echo

done
