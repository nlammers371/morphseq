#!/bin/bash
#$ -N gdino_sam2_ft
#$ -cwd
#$ -q  trapnell-login.q
#$ -l gpgpu=TRUE,cuda=1
#$ -l mfree=30G
#$ -pe serial 1
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err

set -uo pipefail   # safer but allows the job to keep going on non-fatal errors

# ─── Activate your Conda/venv ────────────────────────────────────────────────
# Activate Conda for SGE job scripts
source /net/trapnell/vol1/home/mdcolon/software/miniconda3/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam

# (Optional) explicit CUDA path if the node doesn’t pick it up automatically
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox
CONFIG=$ROOT/configs/pipeline_config.yaml
METADATA=$ROOT/data/raw_data_organized/experiment_metadata.json
FT_GDINO=$ROOT/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json
SAM2_OUT=$ROOT/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json

mkdir -p logs

# echo "=== STEP 3 : GroundedDINO finetuned detection + HQ filtering ==="
# python $ROOT/scripts/03_gdino_detection_with_filtering.py \
#   --config "$CONFIG" \
#   --metadata "$METADATA" \
#   --finetuned-annotations "$FT_GDINO" \
#   --confidence-threshold 0.45 \
#   --iou-threshold 0.5 \
#   | tee logs/step3_gdino.log

echo "=== STEP 4 : SAM2 propagation using finetuned HQ annotations ==="
python $ROOT/scripts/04_sam2_video_processing.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --annotations "$FT_GDINO" \
  --output "$SAM2_OUT" \
  --target-prompt "individual embryo" \
  --segmentation-format rle \
  --verbose \
  | tee logs/step4_sam2.log

echo "✔️  Pipeline finished"