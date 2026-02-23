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
DATA_DIR=$ROOT/data/ #01 script will poop the raw_data_organized folder here
STITCHED_DIR_OF_EXPERIMENTS=/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/
METADATA=$ROOT/data/raw_data_organized/experiment_metadata.json
FT_GDINO=$ROOT/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json
SAM2_OUT=$ROOT/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json
EMBRYO_METADATA=$ROOT/data/embryo_metadata/embryo_metadata_finetuned.json
MASK_EXPORT=$ROOT/data/annotation_and_masks/jpg_masks/grounded_sam_finetuned

mkdir -p logs


EXAMPLE_EXPS="20231206,20240418,20250612_30hpf_ctrl_atf6"

python $ROOT/scripts/01_prepare_videos.py \
  --directory_with_experiments "$STITCHED_DIR_OF_EXPERIMENTS" \
  --output_parent_dir "$DATA_DIR" \
  --workers 8 \
  --verbose
  # --experiments_to_process "$EXAMPLE_EXPS" \


# Step 2 is not implemented, it should perform quality control on the images themselves. 

# echo "=== STEP 3 : GroundedDINO finetuned detection + HQ filtering ==="
python $ROOT/scripts/03_gdino_detection_with_filtering.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --finetuned-annotations "$FT_GDINO" \
  --confidence-threshold 0.45 \
  --iou-threshold 0.5 \
  | tee logs/step3_gdino.log

echo "=== STEP 4 : SAM2 propagation using finetuned HQ annotations ==="
python $ROOT/scripts/04_sam2_video_processing.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --annotations "$FT_GDINO" \
  --output "$SAM2_OUT" \
  --target-prompt "individual embryo" \
  --segmentation-format rle \
  --verbose \
  --save-interval 250 \
  | tee logs/step4_sam2_qc.log
  # --video-ids 20240306_C01 20240306_D06  20231218_F11 \
  # --max-videos 5 \


echo "=== STEP 5 : GSAM QC on propagated SAM2 annotations ==="
python $ROOT/scripts/05_export_embryo_masks.py \
  --embryo_metadata "$EMBRYO_METADATA" \
  --annotations "$SAM2_OUT" \
  --output "$MASK_EXPORT" \
  --workers 8 \
  --verbose \
  | tee logs/step5_sam2_qc.log
echo "✔️  Pipeline finished"