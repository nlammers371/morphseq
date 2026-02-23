#!/bin/bash
#$ -N morphseq_pipeline
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

# (Optional) explicit CUDA path if the node doesn't pick it up automatically
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# ─── Path Configuration ──────────────────────────────────────────────────────
ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox
# ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox
CONFIG=$ROOT/configs/pipeline_config.yaml
DATA_DIR=$ROOT/data/
STITCHED_DIR_OF_EXPERIMENTS=/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/
METADATA=$ROOT/data/raw_data_organized/experiment_metadata.json
GDINO_ANNOTATIONS=$ROOT/data/detections/gdino_detections.json
SAM2_ANNOTATIONS=$ROOT/data/segmentation/grounded_sam_segmentations.json
EMBRYO_METADATA=$ROOT/data/embryo_metadata/embryo_metadata.json
MASK_EXPORT=$ROOT/data/exported_masks

mkdir -p logs

# Example experiments (uncomment to process specific experiments)
# EXAMPLE_EXPS="20250529_30hpf_ctrl_atf6,20231206,20240418,20250305"
EXAMPLE_EXPS="20250529_30hpf_ctrl_atf6"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Video Preparation
# ═══════════════════════════════════════════════════════════════════════════════
log_info "Starting STEP 1: Video Preparation"

python $ROOT/scripts/pipelines/01_prepare_videos.py \
  --directory_with_experiments "$STITCHED_DIR_OF_EXPERIMENTS" \
  --output_parent_dir "$DATA_DIR" \
  --workers 8 \
  --verbose \
  --entities_to_process "$EXAMPLE_EXPS"
  # Fine-grained options for future reference:
  # --experiments_to_process "$EXAMPLE_EXPS" \  # (legacy format)
  # --max-frames-per-video 500 \
  # --quality-threshold 0.8 \
  # --dry-run \

log_success "STEP 1 completed successfully"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Image Quality Control (NOT IMPLEMENTED)
# ═══════════════════════════════════════════════════════════════════════════════
log_warning "STEP 2: Image Quality Control is not implemented yet"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: GroundingDINO Detection + HQ Filtering
# ═══════════════════════════════════════════════════════════════════════════════
log_info "Starting STEP 3: GroundingDINO Detection + HQ Filtering"

python $ROOT/scripts/pipelines/03_gdino_detection.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --annotations "$GDINO_ANNOTATIONS" \
  --confidence-threshold 0.45 \
  --iou-threshold 0.5 \
  --entities_to_process "$EXAMPLE_EXPS" \
  | tee logs/step3_gdino.log
  # Fine-grained options for future reference:
  # --max-images 1000 \

log_success "STEP 3 completed successfully"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: SAM2 Video Processing (NOW IMPLEMENTED)
# ═══════════════════════════════════════════════════════════════════════════════
log_info "Starting STEP 4: SAM2 Video Processing"

python $ROOT/scripts/pipelines/04_sam2_video_processing.py \
  --config "$CONFIG" \
  --metadata "$METADATA" \
  --annotations "$GDINO_ANNOTATIONS" \
  --output "$SAM2_ANNOTATIONS" \
  --entities_to_process "$EXAMPLE_EXPS" \
  --target-prompt "individual embryo" \
  --segmentation-format rle \
  --verbose \
  --save-interval 10 \
  | tee logs/step4_sam2.log
  # Fine-grained options for future reference:
  # --max-videos 5 \
  # --frames-per-batch 10 \
  # --device cuda \

log_success "STEP 4 completed successfully"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAM2 Quality Control Analysis
# ═══════════════════════════════════════════════════════════════════════════════
log_info "Starting STEP 5: SAM2 Quality Control Analysis"

python $ROOT/scripts/pipelines/05_sam2_qc_analysis.py \
  --input "$SAM2_ANNOTATIONS" \
  --output "$SAM2_ANNOTATIONS" \
  --author "pipeline_qc" \
  --process-all \
  --verbose \
  | tee logs/step5_qc_analysis.log
  # Fine-grained options for future reference:
  # --dry-run \
  # --max-entities 100 \
  # --experiment-ids "exp1,exp2" \

log_success "STEP 5 completed successfully"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Export Embryo Masks (SimpleMaskExporter)
# ═══════════════════════════════════════════════════════════════════════════════
log_info "Starting STEP 6: Export Embryo Masks"

python $ROOT/scripts/pipelines/06_export_masks.py \
  --sam2-annotations "$SAM2_ANNOTATIONS" \
  --output "$MASK_EXPORT" \
  --entities-to-process "$EXAMPLE_EXPS" \
  --export-format png \
  --verbose \
  | tee logs/step6_masks.log
  # Fine-grained options for future reference:
  # --overwrite \
  # --dry-run \

log_success "STEP 6 completed successfully"

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════
log_success "Pipeline execution completed!"
log_info "All pipeline stages implemented and executed:"
log_info "✅ STEP 1: Video Preparation (Module 0)"
log_info "✅ STEP 3: GroundingDINO Detection (Module 2)" 
log_info "✅ STEP 4: SAM2 Video Processing (Module 2)"
log_info "✅ STEP 5: SAM2 Quality Control Analysis (Module 2)"
log_info "✅ STEP 6: Export Embryo Masks (Module 2)"
log_info "⚠️  STEP 2: Image Quality Control (Future: Module 1)"

log_info "Output files:"
log_info "  📊 Experiment metadata: $METADATA"
log_info "  🎯 Detection annotations: $GDINO_ANNOTATIONS" 
log_info "  🎬 Segmentation annotations: $SAM2_ANNOTATIONS (with QC flags)"
log_info "  🖼️  Exported masks: $MASK_EXPORT (PNG format)"
