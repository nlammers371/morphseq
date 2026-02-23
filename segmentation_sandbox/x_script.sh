# Set these to your values (example shown for your workspace)
export REPO_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq
export DATA_ROOT=/path/to/data_root
export EXP=20240418

# 1) Run SAM2 QC analysis
python "$REPO_ROOT/segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py" \
  --input "$DATA_ROOT/sam2_pipeline_files/segmentation/grounded_sam_segmentations.json" \
  --experiments "$EXP" --process-all --verbose

# 2) Re-export SAM2 CSV (writes into sam2_expr_files/)
python "$REPO_ROOT/segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py" \
  "$DATA_ROOT/sam2_pipeline_files/segmentation/grounded_sam_segmentations.json" \
  -o "$DATA_ROOT/sam2_pipeline_files/sam2_expr_files/sam2_metadata_${EXP}.csv" \
  --experiment-filter "$EXP" -v

# 3) Rebuild df01 (Build03) to incorporate changes
python -m src.run_morphseq_pipeline.cli build03 \
  --data-root "$DATA_ROOT" --exp "$EXP"

# 4) Rebuild df02 (Build04)
python -m src.run_morphseq_pipeline.cli build04 \
  --data-root "$DATA_ROOT"

# 5) Re-merge df03 (Build06) for just this experiment
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --experiments "$EXP"

# If embeddings need regeneration:
# - To append generation of missing latents:
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --experiments "$EXP" \
  --generate-missing-latents

# - To force regeneration (overwrite):
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --experiments "$EXP" \
  --generate-missing-latents --overwrite-latents