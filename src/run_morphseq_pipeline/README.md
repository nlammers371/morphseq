# MorphSeq Centralized Pipeline Runner

**Status: Production Ready** - Complete SAM2 integration with enhanced Build06 and centralized embedding generation.

## 🚀 Key Features

### ✅ **SAM2 Integration** - First-class CLI citizen
- Direct SAM2 pipeline execution via `sam2` subcommand
- Auto-discovery of SAM2 outputs by Build03  
- Hybrid approach: SAM2 embryo masks + Build02 QC masks
- Complete E2E orchestration with `--run-sam2` flag
- Batch processing for multiple experiments

### ✅ **Enhanced Build06** - Skips Build05 complexity
- Direct df02 → df03 conversion with quality filtering
- Automatic Python 3.9 environment switching for legacy models
- Incremental processing (only missing experiments)
- Centralized embedding generation module

### ✅ **Complete Pipeline** 
**Current Flow**: Build01 → Build02(5 UNets) → SAM2(batch) → Build03(hybrid) → Build04 → Build06(enhanced)

**Commands Available**: `build01`, `build02`, `sam2`, `build03`, `build04`, `build06`, `e2e`, `validate`, `embed`

## Install/Run

- Easiest: use the wrapper script from the repo root:
  - `./morphseq-runner <subcommand> [args]`
  - Examples:
    - `./morphseq-runner status --data-root /data`
    - `./morphseq-runner pipeline --data-root /data`  ← defaults to `e2e`
    - `./morphseq-runner pipeline --data-root /data --action sam2`
    - `./morphseq-runner pipeline --data-root /data --action build01`

- Or run via module:
  - `python -m src.run_morphseq_pipeline.cli <subcommand> [args]`
  - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /data`  ← defaults to `e2e`
  - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /data --action build04`
  - `python -m src.run_morphseq_pipeline.cli pipeline --data-root /data --action build01`

## Data Structure & Path Conventions

### Build Pipeline Data
- Stitched FF images: `<data_root>/built_image_data/stitched_FF_images/{exp}/`
- Build02 QC masks: `<data_root>/segmentation/{model}_predictions/{exp}/`
  - `mask_v0_0100_predictions/` - embryo masks
  - `yolk_v1_0050_predictions/` - yolk masks  
  - `focus_v0_0100_predictions/` - focus masks
  - `bubble_v0_0100_predictions/` - bubble masks
  - `via_v1_0100_predictions/` - viability masks

### SAM2 Pipeline Data
- SAM2 root: `<data_root>/sam2_pipeline_files/`
- Exported masks: `<data_root>/sam2_pipeline_files/exported_masks/{exp}/masks/`
- Metadata CSV: `<data_root>/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv`
- Segmentation data: `<data_root>/sam2_pipeline_files/segmentation/grounded_sam_segmentations.json`
- Detection data: `<data_root>/sam2_pipeline_files/detections/gdino_detections.json`

### Embedding Generation
- Centralized module: `src/analyze/gen_embeddings/`
- CLI tool: `python -m src.analyze.gen_embeddings.cli`
- Python 3.9 subprocess orchestration for legacy model compatibility
- Automatic environment switching in Build06

### Metadata Files
- Per-experiment metadata: `<data_root>/metadata/built_metadata_files/{exp}_metadata.csv`
- Combined df01: `<data_root>/metadata/combined_metadata_files/embryo_metadata_df01.csv`
- Final df03 with embeddings: `<data_root>/metadata/combined_metadata_files/embryo_metadata_df03.csv`
- Pipeline state tracking: `<data_root>/metadata/experiments/{exp}.json`

## Subcommands

### Core Pipeline Steps

**`build01`** - Image stitching and metadata
```bash
python -m src.run_morphseq_pipeline.cli build01 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --microscope Keyence (or YX1) \
```

**`build02`** - Complete QC mask suite (5 UNet models)
```bash  
python -m src.run_morphseq_pipeline.cli build02 \
  --data-root morphseq_playground \
  --mode legacy
```
Runs all 5 models: embryo, yolk, focus, bubble, viability

**`sam2`** - SAM2 segmentation pipeline ⭐ NEW!
```bash
# Single experiment
python -m src.run_morphseq_pipeline.cli sam2 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --confidence-threshold 0.45 \
  --workers 8

# Batch mode (all experiments)  
python -m src.run_morphseq_pipeline.cli sam2 \
  --data-root morphseq_playground \
  --batch
```

**`build03`** - Embryo processing (hybrid masks)
```bash
# Auto-discovers SAM2 CSV or falls back to legacy
python -m src.run_morphseq_pipeline.cli build03 \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --by-embryo 5 --frames-per-embryo 3
```

**`build04`** - QC analysis and stage inference
```bash
python -m src.run_morphseq_pipeline.cli build04 \
  --data-root morphseq_playground
```

**`build06`** - Enhanced embeddings generation (skips Build05) ⭐ ENHANCED!
```bash
# Process new experiments only (default)
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root /path/to/repo \
  --data-root morphseq_playground

# Process specific experiment
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root /path/to/repo \
  --data-root morphseq_playground \
  --experiments "20250529_30hpf_ctrl_atf6"

# Force reprocess with explicit safety
python -m src.run_morphseq_pipeline.cli build06 \
  --morphseq-repo-root /path/to/repo \
  --data-root morphseq_playground \
  --overwrite --experiments "exp1,exp2"
```

### End-to-End Orchestration

**`e2e`** - Complete pipeline with SAM2 ⭐ ENHANCED!
```bash
# Full pipeline with SAM2 + Build06
python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_30hpf_ctrl_atf6 \
  --microscope keyence \
  --run-sam2 \
  --train-name test_sam2_20250906

# Pipeline: Build01 → Build02(5 UNets) → SAM2 → Build03(hybrid) → Build04 → Build06(embeddings)

# Legacy pipeline (no SAM2)
python -m src.run_morphseq_pipeline.cli e2e \
  --data-root morphseq_playground \
  --exp 20250529_30hpf_ctrl_atf6 \
  --microscope keyence \
  --train-name legacy_test_20250906
```

### Utility Commands

**`validate`** - Validation checks
```bash
python -m src.run_morphseq_pipeline.cli validate \
  --data-root morphseq_playground \
  --exp 20250529_24hpf_ctrl_atf6 \
  --checks schema,units,paths
```

## ✨ Key Features & Benefits

### 🎯 **SAM2 Integration**
- **Python orchestration**: Complete segmentation_sandbox pipeline integration
- **Auto-discovery**: Build03 finds SAM2 outputs automatically
- **Hybrid masks**: Superior SAM2 embryo masks + Build02 QC masks
- **Format consistency**: Fixed snip_id format (`_t####`) for pipeline compatibility
- **Batch processing**: Process multiple experiments efficiently

### 🏗️ **Enhanced Build02**
- **Complete QC suite**: All 5 UNet models (embryo, yolk, focus, bubble, viability)
- **Fixed mask processing**: Proper legacy formula for Build02 auxiliary masks
- **Dead flag accuracy**: Correct viability mask processing (fixed critical bug)
- **Quality control**: Full QC flag calculation including accurate dead_flag

### 🚀 **Enhanced Build06** 
- **Skips Build05**: Direct df02 → df03 conversion with same quality filtering
- **Environment switching**: Automatic Python 3.9 subprocess for legacy models
- **Incremental processing**: Only processes missing experiments by default
- **Safe overwrite**: Explicit experiment specification required
- **Centralized generation**: Clean embedding module in `src/analyze/gen_embeddings/`

### 🔄 **Complete Pipeline Flow**
```
Build01 (stitching) → Build02 (5 UNets) → SAM2 (embryo masks) 
    ↓
Build03 (hybrid masks) → Build04 (QC/staging) → Build06 (embeddings)
```
- **Dependency tracking**: Steps run only when inputs change
- **Flexible control**: Skip any step with `--skip-*` flags
- **Progress tracking**: Clear step-by-step indicators
- **Error resilience**: Graceful handling of partial failures

## Environment Setup

```bash
# Main pipeline environment  
conda activate mseq_data_pipeline_env

# For Build06 embedding generation (automatic switching)
# Python 3.9 environment: /net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster

# Verify SAM2 sandbox is available
ls segmentation_sandbox/scripts/pipelines/

# Verify centralized embedding generation
ls src/analyze/gen_embeddings/
```

## Troubleshooting

### SAM2 Issues
- **SAM2 not found**: Ensure `segmentation_sandbox/` exists in repo root
- **Auto-discovery fails**: Manually provide `--sam2-csv` path to Build03
- **Format errors**: Check snip_id uses `_t####` format (should be automatic)
- **GroundingDINO `_C` import errors** (e.g., `NameError: _C`, `undefined symbol`, `libc10.so not found`):
  - Ensure CUDA toolkit matches the PyTorch build (cu118 → `cuda/11.8.0`)
  - Set `CUDA_HOME` and include both CUDA and torch libs in `LD_LIBRARY_PATH`
  - Rebuild the extension:
    - `cd segmentation_sandbox/models/GroundingDINO`
    - `python setup.py build_ext --inplace`

### Build02/Build03 Issues  
- **Missing models**: Check Build02 model availability in conda environment
- **Dead flag errors**: Ensure via masks exist in `segmentation/via_v1_0100_predictions/`

### Build06 Embeddings – Note on Split Logic (Tech Debt)
- Current implementation uses two related paths for embeddings:
  - `src/run_morphseq_pipeline/services/gen_embeddings.py` orchestrates Build06 (df02 filtering, experiment selection, merge) and can generate latents as a fallback.
  - `src/analyze/gen_embeddings/` provides a centralized CLI/wrapper for embedding generation, including Python 3.9 subprocess switching for legacy models.
- Both are wired into the pipeline: the CLI may call the centralized `analyze/gen_embeddings` helpers to (re)generate latents, and Build06 services will also check/generate as needed.
- Future consolidation: unify latent generation under a single module and have Build06 call only that path, keeping Build06 focused on orchestration/merge. This is intentional tech debt to be resolved later.
- **Mask loading failures**: Verify Build02 completed all 5 UNet models successfully

### Build06/Embedding Issues
- **Python version errors**: Build06 automatically switches to Python 3.9 for legacy models
- **Environment not found**: Check `/net/trapnell/vol1/home/nlammers/micromamba/envs/vae-env-cluster` exists
- **Missing embeddings**: Use `--overwrite` with explicit experiment specification
- **Model path issues**: Ensure `--data-root` contains `models/` directory

### General Issues
- **Permission errors**: Ensure write access to data root directory
- **Memory issues**: Reduce workers with `--num-workers` or `--sam2-workers`
- **Disk space**: Pipeline generates substantial mask and embedding data

### Getting Help
- **Status checking**: Use `validate` command to check pipeline state
- **Progress tracking**: Check `metadata/experiments/{exp}.json` for pipeline state
- **Log files**: Check `logs/` directory for detailed error information
