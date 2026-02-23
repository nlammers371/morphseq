# Using ExperimentManager for Keyence Workflows (Theory + Hand‑Off)

Purpose: outline how to use `Experiment` and `ExperimentManager` to batch Keyence export/stitch/segment/snips and then hand off to the centralized CLI for Build03A→Build04→Build05.

Code anchors:
- Experiment classes: `src/build/pipeline_objects.py:65`, `src/build/pipeline_objects.py:343`
- Keyence Build01 helpers: `src/build/build01A_compile_keyence_torch.py`
- Snip extraction and df01 write: `src/build/build03A_process_images.py:1089` and `src/build/build03A_process_images.py:1175`
- CLI entry: `src/run_morphseq_pipeline/cli.py`

Prerequisites
- Root structure under `<root>` with `raw_image_data/Keyence/<YYYYMMDD_...>/...`
- Conda env with repo deps (Keyence stitching, segmentation)
- Sufficient disk for `built_image_data/` and `training_data/`

Quickstart (Python)
```python
from pathlib import Path
from src.build.pipeline_objects import Experiment, ExperimentManager

root = Path("/path/to/morphseq_root")
mgr = ExperimentManager(root=root)
mgr.report()                # See per‑date status flags

# Minimal per‑date run
exp = Experiment(date="20250703_chem3_34C_T01_1457", data_root=root)
exp.export_images()         # Build01: export + FF build (Keyence or YX1)
exp.stitch_images()         # Optional: tiled FF stitching
exp.stitch_z_images()       # Optional: Z stitching (Keyence)
exp.segment_images()        # Legacy segmentation (multiple models)
exp.process_image_masks()   # Track embryos, compute stats, extract snips

# Output hand‑off:
# - Writes date‑scoped metadata and snips
# - Also writes a Build04‑ready df01 at:
#   <root>/metadata/combined_metadata_files/embryo_metadata_df01.csv
```

Hand‑off to Centralized Runner
- After df01 exists, run Build04 via CLI (QC + stage inference):
```bash
python -m src.run_morphseq_pipeline.cli build04 --root /path/to/morphseq_root
```
- Build05 to create training sets from df02:
```bash
python -m src.run_morphseq_pipeline.cli build05 \
  --root /path/to/morphseq_root \
  --train-name keyence_train_YYYYMMDD
```

Stage Reference Generation (required by Build04)
- Rebuild `metadata/stage_ref_df.csv` from your df01 using the included utility:
```python
from src.build.build_utils import generate_stage_ref_from_df01
generate_stage_ref_from_df01(
    root="/path/to/morphseq_root",
    ref_dates=None,       # or ["20230620", "20240626"]
    quantile=0.95,
    max_stage=96,
    pert_key_path=None    # optional; pass a key to restrict to WT/controls
)
```

Perturbation Key (required by Build04)
- Build04 expects `metadata/perturbation_name_key.csv` with columns:
  `master_perturbation,short_pert_name,phenotype,control_flag,pert_type,background`
- If you have run Build04 previously and have df02, you can reconstruct the key:
```python
from src.build.build_utils import reconstruct_perturbation_key_from_df02
reconstruct_perturbation_key_from_df02(root="/path/to/morphseq_root")
```
- Otherwise, curate a minimal CSV manually for your perturbations.

Notes and Caveats
- ExperimentManager is legacy orchestration; the centralized CLI is the source of truth for Build03/04/05 semantics. Prefer the CLI for consistency once df01 is available.
- Segmentation via `segment_images()` uses configured legacy UNet models; confirm checkpoints are available in your environment.
- For SAM2 workflows, skip legacy segmentation and produce df01 via the SAM2 bridge (CLI build03) instead; then continue with Build04/05.

