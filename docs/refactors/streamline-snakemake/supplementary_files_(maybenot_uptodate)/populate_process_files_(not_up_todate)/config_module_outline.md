# `data_pipeline.config` Module Outline

The refactor will introduce a small Python package under `src/data_pipeline/config/`. Each module below is intended to expose plain Python objects (dicts, simple namespaces, enums) that callers can import directly. Values should include concise comments citing their source (legacy script, experiment notes, paper, etc.).

---

## `paths.py`
- `MORPHSEQ_DATA_PIPELINE_ROOT`, `RAW_INPUT_DIR`, `PROCESSED_IMAGE_DIR`, `SEGMENTATION_DIR`, etc. as `pathlib.Path`.
- Filename templates/wildcards (e.g., `SNIP_FILENAME = "{snip_id}.jpg"`).
- Column names used across steps (`USE_EMBRYO_COLUMN`, `TIME_COLUMN`).
- Helper functions: `experiment_dir(date)`, `snip_image_path(experiment_id, snip_id)`.
- Environmental overrides (e.g., `os.environ.get("MORPHSEQ_SANDBOX")`) applied once here.

## `microscopes.py`
- Per-microscope geometry (tile size, overlap, rotation
- Channel ordering and naming for Keyence, YX1, future scopes. (yeah there are some problessm wand dom adhoc filnnlel names 
- Exposure/bit-depth expectations for validation.
- Default preprocessing knobs (e.g., Keyence z-stack focus window).
- Registry helper: `MICROSCOPE_CONFIGS["keyence"]`.

## `models.py`
- Canonical names and checkpoints for SAM2, Grounding DINO, each UNet head, and VAE variants.
- Expected input shape or preprocessing flags per model.
- Version tags / hashes to detect stale weights.
- Convenience accessor: `get_checkpoint("sam2_prod")`.

## `segmentation.py`
- Grounding DINO thresholds: confidence, NMS IoU, max detections per frame.
- SAM2 propagation settings: bidirectional flag, stride, mask smoothing parameters.
- Mask export options: PNG compression, palette mapping, RLE precision.
- Limits for detection filtering (min embryo area, aspect ratio bounds).

## `snips.py`
- Crop padding percentages, guard rails for bounding boxes.
- Rotation parameters: PCA window size, minimum variance thresholds.
- Augmentation toggles and strengths (noise sigma, blur kernels).
- Serialization settings for snip IO (JPEG quality, color profile).

## `qc_thresholds.py`
- Imaging QC distances (edge distance, bubble proximity radius).
- Viability persistence hours, smoothing windows, dead/alive thresholds.
- Tracking QC parameters: speed z-score cutoff, max displacement, smoothing polynomial order.
- Mask QC bounds: area floor/ceiling fractions, overlap IoU cutoff, discontinuity tolerance.
- Size validation percentiles or z-score conversions tied to stage.

## `embeddings.py`
- VAE runtime knobs: batch size, latent dimension, precision, chunk size.
- Minimum snip count or QC pass rate before embedding generation.
- Accepted input file patterns and intermediate cache locations.
- Python 3.9 interpreter path / conda env hints for subprocess wrapper.

## `registry.py` (optional façade)
- Shared API that wraps the individual modules:
  - `get_paths()`, `get_microscope(name)`, `get_qc_threshold(flag_name)`.
- Merge logic for user overrides (e.g., YAML passed via Snakemake).
- Runtime validation helpers that log applied overrides.

---

### Notes on Overrides
- Defaults live in the Python modules.
- Snakemake rules can accept a small YAML to override specific keys; the rule merges YAML dicts into the config objects exposed above.
- Keep YAML-friendly names aligned with module keys (e.g., `qc_thresholds.movement.speed_zscore_cutoff`).

### Documentation Expectations
- Each exported value should have a short comment referencing the legacy script / paper where it was derived.
- When parameters are highly experiment-specific, indicate that in the docstring and provide safe defaults.

