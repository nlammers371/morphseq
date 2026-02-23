Refactor-013 Addendum (2025-09-11): Per‑Experiment Mask Manifest and Video Rendering

- Stage 5 (06_export_masks.py) now writes a per‑experiment mask export manifest when the SAM2 annotations JSON contains exactly one experiment.

- Manifest paths:
  - Per‑experiment: sam2_pipeline_files/exported_masks/<exp_id>/mask_export_manifest_<exp_id>.json
  - Monolithic JSON (multiple experiments): sam2_pipeline_files/exported_masks/mask_export_manifest.json (unchanged)

- Implementation:
  - scripts/utils/simple_mask_exporter.py detects per‑experiment JSONs and uses the suffixed manifest path.
  - Manifest is updated on every run (even when no new masks are exported) to ensure presence and freshness.

- Validation:
  - Verified on 20250529_36hpf_ctrl_atf6 and an additional experiment; CSV export valid; mask file paths validated.

- Video generation utilities:
  - render_eval_video.py: import path made robust; supports per‑experiment JSONs.
  - make_eval_video.sh: auto‑detects per‑experiment JSONs; derives EXP_ID from video IDs; sensible default output dir.

---

Build03 Direct Interface Plan (Per‑Experiment)

Goal: Run Build03 directly on a single experiment (without the full pipeline), consuming per‑experiment SAM2 outputs and emitting a per‑experiment Build03 CSV.

Inputs (per‑experiment)
- sam2_segmentations: sam2_pipeline_files/segmentation/grounded_sam_segmentations_{exp}.json
- sam2_csv (preferred): sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv
- masks_dir: sam2_pipeline_files/exported_masks/{exp}/masks/
- mask_manifest: sam2_pipeline_files/exported_masks/{exp}/mask_export_manifest_{exp}.json
- built01_metadata: metadata/built_metadata_files/{exp}_metadata.csv

Output (per‑experiment)
- metadata/build03_output/expr_embryo_metadata_{exp}.csv

Proposed CLI (run Build03 directly)
- Script: src/run_morphseq_pipeline/steps/run_build03.py (new)
- Usage:
  - python run_build03.py --data-root <root> --exp {exp}
  - Optional overrides:
    - --sam2-csv, --sam2-json, --masks-dir, --mask-manifest, --built01-csv, --out-dir
  - Flags: --overwrite, --validate-only, --no-manifest-check, --no-geometry, --verbose

Intake/Validation
- Prefer sam2_csv if present; otherwise derive from sam2_json + masks_dir.
- Require built01_metadata; if absent, proceed with pixel units and mark scale fields NA.
- If mask_manifest present, cross‑check counts and a few sample paths; warn on mismatches.

Computation (per image/embryo)
- Read labeled mask from exported_mask_path (sam2_csv) or construct path via masks_dir + image_id.
- Extract features: area_px, perimeter_px, centroid; compute time_int from t####; parse video_id/well_id from image_id; enrich with built01 metadata if available.
- If pixel size present, augment with area_um2, perimeter_um, centroid_um; else set NA.

CSV Schema (minimum)
- Identifiers: exp_id, video_id, well_id, image_id, embryo_id, snip_id, time_int
- Geometry: area_px, perimeter_px, centroid_x_px, centroid_y_px
- Scaling: area_um2, perimeter_um, centroid_x_um, centroid_y_um (optional/NA)
- Provenance: exported_mask_path, sam2_source_json, computed_at
- Flags: use_embryo_flag (default true), notes

Write Output
- Path: metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv
- Overwrite only with --overwrite; otherwise skip if exists and report summary.

Test Plan (experiment 20250529_36hpf_ctrl_atf6)
- Data root: morphseq_playground
- Expected inputs:
  - sam2_csv: morphseq_playground/sam2_pipeline_files/sam2_expr_files/sam2_metadata_20250529_36hpf_ctrl_atf6.csv
  - masks_dir: morphseq_playground/sam2_pipeline_files/exported_masks/20250529_36hpf_ctrl_atf6/masks/
  - mask_manifest: morphseq_playground/sam2_pipeline_files/exported_masks/20250529_36hpf_ctrl_atf6/mask_export_manifest_20250529_36hpf_ctrl_atf6.json
  - built01_metadata: morphseq_playground/metadata/built_metadata_files/20250529_36hpf_ctrl_atf6_metadata.csv
- Command (once run_build03.py is added):
  - python src/run_morphseq_pipeline/steps/run_build03.py \
    --data-root morphseq_playground \
    --exp 20250529_36hpf_ctrl_atf6 \
    --verbose
- Expected output:
  - metadata/build03_output/expr_embryo_metadata_20250529_36hpf_ctrl_atf6.csv
- Sanity checks:
  - Row count ≈ number of image×embryo entries with exported masks (68 in the last run).
  - exported_mask_path exists for each row.
  - well_id/video_id parsing match Build01 metadata; warn on wells absent from metadata.

Notes
- This direct Build03 entry point lets us iterate on per‑experiment I/O without invoking upstream stages.
- Once stabilized, ExperimentManager can call run_build03.py after Stage 6 for each experiment.

## Build03 Implementation Updates (2025-09-11)

### Issues Resolved

1. **SAM2 CSV Exporter Fixed**: 
   - Updated `export_sam2_metadata_to_csv.py` to accept `--metadata-json` argument
   - Now reads per-experiment metadata JSON directly instead of searching for monolithic files
   - Updated `run_sam2.py` to pass the per-experiment metadata path automatically

2. **predicted_stage_hpf Calculation Fixed**:
   - Issue: Field was empty in Build03 output CSV
   - Root cause: Column already existed (as empty string), so DataFrame calculation was skipped
   - Solution: Removed existence check, always calculate if input columns present
   - Uses legacy Kimmel et al 1995 formula: `start_age_hpf + (Time Rel (s)/3600) * (0.055*temperature - 0.57)`

3. **Micron Scale Calculations Fixed**:
   - Issue: `area_um2`, `perimeter_um`, `centroid_x_um`, `centroid_y_um` fields empty despite pixel scale data being available
   - Root cause: Script looked for pixel scale in Build01 CSV, but no dedicated `pixel_size_um` column exists there
   - Investigation: Build01 calculates `pixel_size_um = width_um / width_px` but doesn't store it in CSV
   - Solution: Use pixel scale data directly from SAM2 CSV (`width_um`, `width_px`, `height_um`, `height_px`)
   - Calculation: `um_per_pixel_x = width_um / width_px`, `um_per_pixel_y = height_um / height_px`
   - Fallback: If SAM2 data missing, calculate from Build01 CSV Width/Height columns

4. **Geometry Computation Made Default**:
   - Changed from optional `--compute-geometry` flag to default behavior
   - Now use `--no-geometry` to skip (geometry is critical for Build04)
   - Ensures all downstream processing has required geometric measurements

### Pixel Scale Data Flow Verification

**SAM2 CSV** (enriched with per-experiment metadata):
- `width_um=3623.441, width_px=1920, height_um=2717.581, height_px=1440`
- Pixel scale: `x=1.887 um/px, y=1.887 um/px`

**Build01 CSV** (raw image metadata):
- `Width (um)=3623.441, Width (px)=1920, Height (um)=2717.581, Height (px)=1440`
- No stored pixel scale column - calculated on-demand

**Build01 Source** (diffusion-dev:src/build/build01A_compile_keyence_torch.py:L123):
```python
pixel_size_um = meta['Width (um)'] / meta['Width (px)']
```

### Current Status
- ✅ SAM2 CSV export with per-experiment metadata: Working
- ✅ predicted_stage_hpf calculation: Working (36.00 hpf for t=0000 at 24°C)
- ✅ Pixel geometry extraction: Working by default (critical for Build04)
- ✅ Micron conversions: Working using SAM2 CSV pixel scale data
- 🔄 Testing: Complete validation with test experiment 20250529_36hpf_ctrl_atf6
