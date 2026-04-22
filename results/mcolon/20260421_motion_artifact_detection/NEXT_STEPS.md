# Next Steps — Motion Artifact Detection

## Where we are

Exploratory analysis complete. Two orthogonal metrics chosen:
- **NCC** (between-slice motion) — primary detector
- **rel_entropy** (within-slice blur) — catches the NCC-invisible case

Production utility written in:
`src/data_pipeline/quality_control/zstack_motion_qc/`

See that module's README for design rationale and usage.

---

## Immediate next step: full ND2 scan + distribution analysis

**Script to write:** `results/mcolon/20260421_motion_artifact_detection/06_full_nd2_scan.py`

**Goal:** Run the production utility (`zstack_motion_qc`) over an entire raw ND2
file, compute NCC and rel_entropy grids for every (T, P) stack, save per-stack
scalar summaries, and characterize the distribution to find natural QC thresholds.

### What the script should do

1. **Load** the ND2 file (already known: `20250912_WT_tricane_serial_dilution_experiment.nd2`,
   shape T=113, P=95, Z=15, Y=2189, X=2189). Iterate over all (T, P) combinations.

2. **Compute grids** using `compute_local_ncc_grid` + `compute_local_entropy_grid`
   with `tile_size=128`. No masks at this stage — whole-frame grids only.

3. **Reduce to stack summaries** using `ncc_stack_summary` + `entropy_stack_summary`
   from `summaries.py`. Save as a CSV: one row per (T, P) stack.

4. **2D scatter plot**: x = `ncc_min`, y = `entropy_mean` (or `rel_entropy_mean`
   if background tiles can be estimated). Color by density or well identity.
   Goal: visually confirm the two metrics are orthogonal and identify natural clusters.

5. **Find exemplars**: from the scatter, pull out:
   - Top-right cluster (high NCC, high entropy) → Great
   - Bottom-left (low NCC, low entropy) → Bad
   - Middle / edge cases → Okay
   Label a handful by hand and overlay on the scatter.

6. **Distribution panels**: histogram of `ncc_min` and `entropy_mean` separately,
   with the labeled examples marked. Propose thresholds.

### Notes

- Import from `src.data_pipeline.quality_control.zstack_motion_qc` — do not
  copy-paste functions from the exploratory scripts.
- Masks are not available at this stage. Use whole-frame tile means as a proxy;
  per-embryo numbers come later once SAM2 masks exist.
- Background entropy estimation: use corner tiles (outside typical embryo region)
  as a rough bg reference for rel_entropy before real masks are available.
- Memory: loading all (T, P) stacks at once is ~113 × 95 × 15 × 2189² × 4 bytes ≈
  way too large. Stream one stack at a time via dask, compute grids immediately,
  discard the raw stack.

---

## After the scan

- Wire `compute_local_ncc_grid` + `compute_local_entropy_grid` + `save_grids` into
  `_focus_stack()` in `src/data_pipeline/image_building/yx1/stitched_ff_builder.py`.
- Add `embryo_qc_flag` call to the segmentation QC step once masks are available.
- Label more examples (edge cases: partial motion, dim wells, small/early embryos).
