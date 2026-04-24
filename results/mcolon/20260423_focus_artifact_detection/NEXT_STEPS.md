# Next Steps — Focus Artifact Detection

## Current state (2026-04-23)

### What we know
- `rel_entropy_mean` = embryo Shannon entropy − background entropy (per Z-slice, averaged over stack)
- **More negative = worse**: values around −1.0 to −1.2 are catastrophic motion/blur (embryo reduced to a bright streak); values around −0.1 to −0.4 are sharp, well-focused embryos
- A rough threshold of **−0.5** separates clearly bad from recognizable embryos in the 10-embryo labeled set
- Gallery figure (`07_focus_output_tail/rel_entropy_gallery.png`) confirms the signal visually

### What we do NOT yet have
- `rel_entropy_mean` computed for the **full dataset** (~9 K embryo-timepoints)
  - The 10-embryo labeled set (`slice_metrics_relative.csv`) is the only source so far
  - `stack_summaries.csv` from the full nd2 scan has `entropy_mean` (absolute, no background subtraction) but not `rel_entropy`

---

## Immediate next steps

### 1. Compute rel_entropy for all embryos (BLOCKER)
Run a full-scan pass that computes `rel_entropy` (embryo − background) for every `(t, p)` in the nd2, using the SAM2 masks.
- Reference implementation: `02_relative_metrics.py` in `20260421_motion_artifact_detection/`
- Output should be a CSV analogous to `slice_metrics_relative.csv` but covering all ~9 K embryo-timepoints
- This is the prerequisite for everything below

### 2. Generate decile threshold-bin figures
Once rel_entropy is computed for all embryos, produce **10 figures**, one per decile of `rel_entropy_mean`:
- Each figure: **10 rows × 5 columns = 50 embryos**, sampled evenly from that decile's range
- Use the 2D focus-stacked JPEGs from `raw_data_organized/20250912/images/`
- Format mirrors `figures/threshold_bins/v2_ncc_p05_coverage25/` from the motion work
- Script stub: `09_rel_entropy_decile_bins.py` (needs the full rel_entropy CSV as input)

### 3. Pick a threshold
Inspect the decile figures and find the decile boundary where embryos transition from clearly bad to acceptable.
- Use `ncc_p05` (not `ncc_min`) as the motion context axis in any scatter
- Candidate threshold from labeled set: around −0.5; will likely shift with full dataset

### 4. Wire into QC DAG
Add `rel_entropy_mean` as a focus QC flag column in `src/data_pipeline/quality_control/zstack_motion_qc/embryo_qc.py`

---

## Notes
- Do not use `07_focus_analysis.py` for the full recompute — it is slow (writes only at the end). Write a chunked/parallel version.
- `entropy_mean` in `stack_summaries.csv` is absolute (no background correction) — not equivalent to `rel_entropy_mean`.
- Keep the full nd2 scan infrastructure from `06_scan_output/` as the template for the focus scan.
