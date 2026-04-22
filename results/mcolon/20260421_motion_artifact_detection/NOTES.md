# Motion Artifact Detection — Exploration Notes
Date: 2026-04-21

## Problem
During Z-stack acquisition (15 slices, 50µm step, ~1-2s sweep), individual embryos
can move independently. This produces motion artifacts in the focus-stacked output.
Detection must be per-embryo, not per-frame, because well-behaved embryos in the
same frame dilute any whole-image signal.

## Data
- Experiment: 20250912, ND2 shape (T=113, P=95, Z=15, Y=2189, X=2189)
- Z-step: 50µm, pixel size: 3.23µm, home index: Z=7
- 10 labeled examples: 4 Bad, 3 Okay, 3 Great (from frame_nd2_lookup.csv)
- Masks: binary PNGs per embryo per timepoint (emnum_1, emnum_2), same resolution

## Metrics Evaluated

### Per-slice (inside embryo mask)
- `lap_var`, `tenengrad`, `brenner`, `mod_lap` — classical sharpness operators
- `log_mean` — mean |LoG| response inside mask  ✅ CHOSEN
- `entropy` — Shannon entropy of pixel intensities inside mask  ✅ CHOSEN
- `hf_ratio` — high-frequency FFT energy fraction
- `log_sharp_frac` — fraction of mask pixels above 75th pct LoG — USELESS (by
  construction always ~25%, dropped)
- `edge_density`, `grad_anisotropy`

### Adjacent Z-pair (inside mask)
- `ncc` — normalized cross-correlation between consecutive slices  ✅ STRONGEST SIGNAL
- `ssim_score`
- `phase_shift_mag` — lateral shift estimated by phase correlation (pixels)  ✅ USEFUL
- `post_align_residual`, `mask_iou`, `centroid_shift`, `area_ratio`

### Whole-stack summaries
- `bad_pair_frac_ncc` — fraction of pairs with NCC < 0.90  ✅ STRONGEST SIGNAL
- `ncc_min`, `longest_bad_ncc_run`
- `winner_z_entropy_inside`, `winner_z_disc_ratio` — from LoG winner-Z map

## Key Findings

### What works
1. **`ncc_min` and `bad_pair_frac_ncc`** are the strongest discriminators.
   All Great stacks: ncc_min > 0.93, bad_pair_frac = 0.
   Three of four Bad stacks: ncc_min < 0.50, bad_pair_frac > 0.14.
   Threshold suggestion: `ncc_min < 0.85` OR `bad_pair_frac_ncc > 0.10`

2. **`log_mean`** (absolute, not relative) — Bad stacks cluster lower.
   Best used as mean over all Z slices per embryo-stack.

3. **`rel_entropy`** (embryo minus background entropy) — Bad stacks are more
   negative (embryo much less textured than background). Useful for catching
   the tricky case (B10 t=97) that NCC misses.

4. **Winner-Z spatial chaos** — visually compelling. Bad embryos show chaotic
   Z-index maps inside the mask; Great embryos show smooth gradients.
   Captured by `winner_z_disc_ratio` but needs more examples to threshold.

### What doesn't work
- `lap_var_max` — complete overlap between Bad and Great
- `log_sharp_frac` — useless by construction (always ~25%)
- Relative normalization by background (bg_log_mean is nearly flat across Z,
   dividing by a constant doesn't change curve shape or separation)

### Tricky case: B10 t=97
- Labeled Bad, but NCC looks clean (ncc_min=0.915, bad_pair_frac=0)
- Caught by: low `log_mean` overall, strongly negative `rel_entropy` (−0.51)
- Hypothesis: motion happened *within* a single Z slice (blurring one slice)
  rather than *between* slices. NCC-pair approach cannot see within-slice blur.

## Composite Score (proposed)
Simple first-pass threshold combining the two complementary signals:
  BAD if: `ncc_min < 0.85` OR `bad_pair_frac_ncc > 0.10`
  SOFT BAD if: `log_mean_mean < 1.5 * median(log_mean_mean across all embryos in frame)`

The NCC catches between-slice motion; log_mean catches globally soft stacks
(within-slice blur, bad focus, dim embryo).

## Next Steps
1. Test on multi-embryo frames — the real use case is per-embryo comparison
   within the same frame at the same Z. Embryo vs embryo normalization (not
   vs background) is the right relative metric.
2. Wire NCC + log_mean into `_focus_stack()` in `stitched_ff_builder.py` to
   emit a per-embryo quality flag alongside the stacked image.
3. Save winner-Z maps as a diagnostic artifact in built_image_data/.
4. Label more examples, especially edge cases (partially moving embryos,
   dim wells, early timepoints with small embryos).

## Scripts
- `01_zstack_metric_exploration.py` — computes all metrics, saves CSVs + figures
- `02_relative_metrics.py` — background normalization experiment (not useful, kept for reference)

## Outputs
- `slice_metrics.csv` — per (example, embryo, z): all focus metrics
- `pair_metrics.csv` — per (example, embryo, z-pair): NCC, SSIM, phase shift
- `stack_metrics.csv` — per (example, embryo): summaries + winner-Z diagnostics
- `figures/focus_curves.png` — absolute per-slice metrics across Z
- `figures/focus_curves_relative.png` — relative metrics (not useful)
- `figures/pair_metrics.png` — NCC and phase-shift per adjacent pair  ← most diagnostic
- `figures/winner_z_maps.png` — spatial winner-Z maps (chaotic=bad, smooth=good)
- `figures/stack_metrics_dotplot.png` — per-stack metric separation
- `figures/zstrip_*.png` — Z-slice strips with NCC annotations
