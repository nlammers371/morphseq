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
- `ncc` — normalized cross-correlation between consecutive slices  ✅ KEEP
- `phase_shift_mag` — lateral shift estimated by phase correlation (pixels)  DROP (redundant with NCC — NCC already captures that motion occurred)
- `local_ncc_min`, `local_ncc_std` — NCC computed per 128px tile; min=worst tile,
  std=spatial spread of motion. **Not shown in exploratory figure — will be derived
  from saved NCC grids during pipeline QC post-hoc.**
- `ssim_score` — DROP (redundant with NCC)
- `nmi` — DROP (too noisy, ranges overlap across labels)
- `post_align_residual`, `mask_iou`, `centroid_shift`, `area_ratio`

### Whole-stack summaries
- `bad_pair_frac_ncc` — fraction of pairs with NCC < 0.90  ✅ KEEP
- `ncc_min`, `longest_bad_ncc_run`
- `winner_z_entropy_inside`, `winner_z_disc_ratio` — from LoG winner-Z map

## Key Findings

### Chosen metrics (two orthogonal signals)
1. **NCC family** (`ncc_min`, `bad_pair_frac_ncc`) — between-slice motion detector.
   All Great stacks: ncc_min > 0.93, bad_pair_frac = 0.
   Three of four Bad stacks: ncc_min < 0.50, bad_pair_frac > 0.14.
   Threshold: `ncc_min < 0.85` OR `bad_pair_frac_ncc > 0.10`

2. **`rel_entropy`** (embryo entropy − background entropy per Z slice) — catches
   within-slice blur and signal quality. Bad stacks strongly negative (−0.5 to −2.2
   mean); Great stacks near zero (−0.1 to −0.4). Orthogonal to NCC — catches the
   tricky case (B10 t=97: passes NCC, flagged by rel_entropy −0.51).

### What was tried and dropped
- `log_mean`, `lap_var_max` — complete overlap between Bad and Great
- `log_sharp_frac` — useless by construction (always ~25%)
- `ssim_score` — redundant with NCC
- `nmi` — too noisy, ranges overlap
- Background normalization — bg entropy nearly flat across Z, no separation benefit

### Local NCC (not dropped — deferred to pipeline)
- `local_ncc_min` and `local_ncc_std` computed per 128px tile inside embryo bbox.
- local_ncc_std captures spatially non-uniform motion (one end moves, other stays).
- **Plan**: save `(14 pairs, N_tiles_y, N_tiles_x)` NCC grids during `_focus_stack()`.
  Post-hoc, intersect with embryo mask to derive local_ncc_min/std per embryo.
  Also save `(15 slices, N_tiles_y, N_tiles_x)` entropy grids for rel_entropy.
  Grid size: ~17×17 tiles at 128px → ~33KB per frame, negligible storage.

### Tricky case: B10 t=97
- Labeled Bad, but NCC looks clean (ncc_min=0.915, bad_pair_frac=0)
- Caught by: strongly negative `rel_entropy` (−0.51 mean)
- Hypothesis: motion happened *within* a single Z slice (blurring one slice)
  rather than *between* slices. NCC-pair approach cannot see within-slice blur.

## Next Steps (pipeline)
1. Add grid-saving to `_focus_stack()` in `stitched_ff_builder.py`:
   - NCC grid: `(14, N_ty, N_tx)` float32 — adjacent Z-pair NCC per tile
   - Entropy grid: `(15, N_ty, N_tx)` float32 — Shannon entropy per tile per slice
   - Save as .npy alongside focus-stacked TIFFs in built_image_data/
2. Post-hoc QC (after segmentation produces embryo masks):
   - Load grid + mask, intersect tiles with embryo bbox
   - Derive: ncc_min, bad_pair_frac, local_ncc_std, rel_entropy_mean per embryo
   - Flag: `ncc_min < 0.85` OR `bad_pair_frac > 0.10` OR `rel_entropy_mean < threshold`

## Scripts
- `01_zstack_metric_exploration.py` — computes all metrics, saves CSVs + figures
- `02_relative_metrics.py` — background normalization experiment (not useful, kept for reference)
- `03_ranked_metric_viz.py` — 6-column comparison figure (focus image + Z slices + metric bars)
- `04_pair_metrics_plot.py` — NCC and phase-shift per adjacent pair
- `05_add_mi_and_local_ncc.py` — computes NMI + local NCC; generates all-tested-metrics figure

## Outputs
- `slice_metrics.csv` — per (example, embryo, z): all focus metrics
- `pair_metrics.csv` — per (example, embryo, z-pair): NCC, SSIM, phase shift
- `pair_metrics_extended.csv` — pair_metrics + nmi, local_ncc_min/std/n
- `stack_metrics.csv` — per (example, embryo): summaries + winner-Z diagnostics
- `slice_metrics_relative.csv` — rel_entropy per (example, embryo, z)
- `figures/ranked_metric_comparison.png` — main diagnostic figure ← most useful
- `figures/pair_metrics_all_tested.png` — all tested metrics with KEEP/DROP badges
- `figures/pair_metrics_final.png` — NCC and phase-shift only
- `figures/winner_z_maps.png` — spatial winner-Z maps (chaotic=bad, smooth=good)
