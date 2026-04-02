# Pairwise Contrast Coordinates Handoff

## What was implemented

### Classification engine

The binary classification path now supports native pairwise contrast-coordinate outputs via `save_contrast_coordinates=True`.

Files:
- `src/analyze/classification/run_classification.py`
- `src/analyze/classification/engine/loop.py`
- `src/analyze/classification/engine/contrast_coordinates.py`
- `src/analyze/classification/engine/analysis.py`
- `src/analyze/classification/tests/test_run_classification.py`

New persisted layers:
- `raw_contrast_scores_long`
- `contrast_specificity_by_timebin`
- `raw_coordinates`
- `shrunk_coordinates`
- `residual_coordinates`
- `probe_index`

Behavior:
- binary-only API
- requires `n_permutations > 0`
- raw margins are `m_raw = 2 * p_pos - 1`
- shrinkage uses `w = clip((auroc_obs - auroc_null_mean) / 0.5, 0, 1)` per comparison x time bin
- `ClassificationAnalysis.save()` / `load()` now persist these layers

### PBX pairwise runner and comparison

Files:
- `results/mcolon/20260329_pbx_crispant_analysis_cont/08_pairwise_probe_fingerprint.py`
- `results/mcolon/20260329_pbx_crispant_analysis_cont/09_compare_pairwise_vs_multiclass.py`

The pairwise runner now writes:
- parquet-backed analysis bundle from `run_classification()`
- CSV exports of the contrast-coordinate layers
- dense trajectory-ready exports:
  - `pairwise_raw_vectors.csv`
  - `pairwise_shrunk_vectors.csv`

The comparison script now produces:
- aligned comparison metrics across multiclass / pairwise raw / pairwise shrunk
- UMAP coordinate exports
- side-by-side figure output

## Cosmology follow-up completed

File:
- `results/mcolon/20260329_pbx_crispant_analysis_cont/05_pbx_condensation.py`

Changes:
- accepts multiclass or pairwise vector CSV input
- supports `--input-type {auto,multiclass,pairwise}`
- saves `position_history` and `snapshot_iters` into `condensed_positions.npz`
- emits `rotation.gif` and `iterations.gif` when snapshots are available

Executed analyses:
- pairwise raw AlignedUMAP condensation
- pairwise shrunk AlignedUMAP condensation
- Plotly init-vs-final exports for both
- flattened Plotly CSV exports for both
- mirrored both finished bundles into `shared/`

Primary result directories:
- `results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_raw_condensation_aligned_umap_bin4_perm500`
- `results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_shrunk_condensation_aligned_umap_bin4_perm500`
- `results/mcolon/20260329_pbx_crispant_analysis_cont/shared/pairwise_raw_condensation_aligned_umap_bin4_perm500`
- `results/mcolon/20260329_pbx_crispant_analysis_cont/shared/pairwise_shrunk_condensation_aligned_umap_bin4_perm500`

Key outputs in each bundle:
- `trajectories_3d.html`
- `plotly_points_init_and_final.csv`
- `rotation.gif`
- `iterations.gif`
- `plot_trajectories.png`
- `plot_panels.png`
- `plot_stacked_3d.png`
- `metrics.csv`
- `condensed_positions.npz`

## Verification already run

- `python -m py_compile` on modified classification and PBX scripts
- `pytest src/analyze/classification/tests/test_run_classification.py -v`
- smoke run of `08_pairwise_probe_fingerprint.py`
- full pairwise run producing `phenotypic_positioning_pairwise_bin4_perm500`
- full `09_compare_pairwise_vs_multiclass.py`
- full raw and shrunk AlignedUMAP condensation runs

## Current state

The pairwise raw and pairwise shrunk coordinate systems are implemented, saved, and visualized through the trajectory cosmology pipeline.

The Plotly views already include the AlignedUMAP initialization vs final condensed geometry toggle. The GIFs cover:
- final rotating geometry
- optimization progress over saved iterations

## Next step requested

Try to generate a 2D phenotypic phylogeny tree with branch points supported by statistical significance.

Suggested framing:
- derive a 2D condition-level tree / branching graph from the time-evolving pairwise geometry
- quantify branch support instead of drawing a purely visual tree
- candidate support measures: bootstrap over embryos, time-bin stability, centroid-separation significance, or edge persistence across representations
