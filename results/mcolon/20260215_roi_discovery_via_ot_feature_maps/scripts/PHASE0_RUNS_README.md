# Phase 0 Runs Orientation

This note is the quickest map of which scripts generate which `phase0_*` run folders under
`results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/`.

## Script Topology

- `s01_select_reference_mask.py`
  - ranks WT reference candidates on the canonical grid
  - outputs `output/reference_mask_candidates/`
- `s01b_visualize_masks.py`
  - visual QC for a chosen reference plus sampled WT/mutant masks
  - outputs `output/mask_qc/`
- `s02_compute_ot_features.py`
  - feature-cache path
  - writes `feature_dataset/` and `qc/`
  - does not run the full downstream classification/nulls pipeline
- `s02_run_phase0.py`
  - monolithic full-run path
  - calls `run_phase0.py` and writes `feature_dataset/`, `qc/`, `viz/`, and `phase0_summary.json`
- `s03_visualize_differences.py`
  - visualization-only pass from an existing `feature_dataset/`
  - writes `viz/` and `features_sbins.parquet`
- `s04_run_classification.py`
  - AUROC/interval/bootstrap pass from `features_sbins.parquet`
  - writes `results/`
- `s05_run_nulls_and_stability.py`
  - slower null/bootstrap rerun from an existing run directory
  - writes `nulls/`
- `reviz_run003.py`
  - redraws cached figures for a previously generated run
- `reviz_phase0.py`
  - generic cached-run contour redraw with publication/presentation presets

## How To Read A Run Folder

- Presence of `phase0_summary.json` strongly indicates a run came from `s02_run_phase0.py` or a close variant of that full pipeline.
- Presence of `feature_dataset/` without `phase0_summary.json` usually indicates `s02_compute_ot_features.py` or a one-off debug/feature-cache driver.
- `feature_dataset/manifest.json` is the best on-disk source for:
  - `reference_mask_id`
  - `feature_set`
  - `canonical_grid`
- Early runs have inconsistent `stage_window` metadata in the manifest. Use it cautiously.

## Active Run Registry

| Run folder | Reference mask id | Feature set | Samples | Likely generator | Notes |
| --- | --- | --- | ---: | --- | --- |
| `phase0_run_001` | `20251112_H04_e01|frame_39` | `v0_cost` | 40 | `s02_run_phase0.py` | Full run, older outline family |
| `phase0_run_001__archived_20260223_202134` | `20251113_A05_e01|frame_95` | `v0_cost` | 20 | archived full run | Archived predecessor |
| `phase0_run_002` | `20251113_A05_e01|frame_95` | `v0_cost` | 20 | `s02_run_phase0.py` | Full run |
| `phase0_run_003` | `20251112_H04_e01|frame_39` | `v0_cost` | 20 | `s02_run_phase0.py` | Full run, non-preferred outline family |
| `phase0_run_004` | `20250512_B09_e01|frame_113` | `v0_cost` | 20 | `s02_run_phase0.py` | Full run, preferred outline family |
| `phase0_run_005_smoke` | `20250512_B09_e01|frame_113` | `v0_cost` | 8 | `s02_run_phase0.py` smoke variant | Small sample smoke run |
| `phase0_run_006_fast` | `20250512_B09_e01|frame_113` | `v0_cost` | 8 | `s02_run_phase0.py --fast` or similar | Same outline family as run 004 |
| `phase0_run_007_v1_pot_20x20` | `20251112_H04_e01|frame_39` | `v1_dynamics` | 40 | custom full-run variant | 5-channel dynamics run |
| `phase0_run_with_yolk_pivot` | `20250512_B09_e01|frame_113` | unknown on summary | 20 | feature-cache/debug path | Has `feature_dataset/`, no full summary |

## Shape Mismatch Diagnosis

- The Gaussian contour figure shape comes from the cached `feature_dataset/features.zarr/mask_ref` for that run.
- The Gaussian smoothing is not the cause of the outline mismatch.
- The mismatch is mostly a run/reference choice issue:
  - `phase0_run_003` / `phase0_run_001` / `phase0_run_007_v1_pot_20x20`
    - reference family: `20251112_H04_e01|frame_39`
    - narrower, more cylindrical outline
  - `phase0_run_004` / `phase0_run_005_smoke` / `phase0_run_006_fast`
    - reference family: `20250512_B09_e01|frame_113`
    - broader head / ventral bulge outline
- This repo now treats `phase0_run_004` as the preferred cached base for the Gaussian contour redraws.

## Important Provenance Caveat

- Same-reference runs are not guaranteed to have byte-identical `mask_ref` arrays.
- Example: `phase0_run_001` and `phase0_run_003` both point to `20251112_H04_e01|frame_39`, but their cached `mask_ref` arrays differ.
- Interpretation:
  - historical outputs are run-specific artifacts
  - do not assume reference ID alone is enough to reproduce a figure exactly
  - for figure work, treat the run directory itself as the provenance unit

## Recommended Figure Base

- Use `scripts/output/phase0_run_004/` for contour figures when you want the broader embryo shape shown in the preferred comparison image.
- Use `scripts/reviz_phase0.py phase0_run_004 --preset both` to regenerate the two contour variants from cache without recomputing OT.
