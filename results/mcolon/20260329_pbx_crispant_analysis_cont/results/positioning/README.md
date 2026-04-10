# Positioning

This family contains the core phenotypic positioning runs.

High-level script sources:
- `01_bridge_qc_heatmaps.py`
- `02_bridge_multiclass_positioning.py`
- `03_bridge_feature_over_time.py`
- `04_compare_bridge_with_without_wik_ab.py`
- `05_pbx_condensation.py`
- `06_pbx_trajectory_viz.py`
- `08_pairwise_probe_fingerprint.py`
- `09_compare_pairwise_vs_multiclass.py`

Subfolders:
- `bridge/`: bridge-enabled comparisons and QC plots
- `multiclass/`: multiclass positioning bundles
- `pairwise/`: pairwise probe positioning, alignment, and comparisons

Typical outputs:
- `*.csv` positioning tables
- `*.png` UMAP / trajectory panels
- `*.gif` when the geometry is animated
- summary markdown explaining the comparison being tested
