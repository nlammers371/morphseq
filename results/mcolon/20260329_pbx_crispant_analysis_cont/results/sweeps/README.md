# Sweeps

This family contains parameter sweeps over the solver and geometry controls.

High-level script sources:
- `07_k_attract_sweep.py`
- `16_knn_stability_sweep.py`
- `17_coherence_stability_sweep.py`
- `18_force_ablation_coherence.py`
- `19_attraction_weight_sweep.py`
- `20_calibrated_force_band_sweep.py`

Subfolders:
- `force/`: force amplitude, bandwidth, and coherence sweeps
- `knn/`: local-neighborhood sensitivity checks

Typical outputs:
- one folder per parameter setting
- `metrics_history.csv`
- `plot_*` comparison figures
- `*_summary.csv`
- optional `3d_before_after.gif` and `trunk_3d_before_after.png`
