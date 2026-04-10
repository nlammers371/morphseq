# Benchmarks

This family contains synthetic or semi-synthetic benchmark runs used to tune and sanity-check the condensation solver.

High-level script sources:
- `bifurcating_trunk_sandbox.py`
- `temporal_sandbox.py`
- `slice_sandbox.py`
- `void_sandbox.py`
- `solver_tempo_sweep.py`
- `make_solver_tempo_summary_fig.py`
- `run_condensation_experiment.py`
- `force_discovery_sweep.py`
- `force_regime_gif.py`
- `animate_3d_temporal.py`
- `animate_slice_sandbox.py`
- `animate_void_3d.py`

Subfolders:
- `bifurcating_trunk/`: Y / trunk calibration and force-fusion tests
- `condensation/`: condensation parameter sanity checks
- `force/`: force tuning and calibration bundles
- `force_calibration/`: calibration-specific benchmark bundles
- `slice/`: 2D slice-level benchmark geometry
- `solver/`: solver schedule and tempo checks
- `temporal/`: toy temporal stitching benchmark
- `trees/`: tree / principal-graph benchmark runs
- `void/`: void-repulsion benchmark runs

Typical outputs:
- per-condition geometry folders
- `branch_geometry.png`
- `slices_initial.png` / `slices_final.png`
- `trunk_3d_before_after.png`
- `*.gif` rotation or before/after animations
- summary CSVs for threshold and regime sweeps
