# Stream F: Feature Development — Decisions

## Status
- Implemented and validated end-to-end (storage + feature extraction + DCT spike runner + tests).
- All targeted tests pass:
  - `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_feature_compaction_storage.py`
  - `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_feature_compaction_features.py`

## Output Contract (Implemented)

### 1) `ot_pair_metrics.parquet` (catalog)
- Composite key: `(run_id, pair_id)`
- Writer behavior: idempotent upsert (replace on key collision, never duplicate)
- Config categoricals: `backend`, `metric`, `canonical_grid_align_mode`, `mass_mode`, `align_mode`
- Artifact references are tracked separately under `pair_artifacts/<pair_id>/`
- Added robust dtype normalization for identifier/meta columns (`run_id`, `pair_id`, experiment/date IDs, etc.) to avoid mixed int/string parquet failures across reruns

### 2) Pair artifact bundle
- Path root: `.../dct_spike_results/pair_artifacts/<pair_id>/`
- Saved payloads:
  - `fields.npz`: velocity + mass maps (`float32` default)
  - `barycentric_projection.npz`: source points, pushed target points, barycentric velocity, transported mass per source point
  - `metadata.json`: shapes, support counts, pair-frame metadata
- Decision: do not store full coupling matrix in dataframe artifacts

### 3) `ot_feature_matrix.parquet` (ML/PCA view)
- Composite key: `(run_id, pair_id)` with idempotent upsert
- Schema versioned (`feature_schema_version`)
- Stores raw values only (no dataset-level normalization/scaling)
- Feature blocks:
  - OT scalar metrics (cost, transported mass %, created/destroyed %, etc.)
  - Barycentric displacement summaries (`mean/std/p50/p90/p95/max`)
  - DCT radial band-energy fractions on `vx`, `vy`, `divergence`, `curl`
    - Default `n_dct_bands=8` => 32 spectral features per pair

## DCT Spike Execution (Canonical Stream-A pair)

Pair:
- `20251113_A05_e01__f0014__to__20251113_E04_e01__f0014`
- Stage loaded: `48.17 hpf` vs `48.17 hpf`
- `max_support_points=5000`

Runs:
- OTT: `streamf_spike_20260208_ott_gpu_m5000_rel`
- POT: `streamf_spike_20260208_pot_m5000_rel`

Runtimes:
- OTT: `~7.4s`
- POT: `~13.6s`

Compaction result (criterion `cosine>=0.98` and `rmse_rel<=0.10`):
- OTT: recommended `k=110592` (`k_ratio=0.75`, `rmse_rel=0.0349`, `energy=0.9809`)
- POT: recommended `k=110592` (`k_ratio=0.75`, `rmse_rel=0.0349`, `energy=0.9809`)

Alternative operating point (criterion `cosine>=0.98` and `rmse_rel<=0.20`):
- OTT/POT both reach at `k=73728` (`k_ratio=0.50`, `energy=0.9132`)

## Operational Decision
- Do not use raw top-K DCT coefficients as the primary dataframe feature representation.
- Use the compact feature table (`ot_feature_matrix.parquet`) as the default modeling interface.
- Keep top-K reconstruction sweeps as diagnostics/calibration outputs in stream-F artifacts.
- Keep barycentric projection artifacts enabled for registration QA/debugging and group-average visual checks.

## Key Artifacts
- `src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/storage.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/features.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/dct_compaction_spike.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_feature_compaction_storage.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_feature_compaction_features.py`
