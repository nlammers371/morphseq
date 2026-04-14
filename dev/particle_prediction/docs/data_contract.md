# MorphSeq Beta Predictor: Data Contract

## 1. Purpose

This document defines the canonical objects and invariants for the beta predictive model.

The goal is to prevent silent drift between:

- data loading,
- smoothing,
- resampling,
- transition-bank construction,
- prediction,
- and visualization.

If a field is important enough to be used by more than one module, it belongs here.

---

## 2. Core principles

- Every derived object must preserve lineage back to a source embryo and experiment.
- Raw, smoothed, and resampled trajectories are distinct objects.
- Gap handling is canonical data, not an implementation detail.
- History is stored canonically as ordered recent segments.
- Fast summary history features are optional derived fields, not replacements for canonical history.
- Time metadata is preserved for visualization and later stage work, even though it is not central to the v1 transition kernel.

---

## 2.1 Canonical gap semantics

Irregular sampling and missing frames materially affect smoothing, resampling, and transition-window validity. They are therefore part of the canonical contract.

### Definitions

- `observed_dts[i] = time_seconds[i + 1] - time_seconds[i]`
- `missing_frame_counts[i]` is the estimated number of missing nominal frames between samples `i` and `i + 1`
- `interpolatable_gap_mask[i]` marks a small gap that may be linearly filled for smoothing support
- `hard_gap_mask[i]` marks a gap large enough to break continuity for smoothing and transition extraction
- `segment_ids[j]` identifies the contiguous source segment containing sample `j`

### Default beta policy

- Small gaps may be interpolated for local smoothing support only.
- Hard gaps must break trajectory continuity.
- Transition windows must never cross a hard gap.
- Transition windows should record whether they touch an interpolated gap region so downstream weighting or filtering can use that information.
- Gap-aware annotations should survive through resampling so visual inspection and support diagnostics can explain why windows were kept or dropped.

---

## 3. Raw trajectory object

## 3.1 `EmbryoTrajectory`

This is the current base object and should remain close to the existing `loading.py` definition.

### Required fields

- `embryo_id: str`
- `trajectory: np.ndarray` of shape `(T, D)`
- `time_seconds: np.ndarray` of shape `(T,)`
- `delta_t: float`  
  experiment-level median frame interval in seconds
- `temperature: float`
- `perturbation_class: str`
- `experiment_id: str`

### Optional additions

- `metadata: dict[str, Any]`  
  for experiment-specific extra fields
- `frame_index: np.ndarray` if needed for debugging / round-tripping
- `observed_dts: np.ndarray` of shape `(T-1,)`
- `missing_frame_counts: np.ndarray` of shape `(T-1,)`
- `interpolatable_gap_mask: np.ndarray` of shape `(T-1,)`
- `hard_gap_mask: np.ndarray` of shape `(T-1,)`
- `segment_ids: np.ndarray` of shape `(T,)`

### Invariants

- `trajectory.shape[0] == time_seconds.shape[0]`
- `time_seconds` is monotone nondecreasing
- all rows of `trajectory` are finite
- if present, `observed_dts.shape[0] == T - 1`
- if present, `missing_frame_counts.shape[0] == T - 1`
- if present, `interpolatable_gap_mask.shape[0] == T - 1`
- if present, `hard_gap_mask.shape[0] == T - 1`
- if present, `segment_ids.shape[0] == T`
- `segment_ids` must increment only when continuity is broken by a hard gap or equivalent explicit segment boundary

---

## 4. Smoothed trajectory object

## 4.1 `SmoothedTrajectory`

Represents one smoothed latent trajectory.

### Required fields

- `source: EmbryoTrajectory`
- `smoothed: np.ndarray` of shape `(T, D)`
- `time_seconds: np.ndarray` of shape `(T,)`
- `method: Literal["savitzky_golay"]`
- `window_seconds: float`
- `window_frames: int`
- `poly_order: int`

### Optional fields

- `residuals: np.ndarray` of shape `(T, D)`  
  raw minus smoothed
- `diagnostics: dict[str, Any]`

### Invariants

- same number of samples as the raw source trajectory
- no NaNs or infs
- source lineage preserved exactly
- diagnostics should report any gap interpolation or segment-wise smoothing adjustments that materially changed the smoothing support

---

## 5. Arc-length-resampled trajectory object

## 5.1 `ResampledTrajectory`

Represents one smoothed trajectory reparameterized by arc length and resampled at fixed `delta_s`.

### Required fields

- `source: EmbryoTrajectory`
- `smoothed_source: SmoothedTrajectory`
- `resampled: np.ndarray` of shape `(S, D)`
- `arc_length: np.ndarray` of shape `(S,)`
- `delta_s: float`

### Recommended auxiliary fields

- `source_time_interp: np.ndarray` of shape `(S,)`  
  interpolated original time corresponding to each resampled point
- `source_frame_interp: np.ndarray` of shape `(S,)`  
  interpolated original frame index
- `increment_norms: np.ndarray` of shape `(S-1,)`
- gap-aware point annotations derived from the raw source when needed for filtering or diagnostics

### Invariants

- `arc_length[0] == 0`
- `arc_length` is monotone increasing
- adjacent resampled points are approximately `delta_s` apart except near the end
- source lineage preserved exactly
- any propagated gap annotations must remain traceable back to the raw source intervals or segments that generated them

---

## 6. Transition-window object

## 6.1 `TransitionWindow`

This is the canonical modeling unit.

### Required fields

- `state: np.ndarray` of shape `(D,)`  
  the current resampled state \(z_i\)
- `increment: np.ndarray` of shape `(D,)`  
  the next-step increment \(z_{i+1} - z_i\)
- `history_segments: np.ndarray` of shape `(K, D)`  
  canonical ordered history segments ending at the current state
- `embryo_id: str`
- `experiment_id: str`
- `perturbation_class: str`
- `source_segment_id: int`
- `touches_interpolated_gap: bool`
- `resampled_index: int`
- `arc_length_value: float`

### Optional fields

- `mean_recent_position: np.ndarray` of shape `(D,)`
- `mean_recent_direction: np.ndarray` of shape `(D,)`
- `total_recent_displacement: np.ndarray` of shape `(D,)`
- `source_time_estimate: float`
- `support_metadata: dict[str, Any]`

### Invariants

- `increment` must correspond to the transition from `resampled_index` to `resampled_index + 1`
- `history_segments` must be ordered oldest to newest
- the last history segment must terminate at `state`
- the window must lie entirely within one contiguous source segment
- windows that cross a hard gap are invalid and must not be created
- all lineage fields must agree with the source trajectory

---

## 7. Transition bank

## 7.1 `TransitionBank`

Collection of all windows plus lookup structures.

### Required fields

- `windows: list[TransitionWindow]`
- `state_matrix: np.ndarray` of shape `(N, D)`  
  stacked current states for fast search
- `increment_matrix: np.ndarray` of shape `(N, D)`
- `history_tensor: np.ndarray` of shape `(N, K, D)`
- `class_labels: np.ndarray` of shape `(N,)`
- `embryo_ids: list[str]`
- `experiment_ids: list[str]`
- `segment_ids: np.ndarray` of shape `(N,)`
- `touches_interpolated_gap: np.ndarray` of shape `(N,)`

### Optional fast-search fields

- neighbor index structures (KD-tree, FAISS, sklearn NN index, etc.)
- summary feature matrices for fast matching mode

### Invariants

- all stacked arrays agree on first dimension `N`
- bank search structures must be refreshable from canonical stored windows
- bank-level gap flags must agree exactly with the underlying windows

---

## 8. Query object

## 8.1 `PredictionQuery`

Represents one forecasting request.

### Required fields

- `mode: Literal["snapshot", "history"]`
- `current_state: np.ndarray` of shape `(D,)`

### If `mode == "history"`

- `history_segments: np.ndarray` of shape `(K, D)`

### Optional fields

- `recent_points: np.ndarray` of shape `(K+1, D)`  
  if the caller wants point form as well as segment form
- `class_prior: dict[str, float] | None`
- `query_id: str | None`
- `metadata: dict[str, Any]`

### Invariants

- snapshot queries do not require history
- history queries must provide canonical ordered recent segments

---

## 9. Predictor output

## 9.1 `PredictionResult`

Keep the legacy interface concept, but the beta contract cares mainly about:

### Required fields

- `predicted_mean: torch.Tensor` of shape `(B, D)` or `(D,)`
- `predicted_cov_diag: torch.Tensor` of shape `(B, D)` or `(D,)`
- `forward_samples: Optional[torch.Tensor]` of shape `(B, N, D)` or `(N, D)`

### Recommended support diagnostics

- `candidate_count`
- `effective_sample_size`
- `history_mismatch`
- `search_radius`
- `selected_class_weights`

These can be grouped in a diagnostics dict or added as optional explicit fields.

---

## 10. Evaluation task object

## 10.1 `PredictionTask`

A task generated for evaluation.

### Required fields

- `query: PredictionQuery`
- `target_states: np.ndarray` of shape `(H, D)`  
  true future resampled states for one or more horizons
- `horizons: np.ndarray` of shape `(H,)`  
  integer step horizons
- `embryo_id`
- `experiment_id`
- `perturbation_class`

### Optional fields

- `raw_time_targets`
- `source_indices`
- `tier_label`

---

## 11. Shape conventions

Unless explicitly documented otherwise:

- `T` = raw trajectory length in observed frames
- `D` = latent dimension
- `S` = resampled trajectory length in arc-length steps
- `K` = history length in recent segments
- `N` = number of transition windows in bank
- `B` = batch size
- `P` = number of particles / forward samples
- `H` = number of forecast horizons in an evaluation task

---

## 12. Serialization guidance

Prefer simple formats:

- keep canonical bank objects rebuildable from arrays
- store heavy matrices in `npz`, parquet, or pickle only when necessary
- preserve human-readable metadata where possible

Do not serialize approximate search indices as the only copy of the bank.

---

## 13. One-line object lineage

The intended lineage is:

```text
EmbryoTrajectory
  -> SmoothedTrajectory
  -> ResampledTrajectory
  -> TransitionWindow
  -> TransitionBank
  -> PredictionQuery / PredictionTask
  -> PredictionResult
```

That lineage should remain visible in both code and notebooks.
