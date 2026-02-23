# MVP Masterplan UOT: UOT Mask Pipeline Implementation Plan

## Overview
Implement per-pair Unbalanced Optimal Transport (UOT) between embryo masks to quantify morphological dynamics (motion + growth).
CPU-first using POT, enforcing explicit resolution and support budgets to avoid N^2 costs.
Use a pluggable backend so GPU acceleration (JAX/ott-jax) can be added later without rewriting preprocessing.

This version folds in the naming and policy updates (UOTFrame/UOTFramePair, velocity_field, sampling_mode, module renames).

Conceptual/biological objectives live in:
`src/analyze/optimal_transport_morphometrics/docs/analysis_goals_transport_morphodynamics.md`.

## Standardized Naming (frozen for MVP)
Core classes:
- UOTFrame
- UOTFramePair
- UOTSupport
- UOTProblem
- UOTResult

Field renames:
- flow_vectors -> velocity_field_yx_hw2
- reg_m in config -> marginal_relaxation (passed to POT as reg_m)

Module renames:
- densities.py -> density_transforms.py
- pyramid.py -> multiscale_sampling.py
- postprocess.py -> transport_maps.py
- mask_io.py -> frame_mask_io.py
- run_pair.py -> run_transport.py

Coordinate convention:
- Internal always uses (y, x) coordinates and (vy, vx) velocity.
- viz.py converts to (x, y) for plotting.

## MVP Decisions Summary
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Support budget | max_support_points=5000 | Hard cap on memory; sample above this |
| Resolution | Single 1/4 level + benchmark | Fast iteration; benchmark for floor |
| Mass modes | 0A + 0C | Compare uniform vs distance transform |
| Alignment | Centroid only | Yolk-based deferred |
| Backend | POT (CPU), per-pair loop | Correctness before speed |
| Normalization | Normalize -> solve -> rescale | Stable numerics + physical mass preserved |
| Calibration | marginal_relaxation sweep on identity | Prevent destroy/recreate flicker |
| Sampling strategy | Stratified boundary+interior (fallback uniform) | Better coverage than uniform alone |
| Sampling policy | auto (warn + sample) | Avoid crashing on big embryos |

## Key Requirements
- Binary embryo masks as input (optionally grayscale frames later).
- CPU-first using POT (correctness before speed).
- GPU-ready architecture via pluggable backend.
- Outputs: UOT distance, mass-created/destroyed maps, velocity field, summary scalars.

---

## Module Structure

Location: `src/analyze/optimal_transport_morphometrics/uot_masks/`

```
uot_masks/
├── __init__.py
├── config.py
├── frame_mask_io.py          # load frames + masks (RLE/PNG; later brightfield)
├── preprocess.py             # qc, crop, align, downsample orchestration
├── density_transforms.py     # binary->density (0A/0B/0C)
├── multiscale_sampling.py    # downsample + support sampling + budgets
├── backends/
│   ├── __init__.py
│   ├── base.py
│   └── pot_backend.py
├── transport_maps.py         # created/destroyed maps + velocity_field + reconstruction helpers
├── metrics.py
├── viz.py
├── run_transport.py
└── run_timeseries.py
```

---

## Data Structures (MVP)

### UOTFrame + UOTFramePair
```python
@dataclass
class UOTFrame:
    frame: Optional[np.ndarray] = None     # 2D float, optional for MVP
    embryo_mask: Optional[np.ndarray] = None
    meta: Optional[dict] = None

@dataclass
class UOTFramePair:
    src: UOTFrame
    tgt: UOTFrame
    pair_meta: Optional[dict] = None
```

### UOTSupport (explicit yx convention)
```python
@dataclass
class UOTSupport:
    coords_yx: np.ndarray   # (N,2) (row=y, col=x)
    weights: np.ndarray     # (N,) nonnegative, source-mass units
```

### UOTProblem
```python
@dataclass
class UOTProblem:
    src: UOTSupport
    tgt: UOTSupport
    work_shape_hw: tuple[int, int]
    transform_meta: dict    # crop bbox, downsample_factor, orig_shape, etc.
```

### UOTResult (velocity field + mass maps)
```python
Coupling = Union[np.ndarray, sp.coo_matrix]

@dataclass
class UOTResult:
    cost: float
    coupling: Optional[Coupling]

    mass_created_hw: np.ndarray
    mass_destroyed_hw: np.ndarray
    velocity_field_yx_hw2: np.ndarray     # (H,W,2) in (vy,vx)

    support_src_yx: np.ndarray
    support_tgt_yx: np.ndarray
    weights_src: np.ndarray
    weights_tgt: np.ndarray

    transform_meta: dict
    diagnostics: Optional[dict] = None
```

### UOTConfig (sampling policy + marginal relaxation)
```python
class SamplingMode(str, Enum):
    AUTO = "auto"     # sample + warn
    RAISE = "raise"   # raise if too many points

@dataclass
class UOTConfig:
    downsample_factor: int = 4
    max_support_points: int = 5000
    sampling_mode: SamplingMode = SamplingMode.AUTO
    sampling_strategy: str = "stratified_boundary_interior"

    epsilon: float = 1e-2
    marginal_relaxation: float = 10.0  # passed to POT as reg_m
    metric: str = "sqeuclidean"

    mass_mode: str = "uniform"         # or "boundary_band" or "distance_transform"
    align_mode: str = "centroid"

    store_coupling: bool = True
    random_seed: int = 0
```

---

## Data Sources

### Embryo Masks
- CSV (RLE-encoded):
  `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_*.csv`
  - Columns: mask_rle, mask_height_px, mask_width_px
- PNG exports:
  `segmentation_sandbox/data/exported_masks/{experiment_date}/masks/`

### Yolk Masks (optional, for alignment)
- External data root: `$MORPHSEQ_ROOT/segmentation/unet_yolk_v0_0050_predictions/`
- Falls back gracefully when unavailable

### Existing Utilities to Reuse
- `segmentation_sandbox/scripts/utils/mask_utils.py`: decode_mask_rle()
- `segmentation_sandbox/scripts/utils/mask_cleaning.py`: clean_embryo_mask()
- `src/data_pipeline/segmentation/grounded_sam2/mask_export.py`: load_labeled_mask()

---

## Implementation Phases

### Phase 1: Skeleton + Config (Day 1)
Create module structure and dataclasses:
- UOTConfig, UOTFrame, UOTFramePair, UOTSupport, UOTProblem, UOTResult
- MassMode enum (UNIFORM / BOUNDARY_BAND / DISTANCE_TRANSFORM)

### Phase 2: Frame/Mask I/O (Day 1)
- load_mask_from_rle() via decode_mask_rle()
- load_mask_pair_from_csv() for build06 CSV files
- load_mask_from_png() via load_labeled_mask()

### Phase 3: Preprocessing (Days 1-2)
- qc_mask() reuse clean_embryo_mask()
- compute_union_bbox() crop to union + padding (biggest CPU speedup)
  - Watch-out: add safety padding (~5-10 px) around union bbox to avoid clipping mass created at edges in downstream viz
- align_masks(): centroid translation only (MVP)
- preprocess_pair(): full pipeline orchestration

### Phase 4: Density Transforms (Day 2)
- mask_to_density_uniform() (0A)
- mask_to_density_boundary_band() (0B)
- mask_to_density_distance_transform() (0C)
- No semantic normalization across frames (preserve physical mass)
- Watch-out (0C): handle near-zero total mass after distance transform (e.g., single-pixel lines)
  - If mass sum < eps, treat as empty: raise or fall back to uniform per config

Notes on 0C:
- Mass scales superlinearly with object size.
- Treat 0A (area-growth) and 0C (core-growth) as different observables.

### Phase 5: Multiscale + Support Sampling (Days 2-3)
- downsample_density(): sum-pooling (area-preserving)
- pad_to_divisible(divisor=16): avoids edge cases, GPU-friendly
- support policy:
  - include all nonzero pixels if <= max_support_points
  - otherwise sample (stratified boundary+interior; fallback uniform)
- Sampling policy (cleaned up):
  - sampling_mode="auto": sample + warn
  - sampling_mode="raise": raise if sampling disabled
  - always raise on empty mask or total mass == 0

### Phase 6: UOT Backend (Days 3-4)
Backend interface:
```python
class UOTBackend(ABC):
    def solve(self, mu, nu, epsilon, marginal_relaxation, ...) -> (coupling, cost, aux)
```

POT implementation:
- Support = nonzero pixels (point cloud, not dense grid)
- Build cost matrix (squared Euclidean on pixel coords)
- Call ot.unbalanced.sinkhorn_unbalanced()

Sampling guardrail (policy):
```python
if len(points) == 0 or total_mass == 0:
    raise ValueError("Empty mask or zero mass.")
if len(points) > max_support_points:
    if sampling_mode == "raise":
        raise RuntimeError("Too many support points.")
    else:
        warn + sample
```

Normalization convention (critical):
- Normalize both histograms to sum to 1 for the solver.
- Rescale all outputs back to source physical mass (m_src).
- Resulting mass maps are in source-mass units and comparable across time.

### Phase 7: Transport Maps (Day 4)
- compute_marginals()
- mass_created_hw = max(0, nu - nu_hat)
- mass_destroyed_hw = max(0, mu - mu_hat)
- velocity_field_yx_hw2 via barycentric projection:
  - T(x) = sum_y y * pi(x,y) / sum_y pi(x,y)
  - v(x) = T(x) - x
- Apply source mass rescaling to maps (from Phase 6)

### Phase 8: Metrics + Viz (Day 5)
Summary scalars:
- transported_mass, created_mass, destroyed_mass, mean_transport_distance

MVP visualizations (minimum 3):
1) Creation/destruction heatmaps (growth zones)
2) Quiver velocity overlay (downsample arrows)
3) Transport spectrum (distance histogram weighted by mass)

### Phase 9: CLI Runners (Day 5-6)
- run_transport.py: single pair computation
- run_timeseries.py: consecutive frames
- calibrate_marginal_relaxation.py:
  1) Identity pairs (t vs t)
  2) Sweep marginal_relaxation [0.1, 1, 10, 100]
  3) Plot fraction_mass_transported vs relaxation
  4) Pick smallest value with near-zero transport

### Phase 10: Resolution Benchmarking (Day 6)
- benchmark_resolution.py:
  - run same pair at [1/2, 1/4, 1/8, 1/16]
  - compare transport cost, flow direction consistency, runtime
  - set MIN_RESOLUTION constant

---

## Testing Strategy

Golden tests:
1) Identity: mask_t == mask_t1 -> near-zero velocity, low mass_created/destroyed
2) Translation: uniform velocity field (dx, dy)
3) Dilation: creation at boundary, outward velocity
4) Erosion: destruction at boundary, inward velocity

Unit tests:
- density transforms (mass preservation, band thickness)
- preprocessing (island removal, bbox padding, centroid alignment)
- backend (convergence, mass consistency under high marginal_relaxation)

---

## GPU Upgrade Path

Benefits:
- All preprocessing stays NumPy/SciPy (backend-agnostic).
- Only solve() is reimplemented for GPU.
    def solve(self, mu, nu, epsilon, marginal_relaxation, ...):
        # Convert to JAX arrays
        # Use ott.solvers.linear.Sinkhorn
        # vmap for batching many pairs
```

GPU helps most when:
- batching many frame pairs (vmap)
- cropped grids still substantial
- Python overhead becomes dominant

---

## Critical Files / References

| File | Purpose |
|------|---------|
| segmentation_sandbox/scripts/utils/mask_utils.py | RLE decode for mask loading |
| segmentation_sandbox/scripts/utils/mask_cleaning.py | Mask QC pipeline |
| src/data_pipeline/segmentation/grounded_sam2/mask_export.py | PNG mask loading |
| src/analyze/spline_fitting/__init__.py | Module organization pattern |
| src/analyze/optimal_transport_morphometrics/docs/uot_mask_pipeline_mvp_and_gpu_roadmap.md | MVP architecture reference |
| morphseq_playground/metadata/build06_output/df03_final_output_with_latents_*.csv | Mask RLE data source |

---

## Verification Steps
1) Load a mask pair from CSV, verify dimensions match metadata
2) QC + crop; visualize before/after
3) Build pyramid; verify mass preserved at each level
4) Run UOT on coarse level (1/8), check finite coupling
5) Identity test: near-zero velocity + low mass_created/destroyed
6) Visualize velocity field on a real embryo pair
7) Run time series on one embryo, plot distance over time

---

## Dependencies to Add
```
POT  # Python Optimal Transport (CPU)
# Later: jax, ott-jax (GPU)
```
