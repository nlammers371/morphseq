# Quality Control File Organization & Data Outputs

**Date:** 2025-10-09
**Context:** Separating auxiliary-mask-dependent QC from independent QC, with special attention to `sa_outlier` importance

---

## Current State Analysis

### Existing Files in `src/data_pipeline/quality_control/`:
```
death_detection.py                      # Viability + death persistence
surface_area_outlier_detection.py      # SA outlier detection (IMPORTANT)
generate_references/                    # Reference curve generation
    └── build_sa_reference.py
```

### Currently in `src/build/`:
```
qc_utils.py                            # compute_fraction_alive, compute_qc_flags, compute_speed
build04_perform_embryo_qc.py           # Orchestration (uses death_detection + sa_outlier)
```

---

## Proposed QC File Organization

### Principle: Separate by **Data Dependencies**, Not by Domain

**Category 1: UNet Auxiliary Mask-Dependent QC**
- Requires UNet masks (viability, yolk, focus, bubble)
- Cannot run without Build02B UNet segmentation

**Category 2: SAM2-Only QC**
- Only requires SAM2 tracking/masks
- Independent of UNet auxiliary masks

**Category 3: Feature-Based QC**
- Requires computed features (morphology, stage)
- Independent of masks

---

## Proposed Structure

```
src/data_pipeline/quality_control/

├── auxiliary_mask_qc/                  # UNet auxiliary mask-dependent QC
│   ├── __init__.py
│   ├── imaging_quality.py              # frame, yolk, focus, bubble flags
│   └── embryo_death_qc.py              # fraction_alive + dead_flag (unified)
│
├── segmentation_qc/                    # SAM2-only QC
│   ├── __init__.py
│   ├── mask_quality.py                 # SAM2 mask validation
│   └── tracking_quality.py             # Movement speed, trajectory, tracking errors
│
├── morphology_qc/                      # Feature-based QC
│   ├── __init__.py
│   ├── size_validation.py              # SA outlier detection (VERY IMPORTANT)
│   └── stage_consistency.py            # (Future: temporal stage progression checks)
│
└── references/                         # Reference generation utilities
    ├── __init__.py
    └── build_sa_reference.py           # SA reference curve builder
```

---

## Detailed File Specifications

### 1. `auxiliary_mask_qc/imaging_quality.py`

**Purpose:** Detect imaging quality issues using UNet auxiliary masks

**Functions:**
```python
def compute_imaging_qc_flags(
    emb_mask: np.ndarray,
    px_dim_um: float,
    yolk_mask: Optional[np.ndarray] = None,
    focus_mask: Optional[np.ndarray] = None,
    bubble_mask: Optional[np.ndarray] = None,
    qc_scale_um: int = 150,
) -> Dict[str, bool]:
    """
    Compute imaging quality flags using spatial proximity analysis.

    Returns:
        frame_flag: Embryo near image boundary
        no_yolk_flag: Yolk sac missing
        focus_flag: Out-of-focus regions nearby
        bubble_flag: Air bubbles nearby
    """
```

**Dependencies:**
- SAM2 embryo masks (for embryo location)
- UNet yolk, focus, bubble masks

**Output:** `quality_control_flags/{experiment_id}/imaging_quality.csv`
```
Columns:
    - snip_id
    - frame_flag
    - no_yolk_flag
    - focus_flag
    - bubble_flag
```

**Migrated from:** `src/build/qc_utils.py::compute_qc_flags()`

---

### 2. `auxiliary_mask_qc/embryo_death_qc.py`

**Purpose:** Unified viability QC (fraction_alive + death detection)

**Functions:**
```python
def compute_fraction_alive(
    emb_mask: np.ndarray,
    via_mask: Optional[np.ndarray]
) -> float:
    """
    Compute fraction of embryo pixels that are alive.

    Args:
        emb_mask: SAM2 embryo mask
        via_mask: UNet viability mask (1 = necrotic/dead)

    Returns:
        fraction_alive = 1 - (dead_pixels / total_embryo_pixels)
    """


def compute_viability_qc(
    tracking_df: pd.DataFrame,
    sam2_masks_dir: Path,
    unet_via_masks_dir: Path,
    stage_df: pd.DataFrame,
    persistence_threshold: float = 0.25,
    min_decline_rate: float = 0.05,
    buffer_hours: float = 2.0,
) -> pd.DataFrame:
    """
    Unified viability QC with persistence validation.

    Algorithm:
        1. Compute fraction_alive for each snip
        2. Detect persistent death inflection points per embryo
        3. Capture predicted_stage_hpf at inflection
        4. Apply 2hr buffer using predicted_stage_hpf
        5. Flag dead embryos

    Returns DataFrame with:
        - fraction_alive
        - dead_flag (unified, persistence-validated)
        - dead_inflection_time_int
        - death_predicted_stage_hpf
    """
```

**Dependencies:**
- SAM2 embryo masks
- UNet viability masks
- Stage predictions (for buffer calculation)

**Output:** `quality_control_flags/{experiment_id}/embryo_death_qc.csv`
```
Columns:
    - snip_id
    - embryo_id
    - time_int
    - fraction_alive
    - dead_flag
    - dead_inflection_time_int
    - death_predicted_stage_hpf
```

**Migrated from:**
- `src/build/qc_utils.py::compute_fraction_alive()`
- `src/data_pipeline/quality_control/death_detection.py` (entire module)

---

### 3. `segmentation_qc/mask_quality.py`

**Purpose:** Validate SAM2 mask quality

**Functions:**
```python
def validate_sam2_masks(
    propagated_masks_json: Path,
    tracking_csv: pd.DataFrame,
    area_variance_threshold: float = 0.20,
    edge_distance_px: int = 2,
    iou_overlap_threshold: float = 0.1,
    large_mask_frac: float = 0.15,
    small_mask_frac: float = 0.001,
) -> pd.DataFrame:
    """
    Multi-check SAM2 mask validation.

    Flags:
        - HIGH_SEGMENTATION_VAR_SNIP: Area variance >20% vs nearby frames
        - MASK_ON_EDGE: Mask within 2px of image edge
        - DETECTION_FAILURE: Missing expected embryos
        - OVERLAPPING_MASKS: IoU > 0.1 between embryos
        - LARGE_MASK: >15% of frame area
        - SMALL_MASK: <0.1% of frame area
        - DISCONTINUOUS_MASK: Multiple disconnected components
    """
```

**Dependencies:**
- SAM2 propagated_masks.json
- SAM2 tracking_table.csv

**Output:** `quality_control_flags/{experiment_id}/segmentation_quality.csv`
```
Columns:
    - snip_id
    - HIGH_SEGMENTATION_VAR_SNIP
    - MASK_ON_EDGE
    - DETECTION_FAILURE
    - OVERLAPPING_MASKS
    - LARGE_MASK
    - SMALL_MASK
    - DISCONTINUOUS_MASK
```

**Migrated from:** `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`

---

### 4. `segmentation_qc/tracking_quality.py`

**Purpose:** Validate embryo tracking quality (movement, trajectory)

**Functions:**
```python
def compute_speed(
    prev_xy: Optional[Tuple[float, float]],
    prev_t_s: Optional[float],
    curr_xy: Tuple[float, float],
    curr_t_s: float,
    px_dim_um: float,
) -> float:
    """Compute embryo movement speed (µm/s)"""


def smooth_trajectory(
    positions: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay trajectory smoothing"""


def compute_tracking_qc(
    tracking_df: pd.DataFrame,
    morphology_df: pd.DataFrame,
    px_dim_um: float,
    speed_outlier_threshold: float = 100.0,  # µm/s
) -> pd.DataFrame:
    """
    Validate tracking quality: speed, trajectory smoothness, discontinuities.

    Returns DataFrame with:
        - speed_um_per_s
        - trajectory_smoothness_score
        - tracking_error_flag (jumps, ID switches)
    """
```

**Dependencies:**
- SAM2 tracking_table.csv
- Morphology features (for centroid positions)

**Output:** `quality_control_flags/{experiment_id}/tracking_quality.csv`
```
Columns:
    - embryo_id
    - time_int
    - speed_um_per_s
    - trajectory_smoothness_score
    - tracking_error_flag
```

**Migrated from:** `src/build/qc_utils.py::compute_speed()`

---

### 5. `morphology_qc/size_validation.py` ⭐ VERY IMPORTANT

**Purpose:** Two-sided surface area outlier detection

**Functions:**
```python
def compute_sa_outlier_flag(
    df: pd.DataFrame,
    sa_reference_path: Path,
    k_upper: float = 1.2,
    k_lower: float = 0.9,
    stage_col: str = "predicted_stage_hpf",
    sa_col: str = "surface_area_um",
) -> pd.DataFrame:
    """
    Two-sided SA outlier detection using global reference curves.

    Flags:
        - SA > k_upper × p95: Too large (segmentation artifacts, debris)
        - SA < k_lower × p5: Too small (incomplete masks, dead embryos)

    Uses global reference curves from wild-type controls across all experiments.
    Reference file: metadata/sa_reference_curves.csv

    Returns DataFrame with 'sa_outlier_flag' column added.
    """
```

**Dependencies:**
- Morphology features (surface_area_um)
- Stage predictions (predicted_stage_hpf)
- Global reference curves (metadata/sa_reference_curves.csv)

**Output:** `quality_control_flags/{experiment_id}/size_validation.csv`
```
Columns:
    - snip_id
    - sa_outlier_flag
    - sa_outlier_reason  # "too_large" or "too_small" (optional diagnostic)
```

**Already exists:** `src/data_pipeline/quality_control/surface_area_outlier_detection.py`

**Action:** Move to `morphology_qc/size_validation.py`

**Why this is VERY IMPORTANT:**
- Catches segmentation artifacts (too large)
- Catches incomplete masks (too small)
- Catches dead/dying embryos early (shrinking)
- Two-sided detection is robust
- Uses global reference curves (generalizes well)

---

### 6. `references/build_sa_reference.py`

**Purpose:** Generate global SA reference curves from control embryos

**Already exists:** `src/data_pipeline/quality_control/generate_references/build_sa_reference.py`

**Action:** Move to `references/build_sa_reference.py`

**Notes:**
- Run quarterly or after major pipeline changes
- Generates `metadata/sa_reference_curves.csv`
- Uses wild-type controls only

---

## Data Output Summary

### By QC Category:

**Auxiliary Mask-Dependent:**
```
quality_control_flags/{experiment_id}/
    ├── imaging_quality.csv        # frame, yolk, focus, bubble flags
    └── embryo_death_qc.csv        # fraction_alive, dead_flag, dead_inflection_time_int, death_predicted_stage_hpf
```

**SAM2-Only:**
```
quality_control_flags/{experiment_id}/
    ├── segmentation_quality.csv   # 7 SAM2 mask quality flags
    └── tracking_quality.csv       # speed, trajectory, tracking_error_flag
```

**Feature-Based:**
```
quality_control_flags/{experiment_id}/
    └── size_validation.csv        # sa_outlier_flag ⭐
```

---

## Migration Plan

### Files to Move/Create:

1. **Create:** `auxiliary_mask_qc/imaging_quality.py`
   - Extract from: `src/build/qc_utils.py::compute_qc_flags()`

2. **Create:** `auxiliary_mask_qc/embryo_death_qc.py`
   - Extract from: `src/build/qc_utils.py::compute_fraction_alive()`
   - Merge with: `death_detection.py` (entire module)

3. **Create:** `segmentation_qc/mask_quality.py`
   - Extract from: `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`

4. **Create:** `segmentation_qc/tracking_quality.py`
   - Extract from: `src/build/qc_utils.py::compute_speed()`

5. **Move:** `surface_area_outlier_detection.py` → `morphology_qc/size_validation.py`

6. **Move:** `generate_references/build_sa_reference.py` → `references/build_sa_reference.py`

7. **Delete:** `death_detection.py` (merged into embryo_death_qc.py)

8. **Delete:** `qc_utils.py` (functions distributed to proper modules)

9. **Delete:** `build04_perform_embryo_qc.py` (replaced by Snakemake rules)

---

## Advantages of This Organization

### ✅ Clear Dependency Boundaries:
- Know immediately which QC needs UNet masks
- Know which QC can run with SAM2 alone
- Know which QC needs features only

### ✅ Parallel Execution Opportunities:
```python
# Can run in parallel (no shared dependencies):
- imaging_quality (UNet yolk/focus/bubble)
- embryo_death_qc (UNet viability)
- segmentation_quality (SAM2 JSON)
- tracking_quality (SAM2 tracking)
- size_validation (morphology features)
```

### ✅ Modular Testing:
- Test each QC category independently
- Mock UNet masks for auxiliary_mask_qc tests
- Mock SAM2 output for segmentation_qc tests
- Mock features for morphology_qc tests

### ✅ Easy to Extend:
- Add new auxiliary mask QC? → `auxiliary_mask_qc/new_module.py`
- Add new feature-based QC? → `morphology_qc/new_module.py`
- Add new SAM2 validation? → `segmentation_qc/new_module.py`

### ✅ SA Outlier Gets Proper Visibility:
- Separate file in `morphology_qc/size_validation.py`
- Clear importance (catches segmentation + biological issues)
- Easy to find and maintain

---

## Alternative: Flat Structure (Not Recommended)

```
quality_control/
    ├── imaging_quality_qc.py
    ├── embryo_death_qc.py
    ├── mask_quality_qc.py
    ├── tracking_quality_qc.py
    └── size_validation_qc.py
```

**Pros:**
- Fewer directories
- Simpler imports

**Cons:**
- No visual grouping by dependency
- Harder to see which QC needs which inputs
- Harder to parallelize (no clear boundaries)

---

## Recommendation: Use Categorized Structure

**Why:**
1. Clear dependency boundaries enable better Snakemake DAG
2. Easier to parallelize QC rules
3. Easier to test in isolation
4. Scales better as pipeline grows
5. SA outlier gets proper prominence

**Next Steps:**
1. Approve this structure
2. Create directory layout
3. Begin file migrations (Week 1 of implementation plan)
