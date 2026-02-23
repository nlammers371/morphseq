# Plan: Implement Canonical Grid for UOT Masks

## Problem Statement

The current UOT pipeline operates in **raw pixel coordinates** without standardization:
- Each mask pair uses its own cropped grid (no canonical reference)
- No pixel-to-micrometer conversion is applied
- Transport costs and velocities are in arbitrary pixel units
- Comparisons across embryos/timepoints are not physically meaningful

From your progress report:
> "We need an interpretable cost scale (ideally per-pixel). This may require choosing a canonical coordinate system and standardized units rather than ad hoc scaling."

## Design Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid shape | **Fixed 576×256** | Match snip export exactly for pipeline compatibility |
| Resolution | **7.8 μm/px** | Match snip export default (4.5mm × 2.0mm field of view) |
| Overflow handling | **Clip to bounds** | If embryo doesn't fit, indicates pipeline issue upstream |

**Reference implementation**: `src/build/build03A_process_images.py` lines 1436-1467

```python
# The canonical grid parameters we're matching:
outscale: float = 7.8    # Target resolution: 7.8 μm/pixel
outshape = [576, 256]    # Fixed grid: 576×256 pixels = 4.5mm × 2.0mm
```

## Alignment Strategy

**Key insight**: The snip pipeline already has a robust alignment system using yolk-based orientation.

**Existing utilities to reuse**:
- `src/data_pipeline/snip_processing/rotation.py`:
  - `get_embryo_rotation_angle(embryo_mask, yolk_mask)` - PCA + yolk-based orientation
  - `rotate_image(image, angle_degrees)` - OpenCV rotation with canvas expansion
- `src/functions/image_utils.py`:
  - `get_embryo_angle(mask_emb_rs, mask_yolk_rs)` - Original implementation
  - `crop_embryo_image(...)` - Centroid-based cropping to fixed shape

**Orientation logic** (from `get_embryo_rotation_angle`):
1. Get principal axis via PCA (`regionprops.orientation`)
2. Rotate mask to align with principal axis
3. Check if yolk is "above" embryo center (anterior at top)
4. If not, rotate 180° to ensure consistent A-P orientation
5. Fallback when no yolk: use mass distribution heuristic

**Yolk Mask Availability**:
- Yolk masks are generated in `src/build/build02B_segment_bf_main.py`
- Predictions stored in `morphseq_playground/segmentation/yolk_v1_0050_predictions`
- **NOTE**: Yolk masks are not currently exported with embryo mask CSV data
- **TODO**: Future improvement: export yolk mask paths alongside embryo mask metadata
- For now, use `align_mode="centroid"` as fallback when yolk unavailable

---

## Implementation Plan: Three Phases

### Phase 1: Synthetic Data on Canonical Grid

**Goal**: Test the canonical grid system with synthetic masks before touching real data.

**Tasks**:
1. Create `uot_grid.py` module with core transforms:
   - `CanonicalGridConfig` dataclass (shape, resolution, padding)
   - `GridTransform` dataclass (scale factor, offsets, metadata)
   - `compute_grid_transform()` - calculate transform parameters
   - `apply_grid_transform()` - resample mask to canonical grid
   - `transform_coords_to_canonical()` - convert pixel coords to μm

2. Update synthetic test generators:
   - Generate synthetic masks at different resolutions
   - Apply canonical grid transform
   - Verify transport results are resolution-invariant

3. Test with existing synthetic cases:
   - Circle → Oval (shape change)
   - Circle → Circle shifted (pure translation, non-overlapping)
   - Verify velocity fields report in μm/frame

**Validation criteria**:
- [ ] Same shapes at different source resolutions → identical canonical grids
- [ ] Transport cost independent of source resolution
- [ ] Translation test: velocity magnitude = known shift in μm

### Phase 2: Embryo Data + Yolk-Based Alignment

**Goal**: Get real embryo masks onto the canonical grid with proper orientation.

**Tasks**:
1. Integrate alignment from snip processing:
   - Import `get_embryo_rotation_angle` from `src/data_pipeline/snip_processing/rotation.py`
   - Add rotation step before grid transform in preprocessing

2. Update `frame_mask_io.py`:
   - Add `um_per_pixel` extraction from CSV metadata
   - Load yolk masks alongside embryo masks (when available)
   - Store in `UOTFrame.meta`

3. Update `preprocess.py` with canonical grid pipeline:
   ```python
   def preprocess_pair_canonical(src: UOTFrame, tgt: UOTFrame, config: CanonicalGridConfig):
       # 1. QC clean both masks
       # 2. Rotate to canonical orientation (using yolk if available)
       # 3. Rescale to target μm/px (7.8 μm/px)
       # 4. Crop to fixed grid (576×256) centered on embryo
       # 5. Return canonical masks + transform metadata
   ```

4. Add `align_mode: str = "yolk"` to config (with centroid fallback)

**Validation criteria**:
- [ ] All embryos oriented with anterior (yolk) at top
- [ ] Grid dimensions exactly 576×256 after preprocessing
- [ ] Metadata tracks source→canonical transform for reconstruction

### Phase 3: End-to-End Validation on CEP290 Data

**Goal**: Run complete UOT pipeline on real embryo data with interpretable results.

**Tasks**:
1. Run on consecutive frames (same embryo):
   - Verify low transport cost for small time gaps
   - Verify cost increases with time gap
   - Velocities should be 1-10 μm/hour (biologically reasonable)

2. Run on cross-embryo comparison:
   - Compare embryos at ~48 hpf
   - Cost should be higher than consecutive frames
   - Creation/destruction maps should highlight morphological differences

3. Benchmark against `surface_area_um`:
   - If source and target have similar area, created/destroyed mass should be minimal
   - Large area differences should correlate with mass change

**Validation criteria**:
- [ ] Velocity magnitudes in interpretable range (μm/hour)
- [ ] Cost scale consistent across all comparisons
- [ ] Results match visual inspection of mask differences

---

## Key Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `uot_masks/uot_grid.py` | **CREATE** | Core canonical grid transforms |
| `uot_masks/frame_mask_io.py` | MODIFY | Load μm/px ratio and yolk masks |
| `uot_masks/preprocess.py` | MODIFY | Add canonical grid preprocessing |
| `utils/optimal_transport/config.py` | MODIFY | Add `CanonicalGridConfig` to `UOTConfig` |
| `uot_masks/run_transport.py` | MODIFY | Wire up canonical grid pipeline |
| `results/.../synthetic_canonical_tests/` | **CREATE** | Synthetic validation scripts |

## Critical: Downsampling and Cost Interpretation

**Key insight**: At 576×256 with 7.8 μm/px, we may not need additional downsampling.

**The math**:
- Canonical grid: 576 × 256 = 147,456 pixels
- Current `max_support_points = 5000` means we sample if > 5000 foreground pixels
- Typical embryo mask area: ~50-70% of grid = 73,728 - 103,219 pixels
- **Without downsampling**: We'd be sampling heavily (~5% of pixels)
- **With 4x downsample**: 144 × 64 = 9,216 grid → maybe ~5,000 foreground → near budget

**Recommendation**:
- For 576×256 canonical grid, **reduce or eliminate downsampling**
- If downsampling is used, **the scale factor must be tracked and applied to costs/velocities**

### Downsampling Correction Factor

If we downsample by factor `d`:
- **Coordinates** are in downsampled pixel units
- **Transport distance** (in pixels) must be multiplied by `d × um_per_pixel` to get μm
- **Transport cost** (squared distance × mass) must be multiplied by `d²` to get μm² units
- **Velocity** must be multiplied by `d × um_per_pixel` to get μm/frame

**In `GridTransform`**, we must track:
```python
@dataclass
class GridTransform:
    # ... existing fields ...
    downsample_factor: int = 1     # Factor used after canonical grid
    effective_um_per_pixel: float  # = reference_um_per_pixel * downsample_factor
```

**In result post-processing**:
```python
def rescale_results_to_um(result: UOTResult, transform: GridTransform) -> UOTResult:
    """Convert results from downsampled pixel units to micrometers."""
    scale = transform.effective_um_per_pixel  # = 7.8 * downsample_factor

    # Velocity: pixel displacement → μm displacement
    result.velocity_field_yx_hw2 *= scale

    # Transport distance in metrics
    result.mean_transport_distance *= scale
    result.max_transport_distance *= scale

    # Cost: if using squared Euclidean, multiply by scale²
    # (But cost is often left in arbitrary units for relative comparison)
    return result
```

---

## Canonical Grid Module API

```python
# src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py

@dataclass
class CanonicalGridConfig:
    """
    Configuration for canonical grid standardization.

    IMPORTANT: These defaults match the snip export pipeline in
    src/build/build03A_process_images.py (lines 1436-1467).
    Any changes here should be coordinated with snip export parameters.
    """
    reference_um_per_pixel: float = 7.8       # Match snip export default
    grid_shape_hw: tuple[int, int] = (576, 256)  # Match snip output shape
    padding_um: float = 50.0                  # Padding in micrometers
    align_mode: str = "yolk"                  # "yolk" | "centroid" | "none"
    downsample_factor: int = 1                # 1 = no downsampling (recommended for 576×256)

@dataclass
class GridTransform:
    """Records transformation from source to canonical grid."""
    source_um_per_pixel: float      # From CSV: Height(um) / Height(px)
    target_um_per_pixel: float      # = reference_um_per_pixel (7.8)
    scale_factor: float             # = source / target
    rotation_rad: float             # Yolk-based rotation angle
    offset_yx_um: tuple[float, float]  # Translation after rotation
    grid_shape_hw: tuple[int, int]  # (576, 256)
    downsample_factor: int          # Additional downsampling after grid (1 = none)
    effective_um_per_pixel: float   # = target_um_per_pixel * downsample_factor

def compute_grid_transform(
    mask: np.ndarray,
    source_um_per_pixel: float,
    yolk_mask: np.ndarray | None,
    config: CanonicalGridConfig,
) -> GridTransform:
    """
    Compute transform to place mask on canonical grid.

    Steps:
    1. Compute rotation angle (yolk-based or centroid-based)
    2. Compute scale factor to reach target resolution
    3. Compute crop offsets to center on mask centroid
    """
    ...

def apply_grid_transform(
    mask: np.ndarray,
    transform: GridTransform,
) -> np.ndarray:
    """
    Apply transform to resample mask onto canonical grid.

    Steps:
    1. Rotate by transform.rotation_rad
    2. Rescale by transform.scale_factor
    3. Crop/pad to transform.grid_shape_hw centered on centroid
    """
    ...
```

## Verification Checklist

### Synthetic Tests (Phase 1)
- [ ] Identity test: same mask → zero velocity, zero cost
- [ ] Translation test: shifted mask → uniform velocity = shift distance (in μm!)
- [ ] Scale test: different source resolutions → identical results after canonical transform
- [ ] Shape test: circle→oval → outward velocity at expansion zone
- [ ] **Downsampling test**: If d>1, verify `velocity_μm = velocity_px × d × um_per_pixel`

### Alignment Tests (Phase 2)
- [ ] Yolk always at top after rotation
- [ ] Output shape exactly 576×256
- [ ] Transform metadata allows reconstruction of original

### Real Data Tests (Phase 3)
- [ ] Consecutive frames: cost < 0.1 (normalized), velocity < 10 μm/hr
- [ ] Cross-embryo: cost > consecutive, highlights morphological differences
- [ ] Area-matched pairs: minimal mass creation/destruction
