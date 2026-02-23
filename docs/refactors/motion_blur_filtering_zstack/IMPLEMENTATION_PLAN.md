# Z-Stack Motion Blur Filtering - Implementation Plan

## Overview
Implement and validate z-stack filtering methods to remove motion artifacts and out-of-focus frames **before** applying LoG focus stacking. This will improve image quality by ensuring only stable, sharp frames are used.

## Problem Statement

### Current Implementation
The pipeline uses `LoG_focus_stacker` ([export_utils.py:434](../../../src/build/export_utils.py#L434)) which:
1. Applies Laplacian-of-Gaussian filter to all z-slices
2. Picks the slice with maximum |LoG| response per pixel
3. **Does not filter out bad slices** before stacking

### Limitations
1. **Motion Artifacts**: Frames with motion blur can have high local sharpness but are spatially misaligned
2. **Out-of-Focus Frames**: No threshold filtering—even very blurry frames can be selected
3. **No Quality Tracking**: No metrics to identify problematic stacks

## Solution: Test-First Validation Approach

### Strategy
1. **Build & test** filtering methods in notebook with labeled examples
2. **Validate visually** that bad/okay images improve and great images stay unchanged
3. **Integrate** into pipeline with opt-in flags and QA metrics

### Labeled Test Cases
From [bad_image_examples.md](./bad_image_examples.md) and `frame_nd2_lookup.csv`:

**Bad Images** (should improve):
- `20250912_B10_ch00_t0092.jpg` (series 79, time 92)
- `20250912_B10_ch00_t0096.jpg` (series 79, time 96)
- `20250912_B10_ch00_t0097.jpg` (series 79, time 97)
- `20250912_C04_ch00_t0112.jpg` (series 30, time 112)

**Okay Images** (may improve):
- `20250912_C04_ch00_t0111.jpg` (series 30, time 111)
- `20250912_C04_ch00_t0024.jpg` (series 30, time 24)
- `20250912_C04_ch00_t0031.jpg` (series 30, time 31)

**Great Images** (should remain unchanged):
- `20250912_C04_ch00_t0028.jpg` (series 30, time 28)
- `20250912_F02_ch00_t0005.jpg` (series 11, time 5)
- `20250912_G09_ch00_t0031.jpg` (series 71, time 31)

---

## Phase 1: Notebook Development & Validation

**Location**: `results/mcolon/20251019/zstack_filtering_motion.ipynb`

### 1.1 Implement Core Filtering Functions

#### A. Peak-Relative Sharpness Filter (Tier A)
Filters out-of-focus frames relative to the sharpest frame in the stack.

```python
def filter_by_peak_relative(log_scores, alpha=0.7):
    """
    Keep slices with LoG score >= peak * alpha.

    Parameters:
    -----------
    log_scores : np.ndarray, shape (Z,)
        Mean |LoG| response per slice
    alpha : float
        Threshold as fraction of peak (0.5-0.8 typical)

    Returns:
    --------
    keep_mask : np.ndarray, shape (Z,) dtype bool
    """
    peak_score = log_scores.max()
    threshold = peak_score * alpha
    return log_scores >= threshold
```

**Advantages**:
- Self-normalizes per stack (handles dim/bright variations)
- Simple, fast, interpretable
- No global threshold needed

**Tunable Parameter**: `alpha` (recommend testing 0.5, 0.6, 0.7, 0.8)

#### B. Frame-to-Frame Correlation (Motion Detection)
Detects motion artifacts by measuring similarity between consecutive frames.

```python
def compute_frame_correlations(stack):
    """
    Compute Pearson correlation between consecutive frames.

    Parameters:
    -----------
    stack : np.ndarray, shape (Z, Y, X)

    Returns:
    --------
    correlations : np.ndarray, shape (Z-1,)
        corr[i] = correlation(frame[i], frame[i+1])
    """
    correlations = []
    for i in range(len(stack) - 1):
        corr = pearson_corr(stack[i], stack[i+1])
        correlations.append(corr)
    return np.array(correlations)


def filter_by_correlation_bilateral(correlations, threshold=0.9):
    """
    Flag frames involved in low-correlation pairs.

    CRITICAL: When corr(i, i+1) is low, BOTH frames i and i+1 are suspect.

    Parameters:
    -----------
    correlations : np.ndarray, shape (Z-1,)
    threshold : float
        Minimum acceptable correlation

    Returns:
    --------
    keep_mask : np.ndarray, shape (Z,) dtype bool
    """
    Z = len(correlations) + 1
    keep_mask = np.ones(Z, dtype=bool)

    bad_corr_indices = np.where(correlations < threshold)[0]
    for i in bad_corr_indices:
        keep_mask[i] = False      # flag frame i
        keep_mask[i+1] = False    # flag frame i+1

    return keep_mask
```

**Advantages**:
- Detects motion even in sharp frames
- Bilateral flagging prevents using either frame from a bad pair

**Tunable Parameter**: `threshold` (recommend testing 0.85, 0.90, 0.95)

#### C. Hybrid Filter (Recommended)
Combines both metrics—a frame is bad if it's EITHER blurry OR unstable.

```python
def hybrid_filter(stack, log_scores, alpha=0.7, corr_threshold=0.9):
    """
    Combine peak-relative and correlation filtering.

    Returns:
    --------
    keep_mask : np.ndarray, shape (Z,) dtype bool
    metrics : dict with diagnostic info
    """
    # Filter 1: Focus
    is_sharp = filter_by_peak_relative(log_scores, alpha)

    # Filter 2: Motion
    correlations = compute_frame_correlations(stack)
    is_stable = filter_by_correlation_bilateral(correlations, corr_threshold)

    # Combine: keep only frames that are both sharp AND stable
    keep_mask = is_sharp & is_stable

    # Safety: never drop all frames
    if not keep_mask.any():
        keep_mask = is_sharp  # fallback to just sharpness

    metrics = {
        "n_total": len(stack),
        "n_sharp": is_sharp.sum(),
        "n_stable": is_stable.sum(),
        "n_kept": keep_mask.sum(),
        "peak_log": log_scores.max(),
        "mean_log": log_scores.mean(),
        "min_corr": correlations.min(),
        "median_corr": np.median(correlations)
    }

    return keep_mask, metrics
```

### 1.2 Visual Validation Loop

For each labeled example:

```python
# Load example from lookup CSV
example = lookup_df[lookup_df["category"] == "Bad Images"].iloc[0]
nd2_path = Path(example["nd2_path"])
series_num = int(example["nd2_series_num"])
time_int = int(example["time_int"])

# Load full z-stack (assuming temporal neighbors as z-slices)
neighbor_times = [time_int + delta for delta in range(-5, 6)]
stack = np.array([
    read_nd2_frame(nd2_path, series_num, t, z_index=0)
    for t in neighbor_times if t >= 0
])

# Compute LoG scores (mimic what LoG_focus_stacker does)
log_scores = np.array([log_focus_response(frame) for frame in stack])

# Test different filtering methods
results = {}
for alpha in [0.5, 0.6, 0.7, 0.8]:
    for corr_thresh in [0.85, 0.90, 0.95]:
        keep_mask, metrics = hybrid_filter(stack, log_scores, alpha, corr_thresh)
        filtered_stack = stack[keep_mask]

        # Apply focus stacking (simplified)
        ff_original = naive_focus_stack(stack, log_scores)
        ff_filtered = naive_focus_stack(filtered_stack, log_scores[keep_mask])

        results[f"α={alpha}, corr={corr_thresh}"] = {
            "metrics": metrics,
            "ff_original": ff_original,
            "ff_filtered": ff_filtered
        }

# Visualize side-by-side comparisons
# ... plotting code ...
```

### 1.3 Determine Optimal Thresholds

Create a comparison table:

| Image | Category | α | corr_thresh | n_kept | Visual Quality | Notes |
|-------|----------|---|-------------|--------|----------------|-------|
| B10_t92 | Bad | 0.7 | 0.90 | 5/11 | ✅ Improved | Motion removed |
| B10_t92 | Bad | 0.5 | 0.90 | 8/11 | ⚠️ Some artifacts | Too lenient |
| C04_t28 | Great | 0.7 | 0.90 | 10/11 | ✅ Unchanged | Good |
| ... | ... | ... | ... | ... | ... | ... |

**Target**: Find parameters where:
- Bad images show clear improvement
- Great images retain quality (9-10/11 frames kept)
- Okay images show some improvement

---

## Phase 2: Export Utils Integration

**Location**: `src/build/export_utils.py`

### 2.1 Add Quality Metrics Function

```python
def compute_slice_quality_metrics(
    data_zyx: torch.Tensor,      # shape (Z, Y, X) or (N, Z, Y, X)
    log_scores: torch.Tensor,    # shape (Z, Y, X) or (N, Z, Y, X)
    enable_correlation: bool = True
) -> dict:
    """
    Compute per-slice quality metrics from LoG focus stacker outputs.

    Returns dict with:
    - mean_log_per_slice: mean |LoG| response per slice
    - peak_log: maximum mean |LoG| across slices
    - min_correlation: minimum frame-to-frame correlation (if enabled)
    - median_correlation: median correlation
    """
    # Handle batched input
    if data_zyx.ndim == 4:
        # Process first sample only for metrics (or batch average)
        data = data_zyx[0]  # shape (Z, Y, X)
        log_s = log_scores[0]
    else:
        data = data_zyx
        log_s = log_scores

    # Per-slice mean LoG
    mean_log_per_slice = log_s.mean(dim=(1, 2)).cpu().numpy()  # shape (Z,)

    metrics = {
        "mean_log_per_slice": mean_log_per_slice,
        "peak_log": float(mean_log_per_slice.max()),
        "mean_log": float(mean_log_per_slice.mean()),
    }

    if enable_correlation:
        Z = data.shape[0]
        correlations = []
        for i in range(Z - 1):
            a = data[i].cpu().numpy().ravel()
            b = data[i+1].cpu().numpy().ravel()
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            corr = np.dot(a, b) / len(a)
            correlations.append(corr)

        if correlations:
            metrics["min_corr"] = float(np.min(correlations))
            metrics["median_corr"] = float(np.median(correlations))
        else:
            metrics["min_corr"] = 1.0
            metrics["median_corr"] = 1.0

    return metrics
```

### 2.2 Add Filtering Function

```python
def filter_bad_slices(
    data_zyx: torch.Tensor,      # shape (Z, Y, X) or (N, Z, Y, X)
    log_scores: torch.Tensor,    # same shape as data_zyx
    alpha: float = 0.7,
    corr_threshold: float = 0.9,
    method: str = "peak_relative"
) -> tuple[torch.Tensor, np.ndarray, dict]:
    """
    Filter out bad slices before focus stacking.

    Parameters:
    -----------
    method : str
        "peak_relative" - filter by sharpness only
        "correlation" - filter by motion only
        "hybrid" - combine both (recommended)

    Returns:
    --------
    filtered_data : torch.Tensor
        Stack with bad slices removed
    kept_indices : np.ndarray
        Boolean mask of kept slices
    metrics : dict
        Quality metrics
    """
    # Handle batched input (process first sample)
    if data_zyx.ndim == 4:
        data = data_zyx[0]
        log_s = log_scores[0]
        was_batched = True
    else:
        data = data_zyx
        log_s = log_scores
        was_batched = False

    Z = data.shape[0]

    # Compute mean LoG per slice
    mean_log_per_slice = log_s.mean(dim=(1, 2)).cpu().numpy()

    # Filter by sharpness
    if method in ["peak_relative", "hybrid"]:
        peak_score = mean_log_per_slice.max()
        threshold = peak_score * alpha
        is_sharp = mean_log_per_slice >= threshold
    else:
        is_sharp = np.ones(Z, dtype=bool)

    # Filter by correlation
    if method in ["correlation", "hybrid"]:
        is_stable = np.ones(Z, dtype=bool)
        for i in range(Z - 1):
            a = data[i].cpu().numpy().ravel()
            b = data[i+1].cpu().numpy().ravel()
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            corr = np.dot(a, b) / len(a)

            if corr < corr_threshold:
                is_stable[i] = False
                is_stable[i+1] = False
    else:
        is_stable = np.ones(Z, dtype=bool)

    # Combine filters
    keep_mask = is_sharp & is_stable

    # Safety: never drop all frames
    if not keep_mask.any():
        keep_mask = is_sharp if is_sharp.any() else np.ones(Z, dtype=bool)

    # Filter the data
    kept_indices_torch = torch.from_numpy(keep_mask).to(data.device)
    filtered_data = data[kept_indices_torch]

    if was_batched:
        filtered_data = filtered_data.unsqueeze(0)  # restore batch dimension

    metrics = {
        "n_total": Z,
        "n_sharp": int(is_sharp.sum()),
        "n_stable": int(is_stable.sum()),
        "n_kept": int(keep_mask.sum()),
        "rejection_rate": 1.0 - (keep_mask.sum() / Z),
    }

    return filtered_data, keep_mask, metrics
```

### 2.3 Wrapper Function

```python
def LoG_focus_stacker_with_filtering(
    data_zyx: Union[torch.Tensor, np.ndarray],
    filter_size: int,
    device: Union[str, torch.device] = "cpu",
    enable_filtering: bool = False,
    filter_alpha: float = 0.7,
    filter_corr_threshold: float = 0.9,
    filter_method: str = "peak_relative",
    return_metrics: bool = False
):
    """
    LoG focus stacker with optional pre-filtering of bad slices.

    Parameters:
    -----------
    enable_filtering : bool
        If True, filter bad slices before stacking
    filter_alpha : float
        Peak-relative threshold (0.5-0.8)
    filter_corr_threshold : float
        Correlation threshold (0.85-0.95)
    filter_method : str
        "peak_relative", "correlation", or "hybrid"
    return_metrics : bool
        If True, return quality metrics

    Returns:
    --------
    ff : torch.Tensor
        Full-focus image
    abs_log : torch.Tensor
        LoG scores (from filtered stack if filtering enabled)
    metrics : dict (if return_metrics=True)
        Quality and filtering diagnostics
    """
    device = torch.device(device)

    # Convert to tensor if needed
    if not torch.is_tensor(data_zyx):
        data = torch.from_numpy(np.asarray(data_zyx))
    else:
        data = data_zyx
    data = data.to(device, dtype=torch.float32)

    # First pass: compute LoG scores on full stack
    ff_initial, log_scores_initial = LoG_focus_stacker(data, filter_size, device)

    metrics = {}

    if enable_filtering:
        # Filter bad slices
        filtered_data, keep_mask, filter_metrics = filter_bad_slices(
            data, log_scores_initial,
            alpha=filter_alpha,
            corr_threshold=filter_corr_threshold,
            method=filter_method
        )
        metrics.update(filter_metrics)

        # Second pass: focus stack on filtered data
        ff, log_scores = LoG_focus_stacker(filtered_data, filter_size, device)
    else:
        ff = ff_initial
        log_scores = log_scores_initial
        keep_mask = np.ones(data.shape[0] if data.ndim == 3 else data.shape[1], dtype=bool)

    # Compute quality metrics
    if return_metrics:
        quality_metrics = compute_slice_quality_metrics(
            data if not enable_filtering else filtered_data,
            log_scores,
            enable_correlation=True
        )
        metrics.update(quality_metrics)
        return ff, log_scores, metrics

    return ff, log_scores
```

---

## Phase 3: Build Pipeline Integration

### 3.1 Keyence Pipeline (build01A_compile_keyence_torch.py)

**Location**: Around line 382

#### A. Add QA Metrics Collection (Non-Breaking)

```python
# After line 358 (before the processing loop)
qc_rows = []  # NEW: collect quality metrics

# Modify lines 382-383:
ff, log_scores = LoG_focus_stacker(flat, filter_size=filter_size)
# Add QA collection:
qc_metrics = compute_slice_quality_metrics(flat, log_scores, enable_correlation=True)

# NEW: Store metrics per tile
for b in range(ff.shape[0]):
    for p in range(ff.shape[1]):
        qc_rows.append({
            "well": well_names[b],
            "time_int": int(time_indices[b].item()),
            "tile": p,
            "mean_log": float(qc_metrics["mean_log"]),
            "peak_log": float(qc_metrics["peak_log"]),
            "min_corr": float(qc_metrics.get("min_corr", 1.0)),
            "median_corr": float(qc_metrics.get("median_corr", 1.0)),
        })

# After line 428 (after metadata CSV write):
# NEW: Write QA metrics
if qc_rows:
    qc_df = pd.DataFrame(qc_rows)
    qc_path = ff_dir / "focus_quality_metrics.csv"
    qc_df.to_csv(qc_path, index=False)
    print(f"✔️  Wrote QA metrics: {qc_path}")
```

#### B. Add Optional Filtering (Opt-In)

```python
# Add to function signature of build_ff_from_keyence (around line 202):
def build_ff_from_keyence(
    ...,
    enable_slice_filtering: bool = False,       # NEW
    filter_alpha: float = 0.7,                 # NEW
    filter_corr_threshold: float = 0.9,        # NEW
    filter_method: str = "peak_relative",      # NEW
):

# Modify line 382 to use wrapper:
if enable_slice_filtering:
    ff, log_scores, metrics = LoG_focus_stacker_with_filtering(
        flat, filter_size=filter_size,
        enable_filtering=True,
        filter_alpha=filter_alpha,
        filter_corr_threshold=filter_corr_threshold,
        filter_method=filter_method,
        return_metrics=True
    )
    # Add filtering metrics to QA rows
    qc_rows[-1].update({
        "n_slices_total": metrics["n_total"],
        "n_slices_kept": metrics["n_kept"],
        "rejection_rate": metrics["rejection_rate"]
    })
else:
    ff, log_scores = LoG_focus_stacker(flat, filter_size=filter_size)
    qc_metrics = compute_slice_quality_metrics(flat, log_scores)
```

### 3.2 YX1 Pipeline (build01B_compile_yx1_images_torch.py)

**Location**: `_focus_stack` function around line 229

Similar modifications:

```python
def _focus_stack(
    stack_zyx: np.ndarray,
    device: str,
    filter_size: int = 3,
    enable_filtering: bool = False,      # NEW
    filter_alpha: float = 0.7,          # NEW
    filter_method: str = "peak_relative", # NEW
    return_metrics: bool = False         # NEW
) -> Union[np.ndarray, tuple]:

    norm, _, _ = im_rescale(stack_zyx)
    norm = norm.astype(np.float32)
    tensor = torch.from_numpy(norm).to(device)

    if enable_filtering:
        ff_t, _, metrics = LoG_focus_stacker_with_filtering(
            tensor, filter_size, device,
            enable_filtering=True,
            filter_alpha=filter_alpha,
            filter_method=filter_method,
            return_metrics=True
        )
    else:
        ff_t, _ = LoG_focus_stacker(tensor, filter_size, device)
        metrics = {} if return_metrics else None

    arr = ff_t.cpu().numpy()
    arr_clipped = np.clip(arr, 0, 65535)
    ff_i = arr_clipped.astype(np.uint16)
    ff_8 = skimage.util.img_as_ubyte(ff_i)

    if return_metrics:
        return ff_8, metrics
    return ff_8
```

### 3.3 Command-Line Arguments

Add to argument parser in both build scripts:

```python
parser.add_argument(
    "--enable_slice_filtering",
    action="store_true",
    help="Enable pre-filtering of bad z-slices before focus stacking"
)
parser.add_argument(
    "--filter_alpha",
    type=float,
    default=0.7,
    help="Peak-relative threshold for sharpness filtering (0.5-0.8, default: 0.7)"
)
parser.add_argument(
    "--filter_corr_threshold",
    type=float,
    default=0.9,
    help="Correlation threshold for motion detection (0.85-0.95, default: 0.9)"
)
parser.add_argument(
    "--filter_method",
    choices=["peak_relative", "correlation", "hybrid"],
    default="peak_relative",
    help="Filtering method (default: peak_relative)"
)
```

---

## Phase 4: Validation & Testing

### 4.1 Notebook Validation Checklist

- [ ] Load all 10 labeled examples from lookup CSV
- [ ] For each example, test filtering with 3-4 parameter combinations
- [ ] Generate side-by-side comparison plots (original vs filtered)
- [ ] Create quantitative comparison table with visual quality ratings
- [ ] Identify recommended thresholds (α, corr_threshold)

### 4.2 Integration Testing

#### Step 1: QA Metrics Only (Safe)
```bash
python src/build/build01A_compile_keyence_torch.py \
    --exp_name 20250912 \
    --overwrite
```
- Check that `focus_quality_metrics.csv` is created
- Verify bad wells have low `min_corr` or low `mean_log`
- Verify output images are identical to before (no filtering applied)

#### Step 2: Filtering Enabled (Experimental)
```bash
python src/build/build01A_compile_keyence_torch.py \
    --exp_name 20250912 \
    --enable_slice_filtering \
    --filter_alpha 0.7 \
    --filter_method peak_relative \
    --overwrite
```
- Visually compare output for:
  - Well B10, times 92, 96, 97 (should improve)
  - Well C04, time 28 (should stay great)
- Check rejection rates in QA CSV

#### Step 3: Hybrid Method (Advanced)
```bash
python src/build/build01A_compile_keyence_torch.py \
    --exp_name 20250912 \
    --enable_slice_filtering \
    --filter_alpha 0.7 \
    --filter_corr_threshold 0.9 \
    --filter_method hybrid
```

### 4.3 Success Criteria

✅ **Notebook Validation**:
- Bad images show visible improvement (sharper, less motion blur)
- Great images remain unchanged (high retention rate >80%)
- Optimal parameters identified and documented

✅ **Integration Testing**:
- QA metrics CSV correctly identifies problem stacks
- Filtering improves bad images without degrading good ones
- No crashes or errors with various parameter combinations

✅ **Documentation**:
- This plan document completed with findings
- Recommended thresholds documented
- Usage examples provided

---

## Expected Outputs

### 1. QA Metrics CSV
**Location**: `built_image_data/Keyence/FF_images/<exp>/focus_quality_metrics.csv`

| well | time_int | tile | mean_log | peak_log | min_corr | median_corr | n_slices_kept | rejection_rate |
|------|----------|------|----------|----------|----------|-------------|---------------|----------------|
| B10  | 92       | 0    | 0.0234   | 0.0451   | 0.42     | 0.89        | 5             | 0.55           |
| C04  | 28       | 0    | 0.0567   | 0.0823   | 0.97     | 0.98        | 10            | 0.09           |

**Interpretation**:
- Low `min_corr` (<0.85): Motion artifact detected
- Low `mean_log` (<0.03): Generally out of focus
- High `rejection_rate` (>0.4): Problematic stack

### 2. Updated Images
Filtered images should show:
- **Bad → Improved**: Reduced motion blur, sharper details
- **Okay → Improved**: Subtle enhancement
- **Great → Unchanged**: No degradation

---

## Future Enhancements (Phase 5+)

### Advanced Methods (Not Immediately Required)

1. **Anchor Correlation Method**
   - Use sharpest frame as reference
   - Filter by correlation to anchor (not just sequential)
   - Better for drift detection

2. **All-vs-All Correlation Matrix**
   - Visualize global relationships
   - Find contiguous "stable segments"
   - Useful for fragmented stacks

3. **Golden Template Method**
   - Create reference template from perfect stacks
   - Filter all future stacks by correlation to template
   - Most robust but requires template curation

4. **Adaptive Thresholding**
   - Use MAD (Median Absolute Deviation) for robust statistics
   - Auto-adjust thresholds per experiment
   - Reduces manual tuning

---

## References

- [bad_image_examples.md](./bad_image_examples.md) - Labeled test cases
- [frame_nd2_lookup.csv](./frame_nd2_lookup.csv) - ND2 index mapping
- [export_utils.py](../../../src/build/export_utils.py) - Current LoG implementation
- [build01A_compile_keyence_torch.py](../../../src/build/build01A_compile_keyence_torch.py) - Keyence pipeline
- [build01B_compile_yx1_images_torch.py](../../../src/build/build01B_compile_yx1_images_torch.py) - YX1 pipeline

---

## Timeline

| Phase | Tasks | Est. Time | Priority |
|-------|-------|-----------|----------|
| 1     | Notebook validation | 2-3 hours | **HIGH** |
| 2     | Export utils implementation | 1-2 hours | **HIGH** |
| 3A    | QA metrics integration | 1 hour | **HIGH** |
| 3B    | Optional filtering integration | 1-2 hours | **MEDIUM** |
| 4     | Testing & validation | 2-3 hours | **HIGH** |

**Total**: ~8-12 hours

---

## Notes

- **Conservative approach**: QA metrics first, then optional filtering
- **Backward compatible**: No changes to default behavior
- **Test-driven**: Validate on labeled examples before full deployment
- **Tunable**: All thresholds exposed as parameters
