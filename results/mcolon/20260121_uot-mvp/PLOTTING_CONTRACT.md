# UOT Visualization Contract

## Mandatory Requirements for All Velocity/Mass Plots

### 1. Support Mask Must Be Explicit
- **REQUIRED**: Every plot must include a `support_mask` (bool array) indicating where data is defined
- **FORBIDDEN**: Plotting dense arrays initialized with zeros without a mask
- **CHECK**: Assert `support_mask is not None` before plotting

### 2. Non-Support Regions Use NaN, Not Zero
- **REQUIRED**: Dense fields must use `np.nan` for undefined regions
- **RATIONALE**: `0.0` means "stopped", `NaN` means "not sampled"
- **CHECK**: Assert that `np.any(np.isnan(field))` or `support_mask.sum() < field.size`

### 3. Velocity Must Show Support Coverage
- **REQUIRED**: First panel or overlay shows `support_mask` (binary coverage map)
- **LABEL**: "Defined Region (X.XX% of pixels)"
- **CHECK**: Title must include `{support_pct:.2f}%`

### 4. No Fabrication via Smoothing
- **ALLOWED**: Block-splat projection (work cell → s×s canonical footprint)
- **ALLOWED**: Normalized convolution weighted by transported mass (NaN where weight=0)
- **FORBIDDEN**: `gaussian_filter()` on full field (fabricates values in no-data regions)
- **REQUIRED**: If smoothing is applied, it must be labeled and applied ONLY to support regions

### 5. Numeric Annotations Required
Every velocity plot must display:
- `% defined: X.XX%` (support coverage)
- `p50/p90/p99: X.X / Y.Y / Z.Z μm/frame` (on support only)
- `max: W.W μm/frame` (on support only)

### 6. Diagnostics Sidecar
- **REQUIRED**: Include histogram or table of velocity magnitudes on support
- **FORBIDDEN**: Histograms of the full dense field (misleading due to zeros/NaNs)

## Enforcement Checklist

```python
def assert_plotting_contract(field, support_mask, field_name="field"):
    """Enforce plotting contract before visualization."""
    assert support_mask is not None, f"{field_name}: support_mask is None"
    assert support_mask.dtype == bool, f"{field_name}: support_mask must be bool"
    assert support_mask.shape == field.shape[:2], f"{field_name}: shape mismatch"
    
    # Either field has NaNs or support_mask indicates sparsity
    has_nans = np.any(np.isnan(field))
    is_sparse = support_mask.sum() < support_mask.size
    assert has_nans or is_sparse, \
        f"{field_name}: field is fully defined but no NaNs (zeros != no-data)"
    
    # If field has NaNs, they should align with ~support_mask
    if has_nans:
        nan_mask = np.any(np.isnan(field.reshape(field.shape[0], field.shape[1], -1)), axis=-1)
        mismatch = np.sum(nan_mask != ~support_mask)
        assert mismatch / nan_mask.size < 0.01, \
            f"{field_name}: NaN mask doesn't match support_mask ({mismatch} mismatches)"
```

## Example Good Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Support mask
axes[0].imshow(support_mask, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Support Coverage\n{support_pct:.2f}% defined")

# Panel 2: Velocity magnitude (NaN outside support)
velocity_mag = np.sqrt(velocity[...,0]**2 + velocity[...,1]**2)
velocity_mag[~support_mask] = np.nan
im = axes[1].imshow(velocity_mag, cmap="viridis")
axes[1].set_title(f"Velocity (on support)\np50/p90/p99: {p50:.1f}/{p90:.1f}/{p99:.1f} μm/frame")
plt.colorbar(im, ax=axes[1])

# Panel 3: Velocity histogram (support only)
support_velocities = velocity_mag[support_mask]
axes[2].hist(support_velocities, bins=50, alpha=0.7)
axes[2].set_title(f"Velocity Distribution\n(n={support_mask.sum():,} points)")
axes[2].set_xlabel("μm/frame")
```

## What This Prevents

- ✗ "Dense zero field looks like everything stopped" (use NaNs)
- ✗ "3.39% moved but I can't see which pixels" (show support_mask)
- ✗ "Max velocity 2376 but median 0" (plot only support statistics)
- ✗ "Smoothing made it look reasonable" (enforce no-fabrication rule)

---

**Audit Date**: 2026-01-22
**Next Review**: After implementing DenseField helper with support_mask
