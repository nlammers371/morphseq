# Refactor PRD: Segmentation Pipeline Integration MVP

## Objective

Create MVP integration between the segmentation_sandbox pipeline and existing morphseq build scripts, focusing on **minimal working solution first** with clear path to full integration.

## MVP Scope (Phase 1)

**Goal**: Get build scripts working with segmentation_sandbox masks for **ONE experiment** end-to-end.

### MVP Features
- Replace U-Net masks with GroundingDINO+SAM2 integer-labeled masks  
- Build scripts can process integer masks from `segmentation_sandbox/data/exported_masks/`
- Preserve existing functionality for binary mask fallback
- **Track deprecation**: Remove dependency on `region_label` tracking system

## Current State Analysis

### Segmentation Sandbox Pipeline (✅ Working)
- **Input**: `built_image_data/stitched_FF_images/` 
- **Output**: `segmentation_sandbox/data/exported_masks/<exp>/masks/<image>_masks_emnum_<N>.png`
- **Format**: Integer-labeled masks (pixel value = embryo_id, background = 0)
- **Pipeline**: 6 steps via `run_pipeline.sh` → generates labeled masks via `SimpleMaskExporter`

### Build Scripts (🔄 Needs Updates)
**Key mask usage patterns identified:**

1. **`src/build/build03A_process_images.py:84-88`** - `estimate_image_background()`:
   ```python
   im_mask = io.imread(im_emb_path)
   im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)  # Binary conversion
   ```

2. **`src/functions/image_utils.py:41-50`** - `process_masks()`:
   ```python
   im_mask = np.round(im_mask / 255 * 2 - 1).astype(np.uint8)
   im_mask_lb = label(im_mask)  # Creates labels from binary mask
   lbi = row["region_label"]    # 🚨 TRACKING DEPENDENCY TO REMOVE
   ```

3. **`src/build/build03A_process_images.py:341-352`** - `process_mask_images()`:
   ```python
   im_mask = (np.round(im / 255 * 2) - 1).astype(np.uint8)  # Binary conversion
   ```

### Core Issue: Region Label Tracking
- Build scripts expect `row["region_label"]` from tracking system
- Segmentation sandbox provides **pre-labeled** integer masks
- **MVP Solution**: Skip `skimage.measure.label()` step, use sandbox labels directly

## MVP Implementation Plan

### Phase 1.1: Minimal Mask Format Detection (1-2 hours)

Create utility to detect mask format and convert appropriately:

```python
# New utility in src/functions/mask_utils.py
def detect_and_process_mask(mask_path, region_label=None):
    """
    Detect mask format and convert appropriately.
    Returns: (binary_mask, label_value)
    """
    im_mask = io.imread(mask_path)
    
    # Detect format by checking if values are integer labels (>1) or binary (0-255)
    if im_mask.max() > 1 and im_mask.max() < 255:
        # Integer-labeled mask from segmentation_sandbox
        if region_label is not None:
            binary_mask = (im_mask == region_label).astype(np.uint8)
            return binary_mask, region_label
        else:
            # Use first available label
            available_labels = np.unique(im_mask[im_mask > 0])
            if len(available_labels) > 0:
                label_value = available_labels[0]
                binary_mask = (im_mask == label_value).astype(np.uint8)
                return binary_mask, label_value
    else:
        # Legacy binary mask - use original logic
        binary_mask = np.round(im_mask / 255 * 2 - 1).astype(np.uint8)
        # Use skimage.measure.label for region detection
        im_mask_lb = label(binary_mask)
        if region_label is not None:
            binary_mask = (im_mask_lb == region_label).astype(np.uint8)
            return binary_mask, region_label
            
    return None, None
```

### Phase 1.2: Update Key Functions (2-3 hours)

**1. Update `src/functions/image_utils.py:41-50`**:
```python
def process_masks(im_mask, im_yolk, row, close_radius=15):
    # Use new detection utility
    binary_mask, detected_label = detect_and_process_mask(im_mask, row.get("region_label"))
    
    if binary_mask is None:
        raise ValueError(f"Could not process mask for {row}")
        
    im_mask_ft = binary_mask.astype(int)
    # Continue with existing yolk processing...
```

**2. Update `src/build/build03A_process_images.py:84-88`**:
```python
def estimate_image_background(root, embryo_metadata_df, bkg_seed=309, n_bkg_samples=100):
    # ... existing code ...
    
    # Replace mask loading section:
    binary_mask, _ = detect_and_process_mask(im_emb_path, row.get("region_label"))
    if binary_mask is not None:
        im_mask = binary_mask
    else:
        # Fallback to original logic
        im_mask = io.imread(im_emb_path)
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)
```

### Phase 1.3: Path Configuration (30 minutes)

Update segmentation path to point to sandbox output:

```python
# In build scripts, replace:
segmentation_path = os.path.join(root, 'segmentation', '')

# With:
segmentation_path = os.path.join(root, 'segmentation_sandbox', 'data', 'exported_masks', '')
```

### Phase 1.4: MVP Test (1 hour)

Test with single experiment:
1. Run segmentation pipeline: `./segmentation_sandbox/scripts/pipelines/run_pipeline.sh "20240418"`
2. Run build script with modified mask loading
3. Verify end-to-end functionality

## Data Flow (MVP)

```
Input: built_image_data/stitched_FF_images/20240418/
  ↓
Segmentation Pipeline: run_pipeline.sh "20240418" 
  ↓
Output: segmentation_sandbox/data/exported_masks/20240418/masks/
  ↓
Build Scripts: Modified mask loading logic
  ↓  
Training Data: Processed embryo snips
```

## Success Criteria (MVP)

- [ ] Single experiment (e.g., 20240418) processes end-to-end
- [ ] Build scripts load integer masks correctly
- [ ] No crashes or data corruption
- [ ] Backward compatibility with binary masks maintained
- [ ] **Tracking system dependency reduced** (region_label optional)

## Risk Mitigation

### Backward Compatibility
- Maintain fallback to original binary mask logic
- Keep existing mask loading paths as backup
- Test with both mask formats

### Error Handling  
- Graceful fallback when mask detection fails
- Clear error messages for debugging
- Log format detection results

## Future Phases (Post-MVP)

### Phase 2: Full Integration (After MVP Success)
- Process all experiments
- Remove legacy segmentation paths entirely  
- Optimize performance
- Handle missing yolk mask implications

### Phase 3: Deprecation Cleanup
- Remove `region_label` tracking entirely
- Simplify mask processing logic
- Update documentation

## Files to Modify (MVP)

### New Files
1. **`src/functions/mask_utils.py`** - Mask format detection utility

### Modified Files  
2. **`src/functions/image_utils.py:41-50`** - Update `process_masks()` function
3. **`src/build/build03A_process_images.py:84-88`** - Update `estimate_image_background()` 
4. **`src/build/build03A_process_images.py:341-352`** - Update `process_mask_images()` (if needed)

### Configuration Updates
5. Update segmentation paths to point to `segmentation_sandbox/data/exported_masks/`

## Key Advantages of This Approach

1. **MVP-First**: Get basic functionality working quickly
2. **Low Risk**: Maintains backward compatibility throughout
3. **Incremental**: Can test and validate each step
4. **Deprecation-Focused**: Actively removes dependency on tracking system
5. **Clear Path Forward**: Sets up architecture for full integration

## Notes on Yolk Mask Handling

The segmentation sandbox doesn't generate yolk masks. Build scripts have fallback logic when yolk masks are missing:
- `get_embryo_angle()` uses embryo shape instead of yolk landmark
- Focus calculation uses embryo-only region
- This creates systematic differences but doesn't break functionality

**MVP Decision**: Accept yolk mask absence for initial integration, evaluate impact in Phase 2.