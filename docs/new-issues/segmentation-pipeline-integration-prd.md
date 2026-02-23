# Segmentation Pipeline Integration PRD

## Objective

Replace legacy U-Net segmentation with GroundingDINO + SAM2 pipeline in the morphseq build workflow, ensuring seamless integration between the segmentation_sandbox pipeline and existing build scripts.

## Technical Requirements

### Core Integration Goals
- Replace U-Net segmentation with 6-step GroundingDINO + SAM2 pipeline
- Convert integer-labeled masks (embryo_id as pixel value) to binary masks (0/255 format) 
- Maintain compatibility with existing build script expectations
- Preserve current file naming and directory structure where possible
- Support experiment filtering and batch processing

### Pipeline Components to Integrate
- **Main Pipeline**: `segmentation_sandbox/scripts/pipelines/run_pipeline.sh`
- **Configuration**: `segmentation_sandbox/configs/pipeline_config.yaml`
- **Mask Exporter**: `SimpleMaskExporter` class in `segmentation_sandbox/scripts/utils/simple_mask_exporter.py`

## Integration Architecture

### Data Flow
```
Stitched FF Images → Segmentation Pipeline → Integer Masks → Build Scripts
built_image_data/    run_pipeline.sh       exported_masks/   build03A/03B
stitched_FF_images/                        <exp>/masks/      (convert to binary)
```

### Key Integration Points
1. **Input**: `built_image_data/stitched_FF_images/` → Pipeline video preparation
2. **Processing**: 6-step segmentation pipeline execution
3. **Output**: `exported_masks/<experiment>/masks/` → Build script consumption
4. **Format Conversion**: Integer labels → Binary masks (0/255) in build scripts

## Implementation Steps

### Step 1: Build Script Modifications
Modify build scripts to handle integer-labeled masks from segmentation_sandbox:

#### 1.1 Update `src/build/build03A_process_images.py`

**Function: `estimate_image_background` (lines 84-88)**
```python
# Current code:
im_mask = io.imread(im_emb_path)
im_mask = np.round(im_mask / 255 * 2 - 1).astype(int)

# New code:
im_mask_raw = io.imread(im_emb_path)
im_mask = (im_mask_raw == row["region_label"]).astype(int)
```

**Function: `export_embryo_snips` (lines 168-176)**
```python
# Current code:
im_mask = io.imread(im_emb_path)
# ... processing ...

# New code:
im_mask = ((io.imread(im_emb_path) == row["region_label"]) * 255).astype("uint8")
```

**Function: `process_mask_images` (lines 341-352)**
```python
# Current code:
im_mask = (np.round(im / 255 * 2) - 1).astype(np.uint8)

# New code:
im_mask = (io.imread(image_path) > 0).astype(np.uint8)
```

#### 1.2 Update `src/build/build03B_export_z_snips.py`

**Mask loading (lines 87-94)**
```python
# Apply same binarization as in export_embryo_snips
im_mask = ((io.imread(im_emb_path) == row["region_label"]) * 255).astype("uint8")
im_yolk = ((io.imread(im_yolk_path) > 0) * 255).astype("uint8")
```

#### 1.3 Update `src/functions/image_utils.py`

**Function: `process_masks` (lines 41-50)**
```python
# Current code assumes binary input, modify to handle integer labels
def process_masks(im_mask, im_yolk, row, close_radius=15):
    # Check if mask is already binary or needs conversion
    if im_mask.max() > 1:
        # Integer-labeled mask from segmentation_sandbox
        region_label = row["region_label"]
        im_mask = (im_mask == region_label).astype(np.uint8)
    else:
        # Legacy binary mask
        im_mask = np.round(im_mask / 255 * 2 - 1).astype(np.uint8)
    
    # Continue with existing logic...
```

### Step 2: Pipeline Integration Script
Create integration wrapper that manages the full workflow:

#### 2.1 Create `scripts/run_integrated_segmentation.sh`
```bash
#!/bin/bash
# Integrated segmentation pipeline runner

# Configuration
MORPHSEQ_ROOT="/path/to/morphseq"
SANDBOX_ROOT="$MORPHSEQ_ROOT/segmentation_sandbox"
EXPERIMENT_LIST="$1"  # Comma-separated experiment IDs

# Step 1: Run segmentation pipeline
cd $SANDBOX_ROOT
./scripts/pipelines/run_pipeline.sh $EXPERIMENT_LIST

# Step 2: Verify mask export completion
python -c "
from scripts.utils.simple_mask_exporter import SimpleMaskExporter
from pathlib import Path
exporter = SimpleMaskExporter(
    sam2_path=Path('data/segmentation/grounded_sam_segmentations.json'),
    output_dir=Path('data/exported_masks')
)
status = exporter.get_export_status()
print(f'Export Status: {status}')
"

echo "Segmentation pipeline completed. Masks available for build scripts."
```

### Step 3: Configuration Updates

#### 3.1 Update `segmentation_sandbox/configs/pipeline_config.yaml`
Ensure paths point to correct morphseq data directories:
```yaml
paths:
  morphseq_data_dir: "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
  stitched_images_dir: "built_image_data/stitched_FF_images"
```

### Step 4: Build Script Path Updates
Update build scripts to look for masks in segmentation_sandbox output location:

```python
# In build03A_process_images.py and build03B_export_z_snips.py
# Update segmentation path to point to sandbox output
segmentation_path = os.path.join(root, 'segmentation_sandbox', 'data', 'exported_masks', '')
```

## File Modifications

### Files to Modify
1. **`src/build/build03A_process_images.py`**
   - Lines 84-88: `estimate_image_background` function
   - Lines 168-176: `export_embryo_snips` function  
   - Lines 341-352: `process_mask_images` function

2. **`src/build/build03B_export_z_snips.py`**
   - Lines 87-94: Mask loading section

3. **`src/functions/image_utils.py`**
   - Lines 41-50: `process_masks` function

### Files to Create
4. **`scripts/run_integrated_segmentation.sh`** - Integration wrapper script

### Files to Update
5. **`segmentation_sandbox/configs/pipeline_config.yaml`** - Path configuration

## Data Format Specifications

### Input Format (Stitched Images)
- **Location**: `built_image_data/stitched_FF_images/<experiment>/`
- **Format**: Standard image files (PNG/TIFF)
- **Naming**: `<well>_t<timepoint>*` pattern

### Pipeline Output Format (Integer Masks)
- **Location**: `segmentation_sandbox/data/exported_masks/<experiment>/masks/`
- **Format**: PNG files with integer pixel values
- **Naming**: `<image_id>_masks_emnum_<N>.png`
- **Content**: Pixel value = embryo_id (1, 2, 3, etc.), background = 0

### Build Script Expected Format (Binary Masks)
- **Format**: Binary masks with values 0 (background) and 255 (embryo)
- **Conversion**: Integer mask → Binary mask based on `region_label`
- **Logic**: `(integer_mask == region_label) * 255`

## Testing Strategy

### Unit Tests
1. **Mask Conversion Testing**
```python
def test_integer_to_binary_conversion():
    # Test integer mask → binary mask conversion
    integer_mask = np.array([[0, 1, 1], [2, 2, 0], [0, 3, 3]])
    region_label = 2
    expected = np.array([[0, 0, 0], [255, 255, 0], [0, 0, 0]])
    result = ((integer_mask == region_label) * 255).astype("uint8")
    assert np.array_equal(result, expected)
```

2. **File Path Testing**
```python
def test_mask_file_discovery():
    # Test that build scripts can find segmentation_sandbox masks
    # Verify path resolution and file existence
```

### Integration Tests
1. **End-to-End Pipeline Test**
```bash
# Test complete workflow with small dataset
./scripts/run_integrated_segmentation.sh "20240418"
# Verify masks are created and build scripts can process them
```

2. **Build Script Compatibility Test**
```python
# Test that modified build scripts work with both:
# - Legacy binary masks (backward compatibility)
# - New integer masks (forward compatibility)
```

### Validation Criteria
- [ ] Segmentation pipeline runs without errors
- [ ] Masks are exported to correct locations with proper naming
- [ ] Build scripts successfully process integer masks
- [ ] Backward compatibility maintained for existing binary masks
- [ ] No data loss during format conversion
- [ ] Performance acceptable (similar to legacy pipeline)

## Risk Mitigation

### Backward Compatibility
- Keep original mask loading logic as fallback
- Test with both integer and binary mask formats
- Maintain existing file path structure where possible

### Error Handling
- Add validation for mask format detection
- Implement graceful fallback for missing region_labels
- Log conversion errors for debugging

### Performance Considerations
- Integer-to-binary conversion adds minimal overhead
- Batch processing maintained for efficiency
- Memory usage similar to original implementation

## Analysis of Missing Yolk Mask Data

A key difference between the legacy U-Net pipeline and the new GroundingDINO + SAM2 pipeline is that the new pipeline does not generate yolk masks. An analysis of the existing build scripts reveals that the yolk mask is integral to two key downstream scientific calculations:

1.  **Embryo Orientation:** In `src/functions/image_utils.py`, the `get_embryo_angle` function uses the yolk mask as a primary landmark to establish a consistent anterior-posterior orientation for the embryo. Without it, the function uses a less reliable fallback method based on the embryo's shape alone.

2.  **Focus Calculation:** In `src/build/build03B_export_z_snips.py`, the yolk area is subtracted from the embryo mask to create a "body-only" region. The pipeline then analyzes this region to find the most in-focus z-slice.

### Implications of Integration

The build scripts are robust and will not crash if a yolk mask is missing. They will instead default to their fallback behaviors. However, this has direct consequences for data consistency:

*   **Systematic Difference:** Embryos processed without a yolk mask will have their orientation and selected focal plane determined differently than those processed with one.
*   **Potential for Reduced Accuracy:** The fallback methods are likely less accurate, potentially leading to inconsistent embryo alignment and suboptimal focus selection.

This represents a trade-off: the pipeline becomes compatible with the new segmentation output, but at the cost of introducing a known difference in data processing. This should be considered during downstream analysis of the resulting data.

## Success Metrics

1. **Functional**: All build scripts work with segmentation_sandbox output
2. **Performance**: <10% performance degradation vs. legacy pipeline  
3. **Compatibility**: No breaking changes to downstream analysis
4. **Quality**: Segmentation accuracy equal or better than U-Net
5. **Maintainability**: Clear separation between pipeline and build logic

---

## Revised Implementation Strategy

After further analysis of the build scripts, a more robust integration strategy is required. The initial plan to simply convert integer masks to binary masks within the build scripts is flawed because it fails to address the core `region_label` lifecycle and dependency on `skimage.measure.label`.

Here is the revised approach:

**1. Problem: Incompatible Mask Formats & The `region_label` Conflict**

*   **The Issue:** The script assumes one kind of mask (binary 0/255) but is now receiving another (integer-labeled). This causes the critical `region_label` to mismatch between steps, as the script generates its own labels that conflict with the pre-labeled segmentation masks.
*   **The Solution:**
    *   Introduce a `detect_mask_format` utility function within the build script to automatically identify the mask type.
    *   Create conditional logic based on the detected format:
        *   **For new integer-labeled masks:** The script will trust and use the integer labels from the pipeline directly. These values will be used as the definitive `region_label`, ensuring data consistency and correctly leveraging the more informative output of the new segmentation pipeline. This resolves the core conflict.
        *   **For old binary masks:** The script will use the original logic (`skimage.measure.label`) to find objects and generate labels. This ensures full backward compatibility.