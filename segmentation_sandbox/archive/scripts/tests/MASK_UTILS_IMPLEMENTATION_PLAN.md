# Mask Utils Implementation Plan

## Overview
Based on debugging the QC analysis script, we've identified that the complex RLE encoding/decoding in `sam2_utils.py` is causing segmentation format issues. This plan implements a simple, centralized mask utilities system.

## Problem Summary
1. **Complex RLE Function**: The `convert_sam2_mask_to_rle()` function has overcomplicated string conversion that creates inconsistent formats
2. **Scattered Mask Functions**: Mask operations are spread across multiple files 
3. **RLE Decoding Issues**: Current data has base64-encoded RLE that pycocotools can't decode directly
4. **Format Detection Errors**: QC script was looking for format in wrong location

## Three-Step Implementation

### Step 1: Create Simple Mask Utils (`scripts/utils/mask_utils.py`)
**Goal**: Centralize all mask operations with simple, working functions

#### Functions to create:
```python
# Core RLE functions
def encode_mask_rle(binary_mask: np.ndarray) -> Dict
def decode_mask_rle(rle_data: Dict) -> np.ndarray

# Mask conversion functions  
def mask_to_bbox(binary_mask: np.ndarray) -> List[float]
def mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]
def mask_area(binary_mask: np.ndarray) -> float
def polygon_to_mask(polygons: List, height: int, width: int) -> np.ndarray

# Combined function for complete segmentation
def encode_mask_complete(binary_mask: np.ndarray, format: str = "rle") -> Dict

# Simple validation (no strict validation needed)
def validate_rle_format(rle_data: Dict) -> bool
```

#### Key Design Decisions:
- **No base64 encoding**: Let JSON handle bytes natively (simpler)
- **Self-contained segmentation**: Includes RLE + area + bbox - everything needed to recreate or analyze the mask
- **No classes**: Keep it simple with standalone functions
- **Complete encoding function**: `encode_mask_complete()` creates full segmentation object with all derived info
- **Backward compatibility**: Handle existing base64-encoded data in decode function

### Step 2: Update SAM2 Utils (`scripts/detection_segmentation/sam2_utils.py`)
**Goal**: Replace complex functions with simple mask_utils calls

#### Changes:
1. **Replace complex `convert_sam2_mask_to_rle()` (lines 823-859)** with:
   ```python
   def convert_sam2_mask_to_rle(binary_mask: np.ndarray) -> Dict:
       from scripts.utils.mask_utils import encode_mask_rle
       return encode_mask_rle(binary_mask)
   ```

2. **Move `convert_sam2_mask_to_polygon()` to mask_utils** and replace with wrapper

3. **Move `extract_bbox_from_mask()` to mask_utils** and replace with wrapper

### Step 3: Update QC Analysis Script (`scripts/pipelines/05_sam2_qc_analysis.py`)
**Goal**: Use mask_utils for all RLE decoding and validation

#### Changes:
1. **Replace custom RLE decoding** with:
   ```python
   from scripts.utils.mask_utils import decode_mask_rle
   
   def decode_rle_segmentation(segmentation: Dict) -> np.ndarray:
       return decode_mask_rle(segmentation)
   ```

2. **Add format validation** before decoding attempts

3. **Ensure consistent format field access**: Check `segmentation.get("format")` first, then fall back to `embryo_data.get("segmentation_format")` for backward compatibility 

## Testing Strategy

### Test 1: Basic Encode/Decode Round Trip
```python
# Create test mask -> encode -> decode -> verify identical
test_mask = np.zeros((100, 100), dtype=np.uint8)
test_mask[25:75, 25:75] = 1

rle = encode_mask_rle(test_mask)
decoded = decode_mask_rle(rle)
assert np.array_equal(test_mask, decoded)
```

### Test 2: Handle Existing Data Format
```python
# Load real segmentation data and verify decoding works
with open('grounded_sam_segmentations.json.backup_no_flags') as f:
    data = json.load(f)
# Test decode_mask_rle() on existing base64-encoded data
```

### Test 3: QC Script Integration
```python
# Run QC script with new mask_utils
python scripts/pipelines/05_sam2_qc_analysis.py --input data.json --dry-run --verbose
```

## Data Format Specification

### Current Problematic Format:
```json
{
  "segmentation": {
    "counts": "base64_encoded_string_of_utf8_decoded_bytes",
    "size": [1, 2304, 4378]  // 3D size array
  },
  "segmentation_format": "rle"  // Redundant field at embryo level
}
```

### New Simple Format:
```json
{
  "segmentation": {
    "counts": bytes_object,  // JSON serializes natively
    "size": [2304, 4378],    // 2D size array as expected
    "format": "rle",         // Format field in segmentation object
    "area": 12456,           // Area in pixels
    "bbox": [0.1, 0.2, 0.8, 0.9]  // Normalized bounding box [x1,y1,x2,y2]
  }
}
```

### Backward Compatibility:
- `decode_mask_rle()` handles existing base64-encoded data for current QC analysis
- Automatically fixes 3D -> 2D size arrays  
- Checks for format in both `segmentation.format` (new) and `segmentation_format` (old) locations
- **New segmentation data** will use clean format - no need to support old format long-term

## Implementation Decisions

Based on conversation clarifications:

1. **JSON Serialization**: Let JSON handle bytes natively (simpler approach)

2. **Existing Data Migration**: No migration script needed - will regenerate segmentation data with new format

3. **Error Handling**: Fail fast with detailed error messages - raise loud errors for corrupted RLE data that can't be decoded

4. **Additional Mask Functions**: Current set is sufficient - no need for dilation, erosion, connected components

5. **Performance Considerations**: No caching or lazy loading needed for current use case

6. **Format Validation**: No strict validation needed - keep it simple

## Expected Outcomes

After implementation:
- ✅ QC analysis script runs without RLE decoding errors
- ✅ Consistent mask operations across all scripts  
- ✅ Self-contained segmentation data format
- ✅ Backward compatibility with existing data
- ✅ Simple, maintainable codebase
- ✅ Clear documentation of mask format expectations

## Next Steps

1. **Review this plan** and answer implementation questions
2. **Create `scripts/utils/mask_utils.py`** with core functions
3. **Test basic encode/decode** functionality
4. **Update SAM2 utils** to use mask_utils
5. **Update QC script** to use mask_utils  
6. **Run end-to-end test** with real data
7. **Document the new format** and update other scripts as needed
