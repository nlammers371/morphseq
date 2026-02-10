# Pipeline Script 5 Minimal Compatibility Fix

## Overview
Minimal changes to make Pipeline Script 5 compatible with SAM2 utils data structure. Only change `"images"` to `"image_ids"` in key locations.

---

## Required Changes

### 1. Update Entity Tracking Initialization

**File:** `05_sam2_qc_analysis.py`  
**Method:** `_initialize_entity_tracking`  
**Line:** ~167

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

### 2. Update All Entity Processing Methods

**File:** `05_sam2_qc_analysis.py`  
**Method:** `get_all_entities_to_process`  
**Line:** ~272

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

### 3. Update Quality Check Methods

#### 3a. Segmentation Variability Check

**Method:** `check_segmentation_variability`  
**Line:** ~412

**Change:**
```python
# OLD:
images = video_data.get("images", {})

# NEW:
images = video_data.get("image_ids", {})
```

#### 3b. Mask on Edge Check

**Method:** `check_mask_on_edge`  
**Line:** ~528

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

#### 3c. Detection Failure Check

**Method:** `check_detection_failure`  
**Line:** ~588

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

#### 3d. Overlapping Masks Check

**Method:** `check_overlapping_masks`  
**Line:** ~622

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

#### 3e. Large Masks Check

**Method:** `check_large_masks`  
**Line:** ~691

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

#### 3f. Small Masks Check

**Method:** `check_small_masks`  
**Line:** ~738

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

#### 3g. Discontinuous Masks Check

**Method:** `check_discontinuous_masks`  
**Line:** ~784

**Change:**
```python
# OLD:
for image_id, image_data in video_data.get("images", {}).items():

# NEW:
for image_id, image_data in video_data.get("image_ids", {}).items():
```

### 4. Update Summary Methods

**Method:** `get_summary`  
**Line:** ~1047

**Change:**
```python
# OLD:
# New layout uses 'image_ids' mapping; support legacy 'images' as fallback
if 'image_ids' in video_data:
    total_images += len(video_data.get('image_ids', {}))
else:
    total_images += len(video_data.get('images', {}))

# NEW:
total_images += len(video_data.get('image_ids', {}))
```

---

## Summary of Changes

**Total changes:** 8 simple find-and-replace operations
- Change `"images"` → `"image_ids"` in 7 quality check methods
- Remove legacy fallback code in summary method

**Files affected:** 1 file (`05_sam2_qc_analysis.py`)

**Testing:** After changes, run QC on a small SAM2 dataset to verify compatibility.