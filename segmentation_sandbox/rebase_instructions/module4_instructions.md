# Module 4: QC Integration

## Overview
Create a unified QC system with strict separation between **image integrity QC** (Step 02 and before) and **embryo integrity QC** (Step 03 onwards). QC flags are attached at multiple levels and are accessible to the visualization system, but image QC and embryo QC must not be mixed in code, data, or reporting.

**Image Integrity QC**
- Assesses raw image quality (focus, exposure, corruption, etc.)
- Performed before any annotation or mask generation
- Independent of GSAM/embryo IDs
- Results stored and propagated separately

**Embryo Integrity QC**
- Assesses segmentation/mask quality, overlaps, annotation completeness
- Performed after annotation/mask generation
- Dependent on GSAM annotation IDs and embryo IDs
- Includes mask overlap detection, missing annotation tracking, mask validation

## Dependencies
- Modules 1-3 completed
- Integration with ExperimentMetadata and EmbryoMetadata

## Files to Create/Modify

```
utils/
└── qc/
    ├── __init__.py
    ├── qc_flags.py          # Flag definitions and validation (for both QC types)
    ├── image_qc.py          # Image integrity QC (focus, exposure, corruption)
    └── embryo_qc.py         # Embryo integrity QC (mask, overlap, annotation completeness)
```

## Implementation Steps (Pseudocode)

### Step 1: Create `utils/qc/qc_flags.py`

```python
"""Unified QC flag definitions and validation."""

# QC flag definitions - single source of truth
QC_FLAGS = {
    'experiment_level': {
        'POOR_IMAGING_CONDITIONS': {'severity': 'warning', 'desc': 'Suboptimal imaging'},
        'INCOMPLETE': {'severity': 'error', 'desc': 'Missing data'},
        'PROTOCOL_DEVIATION': {'severity': 'warning', 'desc': 'Protocol not followed'}
    },
    'video_level': {
        'DRY_WELL': {'severity': 'critical', 'desc': 'Well dried during imaging'},
        'FOCUS_DRIFT': {'severity': 'warning', 'desc': 'Focus unstable'},
        'NONZERO_SEED_FRAME': {'severity': 'info', 'desc': 'Seed frame not first'},
        'NO_EMBRYO_DETECTED': {'severity': 'error', 'desc': 'No embryos found'}
    },
    'image_level': {
        'BLUR': {'severity': 'warning', 'desc': 'Low sharpness'},
        'DARK': {'severity': 'warning', 'desc': 'Underexposed'},
        'OVEREXPOSED': {'severity': 'warning', 'desc': 'Overexposed'},
        'CORRUPT': {'severity': 'critical', 'desc': 'File corrupted'},
        'EMPTY': {'severity': 'info', 'desc': 'No content'}
    },
    'embryo_level': {
        'DEAD': {'severity': 'critical', 'desc': 'No movement'},
        'MASK_ON_EDGE': {'severity': 'warning', 'desc': 'Touching boundary'},
        'DETECTION_FAILURE': {'severity': 'error', 'desc': 'Lost tracking'},
        'HIGHLY_VAR_MASK': {'severity': 'warning', 'desc': 'Unstable segmentation'}
    }
}

def validate_flag(level, flag):
    """Validate QC flag."""
    # [Implementation]
    pass

def get_flag_severity(level, flag):
    """Get severity level for flag."""
    # [Implementation]
    pass
```


### Step 2: Create `utils/qc/image_qc.py`

```python
"""Image Integrity QC: Automated image quality analysis (focus, exposure, corruption, etc.)"""

import cv2
import numpy as np
from ..core import parse_entity_id

class ImageQualityAnalyzer:
    """Analyze image integrity metrics (independent of annotation/mask)."""
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            'blur_threshold': 100,
            'dark_threshold': 30,
            'bright_threshold': 225
        }
    def analyze_image(self, image_path):
        # 1. Load image
        # 2. Calculate blur (Laplacian variance)
        # 3. Check brightness/contrast
        # 4. Check for corruption/empty
        # 5. Return image-level QC flags only
        pass
    def analyze_batch(self, image_paths, parallel=True):
        # Batch process image integrity QC
        pass
```


### Step 3: Create `utils/qc/embryo_qc.py`

```python
"""Embryo Integrity QC: Quality checks for segmentation/mask results (GSAM/embryo ID dependent)."""

from ..core import parse_entity_id
from .qc_flags import QC_FLAGS

class EmbryoSegmentationQC:
    """Check embryo segmentation/mask quality (GSAM/embryo ID dependent)."""
    def __init__(self, sam_annotations_path, metadata_path):
        # Load annotations and metadata
        pass
    def check_mask_on_edge(self, margin_pixels=5):
        # For each mask: check if any pixels within margin of edge, flag if true
        pass
    def check_mask_variability(self, threshold=0.1):
        # For each embryo track: calculate area for each frame, flag if unstable
        pass
    def check_detection_failures(self):
        # For each embryo: find gaps in frame sequence, flag as detection failures
        pass
    def check_mask_overlap(self):
        # For each image: detect and flag pixel overlaps between embryo masks
        pass
    def check_missing_annotations(self):
        # For each image: ensure all images are tracked, even if not annotated
        pass
    def generate_qc_report(self):
        # Aggregate all embryo QC checks
        pass
```


## Integration Points

1. **ExperimentMetadata**: Integrates image integrity QC only
2. **EmbryoMetadata**: Integrates embryo integrity QC only (GSAM/embryo ID dependent)
3. **Visualization**: Queries both QC types separately for display
4. **GSAM Pipeline**: Runs embryo integrity QC after segmentation/mask generation


## Testing Checklist

- [ ] Test flag validation (for both QC types)
- [ ] Test image integrity QC (focus, exposure, corruption)
- [ ] Test embryo integrity QC (mask, overlap, annotation completeness)
- [ ] Test integration with metadata classes (image QC → ExperimentMetadata, embryo QC → EmbryoMetadata)
- [ ] Test batch processing for both QC types
- [ ] Test report generation (separate reports for image and embryo QC)


## Notes

- QC flags have severity levels for prioritization
- All flags stored with author and timestamp
- Visualization can filter by severity and QC type
- Automated checks supplement manual QC
- **Image integrity QC and embryo integrity QC are strictly separated in code, data, and reporting.**