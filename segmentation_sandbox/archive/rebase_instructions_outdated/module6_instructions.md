# Module 6: Script Updates

## Overview
Update main pipeline scripts to use new refactored utilities. Maintain backward compatibility while leveraging new features.

## Dependencies
- Modules 1-5 completed
- Existing scripts to update

## Scripts to Update

```
scripts/
├── 01_prepare_videos.py
├── 02_image_quality_qc.py  
├── 03_gdino_detection.py
├── 04_sam2_video_processing.py
└── 05_export_embryo_masks.py
```

## Key Updates (Pseudocode)

### Update `01_prepare_videos.py`

```python
# Key changes:
# 1. Use new ExperimentMetadata class
# 2. Recursive search for experiments
# 3. Integrated QC from the start

from utils.metadata.experiment import ExperimentMetadata
from utils.io import get_simplified_filename

def find_experiments(base_dir):
    """Recursively find experiment directories."""
    # [Pseudocode]
    # Look for directories containing stitch images
    # Pattern: *_stitch.png or *_stitch.jpg
    pass

def process_experiment(exp_dir, metadata):
    """Process single experiment."""
    # [Pseudocode]
    # 1. Find all stitch images
    # 2. Group by well
    # 3. Create simplified names (0000.jpg)
    # 4. Generate video with overlays
    # 5. Update metadata with QC flags
    pass
```

### Update `02_image_quality_qc.py`

```python
# Key changes:
# 1. QC integrated into ExperimentMetadata
# 2. Use new ImageQualityAnalyzer

from utils.metadata.experiment import ExperimentMetadata
from utils.qc.image_qc import ImageQualityAnalyzer

def run_qc(metadata_path):
    """Run automated QC."""
    metadata = ExperimentMetadata(metadata_path)
    analyzer = ImageQualityAnalyzer()
    
    # [Pseudocode]
    # For each image in metadata:
    #   Run quality analysis
    #   Add QC flags directly to metadata
    #   Auto-save periodically
    pass
```

### Update `03_gdino_detection.py`

```python
# Key changes:
# 1. Use new GdinoAnnotations class
# 2. Optional visualization during detection

from utils.annotation.detection import GdinoAnnotations
from utils.visualization import PipelineVisualizer

def run_detection(metadata_path, output_path, visualize=False):
    """Run GDINO detection."""
    # [Pseudocode]
    # 1. Load metadata
    # 2. Create GdinoAnnotations manager
    # 3. Run detection with optional viz
    # 4. Generate high-quality annotations
    pass
```

### Update `04_sam2_video_processing.py`

```python
# Key changes:
# 1. Use GroundedSam pipeline
# 2. Automatic QC after segmentation
# 3. Integrated visualization

from utils.annotation.grounded_sam import GroundedSam
from utils.qc.gsam_qc import SegmentationQC

def process_videos(config_path, metadata_path):
    """Process videos with GroundedSam."""
    gsam = GroundedSam(config)
    
    # [Pseudocode]
    # For each video:
    #   Process with full pipeline
    #   Run segmentation QC
    #   Generate analysis video
    pass
```

### Update `05_export_embryo_masks.py`

```python
# Key changes:
# 1. Export with simplified names
# 2. Transfer QC flags via GSAM ID
# 3. Optional visualization export

from utils.metadata.embryo import EmbryoMetadata
from utils.visualization import PipelineVisualizer

def export_masks(sam_path, embryo_path, output_dir):
    """Export masks and update metadata."""
    # [Pseudocode]
    # 1. Load annotations
    # 2. Export masks with simple names
    # 3. Transfer QC flags
    # 4. Generate summary visualization
    pass
```

## Migration Guide

For users with existing data:

1. **Metadata Migration**: 
   - Old format auto-detected and migrated
   - QC data integrated automatically

2. **ID Format**:
   - Complex experiment IDs now supported
   - Parsing handles legacy and new formats

3. **File Organization**:
   - Videos stay in same location
   - New simplified image naming available

## Testing Checklist

- [ ] Test with legacy data
- [ ] Test with new complex experiment IDs
- [ ] Test recursive experiment finding
- [ ] Test QC integration
- [ ] Test visualization generation
- [ ] Test end-to-end pipeline
- [ ] Verify backward compatibility

## Notes

- Scripts maintain same CLI interface
- New features are opt-in
- Performance improvements from batching
- Better error handling throughout