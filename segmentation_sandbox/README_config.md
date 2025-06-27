# MorphSeq Configuration with Base Directory

## Overview

The pipeline configuration now uses a base `morphseq_data_dir` variable to make path management easier and more flexible. This allows you to easily change the location of your morphseq data by updating a single variable.

## Key Features

### 1. Base Directory Management
- **`morphseq_data_dir`**: Base directory for all external morphseq data
- All external data paths (stitched images, masks) are relative to this base directory
- Easy to relocate data by changing one variable

### 2. Path Types
- **morphseq_data_dir-relative**: External data paths (stitched images, masks)
- **sandbox-relative**: Internal paths (models, outputs, logs) relative to `segmentation_sandbox/`

### 3. Configuration Structure
```yaml
paths:
  # Base directory - change this to relocate your data
  morphseq_data_dir: "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
  
  # External data (relative to morphseq_data_dir)
  stitched_images_dir: "built_image_data/stitched_FF_images"
  embryo_mask_root: "mask_data/embryo_mask_data"
  
  # Internal data (relative to sandbox)
  intermediate_dir: "data/intermediate"
  models_dir: "models"
```

## Usage Examples

### Basic Configuration Access
```python
from utils.config_utils import load_config

config = load_config()

# Get detection parameters
threshold = config.get('detection.box_threshold')

# Get full paths
stitched_dir = config.get_stitched_images_dir()
embryo_masks = config.get_mask_dir('embryo')
```

### Path Management
```python
# Base morphseq directory
base_dir = config.get_morphseq_data_path()

# Custom subpaths within morphseq data
experiment_dir = config.get_morphseq_data_path('experiments/batch_1')

# Output file paths
detection_file = config.get_intermediate_path('detections.json')
final_results = config.get_final_path('trajectories.csv')
```

### Mask Directory Access
```python
# Get different mask types
embryo_masks = config.get_mask_dir('embryo')
yolk_masks = config.get_mask_dir('yolk')
via_masks = config.get_mask_dir('via')
```

## Benefits

1. **Easy Relocation**: Change `morphseq_data_dir` to move your data
2. **Clear Separation**: External vs internal path management
3. **Convenience Methods**: Helper functions for common path operations
4. **Validation**: Automatic path resolution and validation
5. **Flexibility**: Support for both relative and absolute paths

## Files Modified

- `configs/pipeline_config.yaml`: Added `morphseq_data_dir` and updated paths
- `utils/config_utils.py`: Enhanced path resolution and convenience methods
- `example_config_usage.py`: Demonstration script showing usage patterns

## Migration

To use this with existing scripts:
1. Replace hardcoded paths with `config.get_*()` methods
2. Use `config.get_morphseq_data_path()` for custom morphseq subdirectories
3. Use `config.get_intermediate_path()` and `config.get_final_path()` for outputs
