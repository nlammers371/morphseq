# 05_sam2_qc_analysis.py Porting Plan

## Overview
Port the working `old_gsam_qc_class.py` into a new pipeline script `05_sam2_qc_analysis.py` that follows the established pipeline patterns while keeping the proven QC logic intact.

## Goals
1. **Minimal changes** to the working QC logic
2. **Pipeline consistency** with other 0X scripts
3. **Modular architecture** for easy maintenance
4. **Proper error handling** and progress reporting

---

## Module Structure

### 1. Core Imports and Setup
```python
#!/usr/bin/env python3
"""
Pipeline Script 5: SAM2 Quality Control Analysis

Analyze SAM2 segmentation results for quality issues and add flags
to the GSAM JSON structure.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
from time import time

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from pycocotools import mask as mask_utils
    from scipy import ndimage
    from skimage.measure import regionprops, label
    _HAS_IMAGE_LIBS = True
except ImportError:
    _HAS_IMAGE_LIBS = False

# Pipeline imports
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.base_file_handler import BaseFileHandler
```

### 2. Utility Functions (Direct Port)
**Functions to port AS-IS:**
- `ensure_json_serializable(obj)` ✅ (No changes needed)

### 3. Core QC Class (Minimal Modifications)

#### 3.1 Class Declaration and Init
```python
class GSAMQualityControl(BaseFileHandler):
    """
    Quality Control for SAM2 annotations. Inherits from BaseFileHandler
    for consistent file operations with the rest of the pipeline.
    """
    
    def __init__(self, gsam_path: str, verbose: bool = True, progress: bool = True):
```

**Porting Strategy:**
- ✅ **Keep**: All logic for loading GSAM data
- ✅ **Keep**: Flag structure initialization 
- ✅ **Keep**: Entity tracking initialization
- 🔄 **Modify**: Inherit from `BaseFileHandler` instead of direct file I/O
- 🔄 **Modify**: Use `self.load_json()` instead of manual JSON loading

#### 3.2 Entity Tracking Methods (Direct Port)
**Methods to port AS-IS:**
- `_initialize_entity_tracking()` ✅
- `diagnose_data_structure()` ✅
- `_mark_entities_checked()` ✅
- `get_new_entities_to_process()` ✅
- `get_all_entities_to_process()` ✅

#### 3.3 Entity Filtering Logic (Direct Port)
**Methods to port AS-IS:**
- `_should_process_experiment()` ✅
- `_should_process_video()` ✅ 
- `_should_process_image()` ✅
- `_should_process_snip()` ✅

#### 3.4 Core QC Infrastructure (Direct Port)
**Methods to port AS-IS:**
- `_add_flag()` ✅
- `_progress_iter()` ✅

#### 3.5 Quality Check Methods (Direct Port)
**All QC check methods to port AS-IS:**
- `check_segmentation_variability()` ✅
- `check_mask_on_edge()` ✅
- `check_detection_failure()` ✅
- `check_overlapping_masks()` ✅
- `check_large_masks()` ✅
- `check_small_masks()` ✅ *(Note: Add this to current implementation)*
- `check_discontinuous_masks()` ✅

**Strategy:** Copy these methods exactly - they contain the core business logic that works.
- and check they are copied exactly
#### 3.6 Main Orchestration Method (Minor Modifications)
```python
def run_all_checks(self,
                  author: str = "auto_qc",
                  process_all: bool = False,
                  target_entities: Optional[Dict[str, List[str]]] = None,
                  force_reprocess: bool = False,
                  save_in_place: bool = True):
```

**Porting Strategy:**
- ✅ **Keep**: All QC orchestration logic
- ✅ **Keep**: Progress reporting and timing
- 🔄 **Modify**: Use `self.save_json()` from BaseFileHandler for saving

#### 3.7 Summary and Reporting Methods (Direct Port)
**Methods to port AS-IS:**
- `_save_qc_summary()` ✅
- `_count_flags_from_new_entities()` ✅
- `_count_flags_in_hierarchy()` ✅
- `generate_overview()` ✅
- `get_flags_summary()` ✅
- `print_summary()` ✅
- `get_flags_for_entity()` ✅
- `get_flags_by_type()` ✅

### 4. Pipeline CLI Interface (New Implementation)

#### 4.1 Argument Parser
```python
def main():
    parser = argparse.ArgumentParser(
        description="Run quality control analysis on SAM2 segmentation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run QC on new entities only (incremental)
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json
  
  # Force reprocess all entities
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json --process-all
  
  # QC specific experiments only
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json \\
    --experiments "20240506,20250703_chem3_28C_T00_1325"
  
  # Custom author and output path
  python 05_sam2_qc_analysis.py --input grounded_sam_annotations.json \\
    --output qc_results.json --author "researcher_name"
        """
    )
```

#### 4.2 Required Arguments
- `--input`: Path to grounded_sam_annotations.json file
- `--output`: Output path (optional, defaults to modifying input file)

#### 4.3 Optional Arguments
- `--author`: QC run author identifier (default: "pipeline_qc")
- `--process-all`: Process all entities instead of just new ones
- `--experiments`: Comma-separated experiment IDs to target
- `--videos`: Comma-separated video IDs to target
- `--images`: Comma-separated image IDs to target  
- `--snips`: Comma-separated snip IDs to target
- `--verbose`: Enable verbose output
- `--no-progress`: Disable progress bars
- `--dry-run`: Run analysis without saving results

#### 4.4 Main Execution Logic
```python
# Parse arguments
args = parser.parse_args()

# Validate inputs
input_path = Path(args.input)
if not input_path.exists():
    print(f"❌ Input file not found: {input_path}")
    sys.exit(1)

# Initialize QC
qc = GSAMQualityControl(
    gsam_path=str(input_path),
    verbose=args.verbose,
    progress=not args.no_progress
)

# Prepare target entities (if specified)
target_entities = None
if any([args.experiments, args.videos, args.images, args.snips]):
    target_entities = {
        "experiment_ids": args.experiments.split(",") if args.experiments else [],
        "video_ids": args.videos.split(",") if args.videos else [],
        "image_ids": args.images.split(",") if args.images else [],
        "snip_ids": args.snips.split(",") if args.snips else []
    }

# Run QC analysis
try:
    qc.run_all_checks(
        author=args.author,
        process_all=args.process_all,
        target_entities=target_entities,
        save_in_place=not args.dry_run and args.output is None
    )
    
    # Save to custom output if specified
    if args.output and not args.dry_run:
        qc.save_json(qc.gsam_data, filepath=args.output, create_backup=True)
    
    # Print summary
    qc.print_summary()
    
except Exception as e:
    print(f"❌ QC analysis failed: {e}")
    sys.exit(1)
```

---

## Testing Strategy

### 1. Create Enhanced Dummy Test File
**Update the existing `create_dummy_gsam.py` to:**
- ✅ Include all violation types from the old class
- ✅ Add `check_small_masks()` violations (currently missing)
- ✅ Ensure proper GSAM structure that matches expected format
- ✅ Add more realistic frame counts for segmentation variability testing

### 2. Test Cases to Validate
1. **Basic functionality**: Script runs without errors
2. **Incremental processing**: Only processes new entities on second run
3. **Targeted processing**: Can process specific experiments/videos/images/snips
4. **Flag detection**: All expected violations are caught
5. **Output consistency**: Results match old class output
6. **Pipeline integration**: Works with real GSAM files from step 4

---

## File Change Summary

### Files to Create
1. `scripts/pipelines/05_sam2_qc_analysis.py` - Main pipeline script
2. `scripts/tests/enhanced_dummy_gsam.py` - Updated test file generator

### Files to Modify
- None (keeping old class as reference)

### Dependencies
- Inherits from `BaseFileHandler` (already exists)
- Uses existing utility imports
- Optional dependencies handled gracefully

---

## Risk Mitigation

### 1. Backwards Compatibility
- Keep old class file as reference
- Test against existing QC outputs
- Maintain same flag structure

### 2. Error Handling
- Graceful degradation when optional libraries missing
- Clear error messages for invalid inputs
- Proper exception handling in main execution

### 3. Performance
- Keep existing progress reporting
- Maintain incremental processing capability
- Preserve memory-efficient iteration patterns

---

## Implementation Checklist

### Phase 1: Core Porting
- [ ] Create basic script structure with imports
- [ ] Port GSAMQualityControl class with BaseFileHandler inheritance
- [ ] Port all utility functions unchanged
- [ ] Port all QC check methods unchanged

### Phase 2: CLI Integration  
- [ ] Implement argument parser
- [ ] Add main execution logic
- [ ] Add error handling and validation

### Phase 3: Testing
- [ ] Update dummy test file generator
- [ ] Test against known violations
- [ ] Validate output consistency with old class
- [ ] Test pipeline integration

### Phase 4: Documentation
- [ ] Add proper docstrings
- [ ] Update help text and examples
- [ ] Create usage documentation

---

## Success Criteria

1. **Functional Parity**: All QC checks work identically to old class
2. **Pipeline Integration**: Follows established 0X script patterns
3. **Incremental Processing**: Efficiently handles new vs. processed entities
4. **Robust Testing**: Catches all expected violations in test file
5. **User Friendly**: Clear CLI interface and helpful error messages

The key principle: **Don't fix what isn't broken** - port the working QC logic with minimal changes while wrapping it in a proper pipeline interface.
