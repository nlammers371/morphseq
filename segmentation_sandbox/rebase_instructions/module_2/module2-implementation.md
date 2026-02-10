# Module 2: Detection & Segmentation Implementation Guide

## Overview
Refactor GroundedDINO and SAM2 utilities to use shared parsing and validation while keeping detection/segmentation logic intact.

## File Structure
```
segmentation_sandbox/
└── detection_segmentation/
    ├── grounded_dino_utils.py
    ├── sam2_utils.py
    ├── gsam_quality_control.py
    └── mask_exporter.py
```

## Task 1: Refactor `grounded_dino_utils.py`

**Source**: Start with `rebase_instructions/module_2/old_gdino_utils.py`

### Key refactoring points

1. **Add imports at top**:
```python
# Add to existing imports
from utils.parsing_utils import extract_experiment_id, get_entity_type
from utils.entity_id_tracker import EntityIDTracker
from metadata.experiment_metadata import ExperimentMetadata
```

2. **Replace custom metadata handling** (around line 140):
```python
# OLD: Custom metadata loading
def set_metadata_path(self, metadata_path: Union[str, Path]):
    self.metadata_path = Path(metadata_path)
    if self.metadata_path.exists():
        self._metadata = load_experiment_metadata(self.metadata_path)
        
# NEW: Use ExperimentMetadata class
def set_metadata_path(self, metadata_path: Union[str, Path]):
    self.metadata_path = Path(metadata_path)
    if self.metadata_path.exists():
        self.exp_metadata = ExperimentMetadata(self.metadata_path)
        self._metadata = self.exp_metadata.metadata  # For compatibility
```

3. **Update save method** (around line 120):
```python
def save(self):
    """Save annotations with entity validation."""
    # Add embedded entity tracker using simplified static method
    try:
        # EntityIDTracker serves as a PURE CONTAINER - it validates and tracks entities
        # but doesn't handle file I/O. We use static methods for format-specific operations.
        self.annotations = EntityIDTracker.update_entity_tracker(
            self.annotations, 
            pipeline_step="module_2_detection"
        )
        
        if self.verbose:
            entities = EntityIDTracker.extract_entities(self.annotations)
            entity_counts = EntityIDTracker.get_counts(entities)
            print(f"📋 Entity tracker updated: {entity_counts}")
            
    except Exception as e:
        if self.verbose:
            print(f"⚠️ Entity tracking update failed: {e}")
    
    # Original save logic continues...
    self.filepath.parent.mkdir(parents=True, exist_ok=True)
    self.annotations["file_info"]["last_updated"] = datetime.now().isoformat()
    # ... rest of save
```

4. **Use ExperimentMetadata for image lists** (in get_missing_annotations):
```python
def _get_filtered_image_ids(self, experiment_ids=None, video_ids=None, image_ids=None):
    if not self.exp_metadata:
        return []
    
    if image_ids:
        return image_ids
    
    # Use ExperimentMetadata method
    target_images = self.exp_metadata.get_images_for_detection(
        experiment_ids=experiment_ids,
        video_ids=video_ids
    )
    
    return [img["image_id"] for img in target_images]
```

### Test checklist
```python
# Test detection with metadata integration
gdino = GroundedDinoAnnotations("test_detections.json")
gdino.set_metadata_path("experiment_metadata.json")

# Should use ExperimentMetadata
missing = gdino.get_missing_annotations(["individual embryo"], experiment_ids=["20240411"])
print(f"Missing annotations: {len(missing['individual embryo'])}")

# Run on one image and save
# Verify entity_tracking added to JSON
```

## Task 2: Refactor `sam2_utils.py`

**Source**: Start with `rebase_instructions/module_2/old_sam2_utils.py`

### Major refactoring

1. **Update imports**:
```python
# Add to imports
from utils.parsing_utils import (
    extract_frame_number, 
    extract_experiment_id,
    extract_embryo_id,
    get_entity_type
)
from utils.entity_id_tracker import EntityIDTracker
from metadata.experiment_metadata import ExperimentMetadata
```

2. **Replace experiment metadata loading** (in __init__, around line 150):
```python
# OLD
self.experiment_metadata = self._load_experiment_metadata()

# NEW
self.exp_metadata = None
if self.experiment_metadata_path and self.experiment_metadata_path.exists():
    self.exp_metadata = ExperimentMetadata(self.experiment_metadata_path)
    self.experiment_metadata = self.exp_metadata.metadata  # Compatibility
```

3. **Fix create_snip_id function** (around line 1450):
```python
def create_snip_id(embryo_id: str, image_id: str) -> str:
    """Create snip_id using standard format."""
    frame_number = extract_frame_number(image_id)
    return f"{embryo_id}_s{frame_number:04d}"
```

4. **Update process_single_video** to use ExperimentMetadata:
```python
# In process_single_video_from_annotations (around line 1200)
# OLD: Custom metadata lookup
video_info = None
if grounded_sam_instance.experiment_metadata:
    for exp_data in grounded_sam_instance.experiment_metadata.get("experiments", {}).values():
        
# NEW: Use ExperimentMetadata
video_info = grounded_sam_instance.exp_metadata.get_video_info(video_id)
if not video_info:
    raise ValueError(f"Video {video_id} not found in experiment metadata")
```

5. **Add entity validation to save** (around line 280):
```python
def save(self):
    """Save with entity validation using embedded tracker approach."""
    # EntityIDTracker is a PURE CONTAINER for entity validation - use static methods for file operations
    try:
        # Update embedded entity tracker (no separate files needed)
        self.results = EntityIDTracker.update_entity_tracker(
            self.results,
            pipeline_step="module_2_segmentation" 
        )
        
        # Validate entity hierarchy
        entities = EntityIDTracker.extract_entities(self.results)
        validation_result = EntityIDTracker.validate_hierarchy(entities, raise_on_violations=False)
        
        if not validation_result.get('valid', True):
            if self.verbose:
                print(f"⚠️ Entity validation warnings: {validation_result.get('violations', [])}")
        
        # Verify snip IDs format
        for snip_id in entities.get("snips", []):
            if not snip_id.count("_s") == 1:
                print(f"⚠️ Non-standard snip_id format: {snip_id}")
                
        if self.verbose:
            entity_counts = EntityIDTracker.get_counts(entities)
            print(f"📋 Entity tracker updated: {entity_counts}")
                
    except Exception as e:
        if self.verbose:
            print(f"⚠️ Entity validation warning: {e}")
    
    # Original save logic
    self.filepath.parent.mkdir(parents=True, exist_ok=True)
    self.results["last_updated"] = datetime.now().isoformat()
    # ... rest
```

### Test checklist
```python
# Test SAM2 with refactored code
gsam = GroundedSamAnnotations(
    "test_segmentations.json",
    seed_annotations_path="test_detections.json",
    experiment_metadata_path="experiment_metadata.json"
)

# Process one video
gsam.process_video("20240411_A01")

# Verify snip_id format
with open("test_segmentations.json") as f:
    data = json.load(f)
    
# Check a snip_id
snip_ids = data.get("snip_ids", [])
assert all("_s" in sid for sid in snip_ids)
print(f"Sample snip_id: {snip_ids[0]}")  # Should be like: 20240411_A01_e01_s0000
```

## Task 3: Update `gsam_quality_control.py`

**Source**: Start with `rebase_instructions/module_2/old_gsam_qc_class.py`

### Minimal changes needed

1. **Add imports**:
```python
from utils.parsing_utils import extract_frame_number, get_entity_type
```

2. **Replace custom frame parsing** (in check_segmentation_variability, around line 380):
```python
# OLD: Custom regex parsing
frame_num = int(snip_id.split("_s")[1])

# NEW: Use parsing utils
frame_num = extract_frame_number(snip_id)
```

3. **Ensure JSON serialization** (already handled in ensure_json_serializable)

### Test
```python
# Run QC on test SAM2 output
qc = GSAMQualityControl("test_segmentations.json")
qc.run_all_checks(author="test_qc", process_all=False)
qc.print_summary()
```

## Task 4: Copy `mask_exporter.py`

**Source**: Copy from `rebase_instructions/module_2/simple_mask_exporter_final.py`

**Add import**:
```python
from utils.entity_id_tracker import EntityIDTracker
from utils.parsing_utils import extract_experiment_id
```

No other changes needed.

## Integration Test

```python
# Full detection → segmentation pipeline
from detection_segmentation import (
    GroundedDinoAnnotations, 
    GroundedSamAnnotations,
    GSAMQualityControl
)

# 1. Detection
gdino = GroundedDinoAnnotations("outputs/detections.json")
gdino.set_metadata_path("experiment_metadata.json")

# Run detection (with model loaded)
gdino.process_missing_annotations(
    model=model,
    prompts="individual embryo",
    experiment_ids=["20240411"]
)

# 2. Segmentation  
gsam = GroundedSamAnnotations(
    "outputs/segmentations.json",
    seed_annotations_path="outputs/detections.json",
    experiment_metadata_path="experiment_metadata.json"
)

# Process videos
gsam.process_missing_annotations(max_videos=1)

# 3. QC
qc = GSAMQualityControl("outputs/segmentations.json")
qc.run_all_checks(author="auto_qc")

print("✓ Detection → Segmentation → QC complete")
```

## Accomplishment Checklist

- [ ] GroundedDINO uses ExperimentMetadata for image lists
- [ ] GroundedDINO adds entity_tracking on save
- [ ] SAM2 uses parsing_utils for ID operations
- [ ] SAM2 creates snip_ids with '_s' prefix format
- [ ] SAM2 validates entity hierarchy on save
- [ ] QC uses parsing utils instead of regex
- [ ] Full pipeline test completes successfully