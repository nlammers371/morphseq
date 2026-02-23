# GroundedDINO → SAM2 Pipeline Refactoring Guide

## Overview

Refactor detection and segmentation pipeline to use our core utilities while maintaining clean separation between technical and biological annotation workflows.

## Current Files

- `grounded_dino_utils.py` (~800 lines) - Detection and annotation management
- `sam2_utils.py` (~1000 lines) - Video segmentation and tracking
- `gsam_quality_control.py` (~800 lines) - Technical QC (minimal changes needed)

## Integration Goals

1. Use `parsing_utils` for consistent ID handling
2. Use `EntityIDTracker` for hierarchy validation
3. Maintain compatibility with `EmbryoMetadata` for downstream integration
4. Keep detection/segmentation separate from biological annotations

## File Structure

Keep current modular organization within Module 2:
- `grounded_dino_utils.py` - Detection and annotation management  
- `sam2_utils.py` - Video segmentation and tracking
- `gsam_quality_control.py` - Technical QC (integrated with SAM2)

## Refactoring Plan

### Step 1: GroundedDinoAnnotations Refactoring

**Replace custom ID parsing:**

```python
# Before: Custom logic
parts = image_id.split('_')
experiment_id = '_'.join(parts[:-1])

# After: Use parsing_utils
from parsing_utils import extract_experiment_id, get_entity_type

experiment_id = extract_experiment_id(image_id)
entity_type = get_entity_type(image_id)
```

**Add entity validation:**

```python
from entity_id_tracker import EntityIDTracker

class GroundedDinoAnnotations:
    def save(self):
        # Validate entity hierarchy before saving
        entities = EntityIDTracker.extract_entities(self.annotations)
        EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
        
        # Compare with experiment metadata if available
        if hasattr(self, 'embryo_metadata') and self.embryo_metadata:
            em_entities = EntityIDTracker.extract_entities(self.embryo_metadata.data)
            missing_from_em = EntityIDTracker.compare_entities(entities, em_entities)
            
            if any(missing_from_em.values()):
                print(f"⚠️ Warning: Detection results include entities not in experiment metadata:")
                for entity_type, missing_ids in missing_from_em.items():
                    if missing_ids:
                        print(f"  {entity_type}: {len(missing_ids)} missing")
        
        # Update entity tracking
        self.annotations["entity_tracking"] = {
            entity_type: list(ids) for entity_type, ids in entities.items()
        }
        
        # Save with base class method
        super().save()
```

**Integrate with EmbryoMetadata:**

```python
def get_target_images_from_metadata(self, embryo_metadata_path: str):
    """Get image list from EmbryoMetadata instead of custom logic."""
    from embryo_metadata import EmbryoMetadata
    
    em = EmbryoMetadata(embryo_metadata_path)
    return em.get_images_for_detection(experiment_ids=self.target_experiments)
```

### Step 2: GroundedSamAnnotations Refactoring

**Replace experiment metadata dependency:**

```python
# Before: Custom metadata loading
self.experiment_metadata = load_experiment_metadata(metadata_path)

# After: Use EmbryoMetadata
from embryo_metadata import EmbryoMetadata

class GroundedSamAnnotations:
    def __init__(self, embryo_metadata_path, ...):
        self.embryo_metadata = EmbryoMetadata(embryo_metadata_path)
        
    def get_video_info(self, video_id):
        return self.embryo_metadata.get_video_data(video_id)
    
    def get_images_for_video(self, video_id):
        return self.embryo_metadata.get_video_images(video_id)
```

**Use parsing_utils for ID operations:**

```python
from parsing_utils import extract_video_id, extract_frame_number

def create_snip_id(self, embryo_id: str, image_id: str) -> str:
    """Create standardized snip_id using parsing_utils format."""
    frame_number = extract_frame_number(image_id)
    return f"{embryo_id}_s{frame_number:04d}"

def assign_embryo_ids(self, video_id: str, num_embryos: int) -> List[str]:
    """Generate embryo IDs using consistent format."""
    return [f"{video_id}_e{i+1:02d}" for i in range(num_embryos)]
```

**Add entity validation:**

```python
def save(self):
    # Validate hierarchy against experiment metadata
    entities = EntityIDTracker.extract_entities(self.results)
    EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
    
    # Compare with experiment metadata for consistency
    em_entities = EntityIDTracker.extract_entities(self.embryo_metadata.data)
    missing_from_em = EntityIDTracker.compare_entities(entities, em_entities)
    
    if any(missing_from_em.values()):
        print(f"⚠️ Warning: SAM2 created entities not in experiment metadata:")
        for entity_type, missing_ids in missing_from_em.items():
            if missing_ids:
                print(f"  {entity_type}: {len(missing_ids)} missing")
    
    # Update tracking
    self.results["entity_tracking"] = {
        entity_type: list(ids) for entity_type, ids in entities.items()
    }
    
    super().save()
```

### Step 3: Pipeline Integration

**Workflow integration:**

```python
def run_detection_segmentation_pipeline(embryo_metadata_path: str, 
                                       output_dir: Path):
    """Integrated GroundedDINO → SAM2 pipeline."""
    
    # Initialize with shared data source
    em = EmbryoMetadata(embryo_metadata_path)
    
    # Step 1: Detection
    gdino = GroundedDinoAnnotations(
        filepath=output_dir / "detections.json"
    )
    
    # Get images from EmbryoMetadata
    target_images = em.get_images_for_detection(experiment_ids=["20240411"])
    
    # Run detection
    gdino.process_missing_annotations(
        model=gdino_model,
        prompts="individual embryo",
        image_ids=[img["image_id"] for img in target_images]
    )
    
    # Step 2: Segmentation
    gsam = GroundedSamAnnotations(
        filepath=output_dir / "segmentations.json",
        seed_annotations_path=gdino.filepath,
        embryo_metadata_path=embryo_metadata_path
    )
    
    # Run segmentation
    gsam.process_missing_annotations()
    
    # Step 3: Quality Control (auto-run)
    qc = GSAMQualityControl(gsam.filepath)
    qc.run_all_checks(author="auto_qc", process_all=False)
    
    return gdino, gsam, qc
```

### Step 4: Output Format Standardization

**GroundedDINO JSON structure:**

```json
{
  "file_info": {"creation_time": "2024-12-15T14:30:22.123456"},
  "image_ids": ["20240411_A01_t0000", "20240411_A01_t0001"],
  "entity_tracking": {
    "experiments": ["20240411"],
    "videos": ["20240411_A01"], 
    "images": ["20240411_A01_t0000", "20240411_A01_t0001"],
    "embryos": [],
    "snips": []
  },
  "images": {
    "20240411_A01_t0000": {
      "annotations": [
        {
          "prompt": "individual embryo",
          "detections": [
            {"box_xyxy": [0.3, 0.1, 0.7, 0.5], "confidence": 0.85}
          ]
        }
      ]
    }
  },
  "high_quality_annotations": {
    "20240411": {
      "prompt": "individual embryo",
      "confidence_threshold": 0.5,
      "iou_threshold": 0.5,
      "filtered": {
        "20240411_A01_t0000": [
          {"box_xyxy": [0.3, 0.1, 0.7, 0.5], "confidence": 0.85}
        ]
      }
    }
  }
}
```

**SAM2 JSON structure:**

```json
{
  "gsam_annotation_id": "1234",
  "snip_ids": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001"],
  "entity_tracking": {
    "experiments": ["20240411"],
    "videos": ["20240411_A01"],
    "images": ["20240411_A01_t0000", "20240411_A01_t0001"], 
    "embryos": ["20240411_A01_e01"],
    "snips": ["20240411_A01_e01_s0000", "20240411_A01_e01_s0001"]
  },
  "experiments": {
    "20240411": {
      "videos": {
        "20240411_A01": {
          "images": {
            "20240411_A01_t0000": {
              "embryos": {
                "20240411_A01_e01": {
                  "snip_id": "20240411_A01_e01_s0000",
                  "segmentation": {...},
                  "bbox": [0.3, 0.1, 0.7, 0.5],
                  "area": 1234.5
                }
              }
            }
          }
        }
      }
    }
  },
  "flags": {
    "qc_meta": {
      "processed_snip_ids": ["20240411_A01_e01_s0000"],
      "last_updated": "2024-12-15T16:00:00"
    },
    "by_snip": {
      "20240411_A01_e01_s0000": {
        "HIGH_SEGMENTATION_VAR_SNIP": [
          {
            "coefficient_of_variation": 0.25,
            "current_area": 1234.5,
            "author": "auto_qc"
          }
        ]
      }
    },
    "by_image": {
      "20240411_A01_t0000": {
        "DETECTION_FAILURE": [
          {
            "missing_embryo_ids": ["20240411_A01_e02"],
            "author": "auto_qc"
          }
        ]
      }
    },
    "flag_overview": {
      "HIGH_SEGMENTATION_VAR_SNIP": {
        "count": 1,
        "snip_ids": ["20240411_A01_e01_s0000"]
      },
      "DETECTION_FAILURE": {
        "count": 1,
        "image_ids": ["20240411_A01_t0000"]
      }
    }
  }
}
```

### Step 5: Downstream Integration Point

**Connect to EmbryoMetadata:**

```python
class EmbryoMetadata:
    def import_segmentation_results(self, gsam_path: str):
        """Import segmentation data and QC flags."""
        # Import snip-level segmentation metadata
        # Import technical QC flags
        
    def import_technical_flags(self, gsam_path: str):
        """Import QC flags from GSAM results."""
        with open(gsam_path) as f:
            gsam_data = json.load(f)
        
        qc_flags = gsam_data.get("flags", {}).get("by_snip", {})
        
        for snip_id, flags in qc_flags.items():
            embryo_id = extract_embryo_id(snip_id)
            for flag_type, flag_instances in flags.items():
                self.add_flag(snip_id, flag_type, "snip", 
                            description=f"Technical QC: {flag_type}")
```

## File Changes Summary

### grounded_dino_utils.py
- Replace custom ID parsing with `parsing_utils`
- Add `EntityIDTracker` validation
- Remove duplicate metadata logic
- Use `EmbryoMetadata.get_images_for_detection()`

### sam2_utils.py  
- Replace experiment metadata with `EmbryoMetadata`
- Use `parsing_utils` for all ID operations
- Standardize snip_id format (`_s` prefix)
- Add entity hierarchy validation

### gsam_quality_control.py
- Add `parsing_utils` imports
- Add `EntityIDTracker` validation
- Keep existing QC logic (minimal changes)

## Benefits

1. **Consistent ID handling** across all pipeline components
2. **Entity validation** prevents orphaned data
3. **Clean integration** with biological annotation pipeline
4. **Separation of concerns** maintained between technical and biological workflows
5. **Reduced code duplication** through shared utilities

## Code Organization Guidelines

### Function Organization
- Group related functions together (detection, validation, I/O)
- Place utility functions at bottom of file
- Keep complex functions under 50 lines
- Split large classes into focused methods

### Documentation Standards
```python
def process_single_video_from_annotations(video_id, video_annotations, ...):
    """Process one video with SAM2 segmentation."""
    
    # Extract video metadata from experiment data
    video_info = get_video_metadata(video_id)
    
    # Validate video directory exists
    video_dir = Path(video_info["processed_jpg_images_dir"])
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Find optimal seed frame (highest detection confidence)
    seed_frame_id, seed_info = find_seed_frame(video_annotations)
    
    # Run SAM2 propagation (bidirectional if seed not first frame)
    if seed_info["is_first_frame"]:
        results = run_forward_propagation(...)
    else:
        results = run_bidirectional_propagation(...)
    
    return results, metadata
```

### SAM2 Temporary Directories
**Only needed for bidirectional propagation:**
- Forward propagation uses existing sequential files on disk (0000.jpg format)
- **Bidirectional only:** Creates temp directory for reversed frame order during backward phase

```python
# Forward propagation: Use existing files directly
if seed_frame_idx == 0:
    results = run_sam2_propagation(predictor, video_dir, ...)

# Bidirectional: Needs temp dir for backward phase only  
if seed_frame_idx > 0:
    with tempfile.TemporaryDirectory() as backward_temp_dir:
        # Create reversed frame order for backward propagation
        results = run_bidirectional_propagation(...)
```