# Module 1: Experiment Metadata Implementation Guide

## Overview
Create ExperimentMetadata class using Module 0 utilities for consistent ID handling and validation.

## File Structure
```
segmentation_sandbox/
└── metadata/
    ├── experiment_metadata.py
    └── schema_manager.py
```

## Task 1: Create `metadata/experiment_metadata.py`

**Reference**: `rebase_instructions/module_1/module_1_reimplemtn_exeperimentmetadataclass.md`

### Required imports
```python
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import Module 0 utilities
from utils.parsing_utils import (
    parse_entity_id, 
    extract_frame_number, 
    get_entity_type,
    extract_experiment_id
)
from utils.entity_id_tracker import EntityIDTracker
from utils.base_file_handler import BaseFileHandler
```

### Class implementation

```python
class ExperimentMetadata(BaseFileHandler):
    """Manage experiment metadata with entity tracking and validation."""
    
    def __init__(self, metadata_path: Union[str, Path], verbose: bool = True):
        super().__init__(metadata_path, verbose)
        self.metadata = self._load_or_initialize()
        self._validate_and_update_tracking()
    
    def _load_or_initialize(self) -> Dict:
        """Load existing metadata or create new structure."""
        if self.filepath.exists():
            return self.load_json()
        else:
            return {
                "file_info": {
                    "creation_time": self.get_timestamp(),
                    "last_updated": self.get_timestamp()
                },
                "experiments": {},
                "entity_tracking": {
                    "experiments": [],
                    "videos": [],
                    "images": [],
                    "embryos": [],
                    "snips": []
                }
            }
    
    def _validate_and_update_tracking(self):
        """Validate entity hierarchy and update tracking."""
        # Extract all entities from metadata
        entities = EntityIDTracker.extract_entities(self.metadata)
        
        # Validate hierarchy (raises on violations)
        EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
        
        # Update entity_tracking section
        self.metadata["entity_tracking"] = {
            entity_type: sorted(list(ids)) 
            for entity_type, ids in entities.items()
        }
        
        if self.verbose:
            counts = EntityIDTracker.get_counts(entities)
            print(f"📊 Loaded metadata: {counts}")
    
    def save(self):
        """Save with validation and tracking update."""
        self._validate_and_update_tracking()
        self.metadata["file_info"]["last_updated"] = self.get_timestamp()
        self.save_json(self.metadata)
        
        if self.verbose:
            print(f"💾 Saved experiment metadata to: {self.filepath}")
    
    def get_images_for_detection(self, 
                                experiment_ids: Optional[List[str]] = None,
                                video_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Get image information for detection pipeline.
        
        Returns list of dicts with:
        - image_id: Full ID with 't' prefix (e.g., '20240411_A01_t0000')
        - image_path: Actual file path (e.g., '.../0000.jpg')
        - video_id, experiment_id, frame_number
        """
        images = []
        
        target_experiments = experiment_ids or self.metadata["entity_tracking"]["experiments"]
        
        for exp_id in target_experiments:
            if exp_id not in self.metadata["experiments"]:
                continue
                
            exp_data = self.metadata["experiments"][exp_id]
            
            for vid_id, vid_data in exp_data.get("videos", {}).items():
                if video_ids and vid_id not in video_ids:
                    continue
                
                images_dir = Path(vid_data["processed_jpg_images_dir"])
                
                for image_id in vid_data.get("image_ids", []):
                    # Extract frame number (removes 't' prefix)
                    frame_num = extract_frame_number(image_id)
                    
                    # Build actual file path (no 't' prefix on disk)
                    image_path = images_dir / f"{frame_num:04d}.jpg"
                    
                    images.append({
                        'image_id': image_id,  # Has 't' prefix
                        'image_path': str(image_path),  # No 't' prefix
                        'video_id': vid_id,
                        'experiment_id': exp_id,
                        'frame_number': frame_num
                    })
        
        return images
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Get video metadata."""
        # Extract experiment_id from video_id
        exp_id = extract_experiment_id(video_id)
        
        exp_data = self.metadata["experiments"].get(exp_id, {})
        return exp_data.get("videos", {}).get(video_id)
    
    def scan_organized_experiments(self, raw_data_dir: Path) -> Dict[str, int]:
        """
        Scan organized directory and compare with current metadata.
        
        Returns counts of new entities found.
        """
        if self.verbose:
            print(f"🔍 Scanning {raw_data_dir} for new content...")
        
        # Use DataOrganizer's scan method
        from data_organization.data_organizer import DataOrganizer
        organizer = DataOrganizer(verbose=False)
        scanned_data = organizer.scan_organized_experiments(raw_data_dir)
        
        # Extract entities from scanned data
        scanned_entities = EntityIDTracker.extract_entities(scanned_data)
        current_entities = EntityIDTracker.extract_entities(self.metadata)
        
        # Find new entities
        new_entities = EntityIDTracker.get_new_entities(
            scanned_entities, current_entities
        )
        
        # Merge new content
        new_counts = {}
        for entity_type, ids in new_entities.items():
            new_counts[entity_type] = len(ids)
        
        if any(new_counts.values()):
            # Merge experiments
            for exp_id, exp_data in scanned_data["experiments"].items():
                if exp_id not in self.metadata["experiments"]:
                    self.metadata["experiments"][exp_id] = exp_data
                else:
                    # Merge videos
                    for vid_id, vid_data in exp_data.get("videos", {}).items():
                        if vid_id not in self.metadata["experiments"][exp_id]["videos"]:
                            self.metadata["experiments"][exp_id]["videos"][vid_id] = vid_data
            
            self._validate_and_update_tracking()
            
            if self.verbose:
                print(f"✅ Found new content: {new_counts}")
        else:
            if self.verbose:
                print("✅ No new content found")
        
        return new_counts
    
    def get_entity_counts(self) -> Dict[str, int]:
        """Get counts of all entity types."""
        entities = EntityIDTracker.extract_entities(self.metadata)
        return EntityIDTracker.get_counts(entities)
```

### Test implementation
```python
# Test 1: Load and validate
em = ExperimentMetadata("test_metadata.json")
assert "entity_tracking" in em.metadata

# Test 2: Get images for detection
images = em.get_images_for_detection(experiment_ids=["20240411"])
if images:
    first_image = images[0]
    assert first_image["image_id"].endswith("_t0000")  # Has 't' prefix
    assert first_image["image_path"].endswith("0000.jpg")  # No 't' prefix

# Test 3: Entity validation
# This should work
em.save()

# Test 4: Directory scanning
new_counts = em.scan_organized_experiments(Path("raw_data_organized"))
print(f"New entities: {new_counts}")
```

## Task 2: Copy `metadata/schema_manager.py`

**Source**: Direct copy from `rebase_instructions/module_1/schema_manager.py`

No modifications needed - works as standalone utility.

### Quick validation test
```python
from metadata.schema_manager import SchemaManager

sm = SchemaManager()

# Test phenotype validation
assert sm.validate_phenotype("EDEMA") == True
assert sm.validate_phenotype("INVALID") == False

# Test adding custom phenotype
sm.add_phenotype("CUSTOM_DEFECT", "Custom description")
assert sm.validate_phenotype("CUSTOM_DEFECT") == True
```

## Accomplishment Checklist

- [ ] ExperimentMetadata inherits from BaseFileHandler
- [ ] Uses parsing_utils for all ID operations
- [ ] Entity validation runs on save
- [ ] get_images_for_detection returns correct format
- [ ] Image paths have no 't' prefix on disk
- [ ] Image IDs have 't' prefix in JSON
- [ ] Directory scanning detects new content
- [ ] Schema manager validates permitted values
- [ ] All tests pass