# Base Annotation Parser Class

## Overview
A foundational class providing common ID parsing, file operations, and utility functions used across all annotation classes (EmbryoMetadata, GroundedDinoAnnotations, GroundedSamAnnotations, ExperimentQC).

## Implementation

```python
# base_annotation_parser.py

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import shutil

class BaseAnnotationParser:
    """
    Base class for all annotation-related classes in the MorphSeq pipeline.
    
    Provides common functionality for:
    - ID parsing and validation
    - File I/O operations
    - Change tracking
    - Common utilities
    """
    
    # ID format patterns shared across all classes
    ID_PATTERNS = {
        "experiment": re.compile(r'^(\d{8})$'),
        "video": re.compile(r'^(\d{8})_([A-H]\d{2})$'),
        "image": re.compile(r'^(\d{8})_([A-H]\d{2})_(\d{4})$'),
        "embryo": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})$'),
        "snip": re.compile(r'^(\d{8})_([A-H]\d{2})_e(\d{2})_(\d{4})$')
    }
    
    def __init__(self, filepath: Union[str, Path], verbose: bool = True):
        """Initialize base parser with common attributes."""
        self.filepath = Path(filepath)
        self.verbose = verbose
        self._unsaved_changes = False
        self._change_log = []
    
    # -------------------------------------------------------------------------
    # File I/O Operations
    # -------------------------------------------------------------------------
    
    def load_json(self, file_path: Path = None) -> Dict:
        """Load JSON file with error handling."""
        if file_path is None:
            file_path = self.filepath
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def save_json(self, data: Dict, file_path: Path = None, 
                  create_backup: bool = True) -> None:
        """Save JSON with atomic write and optional backup."""
        if file_path is None:
            file_path = self.filepath
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested
        if create_backup and file_path.exists():
            self._create_backup(file_path)
        
        # Atomic write
        temp_path = file_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(file_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create timestamped backup of file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup.{timestamp}{file_path.suffix}')
        shutil.copy2(file_path, backup_path)
        if self.verbose:
            print(f"📦 Created backup: {backup_path.name}")
        return backup_path
    
    # -------------------------------------------------------------------------
    # ID Parsing and Validation
    # -------------------------------------------------------------------------
    
    def parse_id(self, entity_id: str) -> Dict[str, str]:
        """Parse any ID type and extract components."""
        for id_type, pattern in self.ID_PATTERNS.items():
            match = pattern.match(entity_id)
            if match:
                return self._extract_id_components(id_type, match.groups(), entity_id)
        
        return {"type": "unknown", "id": entity_id}
    
    def _extract_id_components(self, id_type: str, groups: Tuple, 
                              entity_id: str) -> Dict[str, str]:
        """Extract components based on ID type."""
        if id_type == "experiment":
            return {
                "type": "experiment",
                "experiment_id": groups[0]
            }
        
        elif id_type == "video":
            return {
                "type": "video",
                "experiment_id": groups[0],
                "well_id": groups[1],
                "video_id": entity_id
            }
        
        elif id_type == "image":
            return {
                "type": "image",
                "experiment_id": groups[0],
                "well_id": groups[1],
                "frame": groups[2],
                "video_id": f"{groups[0]}_{groups[1]}",
                "image_id": entity_id
            }
        
        elif id_type == "embryo":
            return {
                "type": "embryo",
                "experiment_id": groups[0],
                "well_id": groups[1],
                "embryo_num": groups[2],
                "video_id": f"{groups[0]}_{groups[1]}",
                "embryo_id": entity_id
            }
        
        elif id_type == "snip":
            return {
                "type": "snip",
                "experiment_id": groups[0],
                "well_id": groups[1],
                "embryo_num": groups[2],
                "frame": groups[3],
                "video_id": f"{groups[0]}_{groups[1]}",
                "embryo_id": f"{groups[0]}_{groups[1]}_e{groups[2]}",
                "image_id": f"{groups[0]}_{groups[1]}_{groups[3]}",
                "snip_id": entity_id
            }
    
    def get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
        """Extract embryo ID from snip ID."""
        parsed = self.parse_id(snip_id)
        if parsed["type"] == "snip":
            return parsed["embryo_id"]
        return None
    
    def extract_frame_number(self, id_str: str) -> int:
        """Extract frame number from image or snip ID."""
        parsed = self.parse_id(id_str)
        if parsed["type"] in ["image", "snip"] and "frame" in parsed:
            return int(parsed["frame"])
        return -1
    
    def validate_id_format(self, entity_id: str, expected_type: str) -> bool:
        """Validate ID matches expected type."""
        parsed = self.parse_id(entity_id)
        return parsed["type"] == expected_type
    
    # -------------------------------------------------------------------------
    # Change Tracking
    # -------------------------------------------------------------------------
    
    def _add_change_log(self, operation: str, details: Dict) -> None:
        """Add entry to change log."""
        self._change_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details
        })
        self._unsaved_changes = True
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent changes."""
        return self._change_log[-limit:]
    
    def clear_change_log(self) -> None:
        """Clear change log after save."""
        self._change_log.clear()
    
    # -------------------------------------------------------------------------
    # Common Utilities
    # -------------------------------------------------------------------------
    
    def get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now().isoformat()
    
    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def mark_saved(self) -> None:
        """Mark as saved and clear change tracking."""
        self._unsaved_changes = False
        self.clear_change_log()
```

## Usage in Derived Classes

```python
# Example: EmbryoMetadata inheriting from BaseAnnotationParser

class EmbryoMetadata(BaseAnnotationParser):
    def __init__(self, sam_annotation_path: Path, embryo_metadata_path: Path, **kwargs):
        # Initialize base class
        super().__init__(embryo_metadata_path, **kwargs)
        
        # EmbryoMetadata specific initialization
        self.sam_annotation_path = sam_annotation_path
        self.data = self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load existing or create new metadata."""
        if self.filepath.exists():
            # Use base class load_json
            return self.load_json()
        else:
            return self._create_empty_metadata()
    
    def save(self):
        """Save metadata using base class method."""
        self.data["file_info"]["last_updated"] = self.get_timestamp()
        self.save_json(self.data)
        self.mark_saved()

# Example: GroundedDinoAnnotations inheriting from BaseAnnotationParser

class GroundedDinoAnnotations(BaseAnnotationParser):
    def __init__(self, filepath: Path, **kwargs):
        super().__init__(filepath, **kwargs)
        self.annotations = self._load_or_initialize()
    
    def add_annotation(self, image_id: str, ...):
        # Validate image ID format
        if not self.validate_id_format(image_id, "image"):
            raise ValueError(f"Invalid image ID format: {image_id}")
        
        # Add annotation...
        self._add_change_log("add_annotation", {"image_id": image_id})
```

## Benefits

1. **DRY Principle**: No duplicate ID parsing code across classes
2. **Consistent File I/O**: All classes use same atomic save pattern
3. **Unified Change Tracking**: Standard change log across pipeline
4. **Shared Validation**: ID format validation available to all
5. **Common Utilities**: Timestamp, backup creation, etc.

## Integration Points

All annotation classes in the pipeline should inherit from this base class:

- `EmbryoMetadata(BaseAnnotationParser)`
- `GroundedDinoAnnotations(BaseAnnotationParser)`
- `GroundedSamAnnotations(BaseAnnotationParser)`
- `ExperimentDataQC(BaseAnnotationParser)`
- Future annotation classes

This ensures consistent behavior and reduces code duplication across the pipeline.