# Module 1: Core Foundation (Revised)

## Overview
Create the base classes and utilities that all other components will inherit from. This establishes consistent patterns for JSON handling, change tracking, ID parsing (building backwards), and common operations across the entire pipeline, obligatory backup creation when saving (important)

**Key Design Change**: ID parsing builds BACKWARDS from most specific to general, since experiment IDs can be complex and auto-detection is unreliable.

## Dependencies
- Python standard library only
- No external dependencies for core functionality

## Files to Create/Modify

```
utils/
├── core/
│   ├── __init__.py
│   ├── base_annotation_parser.py
│   ├── base_models.py
│   └── base_utils.py
```

## Implementation Steps

### Step 1: Create `utils/core/base_utils.py`

```python
"""Common utilities used across all modules."""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import re

def get_timestamp() -> str:
    """Get ISO format timestamp."""
    return datetime.now().isoformat()

def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validate and convert path."""
    path = Path(path)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path

def safe_json_load(filepath: Path) -> Dict:
    """Load JSON with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load {filepath}: {e}")

def safe_json_save(data: Dict, filepath: Path, create_backup: bool = True) -> None:
    """Atomic JSON save with optional backup."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if requested and file exists
    if create_backup and filepath.exists():
        backup_path = filepath.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        shutil.copy2(filepath, backup_path)
    
    # Write to temp file first
    temp_path = filepath.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Atomic rename
    temp_path.replace(filepath)

# =============================================================================
# ID PARSING - BUILD BACKWARDS FROM MOST SPECIFIC
# =============================================================================

def parse_snip_id(snip_id: str) -> Dict[str, str]:
    """
    Parse snip ID by building backwards to construct hierarchy.
    
    Format: {embryo_id}_s{frame_number} OR {embryo_id}_{frame_number}
    
    Examples:
        >>> parse_snip_id("20240411_A01_e01_s0042")
        {
            'experiment_id': '20240411',
            'well_id': 'A01', 
            'video_id': '20240411_A01',
            'embryo_id': '20240411_A01_e01',
            'embryo_number': '01',
            'frame_number': '0042',
            'snip_id': '20240411_A01_e01_s0042'
        }
        
        >>> parse_snip_id("20250624_chem02_28C_T00_1356_H01_e01_034")
        {
            'experiment_id': '20250624_chem02_28C_T00_1356',
            'well_id': 'H01',
            'video_id': '20250624_chem02_28C_T00_1356_H01', 
            'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
            'embryo_number': '01',
            'frame_number': '034',
            'snip_id': '20250624_chem02_28C_T00_1356_H01_e01_034'
        }
    """
    # Strategy: Find the last frame number and work backwards
    
    # Look for frame pattern at the end (3-4 digits, possibly with 's' prefix)
    frame_match = re.search(r'_s?(\d{3,4})$', snip_id)
    if not frame_match:
        raise ValueError(f"Invalid snip_id format: {snip_id} (no frame number found)")
    
    frame_number = frame_match.group(1)
    # Remove frame part to get embryo_id
    embryo_id = snip_id[:frame_match.start()]
    
    # Parse the embryo_id to get hierarchy
    embryo_components = parse_embryo_id(embryo_id)
    
    return {
        **embryo_components,
        'frame_number': frame_number,
        'snip_id': snip_id
    }

def parse_embryo_id(embryo_id: str) -> Dict[str, str]:
    """
    Parse embryo ID by building backwards to construct hierarchy.
    
    Format: {video_id}_e{embryo_number}
    
    Examples:
        >>> parse_embryo_id("20240411_A01_e01")
        {
            'experiment_id': '20240411',
            'well_id': 'A01',
            'video_id': '20240411_A01',
            'embryo_number': '01',
            'embryo_id': '20240411_A01_e01'
        }
        
        >>> parse_embryo_id("20250624_chem02_28C_T00_1356_H01_e01")
        {
            'experiment_id': '20250624_chem02_28C_T00_1356',
            'well_id': 'H01',
            'video_id': '20250624_chem02_28C_T00_1356_H01',
            'embryo_number': '01', 
            'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01'
        }
    """
    # Find embryo pattern at the end
    embryo_match = re.search(r'_e(\d+)$', embryo_id)
    if not embryo_match:
        raise ValueError(f"Invalid embryo_id format: {embryo_id} (no embryo pattern found)")
    
    embryo_number = embryo_match.group(1)
    # Remove embryo part to get video_id
    video_id = embryo_id[:embryo_match.start()]
    
    # Parse the video_id to get hierarchy
    video_components = parse_video_id(video_id)
    
    return {
        **video_components,
        'embryo_number': embryo_number,
        'embryo_id': embryo_id
    }

def parse_image_id(image_id: str) -> Dict[str, str]:
    """
    Parse image ID by building backwards to construct hierarchy.
    
    Format: {video_id}_{frame_number}
    
    Examples:
        >>> parse_image_id("20240411_A01_0042")
        {
            'experiment_id': '20240411',
            'well_id': 'A01',
            'video_id': '20240411_A01',
            'frame_number': '0042',
            'image_id': '20240411_A01_0042'
        }
        
        >>> parse_image_id("20250624_chem02_28C_T00_1356_H01_034")
        {
            'experiment_id': '20250624_chem02_28C_T00_1356',
            'well_id': 'H01',
            'video_id': '20250624_chem02_28C_T00_1356_H01',
            'frame_number': '034',
            'image_id': '20250624_chem02_28C_T00_1356_H01_034'
        }
    """
    # Find frame number at the end (3-4 digits)
    frame_match = re.search(r'_(\d{3,4})$', image_id)
    if not frame_match:
        raise ValueError(f"Invalid image_id format: {image_id} (no frame number found)")
    
    frame_number = frame_match.group(1)
    # Remove frame part to get video_id
    video_id = image_id[:frame_match.start()]
    
    # Parse the video_id to get hierarchy
    video_components = parse_video_id(video_id)
    
    return {
        **video_components,
        'frame_number': frame_number,
        'image_id': image_id
    }

def parse_video_id(video_id: str) -> Dict[str, str]:
    """
    Parse video ID by building backwards to construct hierarchy.
    
    Format: {experiment_id}_{well_id}
    Well ID is always the LAST component and matches pattern [A-H][0-9]{2}
    
    Examples:
        >>> parse_video_id("20240411_A01")
        {
            'experiment_id': '20240411',
            'well_id': 'A01',
            'video_id': '20240411_A01'
        }
        
        >>> parse_video_id("20250624_chem02_28C_T00_1356_H01")
        {
            'experiment_id': '20250624_chem02_28C_T00_1356',
            'well_id': 'H01',
            'video_id': '20250624_chem02_28C_T00_1356_H01'
        }
    """
    # Find well pattern at the end
    well_match = re.search(r'_([A-H]\d{2})$', video_id)
    if not well_match:
        raise ValueError(f"Invalid video_id format: {video_id} (no well pattern found)")
    
    well_id = well_match.group(1)
    # Remove well part to get experiment_id
    experiment_id = video_id[:well_match.start()]
    
    if not experiment_id:
        raise ValueError(f"Invalid video_id format: {video_id} (no experiment_id found)")
    
    return {
        'experiment_id': experiment_id,
        'well_id': well_id,
        'video_id': video_id
    }

def get_parent_ids(entity_id: str, entity_type: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity by parsing backwards.
    
    Args:
        entity_id: The ID to parse
        entity_type: Type hint - 'snip', 'embryo', 'image', or 'video'
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042", "image")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042", "snip")
        {
            'experiment_id': '20240411', 
            'video_id': '20240411_A01', 
            'embryo_id': '20240411_A01_e01'
        }
    """
    if entity_type == 'snip':
        components = parse_snip_id(entity_id)
        return {
            'experiment_id': components['experiment_id'],
            'video_id': components['video_id'],
            'embryo_id': components['embryo_id']
        }
    elif entity_type == 'embryo':
        components = parse_embryo_id(entity_id)
        return {
            'experiment_id': components['experiment_id'],
            'video_id': components['video_id']
        }
    elif entity_type == 'image':
        components = parse_image_id(entity_id)
        return {
            'experiment_id': components['experiment_id'],
            'video_id': components['video_id']
        }
    elif entity_type == 'video':
        components = parse_video_id(entity_id)
        return {
            'experiment_id': components['experiment_id']
        }
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

def construct_child_id(parent_id: str, parent_type: str, child_identifier: str) -> str:
    """
    Construct child ID from parent ID and child identifier.
    
    Examples:
        >>> construct_child_id("20240411", "experiment", "A01")
        "20240411_A01"  # video_id
        
        >>> construct_child_id("20240411_A01", "video", "e01")
        "20240411_A01_e01"  # embryo_id
        
        >>> construct_child_id("20240411_A01", "video", "0042")
        "20240411_A01_0042"  # image_id
        
        >>> construct_child_id("20240411_A01_e01", "embryo", "s0042")
        "20240411_A01_e01_s0042"  # snip_id
    """
    if parent_type == "experiment":
        # Child could be video (well_id like "A01")
        return f"{parent_id}_{child_identifier}"
    elif parent_type == "video":
        # Child could be embryo (like "e01") or image (like "0042")
        if child_identifier.startswith('e'):
            return f"{parent_id}_{child_identifier}"  # embryo
        else:
            return f"{parent_id}_{child_identifier}"  # image
    elif parent_type == "embryo":
        # Child is snip (like "s0042" or "0042")
        if not child_identifier.startswith('s'):
            child_identifier = f"s{child_identifier}"
        return f"{parent_id}_{child_identifier}"
    else:
        raise ValueError(f"Cannot construct child from parent_type: {parent_type}")

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str, entity_type: str) -> str:
    """
    Get simplified filename for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042", "image")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042", "snip")
        "s0042.jpg"
    """
    if entity_type == 'image':
        components = parse_image_id(entity_id)
        return f"{components['frame_number']}.jpg"
    elif entity_type == 'snip':
        components = parse_snip_id(entity_id)
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {entity_type}")

# =============================================================================
# HIERARCHY NAVIGATION UTILITIES
# =============================================================================

def get_experiment_structure(data: Dict, experiment_id: str) -> Dict:
    """
    Get complete structure for an experiment from any nested data.
    Useful for navigating experiment metadata or QC data.
    """
    if 'experiments' in data and experiment_id in data['experiments']:
        return data['experiments'][experiment_id]
    else:
        raise KeyError(f"Experiment {experiment_id} not found in data")

def get_video_structure(data: Dict, video_id: str) -> Dict:
    """Get video structure by parsing video_id to find parent experiment."""
    video_components = parse_video_id(video_id)
    experiment_id = video_components['experiment_id']
    
    exp_structure = get_experiment_structure(data, experiment_id)
    if 'videos' in exp_structure and video_id in exp_structure['videos']:
        return exp_structure['videos'][video_id]
    else:
        raise KeyError(f"Video {video_id} not found in experiment {experiment_id}")

def find_entity_in_structure(data: Dict, entity_id: str, entity_type: str) -> Tuple[Dict, List[str]]:
    """
    Find entity in nested structure and return both the entity data and the path to it.
    
    Returns:
        Tuple of (entity_data, path_components)
        
    Example:
        entity_data, path = find_entity_in_structure(data, "20240411_A01_0042", "image")
        # path = ['experiments', '20240411', 'videos', '20240411_A01', 'images', '20240411_A01_0042']
    """
    parent_ids = get_parent_ids(entity_id, entity_type)
    
    if entity_type == 'experiment':
        path = ['experiments', entity_id]
        return data['experiments'][entity_id], path
        
    elif entity_type == 'video':
        exp_id = parent_ids['experiment_id']
        path = ['experiments', exp_id, 'videos', entity_id]
        return data['experiments'][exp_id]['videos'][entity_id], path
        
    elif entity_type == 'image':
        exp_id = parent_ids['experiment_id']
        video_id = parent_ids['video_id']
        path = ['experiments', exp_id, 'videos', video_id, 'images', entity_id]
        return data['experiments'][exp_id]['videos'][video_id]['images'][entity_id], path
        
    elif entity_type == 'embryo':
        exp_id = parent_ids['experiment_id']
        video_id = parent_ids['video_id']
        path = ['experiments', exp_id, 'videos', video_id, 'embryos', entity_id]
        return data['experiments'][exp_id]['videos'][video_id]['embryos'][entity_id], path
        
    elif entity_type == 'snip':
        exp_id = parent_ids['experiment_id']
        video_id = parent_ids['video_id']
        embryo_id = parent_ids['embryo_id']
        path = ['experiments', exp_id, 'videos', video_id, 'embryos', embryo_id, 'snips', entity_id]
        return data['experiments'][exp_id]['videos'][video_id]['embryos'][embryo_id]['snips'][entity_id], path
    
    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

def ensure_entity_structure(data: Dict, entity_id: str, entity_type: str) -> None:
    """
    Ensure the nested structure exists for an entity.
    Creates missing intermediate structures.
    """
    parent_ids = get_parent_ids(entity_id, entity_type)
    
    # Ensure experiments dict exists
    if 'experiments' not in data:
        data['experiments'] = {}
    
    # Ensure experiment exists
    if entity_type in ['video', 'image', 'embryo', 'snip']:
        exp_id = parent_ids['experiment_id']
        if exp_id not in data['experiments']:
            data['experiments'][exp_id] = {'videos': {}}
    
    # Ensure video exists
    if entity_type in ['image', 'embryo', 'snip']:
        exp_id = parent_ids['experiment_id']
        video_id = parent_ids['video_id']
        if video_id not in data['experiments'][exp_id]['videos']:
            data['experiments'][exp_id]['videos'][video_id] = {
                'images': {},
                'embryos': {}
            }
    
    # Ensure embryo exists (for snips)
    if entity_type == 'snip':
        exp_id = parent_ids['experiment_id']
        video_id = parent_ids['video_id']
        embryo_id = parent_ids['embryo_id']
        if embryo_id not in data['experiments'][exp_id]['videos'][video_id]['embryos']:
            data['experiments'][exp_id]['videos'][video_id]['embryos'][embryo_id] = {
                'snips': {}
            }

# =============================================================================
# BATCH OPERATIONS UTILITIES
# =============================================================================

def group_entities_by_parent(entity_ids: List[str], entity_type: str) -> Dict[str, List[str]]:
    """
    Group entities by their parent for efficient batch operations.
    
    Example:
        >>> group_entities_by_parent(["20240411_A01_0001", "20240411_A01_0002", "20240411_B01_0001"], "image")
        {
            "20240411_A01": ["20240411_A01_0001", "20240411_A01_0002"],
            "20240411_B01": ["20240411_B01_0001"]
        }
    """
    groups = {}
    for entity_id in entity_ids:
        if entity_type == 'video':
            parent_ids = get_parent_ids(entity_id, entity_type)
            parent_key = parent_ids['experiment_id']
        elif entity_type == 'image':
            parent_ids = get_parent_ids(entity_id, entity_type)
            parent_key = parent_ids['video_id']
        elif entity_type == 'embryo':
            parent_ids = get_parent_ids(entity_id, entity_type)
            parent_key = parent_ids['video_id']
        elif entity_type == 'snip':
            parent_ids = get_parent_ids(entity_id, entity_type)
            parent_key = parent_ids['embryo_id']
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")
        
        if parent_key not in groups:
            groups[parent_key] = []
        groups[parent_key].append(entity_id)
    
    return groups

def validate_entity_id_format(entity_id: str, entity_type: str) -> bool:
    """
    Validate that an entity ID has the correct format for its type.
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if entity_type == 'snip':
            parse_snip_id(entity_id)
        elif entity_type == 'embryo':
            parse_embryo_id(entity_id)
        elif entity_type == 'image':
            parse_image_id(entity_id)
        elif entity_type == 'video':
            parse_video_id(entity_id)
        else:
            return False
        return True
    except ValueError:
        return False

def filter_valid_entity_ids(entity_ids: List[str], entity_type: str) -> Tuple[List[str], List[str]]:
    """
    Filter entity IDs into valid and invalid lists.
    
    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    valid_ids = []
    invalid_ids = []
    
    for entity_id in entity_ids:
        if validate_entity_id_format(entity_id, entity_type):
            valid_ids.append(entity_id)
        else:
            invalid_ids.append(entity_id)
    
    return valid_ids, invalid_ids
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, get_parent_ids,
    find_entity_in_structure, ensure_entity_structure,
    group_entities_by_parent, validate_entity_id_format
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking and auto-save functionality
    - Backup management
    - GSAM ID support for cross-referencing
    - Entity navigation using backward-parsing
    - Batch operations with grouping
    - Progress callbacks for long operations
    """
    
    def __init__(self, filepath: Union[str, Path], 
                 auto_save_interval: Optional[int] = None,
                 verbose: bool = True):
        """Initialize parser with file path."""
        self.filepath = validate_path(filepath)
        self.verbose = verbose
        self.auto_save_interval = auto_save_interval
        self._processed_count = 0
        self._unsaved_changes = False
        self.data = self._load_or_initialize()
        
    @abstractmethod
    def _load_or_initialize(self) -> Dict:
        """Load existing data or initialize new structure."""
        pass
    
    @abstractmethod
    def _validate_schema(self, data: Dict) -> None:
        """Validate data structure against expected schema."""
        pass
    
    # -------------------------------------------------------------------------
    # File I/O Operations
    # -------------------------------------------------------------------------
    
    def load_json(self, filepath: Optional[Path] = None) -> Dict:
        """Load JSON data from file."""
        filepath = filepath or self.filepath
        if not filepath.exists():
            return {}
        return safe_json_load(filepath)
    
    def save_json(self, data: Optional[Dict] = None, 
                  filepath: Optional[Path] = None,
                  create_backup: bool = True) -> None:
        """Save JSON data with atomic write."""
        data = data or self.data
        filepath = filepath or self.filepath
        safe_json_save(data, filepath, create_backup)
        self._unsaved_changes = False
        
    def save(self, backup: bool = True) -> None:
        """Save current data to file."""
        self.save_json(self.data, self.filepath, backup)
        if self.verbose:
            print(f"💾 Saved {self.__class__.__name__} to {self.filepath}")
    
    # -------------------------------------------------------------------------
    # Change Tracking
    # -------------------------------------------------------------------------
    
    def mark_changed(self) -> None:
        """Mark data as having unsaved changes."""
        self._unsaved_changes = True
        self._check_auto_save()
    
    def mark_saved(self) -> None:
        """Mark data as saved."""
        self._unsaved_changes = False
        
    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._unsaved_changes
    
    def _check_auto_save(self) -> None:
        """Check if auto-save should trigger."""
        if self.auto_save_interval and self._processed_count >= self.auto_save_interval:
            if self._unsaved_changes:
                self.save()
                self._processed_count = 0
                if self.verbose:
                    print(f"💾 Auto-saved after {self.auto_save_interval} operations")
    
    def increment_processed(self) -> None:
        """Increment processed counter for auto-save."""
        self._processed_count += 1
        self._check_auto_save()
    
    # -------------------------------------------------------------------------
    # Entity Navigation
    # -------------------------------------------------------------------------
    
    def get_entity(self, entity_id: str, entity_type: str) -> Optional[Dict]:
        """
        Get entity data by ID using backward parsing to find location.
        
        Args:
            entity_id: The entity ID to find
            entity_type: Type hint ('experiment', 'video', 'image', 'embryo', 'snip')
        
        Returns:
            Entity data dictionary or None if not found
        """
        try:
            entity_data, _ = find_entity_in_structure(self.data, entity_id, entity_type)
            return entity_data
        except (KeyError, ValueError):
            return None
    
    def ensure_entity_exists(self, entity_id: str, entity_type: str, 
                           default_data: Optional[Dict] = None) -> Dict:
        """
        Ensure entity exists in data structure, creating if necessary.
        
        Args:
            entity_id: The entity ID to ensure exists
            entity_type: Type of entity
            default_data: Default data to populate if creating new entity
        
        Returns:
            The entity data dictionary
        """
        # First ensure the nested structure exists
        ensure_entity_structure(self.data, entity_id, entity_type)
        
        # Get or create the specific entity
        try:
            entity_data, path = find_entity_in_structure(self.data, entity_id, entity_type)
            return entity_data
        except KeyError:
            # Entity structure exists but entity doesn't, create it
            default_data = default_data or {}
            
            # Navigate to parent and add entity
            parent_ids = get_parent_ids(entity_id, entity_type)
            
            if entity_type == 'experiment':
                self.data['experiments'][entity_id] = default_data
                entity_data = self.data['experiments'][entity_id]
            elif entity_type == 'video':
                exp_id = parent_ids['experiment_id']
                self.data['experiments'][exp_id]['videos'][entity_id] = default_data
                entity_data = self.data['experiments'][exp_id]['videos'][entity_id]
            elif entity_type == 'image':
                exp_id = parent_ids['experiment_id']
                video_id = parent_ids['video_id']
                self.data['experiments'][exp_id]['videos'][video_id]['images'][entity_id] = default_data
                entity_data = self.data['experiments'][exp_id]['videos'][video_id]['images'][entity_id]
            elif entity_type == 'embryo':
                exp_id = parent_ids['experiment_id']
                video_id = parent_ids['video_id']
                self.data['experiments'][exp_id]['videos'][video_id]['embryos'][entity_id] = default_data
                entity_data = self.data['experiments'][exp_id]['videos'][video_id]['embryos'][entity_id]
            elif entity_type == 'snip':
                exp_id = parent_ids['experiment_id']
                video_id = parent_ids['video_id']
                embryo_id = parent_ids['embryo_id']
                self.data['experiments'][exp_id]['videos'][video_id]['embryos'][embryo_id]['snips'][entity_id] = default_data
                entity_data = self.data['experiments'][exp_id]['videos'][video_id]['embryos'][embryo_id]['snips'][entity_id]
            
            self.mark_changed()
            return entity_data
    
    def get_entity_parent(self, entity_id: str, entity_type: str) -> Optional[Dict]:
        """Get the parent entity data."""
        try:
            parent_ids = get_parent_ids(entity_id, entity_type)
            
            if entity_type == 'video':
                return self.get_entity(parent_ids['experiment_id'], 'experiment')
            elif entity_type in ['image', 'embryo']:
                return self.get_entity(parent_ids['video_id'], 'video')
            elif entity_type == 'snip':
                return self.get_entity(parent_ids['embryo_id'], 'embryo')
            else:
                return None
        except (KeyError, ValueError):
            return None
    
    def get_entity_children(self, entity_id: str, entity_type: str, 
                          child_type: str) -> Dict[str, Dict]:
        """
        Get all children of an entity.
        
        Args:
            entity_id: Parent entity ID
            entity_type: Type of parent entity
            child_type: Type of children to retrieve
        
        Returns:
            Dictionary mapping child_id -> child_data
        """
        try:
            entity_data = self.get_entity(entity_id, entity_type)
            if not entity_data:
                return {}
            
            if child_type == 'video' and entity_type == 'experiment':
                return entity_data.get('videos', {})
            elif child_type == 'image' and entity_type == 'video':
                return entity_data.get('images', {})
            elif child_type == 'embryo' and entity_type == 'video':
                return entity_data.get('embryos', {})
            elif child_type == 'snip' and entity_type == 'embryo':
                return entity_data.get('snips', {})
            else:
                return {}
        except (KeyError, ValueError):
            return {}
    
    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------
    
    def process_entities_batch(self, entity_ids: List[str], entity_type: str,
                             processor_func: Callable[[str, Dict], Any],
                             progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process entities in batches grouped by parent for efficiency.
        
        Args:
            entity_ids: List of entity IDs to process
            entity_type: Type of entities being processed
            processor_func: Function that takes (entity_id, entity_data) and returns result
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of results from processor_func
        """
        # Validate entity IDs first
        valid_ids, invalid_ids = filter_valid_entity_ids(entity_ids, entity_type)
        
        if invalid_ids and self.verbose:
            print(f"⚠️  Skipping {len(invalid_ids)} invalid {entity_type} IDs")
        
        # Group by parent for efficient processing
        grouped = group_entities_by_parent(valid_ids, entity_type)
        
        results = []
        total_processed = 0
        total_count = len(valid_ids)
        
        for parent_id, child_ids in grouped.items():
            if self.verbose:
                print(f"📦 Processing {len(child_ids)} {entity_type}s in {parent_id}")
            
            for entity_id in child_ids:
                # Get entity data
                entity_data = self.get_entity(entity_id, entity_type)
                if entity_data is not None:
                    # Process entity
                    result = processor_func(entity_id, entity_data)
                    results.append(result)
                
                total_processed += 1
                self.increment_processed()
                
                # Progress callback
                if progress_callback:
                    progress_callback(total_processed, total_count, entity_id)
        
        return results
    
    def update_entities_batch(self, updates: Dict[str, Dict], entity_type: str,
                            merge_mode: str = 'update') -> int:
        """
        Update multiple entities efficiently.
        
        Args:
            updates: Dict mapping entity_id -> update_data
            entity_type: Type of entities being updated
            merge_mode: 'update' (merge), 'replace' (overwrite), or 'append' (for lists)
        
        Returns:
            Number of entities updated
        """
        updated_count = 0
        
        for entity_id, update_data in updates.items():
            try:
                entity_data = self.ensure_entity_exists(entity_id, entity_type, {})
                
                if merge_mode == 'replace':
                    # Replace entirely
                    entity_data.clear()
                    entity_data.update(update_data)
                elif merge_mode == 'update':
                    # Merge/update
                    entity_data.update(update_data)
                elif merge_mode == 'append':
                    # Append to lists
                    for key, value in update_data.items():
                        if key not in entity_data:
                            entity_data[key] = []
                        if isinstance(value, list):
                            entity_data[key].extend(value)
                        else:
                            entity_data[key].append(value)
                
                updated_count += 1
                self.increment_processed()
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to update {entity_id}: {e}")
        
        if updated_count > 0:
            self.mark_changed()
        
        return updated_count
    
    # -------------------------------------------------------------------------
    # GSAM ID Management
    # -------------------------------------------------------------------------
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def get_gsam_id(self) -> Optional[int]:
        """Get GSAM ID if it exists."""
        return self.data.get('gsam_annotation_id')
    
    def set_gsam_id(self, gsam_id: int) -> None:
        """Set GSAM ID manually."""
        self.data['gsam_annotation_id'] = gsam_id
        self.mark_changed()
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def get_summary_stats(self) -> Dict[str, int]:
        """Get summary statistics about the data."""
        stats = {'experiments': 0, 'videos': 0, 'images': 0, 'embryos': 0, 'snips': 0}
        
        for exp_id, exp_data in self.data.get('experiments', {}).items():
            stats['experiments'] += 1
            
            for video_id, video_data in exp_data.get('videos', {}).items():
                stats['videos'] += 1
                stats['images'] += len(video_data.get('images', {}))
                
                for embryo_id, embryo_data in video_data.get('embryos', {}).items():
                    stats['embryos'] += 1
                    stats['snips'] += len(embryo_data.get('snips', {}))
        
        return stats
    
    def validate_all_entity_ids(self) -> Dict[str, List[str]]:
        """
        Validate all entity IDs in the data structure.
        
        Returns:
            Dict with invalid IDs by type
        """
        invalid_ids = {
            'experiments': [],
            'videos': [], 
            'images': [],
            'embryos': [],
            'snips': []
        }
        
        for exp_id, exp_data in self.data.get('experiments', {}).items():
            # Validate experiment ID (just check it's not empty)
            if not exp_id:
                invalid_ids['experiments'].append(exp_id)
            
            for video_id, video_data in exp_data.get('videos', {}).items():
                if not validate_entity_id_format(video_id, 'video'):
                    invalid_ids['videos'].append(video_id)
                
                for image_id in video_data.get('images', {}):
                    if not validate_entity_id_format(image_id, 'image'):
                        invalid_ids['images'].append(image_id)
                
                for embryo_id, embryo_data in video_data.get('embryos', {}).items():
                    if not validate_entity_id_format(embryo_id, 'embryo'):
                        invalid_ids['embryos'].append(embryo_id)
                    
                    for snip_id in embryo_data.get('snips', {}):
                        if not validate_entity_id_format(snip_id, 'snip'):
                            invalid_ids['snips'].append(snip_id)
        
        return invalid_ids
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def __repr__(self) -> str:
        """String representation with summary stats."""
        stats = self.get_summary_stats()
        status = "saved" if not self._unsaved_changes else "unsaved changes"
        gsam_id = self.get_gsam_id()
        
        return (f"{self.__class__.__name__}("
                f"experiments={stats['experiments']}, "
                f"videos={stats['videos']}, "
                f"images={stats['images']}, "
                f"embryos={stats['embryos']}, "
                f"snips={stats['snips']}, "
                f"gsam_id={gsam_id}, "
                f"status={status})")
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
    class Config:
        extra = "allow"  # Allow additional fields

class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    detection_id: Optional[str] = None
    
    @validator('box_xyxy')
    def validate_bbox(cls, v):
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: x2 > x1 and y2 > y1 required")
        return v

class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon|bitmap)$")
    mask_data: Union[Dict, List]  # RLE dict, polygon points, or bitmap
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    segmentation_id: Optional[str] = None
    
    @validator('bbox_xyxy')
    def validate_bbox(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("bbox_xyxy must have exactly 4 elements")
        return v

class QCFlagModel(BaseModel):
    """Model for QC flags at any level."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    entity_id: str
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    severity: Optional[str] = Field(default="medium", regex="^(low|medium|high|critical)$")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['experiment', 'video', 'image', 'embryo', 'snip']
        if v not in valid_levels:
            raise ValueError(f"Invalid level: {v}. Must be one of {valid_levels}")
        return v

class EmbryoMetadataModel(BaseModel):
    """Model for embryo metadata."""
    embryo_id: str
    genotype: Optional[str] = None
    phenotype: Optional[str] = "NONE"
    treatment: Optional[str] = None
    notes: Optional[str] = ""
    
    @validator('phenotype')
    def validate_phenotype(cls, v):
        # Will be expanded based on specific phenotype categories
        return v.upper() if v else "NONE"

class ExperimentMetadataModel(BaseModel):
    """Model for experiment-level metadata."""
    experiment_id: str
    date: Optional[str] = None
    protocol: Optional[str] = None
    investigator: Optional[str] = None
    description: Optional[str] = ""
    conditions: Optional[Dict[str, Any]] = {}

class VideoMetadataModel(BaseModel):
    """Model for video-level metadata."""
    video_id: str
    well_id: str
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    imaging_conditions: Optional[Dict[str, Any]] = {}

class ImageMetadataModel(BaseModel):
    """Model for image-level metadata."""
    image_id: str
    frame_number: int
    timestamp: Optional[datetime] = None
    exposure_time: Optional[float] = None
    resolution: Optional[Dict[str, int]] = None  # {"width": 1024, "height": 768}

class SnipMetadataModel(BaseModel):
    """Model for embryo snip metadata."""
    snip_id: str
    embryo_id: str
    frame_number: int
    bbox_xyxy: Optional[List[float]] = None
    mask_area: Optional[float] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

# Collections for batch validation
class BatchDetectionModel(BaseModel):
    """Model for batch detection results."""
    detections: List[DetectionModel]
    image_id: str
    model_info: Optional[Dict[str, Any]] = {}
    processing_timestamp: datetime

class BatchQCFlagModel(BaseModel):
    """Model for batch QC operations."""
    flags: List[QCFlagModel]
    batch_id: Optional[str] = None
    processing_info: Optional[Dict[str, Any]] = {}

# Validation utilities
def validate_entity_id_structure(entity_id: str, expected_type: str) -> bool:
    """Validate entity ID structure using the parsing functions."""
    from .base_utils import validate_entity_id_format
    return validate_entity_id_format(entity_id, expected_type)

def validate_hierarchy_consistency(parent_id: str, child_id: str, 
                                 parent_type: str, child_type: str) -> bool:
    """Validate that child ID is consistent with parent ID."""
    from .base_utils import get_parent_ids
    
    try:
        child_parents = get_parent_ids(child_id, child_type)
        
        if parent_type == 'experiment' and child_type == 'video':
            return child_parents['experiment_id'] == parent_id
        elif parent_type == 'video' and child_type in ['image', 'embryo']:
            return child_parents['video_id'] == parent_id
        elif parent_type == 'embryo' and child_type == 'snip':
            return child_parents['embryo_id'] == parent_id
        else:
            return False
    except (ValueError, KeyError):
        return False
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, SegmentationModel, 
    QCFlagModel, EmbryoMetadataModel, ExperimentMetadataModel,
    VideoMetadataModel, ImageMetadataModel, SnipMetadataModel,
    BatchDetectionModel, BatchQCFlagModel,
    validate_entity_id_structure, validate_hierarchy_consistency
)
from .base_utils import (
    # Timestamp and file utilities
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    
    # ID parsing (build backwards)
    parse_snip_id, parse_embryo_id, parse_image_id, parse_video_id,
    get_parent_ids, construct_child_id,
    
    # Navigation utilities
    get_experiment_structure, get_video_structure, 
    find_entity_in_structure, ensure_entity_structure,
    
    # Batch utilities
    group_entities_by_parent, validate_entity_id_format, filter_valid_entity_ids,
    
    # Storage utilities
    generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 'SegmentationModel', 
    'QCFlagModel', 'EmbryoMetadataModel', 'ExperimentMetadataModel',
    'VideoMetadataModel', 'ImageMetadataModel', 'SnipMetadataModel',
    'BatchDetectionModel', 'BatchQCFlagModel',
    
    # Validation functions
    'validate_entity_id_structure', 'validate_hierarchy_consistency',
    
    # Timestamp and file utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    
    # ID parsing (build backwards)
    'parse_snip_id', 'parse_embryo_id', 'parse_image_id', 'parse_video_id',
    'get_parent_ids', 'construct_child_id',
    
    # Navigation utilities  
    'get_experiment_structure', 'get_video_structure',
    'find_entity_in_structure', 'ensure_entity_structure',
    
    # Batch utilities
    'group_entities_by_parent', 'validate_entity_id_format', 'filter_valid_entity_ids',
    
    # Storage utilities
    'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test backward ID parsing with complex experiment names
- [ ] Test hierarchy construction: snip → embryo → video → experiment
- [ ] Test parent/child relationship validation  
- [ ] Test entity navigation in nested structures
- [ ] Test batch operations with entity grouping
- [ ] Test auto-save functionality and change tracking
- [ ] Test GSAM ID generation and management
- [ ] Test atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Verify all ID format validations work correctly

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with backward parsing | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for backward parsing | Pending |
| TBD | TBD | Integration tests | Pending |

## Usage Examples

```python
from utils.core import (
    parse_snip_id, parse_embryo_id, parse_image_id, parse_video_id,
    get_parent_ids, construct_child_id, BaseAnnotationParser
)

# Parse complex experiment IDs by building backwards
snip_data = parse_snip_id("20250624_chem02_28C_T00_1356_H01_e01_s034")
# Returns: {
#     'experiment_id': '20250624_chem02_28C_T00_1356',
#     'well_id': 'H01',
#     'video_id': '20250624_chem02_28C_T00_1356_H01',
#     'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
#     'embryo_number': '01',
#     'frame_number': '034',
#     'snip_id': '20250624_chem02_28C_T00_1356_H01_e01_s034'
# }

# Get parent hierarchy for any entity
parents = get_parent_ids("20240411_A01_e01_s0042", "snip")
# Returns: {
#     'experiment_id': '20240411',
#     'video_id': '20240411_A01', 
#     'embryo_id': '20240411_A01_e01'
# }

# Construct child IDs
video_id = construct_child_id("20240411", "experiment", "A01")
# Returns: "20240411_A01"

embryo_id = construct_child_id("20240411_A01", "video", "e01")  
# Returns: "20240411_A01_e01"

snip_id = construct_child_id("20240411_A01_e01", "embryo", "s0042")
# Returns: "20240411_A01_e01_s0042"

# Use base parser for any annotation type
class MyAnnotationParser(BaseAnnotationParser):
    def _load_or_initialize(self):
        data = self.load_json()
        if not data:
            data = {'experiments': {}, 'gsam_annotation_id': self.generate_gsam_id()}
        return data
    
    def _validate_schema(self, data):
        assert 'experiments' in data

# Create parser and use entity navigation
parser = MyAnnotationParser("/path/to/annotations.json")

# Get entity with automatic structure traversal
embryo_data = parser.get_entity("20240411_A01_e01", "embryo")

# Ensure entity exists (creates structure if needed)
snip_data = parser.ensure_entity_exists("20240411_A01_e01_s0042", "snip", 
                                        {"mask_area": 1250, "quality": 0.95})

# Batch operations with automatic grouping
entity_ids = ["20240411_A01_0001", "20240411_A01_0002", "20240411_B01_0001"]

def process_image(image_id, image_data):
    # Your processing logic here
    return {"processed": True, "id": image_id}

results = parser.process_entities_batch(entity_ids, "image", process_image)

# Batch updates
updates = {
    "20240411_A01_0001": {"qc_flags": ["BLUR"]},
    "20240411_A01_0002": {"qc_flags": ["DRY_WELL"]}
}
updated_count = parser.update_entities_batch(updates, "image", merge_mode="update")

parser.save()  # Save all changes
```

## Key Innovations

1. **Backward ID Parsing**: Handles complex experiment IDs by parsing from most specific (frame) to general (experiment)

2. **Automatic Structure Navigation**: Find entities anywhere in nested hierarchy without manual path construction

3. **Smart Batch Operations**: Groups entities by parent for efficient processing

4. **Flexible Entity Management**: Ensure structures exist, get parents/children, validate hierarchies

5. **Change Tracking**: Auto-save, unsaved change detection, atomic writes with backups

6. **GSAM ID Integration**: Cross-reference support for linking between annotation files