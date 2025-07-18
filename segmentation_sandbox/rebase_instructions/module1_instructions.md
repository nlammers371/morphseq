# Module 1: Core Foundation

## Overview
Create the base classes and utilities that all other components will inherit from. This establishes consistent patterns for JSON handling, change tracking, ID parsing, and common operations across the entire pipeline.

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

def parse_image_id(image_id: str) -> Dict[str, str]:
    """
    Parse image ID to extract components.
    
    Format: {experiment_id}_{well_id}_{frame_number}
    Where experiment_id can be complex (e.g., "20250622_chem_35C_T01_1605")
    
    Examples:
        >>> parse_image_id("20250622_chem_35C_T01_1605_H09_0000")
        {
            'experiment_id': '20250622_chem_35C_T01_1605',
            'well_id': 'H09',
            'frame_number': '0000',
            'video_id': '20250622_chem_35C_T01_1605_H09',
            'image_id': '20250622_chem_35C_T01_1605_H09_0000'
        }
    """
    # Split from the right to handle complex experiment IDs
    parts = image_id.rsplit('_', 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid image_id format: {image_id}")
    
    exp_id, well_id, frame = parts
    
    # Validate well_id format
    if not re.match(r'^[A-H]\d{2}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Parse specific entity types
image_data = parse_image_id("20250622_chem_35C_T01_1605_H09_0000")
# Returns: {
#     'experiment_id': '20250622_chem_35C_T01_1605',
#     'well_id': 'H09',
#     'frame_number': '0000',
#     'video_id': '20250622_chem_35C_T01_1605_H09',
#     'image_id': '20250622_chem_35C_T01_1605_H09_0000'
# }

video_data = parse_video_id("20250622_chem_35C_T01_1605_H09")
# Returns: {
#     'experiment_id': '20250622_chem_35C_T01_1605',
#     'well_id': 'H09',
#     'video_id': '20250622_chem_35C_T01_1605_H09'
# }

# Parse with auto-detection (when possible)
level, data = parse_entity_id("20250622_chem_35C_T01_1605_H09_e01_s0042")
# Returns: ('snip', {...})

# Parse with explicit level (recommended for ambiguous cases)
level, data = parse_entity_id("20250622_chem_35C_T01_1605", level='experiment')
# Returns: ('experiment', {'experiment_id': '20250622_chem_35C_T01_1605'})

# Get parent hierarchy
parents = get_parent_ids("20250622_chem_35C_T01_1605_H09_e01_s0042")
# Returns: {
#     'experiment_id': '20250622_chem_35C_T01_1605',
#     'video_id': '20250622_chem_35C_T01_1605_H09',
#     'embryo_id': '20250622_chem_35C_T01_1605_H09_e01'
# }

# Simplified filenames for storage
filename = get_simplified_filename("20250622_chem_35C_T01_1605_H09_0000")
# Returns: "0000.jpg" (not the full ID)
```

## Important Notes on ID Parsing

1. **Experiment IDs are flexible**: Can include dates, conditions, timepoints, etc.
   - Examples: "20250622_chem_35C_T01_1605", "20240411", "control_experiment_001"
   
2. **Well IDs are strict**: Must match pattern [A-H][0-9]{2}
   - Valid: "A01", "H12", "C05"
   - Invalid: "I01", "A1", "AA1"

3. **Frame numbers are 4 digits**: "0000", "0042", "1234"

4. **Auto-detection limitations**: 
   - Works best for snip, embryo, and image IDs
   - May be ambiguous for experiment vs video IDs
   - Use explicit level parameter when uncertain

5. **Parsing strategy**:
   - For image/video/embryo/snip: Parse from the right (rsplit)
   - This handles complex experiment IDs correctly, well_id):
        raise ValueError(f"Invalid well_id format in image_id: {image_id}")
    
    # Validate frame format
    if not re.match(r'^\d{4}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, frame):
        raise ValueError(f"Invalid frame format in image_id: {image_id}")
    
    return {
        'experiment_id': exp_id,
        'well_id': well_id,
        'frame_number': frame,
        'video_id': f"{exp_id}_{well_id}",
        'image_id': image_id
    }

def parse_video_id(video_id: str) -> Dict[str, str]:
    """
    Parse video ID to extract components.
    
    Format: {experiment_id}_{well_id}
    
    Examples:
        >>> parse_video_id("20250622_chem_35C_T01_1605_H09")
        {
            'experiment_id': '20250622_chem_35C_T01_1605',
            'well_id': 'H09',
            'video_id': '20250622_chem_35C_T01_1605_H09'
        }
    """
    # Split from the right to handle complex experiment IDs
    parts = video_id.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid video_id format: {video_id}")
    
    exp_id, well_id = parts
    
    # Validate well_id format
    if not re.match(r'^[A-H]\d{2}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, well_id):
        raise ValueError(f"Invalid well_id format in video_id: {video_id}")
    
    return {
        'experiment_id': exp_id,
        'well_id': well_id,
        'video_id': video_id
    }

def parse_embryo_id(embryo_id: str) -> Dict[str, str]:
    """
    Parse embryo ID to extract components.
    
    Format: {experiment_id}_{well_id}_e{embryo_number}
    
    Examples:
        >>> parse_embryo_id("20250622_chem_35C_T01_1605_H09_e01")
        {
            'experiment_id': '20250622_chem_35C_T01_1605',
            'well_id': 'H09',
            'embryo_number': '01',
            'video_id': '20250622_chem_35C_T01_1605_H09',
            'embryo_id': '20250622_chem_35C_T01_1605_H09_e01'
        }
    """
    # Check for embryo pattern
    if not re.search(r'_e\d+

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, embryo_id):
        raise ValueError(f"Invalid embryo_id format: {embryo_id}")
    
    # Split to get embryo number
    base_id, embryo_part = embryo_id.rsplit('_e', 1)
    
    # Parse the base as a video_id
    video_components = parse_video_id(base_id)
    
    return {
        'experiment_id': video_components['experiment_id'],
        'well_id': video_components['well_id'],
        'video_id': base_id,
        'embryo_number': embryo_part,
        'embryo_id': embryo_id
    }

def parse_snip_id(snip_id: str) -> Dict[str, str]:
    """
    Parse snip ID to extract components.
    
    Format: {experiment_id}_{well_id}_e{embryo_number}_s{frame_number}
    
    Examples:
        >>> parse_snip_id("20250622_chem_35C_T01_1605_H09_e01_s0042")
        {
            'experiment_id': '20250622_chem_35C_T01_1605',
            'well_id': 'H09',
            'embryo_number': '01',
            'frame_number': '0042',
            'video_id': '20250622_chem_35C_T01_1605_H09',
            'embryo_id': '20250622_chem_35C_T01_1605_H09_e01',
            'snip_id': '20250622_chem_35C_T01_1605_H09_e01_s0042'
        }
    """
    # Check for snip pattern
    if not re.search(r'_s\d{4}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, snip_id):
        raise ValueError(f"Invalid snip_id format: {snip_id}")
    
    # Split to get frame number
    base_id, frame_part = snip_id.rsplit('_s', 1)
    
    # Parse the base as an embryo_id
    embryo_components = parse_embryo_id(base_id)
    
    return {
        'experiment_id': embryo_components['experiment_id'],
        'well_id': embryo_components['well_id'],
        'video_id': embryo_components['video_id'],
        'embryo_id': base_id,
        'embryo_number': embryo_components['embryo_number'],
        'frame_number': frame_part,
        'snip_id': snip_id
    }

def parse_entity_id(entity_id: str, level: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
    """
    Parse entity ID, optionally with level hint.
    
    Since experiment IDs can have variable formats, we cannot auto-detect level.
    Either provide the level explicitly, or use the specific parse functions.
    
    Args:
        entity_id: The ID to parse
        level: Required for auto-detection ('experiment', 'video', 'image', 'embryo', 'snip')
    
    Returns:
        Tuple of (level, parsed_components_dict)
    """
    if level == 'experiment':
        # Experiment IDs have no fixed pattern, just return as-is
        return ('experiment', {'experiment_id': entity_id})
    
    elif level == 'video':
        components = parse_video_id(entity_id)
        return ('video', components)
    
    elif level == 'image':
        components = parse_image_id(entity_id)
        return ('image', components)
    
    elif level == 'embryo':
        components = parse_embryo_id(entity_id)
        return ('embryo', components)
    
    elif level == 'snip':
        components = parse_snip_id(entity_id)
        return ('snip', components)
    
    else:
        # Try to auto-detect by checking patterns
        try:
            # Check for snip pattern first (most specific)
            if re.search(r'_s\d{4}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, entity_id):
                return ('snip', parse_snip_id(entity_id))
            
            # Check for embryo pattern
            elif re.search(r'_e\d+

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, entity_id):
                return ('embryo', parse_embryo_id(entity_id))
            
            # Check if ends with 4 digits (likely image)
            elif re.search(r'_\d{4}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, entity_id):
                try:
                    return ('image', parse_image_id(entity_id))
                except ValueError:
                    pass
            
            # Check if ends with well pattern (likely video)
            elif re.search(r'_[A-H]\d{2}

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```, entity_id):
                try:
                    return ('video', parse_video_id(entity_id))
                except ValueError:
                    pass
            
            # Default to experiment if no patterns match
            return ('experiment', {'experiment_id': entity_id})
            
        except ValueError:
            # If parsing fails, assume it's an experiment ID
            return ('experiment', {'experiment_id': entity_id})

def get_parent_ids(entity_id: str) -> Dict[str, str]:
    """
    Get all parent IDs for a given entity.
    
    Examples:
        >>> get_parent_ids("20240411_A01_0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01'}
        
        >>> get_parent_ids("20240411_A01_e01_s0042")
        {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}
    """
    level, components = parse_entity_id(entity_id)
    
    parent_ids = {}
    if 'experiment_id' in components:
        parent_ids['experiment_id'] = components['experiment_id']
    if 'video_id' in components and level not in ['experiment', 'video']:
        parent_ids['video_id'] = components['video_id']
    if 'embryo_id' in components and level == 'snip':
        parent_ids['embryo_id'] = components['embryo_id']
        
    return parent_ids

def generate_gsam_id() -> int:
    """Generate 4-digit GSAM ID for linking annotations."""
    import random
    return random.randint(1000, 9999)

def get_simplified_filename(entity_id: str) -> str:
    """
    Get simplified filename (just frame number) for organized storage.
    
    Examples:
        >>> get_simplified_filename("20240411_A01_0042")
        "0042.jpg"
        
        >>> get_simplified_filename("20240411_A01_e01_s0042")
        "s0042.jpg"
    """
    level, components = parse_entity_id(entity_id)
    
    if level == 'image':
        return f"{components['frame_number']}.jpg"
    elif level == 'snip':
        return f"s{components['frame_number']}.jpg"
    else:
        raise ValueError(f"Cannot create filename for {level} level entity")
```

### Step 2: Create `utils/core/base_annotation_parser.py`

```python
"""Base class for all annotation and metadata parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import json

from .base_utils import (
    safe_json_load, safe_json_save, validate_path, 
    get_timestamp, generate_gsam_id, parse_entity_id
)

class BaseAnnotationParser(ABC):
    """
    Abstract base class for all annotation/metadata parsers.
    
    Provides:
    - Consistent JSON I/O with atomic writes
    - Change tracking
    - Auto-save functionality
    - Backup management
    - GSAM ID support
    - Progress callbacks
    - Entity ID parsing
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
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return get_timestamp()
    
    def ensure_gsam_id(self) -> int:
        """Ensure this annotation has a GSAM ID."""
        if 'gsam_annotation_id' not in self.data:
            self.data['gsam_annotation_id'] = generate_gsam_id()
            self.mark_changed()
        return self.data['gsam_annotation_id']
    
    def parse_entity_id(self, entity_id: str, expected_level: Optional[str] = None):
        """Parse entity ID using unified parser."""
        return parse_entity_id(entity_id, expected_level)
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        import shutil
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.with_suffix(f'.backup_{timestamp}.json')
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity data by ID (must be implemented by subclasses).
        Uses the unified parser to find the entity at any level.
        """
        level, components = self.parse_entity_id(entity_id)
        # Subclasses implement specific lookup logic
        return None
```

### Step 3: Create `utils/core/base_models.py`

```python
"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime

class BaseAnnotationModel(BaseModel):
    """Base model for all annotations."""
    annotation_id: str
    timestamp: datetime
    author: str
    notes: Optional[str] = ""
    
class DetectionModel(BaseModel):
    """Model for object detection results."""
    box_xyxy: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_name: str
    
class SegmentationModel(BaseModel):
    """Model for segmentation masks."""
    mask_format: str = Field(..., regex="^(rle|polygon)$")
    mask_data: Union[Dict, List]  # RLE dict or polygon points
    area: Optional[float] = None
    bbox_xyxy: Optional[List[float]] = None
    
class QCFlagModel(BaseModel):
    """Model for QC flags."""
    flag: str
    level: str = Field(..., regex="^(experiment|video|image|embryo|snip)$")
    author: str
    timestamp: datetime
    details: Optional[str] = ""
    
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
```

### Step 4: Create `utils/core/__init__.py`

```python
"""Core utilities for annotation pipeline."""

from .base_annotation_parser import BaseAnnotationParser
from .base_models import (
    BaseAnnotationModel, DetectionModel, 
    SegmentationModel, QCFlagModel, EmbryoMetadataModel
)
from .base_utils import (
    get_timestamp, validate_path, safe_json_load, safe_json_save,
    parse_entity_id, get_parent_ids, generate_gsam_id, get_simplified_filename
)

__all__ = [
    # Base classes
    'BaseAnnotationParser',
    
    # Models
    'BaseAnnotationModel', 'DetectionModel', 
    'SegmentationModel', 'QCFlagModel', 'EmbryoMetadataModel',
    
    # Utilities
    'get_timestamp', 'validate_path', 'safe_json_load', 'safe_json_save',
    'parse_entity_id', 'get_parent_ids', 'generate_gsam_id', 'get_simplified_filename'
]
```

## Testing Checklist

- [ ] Test unified parse_entity_id with all entity types
- [ ] Test auto-detection of entity levels
- [ ] Test with explicit level hints
- [ ] Test get_parent_ids for hierarchy navigation
- [ ] Test get_simplified_filename for storage organization
- [ ] Verify atomic JSON save/load with concurrent access
- [ ] Test backup creation and recovery
- [ ] Validate auto-save functionality
- [ ] Check GSAM ID generation uniqueness
- [ ] Test path validation edge cases
- [ ] Verify pydantic model validation

## Implementation Log

| Date | Developer | Task | Status |
|------|-----------|------|--------|
| TBD | TBD | Create base_utils.py with unified parser | Pending |
| TBD | TBD | Create base_annotation_parser.py | Pending |
| TBD | TBD | Create base_models.py | Pending |
| TBD | TBD | Create __init__.py | Pending |
| TBD | TBD | Unit tests for unified parser | Pending |
| TBD | TBD | Integration tests | Pending |

## Notes for Implementer

1. The unified `parse_entity_id` function is the KEY INNOVATION - test thoroughly
2. Auto-detection works by trying patterns in order from most specific to least
3. Level hints can speed up parsing if you know the expected type
4. All entity IDs follow strict patterns - enforce these in validation
5. The simplified filename system prevents directory explosion during processing
6. GSAM IDs are critical for cross-referencing between annotation types
7. All paths should use pathlib.Path internally

## Usage Examples

```python
# Auto-detection examples
level, data = parse_entity_id("20240411_A01_0042")
# Returns: ('image', {'experiment_id': '20240411', 'well_id': 'A01', ...})

level, data = parse_entity_id("20240411_A01_e01")
# Returns: ('embryo', {'experiment_id': '20240411', 'well_id': 'A01', ...})

# With level hint (faster)
level, data = parse_entity_id("20240411_A01_0042", level='image')

# Get parent hierarchy
parents = get_parent_ids("20240411_A01_e01_s0042")
# Returns: {'experiment_id': '20240411', 'video_id': '20240411_A01', 'embryo_id': '20240411_A01_e01'}

# Simplified filenames for storage
filename = get_simplified_filename("20240411_A01_0042")
# Returns: "0042.jpg" (not the full ID)
```