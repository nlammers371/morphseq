# Module 0: Core Utilities Implementation Guide

## Overview
Create foundation utilities that all other modules will use. Most code exists in `rebase_instructions/module_0/`.

## File Structure
```
segmentation_sandbox/
└── utils/
    ├── parsing_utils.py
    ├── entity_id_tracker.py
    └── base_file_handler.py
└── data_organization/
    └── data_organizer.py
```

## Task 1: Create `utils/parsing_utils.py`

**Source**: Copy from `rebase_instructions/module_0/module_0_1_parsing_utils.py`

**No modifications needed** - Works as-is

**Test checklist**:
```python
# Test backwards parsing with complex experiment ID
from utils.parsing_utils import parse_entity_id, get_entity_type

# Complex ID test
result = parse_entity_id("20250624_chem02_28C_T00_1356_H01_e01_s0034")
assert result['experiment_id'] == '20250624_chem02_28C_T00_1356'
assert result['well_id'] == 'H01'
assert result['snip_id'] == '20250624_chem02_28C_T00_1356_H01_e01_s0034'

# Entity type detection
assert get_entity_type("exp_A01") == "video"
assert get_entity_type("exp_A01_t0042") == "image"
assert get_entity_type("exp_A01_e01") == "embryo"
assert get_entity_type("exp_A01_e01_s0042") == "snip"
```

## Task 2: Create `utils/entity_id_tracker.py`

**Source**: Copy from `rebase_instructions/module_0/module_0_3_entity_id_tracker.py`

**Modifications needed**:
1. Add import at top:
```python
from parsing_utils import parse_entity_id, extract_experiment_id
```

2. In `_add_ids()` method, ensure it handles string IDs

**Test checklist**:
```python
from utils.entity_id_tracker import EntityIDTracker

# Test entity extraction
test_data = {
    "experiments": {"20240411": {"videos": {"20240411_A01": {
        "images": {"20240411_A01_t0000": {
            "embryos": {"20240411_A01_e01": {
                "snip_id": "20240411_A01_e01_s0000"
            }}
        }}
    }}}}
}

entities = EntityIDTracker.extract_entities(test_data)
assert "20240411" in entities["experiments"]
assert "20240411_A01_e01_s0000" in entities["snips"]

# Test hierarchy validation - should pass
EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)

# Test orphan detection - should fail
entities["snips"].add("20240411_A01_e99_s0000")  # e99 doesn't exist
try:
    EntityIDTracker.validate_hierarchy(entities, raise_on_violations=True)
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected
```

## Task 3: Create `utils/base_file_handler.py`

**Source**: Copy from `rebase_instructions/module_0/module_0_4_basefilehandler.py`

**Modifications needed**:
1. Add missing import:
```python
from pathlib import Path
```

**Test checklist**:
```python
from utils.base_file_handler import BaseFileHandler
import json

# Test save with backup
handler = BaseFileHandler("test_file.json")
test_data = {"test": "data"}
handler.save_json(test_data)

# Verify file exists
assert Path("test_file.json").exists()

# Test load
loaded = handler.load_json()
assert loaded["test"] == "data"

# Verify backup creation on second save
handler.save_json({"test": "data2"})
# Check backup exists with timestamp
```

## Task 4: Create `data_organization/data_organizer.py`

**Source**: Implement based on `rebase_instructions/module_0/module_0_2_simplified_dataorganization.md`

**Key implementation details**:

1. **Directory structure**:
   - Input: `raw_stitches/20240411/A01_t0000_ch00_stitch.png`
   - Output: `organized/raw_data_organized/20240411/images/20240411_A01/0000.jpg`

2. **Critical naming convention**:
   - Disk files: `0000.jpg` (no 't' prefix)
   - JSON metadata: `20240411_A01_t0000` (with 't' prefix)

3. **Core methods to implement**:
```python
import cv2
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from utils.base_file_handler import BaseFileHandler

class DataOrganizer(BaseFileHandler):
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def process_experiments(self, source_dir: Path, output_dir: Path, 
                           experiment_names: Optional[List[str]] = None):
        """Main entry point"""
        # Implementation from markdown guide
        
    def find_experiment_directories(self, base_dir: Path) -> List[Path]:
        """Find directories containing stitch files"""
        # Look for dirs with *_stitch.* files
        
    def parse_stitch_filename(self, filename: str) -> Optional[Tuple[str, str]]:
        """Extract well_id and frame from stitch filename"""
        # Parse: A01_t0000_ch00_stitch.png → ('A01', '0000')
        
    def organize_experiment(self, experiment_dir: Path, output_dir: Path, experiment_id: str):
        """Organize one experiment's files"""
        # Group by well, convert images, create videos
        
    def convert_to_jpeg(self, source_path: Path, target_path: Path, quality: int = 90):
        """Convert any image format to JPEG"""
        # Use OpenCV or PIL
        
    def create_video_from_jpegs(self, jpeg_paths: List[Path], video_path: Path, video_id: str):
        """Create MP4 from JPEG sequence"""
        # Use OpenCV VideoWriter
        
    def scan_organized_experiments(self, raw_data_dir: Path) -> Dict:
        """Scan organized structure and create metadata"""
        # CRITICAL: Add 't' prefix when storing in JSON
```

**Test checklist**:
```python
from data_organization.data_organizer import DataOrganizer

# Test with sample data
organizer = DataOrganizer()
organizer.process_experiments(
    source_dir=Path("test_raw_stitches/"),
    output_dir=Path("test_organized/"),
    experiment_names=["20240411"]
)

# Verify output structure
output_base = Path("test_organized/raw_data_organized")
assert (output_base / "20240411/vids/20240411_A01.mp4").exists()
assert (output_base / "20240411/images/20240411_A01/0000.jpg").exists()

# Verify metadata has 't' prefix
metadata_path = output_base / "experiment_metadata.json"
with open(metadata_path) as f:
    metadata = json.load(f)

image_ids = metadata["experiments"]["20240411"]["videos"]["20240411_A01"]["image_ids"]
assert image_ids[0] == "20240411_A01_t0000"  # Has 't' prefix in JSON
```

## Accomplishment Checklist

- [ ] All parsing utilities handle complex experiment IDs
- [ ] Entity tracker validates parent-child relationships
- [ ] File handler performs atomic writes with backups
- [ ] Data organizer creates correct directory structure
- [ ] Image files saved without 't' prefix (0000.jpg)
- [ ] JSON metadata includes 't' prefix (exp_A01_t0000)
- [ ] All tests pass