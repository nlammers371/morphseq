# Implementation Instructions for Developers

## Overview
This document provides step-by-step instructions for implementing the EmbryoMetadata class system. Follow these steps in order to ensure a smooth implementation.

note to store all tasks completed in the implementation log 
at /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/utils/embryo_metada_dev_instruction/implementation_log.md

## Phase 1: Setup and Core Infrastructure

### Step 1.1: Create Project Structure
```bash
morphseq/
├── scripts/
│   └── utils/
|       └── embryo_metada_dev_instruction/ INSTRUCTIONS LIE HERE
│       ├── embryo_metadata.py          # Main class file
│       ├── embryo_metadata_models.py   # Data models
│       ├── embryo_metadata_batch.py    # Batch processing
│       ├── embryo_metadata_integration.py  # Integration layer
│       └── embryo_metadata_utils.py    # Utilities
└── tests/
    └── test_embryo_metadata.py         # Unit tests


```

### Step 1.2: Implement Core Dependencies
First, implement the utilities module (`embryo_metadata_utils.py`):

```python
# embryo_metadata_utils.py
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import re

# Copy all utility functions from Module 8
# Start with basic file operations
def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    # Implementation from Module 8
    pass

def load_json(file_path: Path, create_if_missing: bool = False, 
              default_content: Optional[Dict] = None) -> Dict:
    # Implementation from Module 8
    pass

# Continue with all utilities...
```

### Step 1.3: Implement Data Models
Next, implement the data models (`embryo_metadata_models.py`):

```python
# embryo_metadata_models.py
from typing import TypedDict, Literal, Optional, Dict, List, Union
from datetime import datetime
from dataclasses import dataclass, field

# Copy all model definitions from Module 2 also think about incoporation Module 0 as well as this is a foundational class. 
@dataclass
class AnnotationBase:
    # Implementation from Module 2
    pass

# Continue with all models...
```

### Step 1.4: Implement Core Class Structure
Create the main class file (`embryo_metadata.py`):

```python
# embryo_metadata.py
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .embryo_metadata_models import *
from .embryo_metadata_utils import *

class EmbryoMetadata:
    # Start with __init__ and basic methods from Module 1
    def __init__(self, sam_annotation_path, embryo_metadata_path=None, 
                 gen_if_no_file=False, auto_validate=True, verbose=True):
        # Implementation from Module 1
        pass
```

## Phase 2: Core Functionality

### Step 2.1: Add Phenotype Management
Add phenotype methods to the main class:

```python
# In embryo_metadata.py, add methods from Module 3
def add_phenotype(self, snip_id: str, phenotype: str, author: str, 
                  notes: str = None, confidence: float = None,
                  force_dead: bool = False) -> bool:
    # Implementation from Module 3
    pass

# Continue with all phenotype methods...
```

### Step 2.2: Add Genotype Management
Add genotype methods from Module 4:

```python
# In embryo_metadata.py
def add_genotype(self, embryo_id: str, genotype: str, author: str,
                notes: str = None, confirmed: bool = False, 
                method: str = None, overwrite_genotype: bool = False) -> bool:
    # Implementation from Module 4
    pass

# Continue with all genotype methods...
```

### Step 2.3: Add Flag Management
Add flag methods from Module 5:

```python
# In embryo_metadata.py
def add_flag(self, entity_id: str, flag: str, level: str, author: str,
            notes: str = None, severity: str = "warning",
            auto_generated: bool = False) -> bool:
    # Implementation from Module 5
    pass

# Continue with all flag methods...
```

## Phase 3: Advanced Features

### Step 3.1: Implement Batch Processing
Create the batch processing module (`embryo_metadata_batch.py`):

```python
# embryo_metadata_batch.py
from typing import Union, List, Dict, Optional
from datetime import datetime

# Copy all batch processing classes from Module 6
class RangeParser:
    # Implementation from Module 6
    pass

class BatchProcessor:
    # Implementation from Module 6
    pass

# Continue with all batch classes...
```

### Step 3.2: Implement Integration Layer
Create the integration module (`embryo_metadata_integration.py`):

```python
# embryo_metadata_integration.py
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Copy all integration classes from Module 7
class SamAnnotationIntegration:
    # Implementation from Module 7
    pass

class ConfigurationManager:
    # Implementation from Module 7
    pass

# Continue with all integration classes...
```

### Step 3.3: Update Main Class with Imports
Update the main class to use batch and integration modules:

```python
# In embryo_metadata.py
from .embryo_metadata_batch import RangeParser, BatchProcessor
from .embryo_metadata_integration import SamAnnotationIntegration, ConfigurationManager

# Add batch methods to main class
def batch_add_phenotype(self, embryo_id: str, phenotype: str, 
                       snip_range: Union[str, List[int], List[str]], 
                       author: str, notes: str = None,
                       skip_existing: bool = True) -> Dict:
    # Implementation using RangeParser
    pass
```

## Phase 4: Testing and Validation

### Step 4.1: Create Unit Tests
Create comprehensive tests (`tests/test_embryo_metadata.py`):

```python
import unittest
import tempfile
from pathlib import Path
import json

class TestEmbryoMetadata(unittest.TestCase):
    def setUp(self):
        # Create temporary files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create mock SAM annotation
        self.create_mock_sam_annotation()
        
    def create_mock_sam_annotation(self):
        # Create realistic test data
        sam_data = {
            "experiments": {
                "20240411": {
                    "videos": {
                        "20240411_A01": {
                            "embryo_ids": ["20240411_A01_e01"],
                            "images": {
                                "20240411_A01_0000": {
                                    "embryos": {
                                        "20240411_A01_e01": {
                                            "snip_id": "20240411_A01_e01_0000"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "embryo_ids": ["20240411_A01_e01"],
            "snip_ids": ["20240411_A01_e01_0000"]
        }
        
        self.sam_path = self.temp_path / "sam_annotations.json"
        with open(self.sam_path, 'w') as f:
            json.dump(sam_data, f)
    
    def test_initialization(self):
        # Test creating new metadata
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True
        )
        
        self.assertIsNotNone(em.data)
        self.assertEqual(len(em.data["embryos"]), 1)
    
    def test_phenotype_operations(self):
        # Test adding phenotypes
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True)
        
        success = em.add_phenotype(
            "20240411_A01_e01_0000", 
            "EDEMA", 
            "test_user"
        )
        self.assertTrue(success)
    
    def test_range_parsing(self):
        # Test range syntax
        from embryo_metadata_batch import RangeParser
        
        items = ["item0", "item1", "item2", "item3", "item4"]
        
        # Test single index
        result = RangeParser.parse_range("[2]", items)
        self.assertEqual(result, ["item2"])
        
        # Test range
        result = RangeParser.parse_range("[1:3]", items)
        self.assertEqual(result, ["item1", "item2"])
        
        # Test open-ended range
        result = RangeParser.parse_range("[3::]", items)
        self.assertEqual(result, ["item3", "item4"])
```

### Step 4.2: Create Integration Tests
Test the full pipeline integration:

```python
def test_full_pipeline_integration():
    """Test complete workflow from SAM annotation to phenotyping."""
    
    # 1. Load SAM annotations
    sam_path = Path("test_data/grounded_sam_annotations.json")
    em = EmbryoMetadata(sam_path, "test_embryo_metadata.json", gen_if_no_file=True)
    
    # 2. Add genotypes
    em.batch_add_genotypes({
        "20240411_A01_e01": "wildtype",
        "20240411_A01_e02": "mutant"
    }, author="geneticist")
    
    # 3. Add phenotypes with range
    em.batch_add_phenotype(
        "20240411_A01_e01",
        "EDEMA",
        "[10:20]",  # Frames 10-19
        author="phenotyper"
    )
    
    # 4. Add flags
    em.add_flag(
        "20240411_A01_0015",
        "MOTION_BLUR",
        "image",
        author="qc_system"
    )
    
    # 5. Save and verify
    em.save()
    
    # 6. Reload and check
    em2 = EmbryoMetadata(sam_path, "test_embryo_metadata.json")
    assert em2.data["embryos"]["20240411_A01_e01"]["genotype"]["value"] == "wildtype"
```

## Phase 5: GroundedSamAnnotation Modifications

### Step 5.1: Modify Existing Class
Follow the modification guide to update `sam2_utils.py`:

```python
# In sam2_utils.py, add to GroundedSamAnnotations class
def _initialize_gsam_id(self):
    """Initialize or retrieve GSAM annotation ID."""
    # Implementation from modification guide
    pass

def get_gsam_id(self) -> int:
    """Get the GSAM annotation ID."""
    # Implementation from modification guide
    pass
```

### Step 5.2: Test Integration
Test the bidirectional linking:

```python
def test_sam_embryo_linking():
    # Create SAM annotation with ID
    gsam = GroundedSamAnnotations("sam_ann.json", ...)
    gsam_id = gsam.get_gsam_id()
    
    # Create EmbryoMetadata linked to it
    em = EmbryoMetadata("sam_ann.json", "embryo_meta.json", gen_if_no_file=True)
    
    # Verify linking
    assert em.data["file_info"]["gsam_annotation_id"] == gsam_id
```

## Phase 6: Performance Optimization

### Step 6.1: Profile Critical Operations
```python
from embryo_metadata_utils import PerformanceUtils

# Profile batch operations
result, time_taken = PerformanceUtils.profile_operation(
    em.batch_add_phenotypes,
    large_batch_data,
    author="batch_test"
)
print(f"Batch operation took {time_taken:.2f} seconds")
```

### Step 6.2: Implement Caching
Add caching for frequently accessed data:

```python
# In embryo_metadata.py
def __init__(self, ...):
    # ... existing init ...
    self._embryo_cache = {}
    self._snip_to_embryo_cache = {}

def _get_embryo_id_from_snip(self, snip_id: str) -> Optional[str]:
    # Check cache first
    if snip_id in self._snip_to_embryo_cache:
        return self._snip_to_embryo_cache[snip_id]
    
    # ... existing lookup logic ...
    
    # Cache result
    self._snip_to_embryo_cache[snip_id] = embryo_id
    return embryo_id
```

## Phase 7: Documentation and Deployment

### Step 7.1: Generate API Documentation
Use docstrings for automatic documentation:

```python
def generate_api_docs():
    """Generate API documentation from docstrings."""
    import pydoc
    
    # Generate HTML documentation
    pydoc.writedoc('embryo_metadata')
    pydoc.writedoc('embryo_metadata_models')
    pydoc.writedoc('embryo_metadata_batch')
```

### Step 7.2: Create User Examples
Create example scripts for common workflows:

```python
# examples/basic_usage.py
from embryo_metadata import EmbryoMetadata

# Example 1: Initialize and add data
em = EmbryoMetadata(
    "path/to/sam_annotations.json",
    "path/to/embryo_metadata.json",
    gen_if_no_file=True
)

# Example 2: Batch phenotyping
em.batch_add_phenotype(
    "20240411_A01_e01",
    "CONVERGENCE_EXTENSION",
    "[23::]",  # From frame 23 onwards
    author="researcher1"
)

# Example 3: Export for analysis
em.export_phenotypes("phenotypes.csv", format="csv")
```

## Implementation Checklist

- [ ] **Phase 1: Setup**
  - [ ] Create project structure
  - [ ] Implement utilities module
  - [ ] Implement data models
  - [ ] Create core class structure

- [ ] **Phase 2: Core Features**
  - [ ] Add phenotype management
  - [ ] Add genotype management
  - [ ] Add flag management

- [ ] **Phase 3: Advanced Features**
  - [ ] Implement batch processing
  - [ ] Implement integration layer
  - [ ] Connect modules together

- [ ] **Phase 4: Testing**
  - [ ] Create unit tests
  - [ ] Create integration tests
  - [ ] Test edge cases

- [ ] **Phase 5: Integration**
  - [ ] Modify GroundedSamAnnotations
  - [ ] Test bidirectional linking
  - [ ] Verify data flow

- [ ] **Phase 6: Optimization**
  - [ ] Profile performance
  - [ ] Add caching
  - [ ] Optimize batch operations

- [ ] **Phase 7: Documentation**
  - [ ] Generate API docs
  - [ ] Create examples
  - [ ] Write deployment guide

## Common Pitfalls to Avoid

1. **Not validating SAM annotation structure** - Always check required fields exist
2. **Forgetting to save** - Remember that changes are local until saved
3. **Ignoring the DEAD phenotype rule** - Implement exclusivity checking
4. **Not handling missing genotypes** - Always show warnings for missing data
5. **Inefficient batch operations** - Use the BatchProcessor for large datasets
6. **Not backing up before major operations** - Always create backups
7. **Hardcoding paths** - Use configuration files for paths

## Debugging Tips

1. **Enable verbose mode** - Set `verbose=True` for detailed output
2. **Check consistency** - Run `em._run_consistency_checks()` after issues
3. **Examine change log** - Use `em._change_log` to see what was modified
4. **Validate IDs** - Use `IdParser.parse_id()` to check ID formats
5. **Test with small data** - Start with a few embryos before scaling up

## Next Steps

After implementation:
1. Integrate with existing MorphSeq pipeline
2. Train team on usage
3. Set up automated testing
4. Create data migration scripts for existing annotations
5. Establish backup and recovery procedures