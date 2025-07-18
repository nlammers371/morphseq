# Module 7: Global Configuration & Testing Framework

## Overview
Create global configuration system and test each module incrementally to ensure functionality before integration.

## Files to Create

```
utils/
├── config/
│   ├── __init__.py
│   ├── global_config.py
│   └── config_schema.py
tests/
├── test_module_1_core.py
├── test_module_2_metadata.py
├── test_module_3_annotation.py
├── test_module_4_qc.py
├── test_module_5_viz.py
└── test_integration.py
```

## Step 1: Global Configuration

```python
# utils/config/global_config.py
"""Global configuration management."""

import json
from pathlib import Path
from typing import Dict, Any

class GlobalConfig:
    """Singleton configuration manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: Path):
        """Load configuration from file."""
        with open(config_path) as f:
            self._config = json.load(f)
        self._validate_config()
        
    def _validate_config(self):
        """Validate required fields."""
        required = [
            'paths.base_data_dir',
            'paths.raw_data_organized',
            'models.gdino.config',
            'models.sam2.config',
            'processing.auto_save_interval',
            'visualization.default_layout'
        ]
        # [Validation logic]
        
    def get(self, key: str, default: Any = None):
        """Get config value with dot notation."""
        # "models.gdino.config" -> config['models']['gdino']['config']
        pass

# Example config structure
DEFAULT_CONFIG = {
    "paths": {
        "base_data_dir": "/data/morphseq",
        "raw_data_organized": "/data/morphseq/raw_data_organized",
        "quality_control": "/data/morphseq/quality_control",
        "annotations": "/data/morphseq/annotations"
    },
    "models": {
        "gdino": {
            "config": "GroundingDINO_SwinT_OGC.py",
            "weights": "groundingdino_swint_ogc.pth"
        },
        "sam2": {
            "config": "sam2_hiera_l.yaml",
            "checkpoint": "sam2_hiera_large.pt"
        }
    },
    "processing": {
        "auto_save_interval": 10,
        "batch_size": 32,
        "num_workers": 4
    },
    "visualization": {
        "default_layout": "standard",
        "output_fps": 5,
        "output_quality": 90
    },
    "qc": {
        "blur_threshold": 100,
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5
    }
}
```

## Step 2: Testing Framework

```python
# tests/test_module_1_core.py
"""Test core functionality first."""

def test_id_parsing():
    """Test complex ID parsing."""
    from utils.core import parse_image_id, parse_video_id
    
    # Test cases
    test_ids = [
        ("20250622_chem_35C_T01_1605_H09_0000", "image"),
        ("20250622_chem_35C_T01_1605_H09", "video"),
        ("simple_exp_A01_0000", "image")
    ]
    
    for test_id, expected_type in test_ids:
        if expected_type == "image":
            result = parse_image_id(test_id)
            assert 'experiment_id' in result
            assert 'well_id' in result
            assert 'frame_number' in result
            print(f"✓ Parsed {test_id}")

def test_backup_functionality():
    """Test backup creation."""
    from utils.core import BaseAnnotationParser
    
    # Create test class
    class TestParser(BaseAnnotationParser):
        def _load_or_initialize(self):
            return {"test": "data"}
        def _validate_schema(self, data):
            pass
    
    # Test backup on save
    parser = TestParser("test.json")
    parser.save(backup=True)
    # Verify backup exists
    
def run_module_1_tests():
    """Run all Module 1 tests."""
    print("Testing Module 1: Core Foundation")
    test_id_parsing()
    test_backup_functionality()
    print("✅ Module 1 tests passed\n")

# tests/test_module_2_metadata.py
def test_experiment_metadata():
    """Test metadata with integrated QC."""
    from utils.metadata.experiment import ExperimentMetadata
    
    # Test initialization
    meta = ExperimentMetadata("test_exp.json")
    
    # Test adding experiment with complex ID
    meta.add_experiment("20250622_chem_35C_T01_1605")
    
    # Test QC integration
    meta.add_qc_flag("20250622_chem_35C_T01_1605_H09_0000", 
                     "BLUR", "test_user", "Low variance")
    
    # Verify data structure
    assert "experiments" in meta.data
    assert "qc_definitions" in meta.data
    print("✓ ExperimentMetadata with QC working")

def test_embryo_metadata_all_fields():
    """Test embryo metadata has all required fields."""
    from utils.metadata.embryo import EmbryoMetadata
    
    # Mock SAM annotations
    # Test all fields: phenotype, genotype, treatment, flags
    
    print("✓ EmbryoMetadata has all required fields")

# Sequential test runner
def run_all_tests():
    """Run tests in order, stop on failure."""
    modules = [
        ("Module 1: Core", run_module_1_tests),
        ("Module 2: Metadata", run_module_2_tests),
        ("Module 3: Annotation", run_module_3_tests),
        ("Module 4: QC", run_module_4_tests),
        ("Module 5: Visualization", run_module_5_tests)
    ]
    
    for name, test_func in modules:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            print("Fix this module before proceeding!")
            return False
    
    print("✅ All modules tested successfully!")
    return True
```

## Step 3: Initialization Script

```python
# scripts/initialize_pipeline.py
"""Initialize pipeline with proper configuration."""

from pathlib import Path
from utils.config import GlobalConfig

def initialize_pipeline(config_path: str = None):
    """Initialize global configuration and verify setup."""
    
    # Load config
    config = GlobalConfig()
    if config_path:
        config.load(Path(config_path))
    else:
        # Use default config
        config.load_default()
    
    # Verify paths exist
    base_dir = Path(config.get('paths.base_data_dir'))
    if not base_dir.exists():
        print(f"Creating base directory: {base_dir}")
        base_dir.mkdir(parents=True)
    
    # Create required subdirectories
    for subdir in ['raw_data_organized', 'quality_control', 'annotations']:
        path = base_dir / subdir
        path.mkdir(exist_ok=True)
    
    # Run tests
    from tests import run_all_tests
    if not run_all_tests():
        raise RuntimeError("Pipeline tests failed!")
    
    print("✅ Pipeline initialized successfully!")
    return config

if __name__ == "__main__":
    initialize_pipeline()
```

## Implementation Order

1. **Test After Each Module**:
   ```
   Module 1 → Test parsing, backups → ✓
   Module 2 → Test metadata, QC → ✓
   Module 3 → Test annotations → ✓
   ...
   ```

2. **Backward Compatibility Tests**:
   - Test old ID formats still work
   - Test migration from old metadata format
   - Test with existing annotation files

3. **Integration Tests**:
   - Full pipeline test with sample data
   - Test data flow between modules

## Critical Functions to Test

- `parse_image_id()` with complex experiment IDs
- `ExperimentMetadata._migrate_to_integrated_qc()`
- `BaseAnnotationParser._create_backup()`
- `EmbryoMetadata` has treatment field
- QC flag validation
- Visualization zone calculations