# Embryo Segmentation Pipeline - Rebasing Developer Instructions

## 🎯 Overview

This document outlines the complete file organization, development practices, and implementation guidelines for the refactored embryo segmentation pipeline. The pipeline has been redesigned with modularity, extensibility, and maintainability as core principles.

## 📚 Implementation Roadmap

Follow this implementation order strictly - each module builds on the previous:

### Phase 1: Foundation (Week 1)
1. **Module 1 - Core Foundation** *(2-3 days)*
   - Implement `BaseAnnotationParser` base class
   - Create unified ID parsing functions
   - Set up backup and atomic save systems
   - Test thoroughly - everything depends on this!

### Phase 2: Data Management (Week 2)
2. **Module 2 - Metadata System** *(3-4 days)*
   - Refactor `ExperimentMetadata` with integrated **image integrity QC only**
   - Implement `EmbryoMetadata` with ALL fields (phenotype, genotype, treatment, flags)
   - Integrate **embryo integrity QC only** in EmbryoMetadata (GSAM/embryo ID dependent)
   - Ensure backward compatibility
   - Test migration from old formats

### Phase 3: Processing Pipeline (Week 3)
3. **Module 3 - Annotation Pipeline** *(4-5 days)*
   - Rename and refactor `grounded_sam_utils` → `gdino_annotations`
   - **CRITICAL**: Fix zero-detection tracking bug - ensure ALL images are recorded in annotations, even with no detections
   - Create unified `GroundedSam` pipeline class
   - Implement SAM2 annotation management with full image tracking
   - Test end-to-end detection → segmentation flow
   - Verify zero-detection images are properly recorded and visible downstream

4. **Module 4 - Mask Export & Embryo QC** *(3-4 days)*
   - Implement incremental mask export system (`EmbryoMaskExporter`)
   - Add mask export manifest tracking with GSAM annotation ID linkage
   - Implement comprehensive **embryo integrity QC** (mask overlap, annotation completeness, GSAM-dependent)
   - Add overlap detection during mask export with QC flag integration
   - Test mask export validation and QC flag propagation

### Phase 4: Quality & Visualization (Week 4)
5. **Module 5 - Image QC Integration** *(2 days)*
   - Create unified QC flag system with **strict separation** between image and embryo QC
   - Implement **image integrity QC** analyzers (Step 02 and before - focus, exposure, corruption)
   - Test multi-level flag propagation with QC type separation

6. **Module 6 - Visualization System** *(3-4 days)*
   - Implement 8-zone modular layout
   - Create extensible data provider system
   - Build overlay and rendering engines
   - Test with various data combinations

### Phase 5: Integration (Week 5)
7. **Module 7 - Script Updates** *(2-3 days)*
   - Update pipeline scripts to use new utilities
   - Validate mask export script (`05_export_embryo_masks.py`) integration
   - Maintain CLI compatibility
   - Add visualization options
   - Test full pipeline including mask export with real data

8. **Module 8 - Global Config & Testing** *(1-2 days)*
   - Create global configuration system
   - Write comprehensive test suite
   - Set up continuous testing
   - Document all changes

## 🚦 Development Flow

```
START → Module 1 (Core) → Test Core → Module 2 (Metadata) → Test Metadata
  ↓                                                              ↓
Module 8 ← Test Scripts ← Module 7 ← Test Viz ← Module 6 ← Module 5 ← Module 4 ← Module 3
(Config)    (Scripts)              (Viz)        (Image QC)  (Mask Export)  (Annotation)
```

### Critical Checkpoints:
- ✅ After Module 1: All ID parsing must work perfectly
- ✅ After Module 2: Old data must migrate seamlessly
- ✅ After Module 3: Detection/segmentation pipeline functional + **ALL images tracked (including zero-detection)**
- ✅ After Module 4: Mask export functional + embryo QC integrated + overlap detection working
- ✅ After Module 5: Image QC flags properly integrated with strict separation
- ✅ After Module 6: Visualization shows all data correctly
- ✅ After Module 7: Scripts work with existing data and mask export validated
- ✅ After Module 8: All tests pass, ready for production

## ⚠️ Before You Start

1. **Read ALL module instructions first** - Understand the full scope
2. **Set up your development environment** - Python 3.8+, all dependencies
3. **Create a test dataset** - Small subset for quick testing
4. **Initialize git branch** - Work on feature branch, not main
5. **Set up logging from day 1** - Use centralized logger

## 🚦 QC Separation Principles (CRITICAL)

**Image Integrity QC (Step 02 and before):**
- Assesses raw image quality (focus, exposure, corruption, artifacts)
- Performed before any annotation or mask generation
- Independent of GSAM/embryo IDs
- Integrated with ExperimentMetadata only
- Results stored and propagated separately

**Embryo Integrity QC (Step 03 onwards):**
- Assesses segmentation/mask quality, overlaps, annotation completeness
- Performed after annotation/mask generation
- Dependent on GSAM annotation IDs and embryo IDs
- Integrated with EmbryoMetadata only
- Includes mask overlap detection, missing annotation tracking, mask validation

**DO NOT MIX IMAGE QC AND EMBRYO QC** in code, data, or reporting.

## 🧬 Embryo QC Functions (Module 4 - CRITICAL IMPLEMENTATION)

**All embryo QC functions must be implemented in Module 4. These are GSAM/embryo ID dependent and performed after annotation/mask generation:**

### Core Embryo Integrity QC Functions:
1. **`check_mask_overlap_detection()`** - Detect pixel overlaps between embryo masks during export
2. **`validate_annotation_completeness()`** - Ensure all GSAM annotations have corresponding masks
3. **`check_mask_boundary_integrity()`** - Validate mask boundaries are closed and complete
4. **`detect_mask_fragmentation()`** - Identify severely fragmented masks
5. **`validate_mask_size_consistency()`** - Check for abnormally small/large masks
6. **`check_temporal_mask_continuity()`** - Validate mask consistency across video frames
7. **`detect_mask_displacement_anomalies()`** - Identify sudden position jumps in tracking
8. **`validate_gsam_annotation_linkage()`** - Ensure proper GSAM ID → mask mapping
9. **`check_mask_export_consistency()`** - Validate exported masks match annotation data
10. **`detect_missing_embryo_annotations()`** - Find embryos missing from annotation files

### Overlap-Specific QC (Integrated with Mask Export):
1. **`calculate_overlap_percentages()`** - Quantify overlap severity between embryo pairs
2. **`flag_critical_overlaps()`** - Mark overlaps above threshold as QC failures
3. **`track_overlap_resolution()`** - Monitor how overlaps are resolved during export
4. **`validate_overlap_assignment()`** - Ensure overlapping pixels assigned correctly

### GSAM-Dependent Validation:
1. **`validate_gsam_id_uniqueness()`** - Ensure GSAM IDs are unique within experiment
2. **`check_gsam_metadata_consistency()`** - Validate GSAM annotation metadata completeness
3. **`verify_gsam_temporal_ordering()`** - Check GSAM annotations maintain temporal sequence
4. **`validate_gsam_export_manifest()`** - Ensure export manifest matches current GSAM data

### Integration Requirements:
- All functions must integrate with `EmbryoMetadata` QC flags
- Must NOT overlap with image integrity QC functions
- Results stored separately from image QC
- Must work with incremental mask export system
- Should trigger appropriate QC flag severity levels

## 🛠️ Development Rules

1. **Log everything** - Use centralized logging, not print statements
2. **Create backups** - Every save operation should backup first
3. **Document as you go** - Update module logs after each session
4. **Test at module boundaries** - Run the module tests before moving on
5. **Maintain backward compatibility** - Existing data must still work

## 📂 File Organization

### Utils Directory Structure

```
utils/
├── core/                               # Foundation classes
│   ├── __init__.py
│   ├── base_annotation_parser.py      # Base class for ALL parsers
│   ├── base_models.py                 # Pydantic validation models
│   └── base_utils.py                  # Common utilities
│
├── metadata/                           # Metadata management
│   ├── __init__.py
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── experiment_metadata.py     # Video/image organization
│   │   ├── experiment_qc.py           # Integrated QC flags
│   │   └── experiment_utils.py        # Path helpers, metadata ops
│   └── embryo/
│       ├── __init__.py
│       ├── embryo_metadata.py         # Phenotype/genotype tracking
│       ├── embryo_managers.py         # Specialized managers
│       └── embryo_batch.py            # Batch operations
│
├── annotation/                         # Detection & segmentation
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── gdino_annotations.py       # GDINO annotation management
│   │   └── gdino_inference.py         # GDINO model inference
│   ├── grounded_sam/
│   │   ├── __init__.py
│   │   ├── gsam_pipeline.py           # All-in-one GroundedSam class
│   │   └── gsam_utils.py              # Helper functions
│   └── segmentation/
│       ├── __init__.py
│       ├── sam2_annotations.py        # SAM2 annotation management
│       ├── sam2_predictor.py          # SAM2 video prediction
│       └── mask_utils.py              # Mask operations (RLE, polygon)
│
├── visualization/                      # Comprehensive viz system
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── layout_manager.py          # 8-zone modular layouts
│   │   ├── data_providers.py          # Extensible data sources
│   │   ├── frame_data.py              # Image-level aggregation
│   │   ├── zone_renderer.py           # Zone rendering engine
│   │   └── color_schemes.py           # Visual themes
│   ├── components/
│   │   ├── __init__.py
│   │   ├── info_panels.py             # Information displays
│   │   ├── annotation_layers.py       # Detection/mask overlays
│   │   └── comparison_views.py        # Side-by-side views
│   └── renderers/
│       ├── __init__.py
│       ├── video_renderer.py          # MP4 generation
│       ├── frame_renderer.py          # Single frame rendering
│       └── grid_renderer.py           # Multi-video grids
│
├── qc/                                 # Quality control
│   ├── __init__.py
│   ├── qc_flags.py                    # Flag definitions & validation (both QC types)
│   ├── image_qc.py                    # Image integrity QC (Step 02 and before)
│   └── embryo_qc.py                   # Embryo integrity QC (Step 03 onwards, GSAM-dependent)
│
├── io/                                 # Input/Output utilities
│   ├── __init__.py
│   ├── naming_utils.py                # ID parsing, path resolution
│   ├── video_utils.py                 # Video read/write operations
│   └── mask_export_utils.py           # Incremental mask export with manifest tracking
│
├── config/                             # Configuration management
│   ├── __init__.py
│   ├── global_config.py               # Singleton config manager
│   └── config_schema.py               # Config validation
│
└── logging/                            # Centralized logging
    ├── __init__.py
    ├── pipeline_logger.py              # Main logger setup
    └── log_aggregator.py               # Log collection/analysis
```

### Scripts Directory

```
scripts/
├── 01_prepare_videos.py               # Initial data preparation
├── 02_image_quality_qc.py             # Image integrity QC (Step 02 - independent of annotation)
├── 03_gdino_detection.py              # Object detection
├── 04_sam2_video_processing.py        # Segmentation & tracking + embryo integrity QC
├── 05_export_embryo_masks.py          # Mask export & metadata + embryo integrity QC
├── initialize_pipeline.py             # Pipeline setup & testing
└── utils/                             # Script-specific helpers
    └── embryo_metada_dev_instruction/ # Development modules
```

### Tests Directory

```
tests/
├── test_module_1_core.py              # Core functionality tests
├── test_module_2_metadata.py          # Metadata system tests
├── test_module_3_annotation.py        # Annotation pipeline tests
├── test_module_4_qc.py                # QC system tests
├── test_module_5_viz.py               # Visualization tests
├── test_module_6_scripts.py           # Script integration tests
├── test_integration.py                # End-to-end tests
└── test_data/                         # Sample test data
```

## 🔧 Development Practices

### 1. Centralized Logging

All modules must use the centralized logging system:

```python
# utils/logging/pipeline_logger.py
import logging
from pathlib import Path
from datetime import datetime

class PipelineLogger:
    """Centralized logging for all pipeline components."""
    
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._setup_logger()
        return cls._instance
    
    @classmethod
    def _setup_logger(cls):
        """Configure centralized logger."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger("embryo_pipeline")
        logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        cls._logger = logger
    
    @classmethod
    def get_logger(cls, module_name):
        """Get logger for specific module."""
        if cls._logger is None:
            cls._setup_logger()
        return cls._logger.getChild(module_name)

# Usage in any module:
from utils.logging import PipelineLogger
logger = PipelineLogger.get_logger(__name__)

logger.info("Processing started")
logger.debug(f"Processing image: {image_id}")
logger.error(f"Failed to process: {error}")
```

### 2. Intelligent File Parsing

All file operations must handle complex experiment IDs correctly:

```python
# Key parsing functions in utils/core/base_utils.py

def parse_image_id(image_id: str) -> Dict[str, str]:
    """
    Parse complex image IDs correctly.
    Example: "20250622_chem_35C_T01_1605_H09_0000"
    """
    parts = image_id.rsplit('_', 2)  # Split from right
    # Returns: {
    #   'experiment_id': '20250622_chem_35C_T01_1605',
    #   'well_id': 'H09',
    #   'frame_number': '0000',
    #   'video_id': '20250622_chem_35C_T01_1605_H09'
    # }

# Recursive experiment discovery
def find_experiments(base_dir: Path) -> List[Path]:
    """Recursively find experiment directories."""
    experiments = []
    for path in base_dir.rglob("*_stitch.png"):
        exp_dir = path.parent
        if exp_dir not in experiments:
            experiments.append(exp_dir)
    return experiments
```

### 3. Testing at Key Points

Implement progressive testing after each module:

```python
# tests/test_integration.py
def test_module_progression():
    """Test each module builds on previous."""
    
    # Module 1: Core
    assert test_id_parsing(), "Core parsing failed"
    assert test_backup_creation(), "Backup system failed"
    
    # Module 2: Metadata (depends on Core)
    assert test_experiment_metadata(), "Experiment metadata failed"
    assert test_embryo_metadata_fields(), "Embryo metadata incomplete"
    
    # Module 3: Annotation (depends on Core + Metadata)
    assert test_gdino_annotations(), "GDINO annotations failed"
    assert test_gsam_pipeline(), "GroundedSam pipeline failed"
    
    # Continue for each module...
```

### 4. Key Testing Checkpoints

```python
# Critical functions to test at each stage:

CRITICAL_TESTS = {
    "Module 1": [
        "parse_image_id with complex IDs",
        "BaseAnnotationParser backup creation",
        "Atomic JSON save/load",
        "GSAM ID generation"
    ],
    "Module 2": [
        "ExperimentMetadata image integrity QC integration only",
        "EmbryoMetadata has ALL fields (treatment!) + embryo integrity QC only",
        "Backward compatibility migration",
        "Recursive experiment finding",
        "QC type separation maintained"
    ],
    "Module 3": [
        "GDINO high-quality filtering",
        "GroundedSam full pipeline",
        "Seed frame selection", 
        "Bidirectional propagation",
        "Zero-detection image tracking (CRITICAL BUG FIX)",
        "All images recorded in annotations even with no detections"
    ],
    "Module 4": [
        "Incremental mask export with manifest tracking",
        "GSAM annotation ID linkage in export manifest",
        "Mask export validation and consistency checks",
        "Embryo mask overlap detection during export",
        "Comprehensive embryo integrity QC system",
        "Mask overlap QC flags",
        "Annotation completeness validation",
        "GSAM-dependent QC checks",
        "QC flag integration with mask export"
    ],
    "Module 5": [
        "QC flag validation (both image and embryo QC types)",
        "Image integrity QC system (Step 02 and before)",
        "QC type separation in code, data, and reporting", 
        "Severity levels",
        "Integration with metadata (image QC → ExperimentMetadata, embryo QC → EmbryoMetadata)"
    ],
    "Module 6": [
        "8-zone layout system",
        "Custom metric registration",
        "Image-level data aggregation",
        "Zone plug-and-play"
    ]
}
```

## 🚀 Implementation Guidelines

### 1. Module Implementation Order

1. **Start with Core (Module 1)**
   - Implement and test thoroughly
   - All other modules depend on this

2. **Metadata System (Module 2)**
   - Ensure backward compatibility
   - Test QC integration

3. **Annotation Pipeline (Module 3)**
   - Test with sample data
   - Verify ID parsing

4. **QC Integration (Module 4)**
   - Must work before visualization
   - Test flag propagation

5. **Visualization (Module 5)**
   - Depends on QC being ready
   - Test zone system

6. **Script Updates (Module 7)**
   - Update incrementally
   - Maintain CLI compatibility

### 2. Data Flow

```
Raw Images → ExperimentMetadata → Image Integrity QC → GDINO Detection 
    ↓              ↓                 ↓                      ↓
  Videos     Track Everything   Flag Image Issues    Find Embryos (ALL images tracked)
    ↓              ↓                 ↓                      ↓
SAM2 Segmentation → Mask Export → Embryo Integrity QC → EmbryoMetadata → Visualization → Analysis
                         ↓              ↓                    ↓
                 Overlap Detection  GSAM Validation   QC Flag Integration
```

**Critical Flow Notes:**
- **Zero-detection tracking**: GDINO must record ALL images, even those with no detections
- **Mask export with QC**: Overlap detection happens during export, flags propagated to EmbryoMetadata
- **GSAM linkage**: Export manifest maintains version links to GSAM annotations

**QC Separation:**
- **Image Integrity QC**: Applied to ExperimentMetadata (focus, exposure, corruption)
- **Embryo Integrity QC**: Applied to EmbryoMetadata (mask overlap, annotation completeness, GSAM-dependent)

### 3. Key Design Principles

1. **Everything inherits from BaseAnnotationParser**
   - Consistent save/load
   - Automatic backups
   - Change tracking
   - GSAM ID management

2. **Unified ID parsing**
   - Handle complex experiment IDs
   - Parse from right with rsplit()
   - Validate components

3. **Separated QC System**
   - **Image Integrity QC** at experiment/video/image levels (Step 02 and before)
   - **Embryo Integrity QC** at embryo/mask levels (Step 03 onwards, GSAM-dependent)
   - No mixing of QC types in code, data, or reporting
   - Single source of truth for each QC type
   - Accessible to visualization with type separation

4. **Modular visualization**
   - 8-zone system
   - Plug-and-play content
   - Extensible for future metrics

### 4. Error Handling

```python
# Standard error handling pattern
try:
    result = process_function(data)
    logger.info(f"Successfully processed {data_id}")
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Create backup before failing
    backup_path = create_emergency_backup(data)
    logger.info(f"Emergency backup created: {backup_path}")
    raise
```

### 5. Performance Considerations

1. **Batch Processing**
   - Use ThreadPoolExecutor for parallel ops
   - Auto-save at intervals
   - Progress tracking

2. **Caching**
   - Image-level data aggregation
   - Lazy model loading
   - Metadata caching

3. **Memory Management**
   - Stream video processing
   - RLE mask compression
   - Incremental saves

## 📋 Checklist for Developers

- [ ] Set up centralized logging in your module
- [ ] Use proper ID parsing (rsplit for complex IDs)
- [ ] Implement backup functionality
- [ ] Add progress tracking for long operations
- [ ] Test after implementing each major function
- [ ] Document all public APIs
- [ ] Handle backward compatibility
- [ ] Add to critical tests list
- [ ] Update integration tests
- [ ] **Maintain strict QC type separation (image vs embryo)**
- [ ] **Ensure image QC goes to ExperimentMetadata only**
- [ ] **Ensure embryo QC goes to EmbryoMetadata only**

## 🔍 Debugging Tips

1. **Check logs first** - All operations logged centrally
2. **Verify ID parsing** - Most issues from incorrect parsing
3. **Test with simple data** - Use test fixtures
4. **Check file permissions** - Atomic saves need write access
5. **Validate against schema** - Use pydantic models

## 📝 Final Notes

- This refactoring maintains backward compatibility
- All JSON formats unchanged (only internal structure improved)
- Scripts maintain same CLI interface
- Performance should be equal or better
- Code reuse reduces maintenance burden

Remember: **Test early, test often, log everything!**