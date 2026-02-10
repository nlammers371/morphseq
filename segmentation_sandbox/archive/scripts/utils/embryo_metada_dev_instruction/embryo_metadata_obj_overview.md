# EmbryoMetadata System: File Structure & Navigation

## Directory Overview

```
segmentation_sandbox/scripts/utils/
├── base_annotation_parser.py                # Foundational class for all annotation objects
├── embryo_metada_dev_instruction/
│   ├── embryo_metadata_refactored.py        # Core EmbryoMetadata class (uses mixins)
│   ├── embryo_phenotype_manager.py          # Phenotype manager mixin
│   ├── embryo_genotype_manager.py           # Genotype manager mixin
│   ├── embryo_flag_manager.py               # Flag manager mixin
│   ├── embryo_treatment_manager.py          # Treatment manager mixin
│   ├── embryo_metadata_batch.py             # Batch processing engine
│   ├── embryo_metadata_integration.py       # Integration layer (SAM, GSAM, config)
│   ├── embryo_metadata_models.py            # Data models & validation
│   ├── embryo_metadata_utils.py             # Utilities & helpers
│   ├── permitted_values_manager.py          # Permitted values schema manager
│   ├── add_permitted_value.py               # CLI utility for schema extension
│   ├── sam2_utils.py                        # SAM2 annotation helpers (GSAM ID, etc.)
│   ├── inspect_data_structure.py            # Data inspection/debugging
│   ├── debug_batch.py                       # Debugging batch operations
│   ├── debug_batch_operations.py            # Debugging batch operations
│   ├── demo_base_parser_pipeline.py         # Demo pipeline for base parser
│   ├── config/                              # Permitted values schema JSON, configs
│   ├── archive_embryo_metadata_dev/         # Archive: logs, tests, module docs
│   │   ├── test_files/                      # All test scripts
│   │   ├── logs/                            # Implementation logs
│   │   ├── module_docs/                     # Module documentation
│   │   └── debug_scripts/                   # Debug scripts
│   ├── module_0_base_annotation_parser.md   # Module documentation
│   ├── module_1_embryometadata_core_class_structure.md
│   ├── module_2_data_models_and_validation.md
│   ├── module_3_phenotype_management.md
│   ├── module_4_genotype_management.md
│   ├── module_5_multi_level_flag_management.md
│   ├── module_6_batch_processing_engine.md
│   ├── module_7_integration_layer.md
│   ├── module_8_utilities_and_helpers.md
│   ├── permitted_values_schema.md           # Permitted values schema documentation
```

## File Summaries

### Top-Level
- **base_annotation_parser.py**: Foundational class for ID parsing, file I/O, change tracking, and entity hierarchy. Used by all annotation classes.

### Core EmbryoMetadata System
- **embryo_metadata_refactored.py**: Main EmbryoMetadata class, uses mixins for modularity.
- **embryo_phenotype_manager.py**: Handles phenotype CRUD, validation, and snip-level operations.
- **embryo_genotype_manager.py**: Manages genotype CRUD, single genotype enforcement.
- **embryo_flag_manager.py**: Manages flag CRUD, multi-level flagging.
- **embryo_treatment_manager.py**: Manages treatments, multi-treatment support, warnings.
- **embryo_metadata_batch.py**: Batch processing engine for large-scale annotation, parallelism, range parsing.
- **embryo_metadata_integration.py**: Integration layer for linking EmbryoMetadata with SAM/GSAM annotations, config inheritance.
- **embryo_metadata_models.py**: Data models (Phenotype, Genotype, Flag, Treatment, AnnotationBase) and validation logic.
- **embryo_metadata_utils.py**: Utilities for file operations, path validation, data conversion, export, performance, and more.
- **permitted_values_manager.py**: Manages permitted values schema, validation, and schema extension.
- **add_permitted_value.py**: CLI utility to add new permitted values to schema.
- **sam2_utils.py**: Helper functions for SAM2 annotation files, GSAM ID management, and annotation linking.
- **inspect_data_structure.py**: Debugging and inspection of annotation data structures.
- **debug_batch.py / debug_batch_operations.py**: Debugging scripts for batch operations.
- **demo_base_parser_pipeline.py**: Demo pipeline showing usage of base annotation parser.

### Config & Documentation
- **config/**: Contains permitted values schema JSON and other config files.
- **module_X_*.md**: Documentation for each module (architecture, usage, design).
- **permitted_values_schema.md**: Documentation for permitted values schema.

### Archive
- **archive_embryo_metadata_dev/**: Contains logs, test scripts, module docs, and debug scripts.
  - **test_files/**: All test scripts for core, batch, integration, and edge cases.
  - **logs/**: Implementation logs and progress tracking.
  - **module_docs/**: Markdown documentation for each module.
  - **debug_scripts/**: Debugging scripts and utilities.

## Quick Usage Example

### Basic Usage
```python
from embryo_metadata_refactored import EmbryoMetadata
em = EmbryoMetadata(
    sam_annotation_path="path/to/sam.json",
    embryo_metadata_path="path/to/metadata.json",
    gen_if_no_file=True,
    verbose=True
)
```

### Batch Operations
```python
from embryo_metadata_batch import BatchProcessor
batch = BatchProcessor(em)
batch.assign_phenotype_to_range(
    phenotype="EDEMA",
    range_spec="[10:20]",
    author="user"
)
```

### Integration with SAM
```python
from embryo_metadata_integration import create_embryo_metadata_from_sam
em = create_embryo_metadata_from_sam(
    sam_path="path/to/sam.json",
    metadata_path="path/to/metadata.json",
    verbose=True
)
```

## Navigation Tips
- Use **base_annotation_parser.py** for foundational utilities in new annotation classes.
- Use **embryo_metadata_refactored.py** and manager mixins for all embryo-level annotation logic.
- Use **embryo_metadata_batch.py** for batch operations and large-scale annotation.
- Use **embryo_metadata_integration.py** and **sam2_utils.py** for linking with SAM/GSAM annotations.
- Refer to **archive_embryo_metadata_dev/** for tests, logs, and module documentation.

---

*For further details, see the module documentation files in the directory.*
