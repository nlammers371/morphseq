# EmbryoMetadata Class Implementation Overview

## Executive Summary

The EmbryoMetadata class represents the culmination of the MorphSeq pipeline, providing a comprehensive data management system for storing and organizing phenotype, genotype, and quality control information about tracked embryos. This class integrates with the existing pipeline components while providing a flexible API for both manual and automated annotation workflows.

## Pipeline Position

```
Step 01: ExperimentMetadata → Step 02: ExperimentQC → Step 03: GDinoAnnotation 
    → Step 04: GroundedSamAnnotation → **Step 05: EmbryoMetadata** (NEW)
```

## Architecture Overview

### Core Components

1. **EmbryoMetadata Class** (`embryo_metadata.py`)
   - Main class managing all embryo-related metadata
   - Handles data loading, validation, and persistence
   - Provides unified API for all operations

2. **Data Models** (`embryo_metadata_models.py`)
   - Type definitions for phenotypes, genotypes, and flags
   - Validation schemas
   - Data structure specifications

3. **Batch Processing Engine** (`embryo_metadata_batch.py`)
   - Efficient batch operations for large-scale processing
   - Range syntax parser for temporal specifications
   - Parallel processing capabilities

4. **Integration Module** (`embryo_metadata_integration.py`)
   - GroundedSamAnnotation integration
   - Source data inheritance
   - Configuration tracking

5. **Utilities** (`embryo_metadata_utils.py`)
   - Helper functions
   - Data conversion utilities
   - Validation helpers

## Key Design Principles

### 1. Hierarchical Data Organization
- **Experiment** → **Video** → **Image** → **Snip** hierarchy
- Flags can be applied at any level
- Phenotypes primarily at snip level
- Genotypes at embryo level

### 2. Immutability with Local Changes
- All modifications are local until explicitly saved
- Atomic saves with backup creation
- Change tracking for audit trail

### 3. Extensible Validation System
- Permitted values stored in JSON
- Runtime validation of all inputs
- Custom validation rules (e.g., DEAD phenotype exclusivity)

### 4. Efficient Batch Processing
- Lazy loading of data
- Batch operations with progress tracking
- Auto-save intervals for long operations

### 5. Rich Metadata Tracking
- Authorship for every annotation
- Timestamps for all changes
- Configuration tracking from source models

## Implementation Modules

### Module 1: Core Class Structure
- `embryo_metadata_core.md` - Core class implementation
- Handles initialization, loading, saving
- Manages internal data structures

### Module 2: Data Models and Validation
- `embryo_metadata_models.md` - Data model specifications
- Validation logic
- Permitted values management

### Module 3: Phenotype Management
- `embryo_metadata_phenotype.md` - Phenotype-specific operations
- DEAD phenotype special handling
- Temporal phenotype tracking

### Module 4: Genotype Management
- `embryo_metadata_genotype.md` - Genotype operations
- Missing genotype warnings
- Overwrite protection

### Module 5: Flag Management
- `embryo_metadata_flags.md` - Multi-level flag system
- Flag categories and validation
- Batch flag operations

### Module 6: Batch Processing
- `embryo_metadata_batch.md` - Batch operation infrastructure
- Range syntax parsing
- Progress tracking

### Module 7: Integration Layer
- `embryo_metadata_integration.md` - External data integration
- GroundedSamAnnotation linking
- Configuration inheritance

### Module 8: Utilities and Helpers
- `embryo_metadata_utils.md` - Utility functions
- Data converters
- Helper methods

## Data Flow

```
GroundedSamAnnotation → Load embryo_ids/snip_ids
                    ↓
    Initialize EmbryoMetadata with source references
                    ↓
    Apply flags/phenotypes/genotypes (manual or automated)
                    ↓
    Validate against permitted values
                    ↓
    Store locally with change tracking
                    ↓
    Save to persistent storage with backups
```

## Integration Points

1. **Input Sources**
   - `grounded_sam_ft_annotations.json` - Source for embryo tracking
   - `experiment_metadata.json` - Experiment structure
   - Future ML models for automated phenotyping

2. **Output Consumers**
   - Analysis pipelines
   - Visualization tools
   - Statistical frameworks

## Usage Patterns

### Basic Usage
```python
# Initialize
em = EmbryoMetadata(
    sam_annotation_path="/path/to/sam2_annotations.json",
    embryo_metadata_path="/path/to/embryo_metadata.json",
    gen_if_no_file=True
)

# Add phenotype
em.add_phenotype("20240411_A01_e01_0001", "EDEMA", author="researcher1")

# Add genotype
em.add_genotype("20240411_A01_e01", "wildtype", author="researcher1")

# Save
em.save()
```

### Batch Processing
```python
# Batch phenotype assignment with range syntax
em.batch_add_phenotype(
    embryo_id="20240411_A01_e01",
    phenotype="CONVERGENCE_EXTENSION", 
    snip_range="[23::]",  # From frame 23 onwards
    author="batch_processor"
)

# Batch flag processing
flags = {
    "20240411_A01_0001": ["MOTION_BLUR", "MASK_ON_EDGE"],
    "20240411_A01_0002": "DETECTION_FAILURE"
}
em.batch_add_flags(flags, level="image", author="qc_system")
```

## Performance Considerations

1. **Memory Efficiency**
   - Lazy loading of large datasets
   - Incremental saves for batch operations
   - Efficient indexing structures

2. **Scalability**
   - Designed for 100,000+ embryos
   - Parallel processing support
   - Optimized query patterns

3. **Reliability**
   - Atomic saves with rollback
   - Data integrity checks
   - Recovery from corruption

## Future Extensions

1. **ML Integration**
   - Automated phenotype detection
   - Confidence scores for predictions
   - Active learning interfaces

2. **Visualization**
   - Timeline views of phenotypes
   - Statistical dashboards
   - Quality control reports

3. **Export Formats**
   - Analysis-ready datasets
   - Standardized phenotype ontologies
   - Interoperability with external tools

## Implementation Timeline

1. **Phase 1**: Core infrastructure (Modules 1-2)
2. **Phase 2**: Basic operations (Modules 3-5)
3. **Phase 3**: Batch processing (Module 6)
4. **Phase 4**: Integration (Module 7)
5. **Phase 5**: Testing and optimization

## Success Metrics

- Load time < 1s for 10,000 embryos
- Batch operations > 1000 items/second
- Zero data loss with proper error handling
- 100% validation coverage
- Intuitive API with minimal learning curve