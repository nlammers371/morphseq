# Module 3 Comprehensive Test Plan

**Created**: 2025-08-19  
**Purpose**: Comprehensive testing strategy for embryo annotation system implementation

---

## Testing Philosophy

### Test-Driven Approach ✅
- **Test early, test often** - Run tests after each component implementation
- **Real data focus** - Use actual SAM2 segmentation files for validation
- **Incremental validation** - Test components individually before integration
- **Error scenario coverage** - Comprehensive error handling validation

### Test Data Sources
- **Real SAM2 data**: `/data/segmentation/grounded_sam_segmentations.json`
- **Current experiment**: `20250612_30hpf_ctrl_atf6` with 96 wells
- **Test fixtures**: Will create minimal synthetic data for unit tests

---

## Test Structure

### Phase 1: Unit Tests 🔬

#### 1.1 EmbryoMetadata Core Tests
**File**: `test_embryo_metadata.py`

**Test Cases**:
- ✅ **SAM2 Import**
  - `test_create_from_sam()` - Import real SAM2 data
  - `test_sam_consistency_check()` - Validate against source SAM2
  - `test_entity_tracking_initialization()` - Proper hierarchy setup
  
- ✅ **File I/O Operations**
  - `test_save_and_reload()` - Atomic saves with backup
  - `test_auto_metadata_path_generation()` - Default path construction
  - `test_file_not_found_handling()` - Error handling for missing files
  
- ✅ **Data Access and Caching**
  - `test_cache_building()` - Lookup cache performance
  - `test_get_embryo_id_from_snip()` - Snip-to-embryo mapping
  - `test_get_available_snips()` - Snip enumeration
  
- ✅ **Basic CRUD Operations**
  - `test_get_genotype()` - Genotype retrieval
  - `test_get_phenotypes()` - Phenotype retrieval
  - `test_get_flags()` - Flag retrieval with level detection

#### 1.2 UnifiedEmbryoManager Tests
**File**: `test_unified_managers.py`

**Test Cases**:
- ✅ **Data Models**
  - `test_phenotype_model()` - Phenotype object creation and serialization
  - `test_genotype_model()` - Genotype object with allele/zygosity
  - `test_flag_model()` - Flag object with type/priority
  - `test_treatment_model()` - Treatment object with dosage/timing
  
- ✅ **Phenotype Management**
  - `test_add_phenotype_basic()` - Basic phenotype addition
  - `test_dead_exclusivity()` - DEAD cannot coexist with other phenotypes
  - `test_dead_force_override()` - force_dead=True behavior
  - `test_invalid_phenotype_validation()` - Schema validation
  
- ✅ **Genotype Management**
  - `test_add_genotype()` - Single genotype per embryo
  - `test_genotype_overwrite()` - overwrite_genotype behavior
  - `test_zygosity_validation()` - Valid zygosity types
  
- ✅ **DEAD Logic Validation**
  - `test_dead_temporal_consistency()` - Frame number ordering
  - `test_dead_warning_system()` - Inconsistency warnings
  - `test_embryo_phenotype_extraction()` - Cross-snip phenotype analysis

#### 1.3 AnnotationBatch Tests  
**File**: `test_annotation_batch.py`

**Test Cases**:
- ✅ **Batch Structure**
  - `test_batch_initialization()` - Author, description, stats
  - `test_inheritance_from_metadata()` - Business logic inheritance
  - `test_batch_data_structure()` - Internal data organization
  
- ✅ **Add Operations**
  - `test_add_phenotype_to_batch()` - Phenotype operation storage
  - `test_mark_dead_convenience()` - mark_dead() method
  - `test_add_genotype_to_batch()` - Genotype operation storage
  - `test_batch_statistics_tracking()` - Operation counting
  
- ✅ **Frame Range Processing**  
  - `test_frame_range_parsing()` - "all", "30:50", [30,31] formats
  - `test_frame_validation()` - Invalid frame specifications
  - `test_missing_frame_handling()` - Gaps in frame sequences

### Phase 2: Integration Tests 🔗

#### 2.1 SAM2 Integration Tests
**File**: `test_sam2_integration.py`

**Test Cases**:
- ✅ **Real Data Processing**
  - `test_import_real_sam2_data()` - Use actual segmentation file
  - `test_embryo_extraction_20250612()` - Current experiment data
  - `test_entity_hierarchy_validation()` - Complete hierarchy check
  
- ✅ **Metadata Creation**
  - `test_metadata_from_real_sam2()` - End-to-end creation
  - `test_snip_id_generation()` - Proper snip ID format
  - `test_embryo_count_validation()` - Expected vs actual counts

#### 2.2 Batch Processing Integration
**File**: `test_batch_integration.py`

**Test Cases**:
- ✅ **Batch Workflows**
  - `test_batch_apply_to_metadata()` - Apply operations to main metadata
  - `test_batch_preview_functionality()` - Preview before apply
  - `test_batch_rollback_capability()` - Undo batch operations
  
- ✅ **DEAD Safety Integration**
  - `test_overwrite_dead_false_default()` - Silent skip behavior
  - `test_overwrite_dead_true_explicit()` - Death frame correction
  - `test_batch_dead_workflow_integration()` - Complex scenarios

#### 2.3 Data Access Helpers Integration
**File**: `test_data_helpers_integration.py`

**Test Cases**:
- ✅ **Helper Method Functionality**
  - `test_get_embryo_data_helper()` - Consistent embryo data access
  - `test_get_snip_data_helper()` - Consistent snip data access
  - `test_helper_error_handling()` - Centralized error messages
  
- ✅ **DEAD Safety with Helpers**
  - `test_dead_safety_with_helpers()` - Integration with data access
  - `test_helper_performance()` - Caching and performance validation

### Phase 3: Performance Tests ⚡

#### 3.1 Load Testing
**File**: `test_performance.py`

**Test Cases**:
- ✅ **Large Dataset Handling**
  - `test_large_embryo_count_performance()` - Scale testing
  - `test_memory_usage_monitoring()` - Memory efficiency
  - `test_cache_performance_validation()` - Lookup speed
  
- ✅ **Batch Operation Performance**
  - `test_large_batch_processing_time()` - Batch size scaling
  - `test_concurrent_batch_safety()` - Thread safety validation

---

## Test Data Setup

### Synthetic Test Data
```python
# Minimal SAM2 structure for unit tests
MINIMAL_SAM2_DATA = {
    "experiments": {
        "test_exp": {
            "videos": {
                "test_exp_A01": {
                    "embryo_ids": ["test_exp_A01_e01", "test_exp_A01_e02"],
                    "images": {
                        "test_exp_A01_ch00_t0100": {
                            "embryos": {
                                "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0100"},
                                "test_exp_A01_e02": {"snip_id": "test_exp_A01_e02_s0100"}
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### Real Data Tests
- **Primary**: `/data/segmentation/grounded_sam_segmentations.json`
- **Backup**: Archive versions for regression testing
- **Experiment**: `20250612_30hpf_ctrl_atf6` (96 wells, 1 frame each)

---

## Test Execution Strategy

### Incremental Testing ⚡
1. **Component completion** → Run related unit tests
2. **Integration milestone** → Run integration tests  
3. **Feature completion** → Full test suite
4. **Before commit** → Complete validation

### Test Commands
```bash
# Unit tests by component
python -m pytest tests/test_embryo_metadata.py -v
python -m pytest tests/test_unified_managers.py -v
python -m pytest tests/test_annotation_batch.py -v

# Integration tests
python -m pytest tests/test_*_integration.py -v

# Full test suite
python -m pytest tests/ -v --tb=short

# Performance tests
python -m pytest tests/test_performance.py -v -s
```

### Success Criteria ✅
- **All unit tests pass** (100% success rate)
- **Integration tests pass with real data** (no errors)
- **Performance within acceptable bounds** (< 5s for typical operations)
- **Error handling comprehensive** (graceful failure modes)
- **Documentation examples work** (all code snippets functional)

---

## Risk Mitigation

### Test Data Backup
- Keep working copies of SAM2 data
- Version control test fixtures
- Document data dependencies

### Test Environment Isolation
- Use temporary files for write tests
- Clean up test artifacts
- Mock external dependencies where appropriate

### Continuous Validation
- Run tests after each major change
- Validate with multiple experiment datasets when available
- Monitor performance regressions

---

*This test plan will guide the comprehensive validation of Module 3 implementation*