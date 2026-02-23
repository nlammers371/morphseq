# Module 3 Fresh Implementation Plan (Phased MVP Approach)

## Overview
Phased implementation starting simple and building up complexity:
- **Pre-Phase**: CLI script creation for pipeline integration
- **Phase 1**: Core MVP with basic functionality (Days 1-2)
- **Phase 2**: Parameter validation and API enhancement (Days 3-4)  
- **Phase 3**: Business rules and validation (Days 5-7)
- **Phase 4**: Batch system and tutorials (Week 2)

**Key Principles**: Tiny API surface, unambiguous parameters, atomic operations, clear error messages

---

## Step 1: Archive Existing Work ✅ **COMPLETED**
**Target**: Preserve current implementation while starting fresh

**Actions**:
- ✅ Move `scripts/annotations/embryo_metadata.py` → `scripts/annotations/embryo_metadata_old.py`
- ✅ Move `scripts/annotations/annotation_batch.py` → `scripts/annotations/annotation_batch_old.py`  
- ✅ Move `scripts/annotations/unified_managers.py` → `scripts/annotations/unified_managers_old.py`
- ✅ Keep as reference in case we need to extract working validation logic

---

## Phase 1: Core MVP Foundation (Days 1-2) 🚀

### Minimal Viable Implementation
**Target**: Get basic functionality working with simplest possible approach

**Core MVP Design** (Composition over inheritance):
```python
class EmbryoMetadata:
    """MVP: Use composition with BaseFileHandler, hardcoded validation, auto-update from SAM2"""
    
    # Hardcoded validation lists (no config files yet)
    VALID_PHENOTYPES = ["NORMAL", "EDEMA", "DEAD", "CONVERGENCE_EXTENSION"]
    
    def __init__(self, sam2_path, annotations_path=None):
        """Composition with BaseFileHandler"""
        from scripts.utils.base_file_handler import BaseFileHandler
        self.sam2_path = sam2_path
        self.annotations_path = annotations_path or sam2_path.replace('.json', '_biology.json')
        self.file_handler = BaseFileHandler(self.annotations_path)
        
        # Load existing or create from SAM2
        if Path(self.annotations_path).exists():
            self.data = self.file_handler.load_json()
            # Auto-update with any new embryos from SAM2
            self._update_from_sam2()
        else:
            self.data = self._create_from_sam2()
    
    def _update_from_sam2(self):
        """Auto-detect new embryos from SAM2 and merge without overwriting existing annotations"""
        sam2_embryos = self._extract_embryos_from_sam2()
        
        for embryo_id, embryo_structure in sam2_embryos.items():
            if embryo_id not in self.data["embryos"]:
                # New embryo - add complete structure
                self.data["embryos"][embryo_id] = embryo_structure
            else:
                # Existing embryo - only add new snips, preserve annotations
                existing_snips = self.data["embryos"][embryo_id].get("snips", {})
                new_snips = embryo_structure.get("snips", {})
                
                for snip_id, snip_data in new_snips.items():
                    if snip_id not in existing_snips:
                        existing_snips[snip_id] = snip_data  # Add new snip with empty annotations
    
    def add_phenotype(self, phenotype, author, embryo_id, target="all"):
        """MVP: Only supports embryo_id + target='all' - no parameter validation yet"""
        if target != "all":
            raise ValueError("MVP only supports target='all'")
        
        if phenotype not in self.VALID_PHENOTYPES:
            raise ValueError(f"Invalid phenotype. Use: {self.VALID_PHENOTYPES}")
        
        # Simple: add to all snips for this embryo
        embryo_data = self.data["embryos"].get(embryo_id)
        if not embryo_data:
            raise ValueError(f"Embryo {embryo_id} not found")
        
        for snip_id in embryo_data.get("snips", {}):
            snip_data = embryo_data["snips"][snip_id]
            if "phenotypes" not in snip_data:
                snip_data["phenotypes"] = []
            
            snip_data["phenotypes"].append({
                "value": phenotype,
                "author": author,
                "timestamp": datetime.now().isoformat()
            })
    
    def save(self):
        """Delegate to BaseFileHandler"""
        self.file_handler.save_json(self.data)
```

**Phase 1 Success Criteria** (3 core actions):
1. **✅ Archive existing files** (completed)
2. **🔄 Create update script** `scripts/pipelines/07_embryo_metadata_update.py`:
   - Standardized argument parsing for embryo metadata updates
   - **Auto-detect new embryos** from SAM2 reference and add to existing metadata
   - Save outputs to `data/embryo_metadata/` directory
   - Integrate with existing pipeline workflow
3. **🔄 Create working MVP** that can:
   - Load real SAM2 file and extract embryo structure
   - **Auto-merge new embryos** from SAM2 into existing metadata without overwriting annotations
   - Add phenotype using `embryo_id + target="all"` only
   - Save using BaseFileHandler atomic writes to `data/embryo_metadata/`
   - Pass basic test with real data including embryo detection updates

**Phase 1 Required Tests**:
- `test_update_script.py` - Test script argument parsing and data directory creation
- `test_mvp_basic.py` - Load SAM2, add phenotype, save/reload to `data/embryo_metadata/`
- `test_sam2_extraction.py` - Verify embryo structure extraction
- `test_embryo_detection.py` - Test auto-detection and merging of new embryos from SAM2
- `test_pipeline_integration.py` - Verify script works with existing pipeline
- Performance: Process 10 embryos in <2 seconds

**Logging Checkpoint**: 
- 📝 Document test results in `implementation_log.md`
- 📝 Record any SAM2 structure surprises
- 📝 Benchmark loading/saving times
- 📝 Track new embryo detection accuracy and merge behavior

**Git Workflow Reminders**:
- 🔄 First commit: `git commit -m "Add script 07_embryo_metadata_update.py"` when script works
- 🔄 Second commit: `git commit -m "Phase 1: MVP implementation working"` when MVP passes tests
- 🔄 Create branch: `git checkout -b phase1-mvp-implementation` 
- 🔄 Commit frequently: After each working feature, not just at phase end
- 📝 Log all git commits in `implementation_log.md` with commit hashes

**Deferred to Phase 2**:
- Parameter validation (_select_mode)
- snip_ids support  
- Frame range parsing
- Complex validation

## Phase 2: Parameter Validation (Days 3-4) 🧠

### Either/Or API Implementation
**Target**: Add proper parameter validation and snip_ids support

**Enhanced API Design**:
```python
def _select_mode(self, embryo_id=None, target=None, snip_ids=None):
    """Prevent ambiguous parameter combinations"""
    by_embryo = embryo_id is not None or target is not None
    by_snips = snip_ids is not None and len(snip_ids) > 0
    
    if by_embryo and by_snips:
        print("❌ ERROR: Provide either (embryo_id + target) OR snip_ids, not both")
        raise ValueError("Ambiguous parameters")
    
    if not by_embryo and not by_snips:
        print("❌ ERROR: Must provide either (embryo_id + target) OR snip_ids")
        raise ValueError("Missing parameters")
    
    return "embryo" if by_embryo else "snips"

def add_phenotype(self, phenotype, author, embryo_id=None, target=None, snip_ids=None):
    """Enhanced: Support both API approaches with validation"""
    mode = self._select_mode(embryo_id, target, snip_ids)
    
    if mode == "embryo":
        resolved_snips = self._resolve_target_to_snips(embryo_id, target)
    else:
        resolved_snips = snip_ids
    
    for snip_id in resolved_snips:
        self._add_phenotype_to_snip(snip_id, phenotype, author)
```

**Phase 2 Success Criteria**:
1. **🔄 Parameter Validation**: `_select_mode()` prevents ambiguous usage
2. **🔄 Snip IDs Support**: `add_phenotype(snip_ids=["s1", "s2"])` works
3. **🔄 Frame Range Parsing**: Basic support for `target="30:50"`, `target="all"`
4. **🔄 Clear Error Messages**: Print statements guide users to correct usage

**Phase 2 Required Tests**:
- `test_parameter_validation.py` - Test both valid and invalid parameter combinations
- `test_both_apis.py` - Compare embryo_id vs snip_ids approaches
- `test_frame_parsing.py` - Verify target parsing works correctly
- Performance: Both approaches complete in similar time

**Logging Checkpoint**:
- 📝 Document API usability testing results
- 📝 Record any parameter validation edge cases discovered
- 📝 Compare performance between embryo_id and snip_ids approaches

**Git Workflow Reminders**:
- 🔄 `git add .` and `git commit -m "Phase 2: Parameter validation implemented"` when validation works
- 🔄 Commit after each API addition: `git commit -m "Add snip_ids support"`, `git commit -m "Add frame parsing"`
- 🔄 Test before committing: All Phase 2 tests must pass
- 📝 Log commit hashes and what each commit contains

**Frame Resolution Logic**:
- `target="all"` → all snips for embryo
- `target="30:50"` → frame range parsing (start:end)
- `target="200:"` → open-ended ranges (start to end of data)
- `snip_ids=["snip1", "snip2"]` → direct snip specification

---

## Phase 3: Business Rules (Days 5-7) 📦

### Business Logic Implementation
**Target**: Add essential validation rules and clear error handling

**DEAD Phenotype Validation**:
```python
def _validate_dead_exclusivity(self, embryo_id, phenotype):
    """DEAD cannot coexist with other phenotypes at embryo level"""
    if phenotype != "DEAD":
        return  # Non-DEAD phenotypes don't need this check
    
    embryo_data = self.data["embryos"].get(embryo_id)
    if not embryo_data:
        return
    
    # Check if any snip has non-DEAD phenotypes
    conflicts = []
    for snip_id, snip_data in embryo_data.get("snips", {}).items():
        for existing in snip_data.get("phenotypes", []):
            if existing["value"] != "DEAD":
                conflicts.append(f"snip {snip_id} has {existing['value']}")
    
    if conflicts:
        print(f"❌ ERROR: Cannot add DEAD to embryo {embryo_id}")
        print(f"   Existing phenotypes: {conflicts[:3]}...")  # Show first 3
        raise ValueError(f"DEAD exclusivity violation for embryo {embryo_id}")

def _validate_phenotype_value(self, phenotype):
    """Check phenotype against controlled vocabulary"""
    if phenotype not in self.VALID_PHENOTYPES:
        print(f"❌ ERROR: Invalid phenotype '{phenotype}'")
        print(f"   Valid options: {self.VALID_PHENOTYPES}")
        raise ValueError(f"Invalid phenotype: {phenotype}")
```

**Phase 3 Success Criteria**:
1. **🔄 DEAD Exclusivity**: Embryo-level validation prevents conflicts
2. **🔄 Controlled Vocabularies**: Phenotypes validated against approved lists  
3. **🔄 Clear Error Messages**: Print statements explain problems and solutions
4. **🔄 Duplicate Handling**: Gracefully handle duplicate phenotype additions

**Phase 3 Required Tests**:
- `test_dead_validation.py` - DEAD exclusivity rules work correctly
- `test_business_rules.py` - Phenotype validation and error handling
- `test_error_messages.py` - Verify error messages are clear and actionable
- `test_duplicate_handling.py` - Test duplicate phenotype scenarios

**Logging Checkpoint**:
- 📝 Document business rule test scenarios and edge cases
- 📝 Record user feedback on error message clarity
- 📝 Test performance impact of validation checks

**Git Workflow Reminders**:
- 🔄 `git add .` and `git commit -m "Phase 3: Business rules implemented"` when validation passes
- 🔄 Commit each rule separately: `git commit -m "Add DEAD exclusivity validation"`, `git commit -m "Add phenotype vocabulary validation"`
- 🔄 Commit test improvements: `git commit -m "Improve error message clarity"`
- 📝 Document any business rule changes or exceptions discovered

**Validation Rules**:
- **DEAD Exclusivity**: Cannot coexist with other phenotypes (embryo-level)
- **Controlled Vocabularies**: All phenotypes must be in approved list
- **Level Enforcement**: Phenotypes at snip level, genotypes at embryo level
- **Duplicate Prevention**: Same phenotype + author = no-op (warn but don't fail)

## Phase 4: Batch System & Tutorials (Week 2) ✨

### AnnotationBatch Implementation
**Target**: Inherit all functionality with batch-specific features

**Batch System Design**:
```python
class AnnotationBatch(EmbryoMetadata):
    """Inherit all EmbryoMetadata functionality with batch conveniences"""
    
    def __init__(self, data_structure, author, validate=True):
        # Author required for batch operations
        if not author:
            raise ValueError("Author required for AnnotationBatch")
        
        # Manual setup to avoid file loading
        self.author = author
        self.data = data_structure
        self.validate = validate
        self.sam2_path = None  # No file association
        self.annotations_path = None
        
    def add_phenotype(self, phenotype, author=None, embryo_id=None, target=None, snip_ids=None):
        """Inherit parent method, default to batch author"""
        return super().add_phenotype(
            phenotype, 
            author or self.author,  # Use batch author as default
            embryo_id=embryo_id,
            target=target, 
            snip_ids=snip_ids
        )

def apply_batch(self, batch, on_conflict="error", dry_run=False):
    """Apply batch changes with conflict resolution"""
    # Simple data merge with conflict handling
    # Return validation report with counts and errors
```

### Tutorial Notebooks Implementation

**1. Biologist Tutorial (`biologist_tutorial.ipynb`)**:
- **Loading Data**: How to initialize with SAM2 files
- **Basic Annotation**: Adding phenotypes using embryo_id + target="all"
- **Common Workflows**: Annotating developmental stages
- **Error Handling**: Understanding and fixing common mistakes
- **Best Practices**: Recommended annotation workflows

**2. Pipeline Integration Tutorial (`pipeline_integration_tutorial.ipynb`)**:
- **Batch Processing**: Processing multiple SAM2 files
- **Advanced API**: Using snip_ids for precise control
- **Integration Patterns**: How to integrate with existing pipelines
- **Performance**: Optimizing for large datasets
- **Error Recovery**: Handling validation failures in automated systems

**Phase 4 Success Criteria**:
1. **🔄 AnnotationBatch**: Inheritance works correctly, batch author defaulting
2. **🔄 Apply Mechanism**: Batch changes merge correctly with conflict resolution
3. **🔄 Tutorial Notebooks**: Both tutorials are complete and tested
4. **🔄 Integration Testing**: System works end-to-end with real pipeline data

**Phase 4 Required Tests**:
- `test_annotation_batch.py` - Batch inheritance and functionality
- `test_apply_mechanism.py` - Batch application and conflict resolution
- `test_tutorial_examples.py` - All tutorial code examples work
- `test_integration_full.py` - End-to-end pipeline integration

**Logging Checkpoint**:
- 📝 Document batch operation performance benchmarks
- 📝 Record tutorial user testing feedback
- 📝 Integration test results with real pipeline data
- 📝 Final system performance metrics

**Git Workflow Reminders**:
- 🔄 `git add .` and `git commit -m "Phase 4: Batch system and tutorials complete"` when integration passes
- 🔄 Commit tutorials separately: `git commit -m "Add biologist tutorial notebook"`, `git commit -m "Add pipeline integration tutorial"`
- 🔄 Final integration commit: `git commit -m "Complete Module 3 implementation with full test suite"`
- 🔄 Create PR: `gh pr create --title "Module 3: Simplified embryo metadata system" --body "Incremental MVP implementation"`
- 📝 Document final performance benchmarks and system capabilities

**Tutorial Content Requirements**:
- **Interactive Examples**: All code runnable with provided sample data
- **Biological Context**: Examples use realistic developmental biology scenarios
- **Error Scenarios**: Show common mistakes and how to fix them
- **Performance Notes**: Guidance on optimizing for different use cases

---

## Data Flow Architecture 📊

### Input Sources
1. **Primary**: SAM2 annotations (embryo_ids, snip_ids from grounded_sam_segmentations.json)
2. **Secondary**: experimentQC flags (quality issues)
3. **Manual**: User annotations via API

### Processing Steps
1. **Load SAM2** → extract embryo structure with snip generation
2. **Initialize empty** phenotype/genotype fields for all embryos
3. **Apply QC flags** from experimentQC (if available)
4. **Accept user annotations** with parameter validation
5. **Validate business rules** (warnings vs errors)
6. **Atomic save** to persistent storage with versioning

### Output Format
- **JSON file** with complete biological annotations
- **Schema versioned** for future compatibility  
- **Compatible** with downstream analysis tools
- **Atomic writes** prevent corruption

---

## SAM2 Import Implementation
**Target**: 4-step process from `data_structure_and_sam2_import.md`

### Exact Implementation from Design:
1. **Load SAM2 data** from JSON file
2. **`_extract_embryos_from_sam2()`** - scan experiments → videos → images → embryos  
3. **`_create_embryo_structure(embryo_id)`** - genotype/treatments/snips structure
4. **`_create_snip_structure(snip_id, frame_number)`** - phenotypes/flags arrays
5. **Generate snip IDs** as `f"{embryo_id}_s{frame_num:04d}"`

### SAM2 Import Logic:
```python
def _extract_embryos_from_sam2(self):
    """Extract embryo structure from SAM2 following design specification"""
    embryos = {}
    
    for exp_id, exp_data in self.sam2_data["experiments"].items():
        for video_id, video_data in exp_data["videos"].items():
            for image_id, image_data in video_data.get("images", {}).items():
                frame_num = self._extract_frame_number(image_id)  # "t0100" → 100
                
                for embryo_id in image_data.get("embryos", {}):
                    if embryo_id not in embryos:
                        embryos[embryo_id] = self._create_embryo_structure(embryo_id)
                    
                    # Add snip for this frame
                    snip_id = f"{embryo_id}_s{frame_num:04d}"
                    embryos[embryo_id]["snips"][snip_id] = self._create_snip_structure(snip_id, frame_num)
    
    return embryos
```

---

## Step 5: Business Rules from Design Document
**Target**: Exact validation rules from design spec

### DEAD Phenotype Logic:
- **Exclusivity**: DEAD cannot coexist with other phenotypes at same frame
- **Permanence**: Once dead at frame N, all frames ≥ N must be DEAD
- **Safety**: `overwrite_dead=False` silently skips DEAD frames (from design)

### Other Rules from Design:
- **Single Genotype**: One per embryo unless `overwrite=True`
- **Controlled Vocabularies**: Validate against approved lists from design:
  ```python
  VALID_PHENOTYPES = ["NORMAL", "EDEMA", "CONVERGENCE_EXTENSION", "DEAD"]
  VALID_GENES = ["WT", "tmem67", "lmx1b", "sox9a", "cep290", "b9d2", "rpgrip1l"]
  VALID_ZYGOSITY = ["homozygous", "heterozygous", "compound_heterozygous", "crispant", "morpholino"]
  VALID_TREATMENTS = ["control", "DMSO", "PTU", "BIO", "SB431542", "DAPT", "heat_shock", "cold_shock"]
  VALID_FLAGS = ["MOTION_BLUR", "OUT_OF_FOCUS", "DARK", "CORRUPT"]
  ```
- **Level Enforcement**: Genotypes/treatments at embryo level, phenotypes/flags at snip level

---

## Step 6: Apply Mechanism from Design
**Target**: Simple data merge from `annotation_batch_design.md`

### Implementation from Design:
- `apply_batch(batch, on_conflict="error"|"skip"|"overwrite"|"merge", dry_run=False)`
- `_merge_batch_data()` - direct data structure merging (no operation logging)
- Conflict resolution for genotypes and phenotypes
- Validation report with applied counts and errors

### Apply Logic:
```python
def apply_batch(self, batch, on_conflict="error", dry_run=False):
    """Apply batch changes to metadata with conflict resolution"""
    # 1. Validate batch data consistency
    validation_report = self._validate_batch_data(batch.data, on_conflict)
    
    if validation_report["errors"] and on_conflict == "error":
        raise ValueError(f"Batch validation failed: {validation_report['errors']}")
    
    # 2. Apply changes if not dry run
    if not dry_run:
        applied_count = self._merge_batch_data(batch.data, on_conflict)
        validation_report["applied_count"] = applied_count
    
    return validation_report
```

---

## Step 7: Error Handling and Validation
**Target**: Clear feedback and robust validation

### **NEW: Print Statement Error Messages**:
```python
# When both embryo_id+target AND snip_ids provided
if embryo_approach and snip_approach:
    print(f"❌ ERROR: add_phenotype() called with both approaches:")
    print(f"   Embryo approach: embryo_id='{embryo_id}', target='{target}'")
    print(f"   Snip approach: snip_ids={snip_ids}")
    print(f"   SOLUTION: Use either (embryo_id + target) OR snip_ids, not both")
    raise ValueError("Ambiguous parameters")

# When neither approach provided
if not embryo_approach and not snip_approach:
    print(f"❌ ERROR: add_phenotype() called without specifying target:")
    print(f"   Missing parameters: embryo_id={embryo_id}, target={target}, snip_ids={snip_ids}")
    print(f"   SOLUTION: Provide either (embryo_id='embryo_e01', target='all') OR snip_ids=['snip1', 'snip2']")
    raise ValueError("Missing parameters")
```

### Contract Verification (from design):
- `_verify_contract()` in AnnotationBatch to ensure inheritance works
- Required attributes: data, validate, validator, frame_parser

---

## Testing Strategy & Requirements 🧪

### Test Organization
**All test scripts should be created in `tests/` directory with descriptive names**

**Test Naming Convention**:
- `test_[phase]_[feature].py` - Phase-specific functionality tests
- `test_integration_[scenario].py` - End-to-end integration tests
- `test_performance_[aspect].py` - Performance and benchmark tests

### Phase-by-Phase Test Requirements

**Phase 1 Tests (MVP Foundation)**:
```python
# test_mvp_basic.py - Core functionality
def test_sam2_loading():
    metadata = EmbryoMetadata("real_sam2.json")
    assert metadata.data["embryos"]  # Has embryo data

def test_basic_phenotype_addition():
    metadata = EmbryoMetadata("real_sam2.json")
    result = metadata.add_phenotype("EDEMA", "test_user", embryo_id="e01", target="all")
    assert "EDEMA" in str(metadata.data)

# test_sam2_extraction.py - SAM2 structure parsing
def test_embryo_extraction():
    # Verify experiments -> videos -> image_ids -> embryos structure works
    
# Performance target: Process 10 embryos in <2 seconds
```

**Phase 2 Tests (Parameter Validation)**:
```python
# test_parameter_validation.py - API validation
def test_ambiguous_parameters():
    # Should fail with clear error
    with pytest.raises(ValueError, match="Ambiguous parameters"):
        metadata.add_phenotype("EDEMA", "user", embryo_id="e01", snip_ids=["s1"])

def test_missing_parameters():
    # Should fail with clear error
    with pytest.raises(ValueError, match="Missing parameters"):
        metadata.add_phenotype("EDEMA", "user")

# test_both_apis.py - Compare approaches
def test_embryo_id_vs_snip_ids():
    # Both should produce same result for equivalent targets
    
# test_frame_parsing.py - Target parsing
def test_frame_ranges():
    # target="30:50" vs target="all" vs snip_ids
```

**Phase 3 Tests (Business Rules)**:
```python
# test_dead_validation.py - DEAD exclusivity
def test_dead_blocks_other_phenotypes():
    metadata.add_phenotype("EDEMA", "user", embryo_id="e01", target="all")
    with pytest.raises(ValueError, match="DEAD exclusivity"):
        metadata.add_phenotype("DEAD", "user", embryo_id="e01", target="all")

# test_business_rules.py - Validation rules
def test_invalid_phenotype():
    with pytest.raises(ValueError, match="Invalid phenotype"):
        metadata.add_phenotype("INVALID", "user", embryo_id="e01", target="all")

# test_error_messages.py - User experience
def test_error_message_clarity():
    # Verify error messages are actionable
```

**Phase 4 Tests (Batch & Integration)**:
```python
# test_annotation_batch.py - Inheritance
def test_batch_inherits_all_methods():
    batch = AnnotationBatch(data, "user")
    # Should have all EmbryoMetadata methods

# test_tutorial_examples.py - Documentation
def test_all_tutorial_code_works():
    # Every code example in tutorials must pass

# test_integration_full.py - End-to-end
def test_real_pipeline_integration():
    # Full workflow with actual SAM2 files
```

### Performance Benchmarks

**Target Performance Metrics**:
- **SAM2 Loading**: <1s for typical file (50-100 embryos)
- **Phenotype Addition**: <0.1s per operation  
- **Save Operations**: <2s including backup creation
- **Batch Operations**: Process 100 embryos in <5s
- **Memory Usage**: <100MB for typical dataset

**Benchmark Test Requirements**:
```python
# test_performance_loading.py
def test_sam2_loading_speed():
    start = time.time()
    metadata = EmbryoMetadata("large_sam2.json")
    duration = time.time() - start
    assert duration < 1.0, f"Loading took {duration}s, expected <1s"

# test_performance_operations.py  
def test_phenotype_addition_speed():
    # Measure bulk operations
    
# test_performance_memory.py
def test_memory_usage():
    # Monitor memory consumption during typical workflows
```

### Test Data Requirements

**Sample Data Files Needed**:
- `test_sam2_small.json` - 5-10 embryos for basic testing
- `test_sam2_large.json` - 100+ embryos for performance testing  
- `test_sam2_edge_cases.json` - Unusual structures for robustness testing
- `expected_outputs/` - Golden files for regression testing

### Error Testing Strategy

**Error Scenarios to Test**:
```python
# Parameter validation errors
metadata.add_phenotype("EDEMA", "user", embryo_id="e01", target="all", snip_ids=["s1"])

# Business rule violations  
metadata.add_phenotype("INVALID_PHENOTYPE", "user", embryo_id="e01", target="all")

# Data consistency issues
metadata.add_phenotype("EDEMA", "user", embryo_id="NONEXISTENT", target="all")

# File I/O problems
metadata = EmbryoMetadata("nonexistent_file.json")
```

### Success Criteria Summary

**Each phase must pass ALL tests before proceeding to next phase**:
- **Phase 1**: 2 test scripts, basic functionality works
- **Phase 2**: 5 test scripts, both APIs work with validation  
- **Phase 3**: 8 test scripts, business rules enforced
- **Phase 4**: 12+ test scripts, full integration works

**Performance Gates**:
- All operations complete within target times
- Memory usage stays within bounds
- Error messages are clear and actionable

---

## Step 9: Tutorial Documentation
**Target**: Create user-friendly tutorial notebooks for different audiences

### Tutorial Requirements:
1. **Biologist Tutorial** (`biologist_tutorial.ipynb`)
   - **Target Audience**: Biology researchers with minimal programming experience
   - **Focus**: Simple, practical examples with real biological scenarios
   - **Content**:
     - Basic loading of SAM2 data
     - Adding phenotypes using embryo_id + target approach
     - Common workflows for annotating developmental stages
     - How to interpret error messages
     - Best practices for biological annotation workflows
     - Visual examples with sample data

2. **Pipeline Integration Tutorial** (`pipeline_integration_tutorial.ipynb`)
   - **Target Audience**: Bioinformaticians and pipeline developers
   - **Focus**: Integration patterns and advanced usage
   - **Content**:
     - Loading and processing multiple SAM2 files
     - Batch operations for high-throughput annotation
     - Pipeline integration patterns
     - Error handling and validation strategies
     - Performance considerations for large datasets
     - Advanced API usage with snip_ids
     - Integration with downstream analysis tools

### Tutorial Structure:
- **Interactive Examples**: All code should be runnable with sample data
- **Clear Explanations**: Each step explained in biological/technical context
- **Error Scenarios**: Common mistakes and how to fix them
- **Best Practices**: Recommended workflows and patterns
- **Real Data**: Use actual SAM2 data examples where possible

---

## Key Improvements Over Original Design
1. **Clear Either/Or API**: Eliminates ambiguity between embryo_id+target vs snip_ids
2. **Print Error Messages**: Clear feedback when API used incorrectly
3. **Parameter Validation**: Explicit checks for ambiguous usage
4. **Unified Implementation**: Single method handles both approaches internally

---

## Critical Success Metrics 🎯

- **Correctness**: All business rules enforced consistently
- **Performance**: Process 1000 embryos in < 10 seconds
- **Reliability**: Zero data corruption across 1000 operations
- **Usability**: Clear error messages guide users
- **Maintainability**: New developers understand in < 1 hour

## Key Implementation Decisions 🔧

### Conflict Policy
- **Default**: duplicate phenotype = no-op, print `WARN(dupe)`
- **Explicit**: `replace=True` allows swapping existing phenotype
- **DEAD**: embryo-level check prevents conflicts

### Range Grammar
- Support one format initially: `"30:50"`, `"200:"`, `"all"`
- Defer complex parsing until proven needed

## Expected Result
- **Simple architecture**: ~800 lines total vs 1500+ previous
- **Phased approach**: Working functionality at each stage
- **Unambiguous API**: Clear either/or parameter usage with _select_mode()
- **Atomic operations**: Corruption-proof I/O
- **Clear error messages**: Users understand mistakes immediately
- **Full inheritance**: AnnotationBatch gets all functionality automatically

---

---

## Implementation Logging Strategy 📋

### Daily Progress Tracking
**All implementation work must be logged in `implementation_log.md`**

**Required Log Entries:**
- **Start of each phase**: Goals, expected timeline
- **End of each day**: What worked, what failed, decisions made
- **Test results**: Script names, pass/fail, performance metrics
- **Problem resolution**: Issues encountered and solutions
- **Decision rationale**: Why specific approaches were chosen
- **Git commits**: All commit hashes, what each commit contains, branch information

**Log Format:**
```markdown
## [DATE] - Phase X Day Y

### Goals for Today
- [ ] Specific task 1
- [ ] Specific task 2

### Work Completed
- ✅ Task completed with results
- ❌ Task failed with reason

### Tests Created/Run
- `test_script_name.py` - PASS/FAIL - 0.5s runtime
- Performance: Processed N embryos in X seconds

### Git Activity
- Branch: `branch-name`
- Commits made:
  - `abc1234` - "Add feature X" 
  - `def5678` - "Fix bug in Y"
- Files changed: `file1.py`, `file2.py`

### Problems & Solutions
- Problem: Description
- Solution: What fixed it
- Impact: How this affects timeline

### Next Steps
- Tomorrow's priority tasks
```

---

## Implementation Priority 📋

**Phase 1 - MVP Foundation (Days 1-2)**
1. ✅ Archive files 
2. 🔄 Create working MVP with BaseFileHandler composition
3. 🔄 Simple SAM2 extraction and embryo structure creation
4. 🔄 Basic `add_phenotype()` with embryo_id + target="all" only
5. 🔄 Required tests: `test_mvp_basic.py`, `test_sam2_extraction.py`

**📝 Phase 1 Logging Requirements:**
- Document SAM2 structure analysis results
- Record test pass/fail for each script
- Benchmark loading times for real data files

**Phase 2 - Parameter Validation (Days 3-4)**
1. ⏳ Add `_select_mode()` for either/or parameter validation
2. ⏳ Support snip_ids approach: `add_phenotype(snip_ids=["s1", "s2"])`
3. ⏳ Basic frame range parsing for target parameter
4. ⏳ Required tests: `test_parameter_validation.py`, `test_both_apis.py`

**📝 Phase 2 Logging Requirements:**
- Document API design decisions and trade-offs
- Record validation error message clarity testing
- Performance comparison: embryo_id vs snip_ids approaches

**Phase 3 - Business Rules (Days 5-7)**
1. ⏳ DEAD exclusivity validation (embryo-level)
2. ⏳ Basic phenotype validation with controlled vocabularies
3. ⏳ Simple error messages with print statements
4. ⏳ Required tests: `test_dead_validation.py`, `test_business_rules.py`

**📝 Phase 3 Logging Requirements:**
- Document business rule test scenarios and results
- Record edge cases discovered during testing
- Validate error message user-friendliness

**Phase 4 - Batch System & Tutorials (Week 2)**
1. ⏳ AnnotationBatch inheriting from EmbryoMetadata
2. ⏳ Apply mechanism with conflict resolution
3. ⏳ Create tutorial notebooks:
   - `biologist_tutorial.ipynb` - For biology researchers  
   - `pipeline_integration_tutorial.ipynb` - For pipeline developers
4. ⏳ Integration testing with real pipeline data

**📝 Phase 4 Logging Requirements:**
- Document batch operation performance benchmarks
- Record tutorial feedback from test users
- Final system integration test results

**Pre-Implementation Requirement: Update Script Setup**
- **🔄 Create `scripts/pipelines/07_embryo_metadata_update.py`** - Standardized script for updating embryo metadata
- **🔄 Save outputs to `data/embryo_metadata/`** - Organized storage location
- **🔄 Integrate with existing pipeline numbering** - Follows 05_sam2_qc, 06_xxx pattern

**Status**: Ready to begin Phase 1 after script creation