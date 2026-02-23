# SAM2 Pipeline Validation Commands

**CRITICAL**: The refactor-008 document incorrectly claimed "production ready" status without actual testing. These commands will perform the first real validation.

## Prerequisites
```bash
# Ensure correct environment
conda activate segmentation_grounded_sam
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
```

## Validation Tests (Run in Order)

### 1. Minimal Build03A Test (2-3 minutes)
**Purpose**: Test SAM2→Build03A integration with minimal sample size
**Expected**: Should process 2 embryos and create metadata files

```bash
python -m src.run_morphseq_pipeline.cli build03a \
  --root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --test-suffix minimal_build03a_test \
  --exp 20250612_30hpf_ctrl_atf6 \
  --sam2-csv sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv \
  --by-embryo 2 --frames-per-embryo 1
```

**Success Indicators**:
- Creates: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/minimal_build03a_test/`
- Files: `embryo_metadata_files/` and `combined_metadata_files/embryo_metadata_df01.csv`
- No KeyError crashes

### 2. Build03A + Build04 Test (3-5 minutes)
**Purpose**: Test Build03A→Build04 format compatibility
**Run only if Test 1 succeeds**

```bash
python -m src.run_morphseq_pipeline.cli build04 \
  --root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --test-suffix minimal_build03a_test \
  --exp 20250612_30hpf_ctrl_atf6
```

**Success Indicators**:
- Creates: `embryo_metadata_df02.csv` 
- No `predicted_stage_hpf` KeyError
- QC processing completes

### 3. Full E2E Pipeline Test (5-10 minutes)
**Purpose**: Complete Build03A→Build04→Build05 chain
**Run only if Tests 1 & 2 succeed**

```bash
python -m src.run_morphseq_pipeline.cli e2e \
  --root /net/trapnell/vol1/home/nlammers/projects/data/morphseq \
  --test-suffix full_e2e_test \
  --exp 20250612_30hpf_ctrl_atf6 \
  --sam2-csv sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv \
  --by-embryo 3 --frames-per-embryo 1 \
  --train-name sam2_validation_test
```

**Success Indicators**:
- Creates training folder structure
- All metadata files generated
- No crashes in VAE/pythae components

## Expected Failure Points

### Most Likely Failures:
1. **Missing SAM2 CSV columns**: Format bridge incomplete
2. **Path resolution errors**: Mask files not found  
3. **VAE/pythae errors**: Despite being installed, integration may fail
4. **Memory/GPU issues**: Large model loading

### How to Interpret Results:

**COMPLETE SUCCESS**: All 3 tests pass → Pipeline actually works
**PARTIAL SUCCESS**: Test 1 works, others fail → SAM2 integration OK, downstream issues
**IMMEDIATE FAILURE**: Test 1 crashes → Core SAM2 integration broken

## Files to Check After Each Test:

### After Test 1 (Build03A):
```bash
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/minimal_build03a_test/
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/minimal_build03a_test/combined_metadata_files/
```

### After Test 2 (Build04):
```bash
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/minimal_build03a_test/combined_metadata_files/embryo_metadata_df02.csv
```

### After Test 3 (E2E):
```bash
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/full_e2e_test/
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/training_data/sam2_validation_test/
```

## Next Steps Based on Results:

- **All pass**: Update refactor-008 to "ACTUALLY PRODUCTION READY"
- **Partial success**: Document specific failure points and fix iteratively  
- **Complete failure**: Major debugging required, "production ready" claim was premature

---

*Created August 31, 2025 - First real validation attempt after discovering false "production ready" claims*