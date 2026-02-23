# YX1 Test Data Subset - Extraction Documentation

**Status:** PLACEHOLDER - Raw data extraction pending
**Created:** 2025-11-06
**Purpose:** Test subset for streamline-snakemake refactor (Wave 1, Agent 4)

---

## Critical Finding: Experiment ID Discrepancy

The original task requested extraction from experiment **20250911**, but this experiment **does not exist** in the repository.

### Available Alternatives:

1. **Experiment 20250912** (Recommended)
   - Date: September 12, 2025 (one day after requested date)
   - Plate metadata: EXISTS at `/metadata/plate_metadata/20250912_well_metadata.xlsx`
   - Microscope: To be confirmed (metadata format is plate-map style)
   - Status: Likely candidate for extraction

2. **Experiment 20240418** (Confirmed YX1)
   - Date: April 18, 2024
   - Microscope: **YX1** (confirmed in built_metadata_files)
   - Wells available: A01, C01, and others
   - Status: Confirmed YX1 experiment with complete metadata

---

## Extraction Requirements

### Wells to Extract
- **Well A6**: Primary test well (different condition 1)
- **Well B4**: Secondary test well (different condition 2)

**Selection Criteria:**
- Known good segmentation quality
- Different experimental conditions (for testing metadata propagation)
- Multiple embryos per well (1-2 embryos preferred for A6, 1 for B4)
- Clean tracking across timepoints

### Timepoints
- **Frames 0-9** (first 10 timepoints only)
- Approximately 30-minute intervals typical for YX1 experiments

### Channels
- **BF (Brightfield)**: Required - primary segmentation channel
- **Fluorescence**: Optional - include if available (e.g., GFP, RFP)

Expected YX1 channel names (pre-normalization):
- "EYES - Dia" → normalizes to "BF"
- Additional fluorescence channels if present

---

## Data Source Location

### Expected Raw Data Structure

The raw YX1 data should be located on the file server at:
```
[DATA_ROOT]/raw_image_data/YX1/[EXPERIMENT_ID]/
```

For experiment 20250912 or suitable YX1 alternative, expected structure:
```
YX1/
└── 20250912/  (or confirmed YX1 experiment)
    ├── [EXPERIMENT]_A06_Seq0001.nd2  (Well A6 time series)
    ├── [EXPERIMENT]_B04_Seq0001.nd2  (Well B4 time series)
    └── ... (other wells)
```

OR already-stitched format:
```
YX1/
└── 20250912/
    └── stitched_FF_images/
        ├── [EXPERIMENT]_A06_t0000.tif
        ├── [EXPERIMENT]_A06_t0001.tif
        ... (through t0009 for both wells)
        ├── [EXPERIMENT]_B04_t0000.tif
        ... (through t0009)
```

---

## Extraction Instructions

### Step 1: Locate Source Data

```bash
# On the file server or data storage system:
DATA_ROOT="/path/to/morphseq_playground"  # or appropriate data root
EXP_ID="20250912"  # or confirmed YX1 experiment

# Check for raw ND2 files
ls -lh ${DATA_ROOT}/raw_image_data/YX1/${EXP_ID}/*A06*.nd2
ls -lh ${DATA_ROOT}/raw_image_data/YX1/${EXP_ID}/*B04*.nd2

# OR check for already-stitched images
ls -lh ${DATA_ROOT}/built_image_data/stitched_FF_images/${EXP_ID}/*A06*
ls -lh ${DATA_ROOT}/built_image_data/stitched_FF_images/${EXP_ID}/*B04*
```

### Step 2: Extract Well A6 (Frames 0-9)

```bash
# For ND2 format:
# Use nd2 extraction tool to get first 10 frames
python -m src.image_building.yx1.extract_frames \
    --input ${DATA_ROOT}/raw_image_data/YX1/${EXP_ID}/${EXP_ID}_A06_Seq0001.nd2 \
    --output test_data/real_subset_yx1/raw_image_data/YX1/test_yx1_001/ \
    --frames 0-9 \
    --well A06

# For stitched format:
# Copy first 10 timepoints
for t in $(seq -w 0 9); do
    cp ${DATA_ROOT}/built_image_data/stitched_FF_images/${EXP_ID}/*A06*t00${t}*.tif \
       test_data/real_subset_yx1/raw_image_data/YX1/test_yx1_001/
done
```

### Step 3: Extract Well B4 (Frames 0-9)

```bash
# Repeat Step 2 for Well B4
# Ensure consistent naming: test_yx1_001_B04_t0000.tif format
```

### Step 4: Extract/Create Plate Metadata

```bash
# Extract relevant rows from source experiment plate layout
python << 'EOF'
import pandas as pd

# Load source plate metadata
source_meta = pd.read_excel(
    'metadata/plate_metadata/${EXP_ID}_well_metadata.xlsx'
)

# Filter for wells A6 and B4
subset = source_meta[source_meta['well'].isin(['A06', 'B04'])].copy()

# Rename experiment ID to test_yx1_001
subset['experiment_date'] = 'test_yx1_001'

# Save to test data location
subset.to_csv(
    'test_data/real_subset_yx1/plate_metadata/test_yx1_001_plate_layout.csv',
    index=False
)
EOF
```

---

## Expected Contents After Extraction

```
test_data/real_subset_yx1/
├── raw_image_data/
│   └── YX1/
│       └── test_yx1_001/
│           ├── test_yx1_001_A06_t0000_BF.tif  (or .nd2 format)
│           ├── test_yx1_001_A06_t0001_BF.tif
│           ... (through t0009 for Well A6, all channels)
│           ├── test_yx1_001_B04_t0000_BF.tif
│           ... (through t0009 for Well B4, all channels)
│           └── [fluorescence channels if available]
│
├── plate_metadata/
│   └── test_yx1_001_plate_layout.csv
│       Columns expected:
│       - well: "A06", "B04"
│       - experiment_date: "test_yx1_001"
│       - genotype, treatment, temperature, etc.
│
└── README.md (this file)
```

---

## Well Selection Rationale

### Why Wells A6 and B4?

These wells should be selected based on:

1. **Different Experimental Conditions**
   - A6: Condition 1 (e.g., wildtype control, specific genotype)
   - B4: Condition 2 (e.g., treatment, different genotype)
   - Purpose: Test metadata propagation and grouping logic

2. **Known Segmentation Quality**
   - Clean embryo boundaries
   - Minimal occlusion or overlap
   - Good tracking continuity across all 10 frames
   - Purpose: Validate segmentation pipeline without QC noise

3. **Embryo Count**
   - A6: 1-2 embryos (tests multi-embryo tracking)
   - B4: 1 embryo (simpler case for validation)
   - Purpose: Test both single and multi-embryo scenarios

4. **Spatial Distribution**
   - Different plate regions (row A vs row B)
   - Purpose: Test well position encoding

---

## Data Provenance

### Source Information (to be filled after extraction)

- **Source Experiment:** _[To be determined: 20250912 or alternative]_
- **Microscope:** YX1
- **Acquisition Date:** _[To be filled]_
- **Original Well IDs:** A6, B4
- **Original Frame Range:** _[To be filled with source frame indices]_
- **Extraction Date:** _[To be filled]_
- **Extracted By:** _[To be filled]_

### Known QC Issues (for testing QC flags)

Document any known issues in the source wells that should trigger QC flags:

- **Well A6:**
  - [ ] Edge contact: _[Document if present]_
  - [ ] Death event: _[Document if present]_
  - [ ] Focus issues: _[Document if present]_
  - [ ] Tracking discontinuities: _[Document if present]_

- **Well B4:**
  - [ ] Edge contact: _[Document if present]_
  - [ ] Death event: _[Document if present]_
  - [ ] Focus issues: _[Document if present]_
  - [ ] Tracking discontinuities: _[Document if present]_

---

## Expected Segmentation Quality

### Well A6 (Target: HIGH QUALITY)
- Clear embryo boundaries
- Consistent size and shape across frames
- No edge contact in first 10 frames
- Good contrast in BF channel
- Expected embryo count: 1-2

### Well B4 (Target: HIGH QUALITY)
- Clear embryo boundaries
- Consistent tracking
- No artifacts or debris
- Good focus across all frames
- Expected embryo count: 1

---

## Channels Available

### Confirmed Channels (to be updated after extraction)
- [ ] BF (Brightfield) - **REQUIRED**
- [ ] GFP - _[Check if available]_
- [ ] RFP - _[Check if available]_
- [ ] Other: _[Specify]_

### Channel Naming Convention

YX1 raw channel names → Normalized names:
- "EYES - Dia" → "BF"
- "EYES - GFP" → "GFP" (if present)
- "EYES - RFP" → "RFP" (if present)

---

## Testing Integration

Once extracted, this test data will be used for:

1. **Phase 1-2 Testing (Metadata + Image Building)**
   - Plate metadata normalization
   - YX1 scope metadata extraction
   - Series-to-wells mapping
   - Channel name normalization
   - Image manifest generation

2. **Phase 3 Testing (Segmentation)**
   - GroundingDINO detection
   - SAM2 tracking
   - Mask export validation
   - CSV formatting

3. **Phase 4-8 Testing (Snips → Analysis)**
   - Snip extraction and processing
   - Feature computation
   - QC flag generation
   - Analysis-ready table assembly

### Test Command

```bash
# After extraction, test pipeline through Phase 2:
snakemake \
    --config experiments=test_yx1_001 \
    --until rule_generate_image_manifest \
    --cores 2

# Validate outputs:
python scripts/validate_phase2_outputs.py test_yx1_001 yx1
```

---

## Next Steps

1. **Identify Data Location**: Confirm where YX1 experiment raw data is stored
2. **Select Final Experiment**: Choose between 20250912 or alternative YX1 experiment
3. **Verify Well Quality**: Check wells A6 and B4 meet quality criteria
4. **Perform Extraction**: Run extraction scripts above
5. **Validate Extraction**: Check file counts, sizes, and metadata
6. **Update This README**: Fill in provenance and actual extraction details
7. **Test Pipeline**: Run validation scripts

---

## Contact / Questions

If the source experiment 20250911 was a typo or placeholder:
- Check experiment 20250912 (exists, needs microscope confirmation)
- Or use experiment 20240418 (confirmed YX1)

The critical requirement is: **Any YX1 experiment with known good quality in wells A6 and B4, or equivalent wells that can be renamed for testing.**

---

**Status Summary:**
- Directory structure: CREATED
- Raw data: NOT YET EXTRACTED (no image data in repository)
- Plate metadata: NOT YET EXTRACTED (template needed)
- Pipeline testing: PENDING data extraction

**Ready for:** Data extraction when raw images are accessible
