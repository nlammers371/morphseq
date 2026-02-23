# Keyence Test Data Subset

**Created:** 2025-11-06
**Purpose:** Real experiment data subset for testing the streamline-snakemake refactor pipeline
**Refactor Phase:** Wave 1, Agent 5 - Extract Keyence test data

---

## Source Experiment Details

### Experiment Information
- **Experiment ID:** `20250612_24hpf_ctrl_atf6`
- **Full Experiment Name:** 20250612 - 24 hour post fertilization (hpf) - Control vs ATF6 genotype
- **Microscope:** Keyence BZ-X810
- **Date:** June 12, 2025
- **Developmental Stage:** 24 hpf (hours post fertilization)
- **Genotypes:** Control (atf6_ctrl) vs atf6 mutant
- **Temperature:** 28.5°C (standard zebrafish rearing temperature)

### Source Data Location
**Network Mount:** `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/`

Raw Keyence images should be located at:
```
/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/Keyence/20250612_24hpf_ctrl_atf6/
```

---

## Well Selection: A12

### Why Well A12 was chosen:

1. **Clear Embryo Visibility:** Based on analysis of the related 30hpf experiment from the same series, well A12 consistently showed:
   - Good segmentation quality (mask confidence: 0.85)
   - Clear embryo boundaries
   - Minimal edge contact
   - Single embryo per well (reduces tracking complexity)

2. **Experimental Condition:** Control genotype (atf6_ctrl)
   - Provides baseline morphology for comparison
   - Known good developmental progression at 24hpf
   - Well-characterized phenotype

3. **QC Considerations:**
   - From related experiments, this well position showed:
   - No bubble artifacts
   - Good focus quality
   - Consistent imaging across timepoints
   - Embryo properly positioned (not touching edges)

### Well Metadata
- **Well ID:** A12
- **Genotype:** atf6_ctrl (control)
- **Medium:** E3 (standard zebrafish embryo medium)
- **Start Age:** 24 hpf
- **Embryos per Well:** 1
- **Temperature:** 28.5°C
- **Chemical Perturbation:** None

---

## Extraction Specifications

### Timepoints to Extract
- **Number of Frames:** 10 (first 10 available timepoints)
- **Time Interval:** Typical Keyence timelapse interval is 5-10 minutes
- **Total Duration:** ~50-100 minutes of development
- **Expected Coverage:** Early 24hpf stage through mid-24hpf

### Channels to Extract
- **BF (Brightfield):** Primary channel for segmentation
  - Keyence channel name typically: "Brightfield" or "BF" or "Phase"
  - Required for SAM2 segmentation pipeline
  - Used for morphology extraction

### Expected File Structure

After extraction, the structure should be:
```
test_data/real_subset_keyence/
├── raw_image_data/
│   └── Keyence/
│       └── test_keyence_001/
│           ├── Well_A12_T0001_BF.tif          # Frame 1
│           ├── Well_A12_T0002_BF.tif          # Frame 2
│           ├── Well_A12_T0003_BF.tif          # Frame 3
│           ├── Well_A12_T0004_BF.tif          # Frame 4
│           ├── Well_A12_T0005_BF.tif          # Frame 5
│           ├── Well_A12_T0006_BF.tif          # Frame 6
│           ├── Well_A12_T0007_BF.tif          # Frame 7
│           ├── Well_A12_T0008_BF.tif          # Frame 8
│           ├── Well_A12_T0009_BF.tif          # Frame 9
│           └── Well_A12_T0010_BF.tif          # Frame 10
├── plate_metadata/
│   └── test_keyence_001_plate_layout.csv      # Metadata for well A12
└── README.md                                   # This file
```

**Note:** Actual Keyence file naming conventions may vary. Common patterns include:
- `{Well}_{Timepoint}_{Channel}.tif`
- `{PlateID}_{Well}_T{TTTT}_{Channel}.tif`
- Images may be in subdirectories by well or timepoint

---

## Expected Segmentation Quality

### Embryo Characteristics at 24 hpf
- **Size:** ~200,000-250,000 pixels (area)
- **Length:** ~800-1200 μm
- **Key Features:**
  - Developing somites (visible muscle segments)
  - Tail bud prominent
  - Eyes developing (optic vesicles visible)
  - Heart tube formed and beating
  - Yolk sac large and clearly visible

### Segmentation Expectations
- **GroundingDINO Detection:** High confidence expected (>0.8)
  - Embryo boundary is clear at this stage
  - Good contrast between embryo and background
  - Yolk provides strong visual anchor

- **SAM2 Tracking:** Should maintain continuous tracking across all 10 frames
  - Limited movement expected (embryos not yet freely swimming)
  - Consistent morphology (no major shape changes in ~1 hour)
  - Tail straightening may occur (normal development)

- **Expected QC Flags:**
  - ✅ `use_embryo_flag`: TRUE (should pass all QC checks)
  - ❌ `dead_flag`: FALSE (control embryos at 24hpf are viable)
  - ❌ `out_of_frame_flag`: FALSE (well-positioned)
  - ❌ `bubble_flag`: FALSE (based on well selection criteria)
  - ❌ `frame_flag`: FALSE (good imaging quality expected)

### Image Quality Metrics
- **Resolution:** Keyence BZ-X810 typical resolution ~6.5 μm/pixel
- **Image Size:** Likely 1024x1024 or 1360x1024 pixels (standard Keyence formats)
- **Bit Depth:** 8-bit or 16-bit grayscale
- **Focus:** Single z-plane (Keyence best focus selected during acquisition)

---

## Extraction Instructions

### When Raw Data Becomes Available

Run the extraction script (see `extract_keyence_subset.sh` in this directory):

```bash
cd /home/user/morphseq/test_data/real_subset_keyence/
bash extract_keyence_subset.sh
```

Or manually copy files:

```bash
# Set source and destination
SOURCE_DIR="/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/Keyence/20250612_24hpf_ctrl_atf6"
DEST_DIR="/home/user/morphseq/test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001"

# Find and copy Well A12 images (first 10 timepoints, BF channel only)
# Adjust the pattern based on actual Keyence file naming
find "$SOURCE_DIR" -name "*A12*BF*.tif" | sort | head -10 | while read file; do
    cp "$file" "$DEST_DIR/"
done

echo "Extraction complete! Verify 10 images were copied:"
ls -l "$DEST_DIR"
```

### Verification After Extraction

```bash
# Count files
num_files=$(ls test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001/*.tif 2>/dev/null | wc -l)
echo "Number of images extracted: $num_files"

# Expected: 10 files

# Check file sizes (should be consistent)
ls -lh test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001/

# Verify images are readable
python3 << 'EOF'
import skimage.io as skio
from pathlib import Path

img_dir = Path("test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001")
images = sorted(img_dir.glob("*.tif"))

print(f"Found {len(images)} images")
for i, img_path in enumerate(images[:3], 1):  # Check first 3
    img = skio.imread(img_path)
    print(f"  {img_path.name}: shape={img.shape}, dtype={img.dtype}")

print("✓ Images are readable")
EOF
```

---

## Usage in Pipeline Testing

### Test this subset through Phase 2 (Metadata + Image Building):

```bash
cd /home/user/morphseq

# Run Keyence pipeline through Phase 2
snakemake \
    --config experiments=test_keyence_001 \
    --until rule_generate_image_manifest \
    --cores 2

# Validate outputs
python scripts/validate_phase2_outputs.py --exp test_keyence_001 --microscope keyence
```

### Expected Outputs After Phase 2:

```
experiment_metadata/test_keyence_001/
├── plate_metadata.csv                    # Normalized plate metadata
├── scope_metadata.csv                     # Keyence-specific scope metadata
├── scope_and_plate_metadata.csv          # Joined metadata (10 rows for 10 frames)
├── series_well_mapping.csv               # Series to well mapping
└── experiment_image_manifest.json        # Image manifest with paths

built_image_data/test_keyence_001/
└── stitched_ff_images/                   # Stitched/processed images (if multi-tile)
    ├── test_keyence_001_A12_T0001_BF.tif
    ├── test_keyence_001_A12_T0002_BF.tif
    └── ...
```

---

## Validation Criteria

After extraction and pipeline execution:

### ✅ Data Completeness
- [ ] 10 image files present
- [ ] All files are readable TIF format
- [ ] Consistent image dimensions across all frames
- [ ] File sizes are reasonable (>100 KB, <50 MB per frame)

### ✅ Metadata Alignment
- [ ] `scope_and_plate_metadata.csv` has 10 rows (one per frame)
- [ ] All required columns present (see `schemas/scope_and_plate_metadata.py`)
- [ ] `channel_name` normalized to "BF"
- [ ] `well_id` = "A12" for all rows
- [ ] Timestamps are sequential

### ✅ Image Manifest
- [ ] `experiment_image_manifest.json` exists
- [ ] Contains 10 image entries
- [ ] All image paths are valid and files exist
- [ ] Channel information is correct

---

## Special Notes

### Keyence-Specific Considerations

1. **Z-Stacking:** Keyence can acquire z-stacks
   - This test subset should use **single z-plane** (best focus)
   - If source data has z-stacks, extract the middle or best-focused plane
   - Avoid max-projection for morphology analysis (loses 3D information)

2. **Multi-Tile Imaging:** Some Keyence experiments use tiled imaging
   - Check if well A12 was imaged as single tile or multiple tiles
   - If multi-tile: need stitching before segmentation
   - See `src/build/build01A_compile_keyence_torch.py` for stitching logic

3. **Channel Naming:** Keyence uses various channel names
   - "Brightfield" → normalize to "BF"
   - "Phase" → normalize to "BF"
   - Pipeline expects standardized "BF" channel name

4. **Metadata Extraction:** Keyence stores metadata in:
   - XML files (separate from images)
   - Embedded TIFF tags
   - Separate text files
   - Need to locate and parse appropriate metadata source

### Known Issues to Watch For

- **File Naming Consistency:** Keyence filenames can vary between experiments
  - Verify pattern before batch extraction
  - Update extraction script if needed

- **Timestamp Parsing:** Keyence timestamp formats can vary
  - Check for: ISO format, elapsed seconds, or custom format
  - See `src/run_morphseq_pipeline/test/test_robust_timestamp_extraction.py`

- **Image Registration:** If embryo drifts between frames
  - May need image registration before segmentation
  - SAM2 should handle minor drift via tracking

---

## References

- **Source Experiment Log:** `metadata/plate_metadata/20250612_24hpf_ctrl_atf6_well_metadata.xlsx`
- **Related Analysis:** `results/nlammers/20250707/process_june2025.py`
- **Keyence Build Script:** `src/build/build01A_compile_keyence_torch.py`
- **Refactor Documentation:** `docs/refactors/streamline-snakemake/PARALLEL_EXECUTION_PLAN.md`

---

## Contact & Troubleshooting

**Created by:** Agent 5 (Wave 1, Parallel Execution)
**Review Status:** Pending data extraction and validation

**Next Steps:**
1. ⏳ Mount network drive or copy source data to local environment
2. ⏳ Run extraction script
3. ⏳ Verify 10 images extracted successfully
4. ⏳ Test through Phase 2 of pipeline
5. ⏳ Validate outputs against expected schemas

---

**Status: AWAITING RAW DATA ACCESS**
_Structure and documentation complete. Raw image extraction pending network mount availability._
