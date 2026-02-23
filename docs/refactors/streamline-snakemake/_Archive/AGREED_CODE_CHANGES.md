# Agreed-Upon Code Changes to Refactoring Plan

**Date:** 2025-11-06
**Status:** APPROVED

This document summarizes the code changes agreed upon during plan review.

---

## 1. Series-to-Wells Mapping: Keep Separate, Make Microscope-Specific

**Original Plan:** Single shared `rule map_series_to_wells`

**Agreed Change:** Separate microscope-specific implementations

**Rationale:** Series mapping logic differs significantly between microscopes:
- YX1 uses implicit positional or explicit series_number_map
- Keyence has different file structure and conventions

**Implementation:**
```
metadata_ingest/mapping/
├── series_well_mapper_yx1.py       # YX1-specific logic
├── series_well_mapper_keyence.py   # Keyence-specific logic
└── align_scope_plate.py            # Shared join (unchanged)
```

**Snakemake:**
```python
rule map_series_to_wells_yx1:
    input:
        plate = "input_metadata_alignment/{exp}/raw_inputs/plate_layout.csv",
        scope = "input_metadata_alignment/{exp}/raw_inputs/yx1_scope_raw.csv"
    output:
        mapping = "input_metadata_alignment/{exp}/series_mapping/series_well_mapping.csv",
        provenance = "input_metadata_alignment/{exp}/series_mapping/mapping_provenance.json"

rule map_series_to_wells_keyence:
    # Similar structure for Keyence
```

**Files to Update in Original Plan:**
- `preliminary_rules.md`: Split `rule map_series_to_wells` into microscope-specific versions
- `processing_files_pipeline_structure_and_plan.md`: Update module listing to show both mappers

---

## 2. Snip Processing: Combine Rules with Config Flag

**Original Plan:** Separate `rule extract_snips` (TIF) → `rule process_snips` (JPEG)

**Agreed Change:** Single `rule process_snips` with `--save-raw-crops` config flag (default=True)

**Rationale:**
- Raw TIF crops only used for debugging
- Combining reduces I/O and simplifies DAG
- Config flag preserves debugging capability when needed

**Implementation:**
```python
# config/defaults.yaml
snip_processing:
  save_raw_crops: true  # Default ON for debugging; set false in production

# snip_processing/process_snips.py
def process_snips(tracking_csv, images_dir, output_dir, config):
    """
    Extract, rotate, augment snips in single pass.

    Outputs:
        - processed/{snip_id}.jpg  (always)
        - raw_crops/{snip_id}.tif  (if save_raw_crops=True)
    """
    for snip in tracking:
        raw_crop = extract_crop(snip)

        if config['save_raw_crops']:
            save_tif(raw_crop, output_dir / 'raw_crops' / f'{snip.id}.tif')

        processed = apply_processing_pipeline(raw_crop, config)
        save_jpeg(processed, output_dir / 'processed' / f'{snip.id}.jpg')
```

**Snakemake:**
```python
rule process_snips:
    input:
        tracking = "segmentation/{exp}/segmentation_tracking.csv",
        images = "built_image_data/{exp}/stitched_ff_images/"
    output:
        processed = directory("processed_snips/{exp}/processed/"),
        manifest = "processed_snips/{exp}/snip_manifest.csv"
    params:
        save_raw = config.get('snip_processing', {}).get('save_raw_crops', True)
```

**Files to Update:**
- `preliminary_rules.md`: Replace separate `rule extract_snips` and `rule process_snips` with combined rule
- `processing_files_pipeline_structure_and_plan.md`: Update snip_processing/ module description

---

## 3. QC Reports: Add as "Dump" Rule (Beyond Initial Refactor)

**Original Plan:** No QC reporting

**Agreed Change:** Add `rule generate_qc_reports` as optional final step (no dependencies on it)

**Rationale:**
- Useful for human review and debugging
- No other rules need these reports (doesn't block pipeline)
- Beyond scope of initial refactor but easy to add

**Implementation:**
```python
# quality_control/reporting/generate_qc_summary.py
def generate_qc_reports(consolidated_qc_csv, output_dir):
    """
    Generate human-readable QC summaries and plots.

    Outputs:
        - qc_summary.html  (interactive dashboard)
        - flag_distributions.png  (% embryos per flag)
        - flag_correlations.png  (heatmap of flag co-occurrence)
        - per_well_summary.csv  (aggregated stats)
    """
    pass
```

**Snakemake:**
```python
rule generate_qc_reports:
    input:
        qc = "quality_control/{exp}/consolidated_qc_flags.csv",
        features = "computed_features/{exp}/consolidated_snip_features.csv"
    output:
        html = "quality_control/{exp}/reports/qc_summary.html",
        plots = directory("quality_control/{exp}/reports/plots/")
    run:
        from quality_control.reporting.generate_qc_summary import generate_qc_reports
        generate_qc_reports(input.qc, output.html)
```

**Notes:**
- NOT included in main `all` target initially
- Can be run manually: `snakemake generate_qc_reports --config experiments=<exp>`
- Implementation deferred to post-MVP

**Files to Update:**
- Add to `processing_files_pipeline_structure_and_plan.md` under `quality_control/reporting/`
- Note as "optional, beyond initial scope" in implementation plan

---

## 4. Testing Strategy: Both Microscopes → Single Experiment

**Original Plan:** Not specified

**Agreed Change:** Test BOTH microscopes through Phase 2, then ONE experiment through Phase 8

**Rationale:**
- Metadata extraction is microscope-specific (must validate both)
- Segmentation onwards is microscope-agnostic (one test sufficient)
- Focuses testing effort on highest-risk areas (scope-specific logic)

**Test Data Structure:**
```
test_data/
├── real_subset_yx1/          # YX1: 2 wells, 10 frames each
│   └── test_yx1_001/
├── real_subset_keyence/      # Keyence: 1 well, 10 frames
│   └── test_keyence_001/
└── expected_outputs/         # Golden outputs for validation
```

**Test Schedule:**
- **Phase A:** Run YX1 + Keyence through Phase 2, validate both
- **Phase B:** Continue YX1 through Phases 3-8, validate end-to-end

**Files to Create:**
- `scripts/extract_test_subset.sh` (extract wells from existing experiments)
- `scripts/validate_phase2_outputs.py` (validate metadata for both microscopes)
- `scripts/validate_full_pipeline.py` (validate end-to-end)
- `scripts/rapid_test.sh` (quick iteration testing)

---

## Summary of Documentation Updates Needed

**Original planning documents to update:**

1. **`refactoring_plan.md`**
   - Note series_well_mapper is microscope-specific (update Week 2 tasks)
   - Update snip_processing module description (combined rule)
   - Add quality_control/reporting/ directory (optional)

2. **`preliminary_rules.md`**
   - Split `rule map_series_to_wells` into `_yx1` and `_keyence` versions
   - Combine `rule extract_snips` + `rule process_snips` → `rule process_snips` (with config)
   - Add `rule generate_qc_reports` (Phase 6+, optional)

3. **`processing_files_pipeline_structure_and_plan.md`**
   - Update `metadata_ingest/mapping/` to show two mapper files
   - Update `snip_processing/` to describe combined processing
   - Add `quality_control/reporting/` package (optional)

4. **`data_validation_plan.md`**
   - No changes needed (validation strategy unchanged)

5. **`data_ouput_strcutre.md`**
   - Add `quality_control/{exp}/reports/` directory (optional)
   - Note that `processed_snips/{exp}/raw_crops/` is conditional on config

---

## Config File Changes

**New config structure:**

```yaml
# config/defaults.yaml

# Snip processing options
snip_processing:
  save_raw_crops: true       # Save TIF crops (default ON for debugging)
  rotation: true             # Apply PCA rotation
  clahe: true               # Apply CLAHE equalization
  noise_augmentation: true  # Add background noise

# QC reporting (optional)
qc_reporting:
  enabled: false            # Generate QC reports (default OFF)
  plot_formats: [png, svg]  # Output formats
```

---

**All changes captured and ready for implementation! ✓**
