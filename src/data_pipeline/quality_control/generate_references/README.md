# SA Reference Generation Scripts

**Purpose**: Generate surface area reference curves (p5, p50, p95 vs stage) from all build04 data.

**Date Created**: 2025-10-08

---

## Overview

These scripts build global reference curves for surface area QC by aggregating data across all experiments. The references are used by two-sided SA outlier detection to flag embryos that are too large (segmentation artifacts) or too small (incomplete masks, dead embryos).

---

## Usage

### 1. Generate Reference Curves

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/src/data_pipeline/quality_control/generate_references
conda activate segmentations_grounded_sam
python build_sa_reference.py
```

**Output**:
- `sa_reference_curves.csv` - p5/p50/p95 vs stage_hpf (saved to configured output directory)
- `reference_plot.png` - Visualization of curves and sample sizes

---

## Reference Dataset Filters

The reference is built from **true wild-type controls only**:

- `genotype in ['wik', 'ab', 'wik-ab']` OR `phenotype == 'wt'` OR `control_flag == True`
- `chem_perturbation == 'None'` or missing (no chemical treatments)
- `use_embryo_flag == True` (passed basic QC)

This ensures the reference represents **normal embryo development** without experimental perturbations.

---

## Algorithm

1. **Load all build04 CSVs** from `metadata/build04_output/qc_staged_*.csv`
2. **Filter for controls** (see above)
3. **Bin by stage** (0.5 hpf increments with ±0.25 hpf windows)
4. **Calculate percentiles** (p5, p50, p95) for each bin (minimum n=5)
5. **Fill edge bins** (forward/backward fill for early/late stages with sparse data)
6. **Smooth** with Savitzky-Golay filter (window=5, poly=2)
7. **Save** to CSV and generate plot

---

## Output Format

**sa_reference_curves.csv**:
```csv
stage_hpf,p5,p50,p95,n
0.25,400000,450000,500000,0
0.75,400000,450000,500000,0
...
30.25,850000,950000,1100000,637
...
```

Columns:
- `stage_hpf`: Center of 0.5 hpf bin
- `p5`: 5th percentile surface area (µm) - lower bound reference
- `p50`: 50th percentile (median) surface area (µm)
- `p95`: 95th percentile surface area (µm) - upper bound reference
- `n`: Sample size in this bin

---

## Maintenance

**When to regenerate**:
- After adding new experiments to build04
- If reference looks outdated (check plot for sample sizes)
- After major changes to segmentation pipeline

**Frequency**:
- Ad hoc when new data batches are processed
- Consider quarterly regeneration for large datasets

---

## Future Refactoring (TODO)

**Note**: In the next refactoring, this will be updated to use **surface_area files** directly instead of full build04 output CSVs. The current approach loads entire `qc_staged_*.csv` files which contain many unnecessary columns. A leaner approach would:

1. Use dedicated surface_area files with only: `embryo_id, stage_hpf, surface_area_um, genotype, chem_perturbation, use_flag`
2. Reduce memory footprint and I/O time
3. Make reference generation more modular and maintainable

This is noted for future pipeline reorganization.

---

## History

- **2025-10-08**: Initial implementation
  - Aggregates all build04 data
  - Filters for wt controls only
  - 0.5 hpf binning with Savitzky-Golay smoothing
  - Outputs CSV + plot
  - **Note**: Uses full build04 CSVs (will switch to surface_area files in future refactor)
