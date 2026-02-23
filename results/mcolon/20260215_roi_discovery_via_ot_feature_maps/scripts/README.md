# Phase 0 Execution Scripts

This directory contains numbered scripts for executing Phase 0 (1D S-bin localization) on real CEP290 data.

## Prerequisites

- Python environment: `/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python`
- CEP290 data CSV: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- Build02 yolk masks: `${PROJECT_ROOT}/morphseq_playground/segmentation/...` (see `--data-root` below)

## Execution Order

### Script 1: Reference Mask Selection

**Purpose:** Select a WT reference mask from the 47-49 hpf window.

**Command:**
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" scripts/s01_select_reference_mask.py

# If yolk masks are stored elsewhere:
"$PYTHON" scripts/s01_select_reference_mask.py \
    --data-root /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground
```

**Outputs:**
- `scripts/output/reference_mask_candidates/top_candidate_0.png` (top 3 candidates)
- `scripts/output/reference_mask_candidates/candidates_summary.csv`

**Action required:** Review visualizations and select one reference by `embryo_id` + `frame_index`.

---

### Script 2: Compute Optimal Transport Features

**Purpose:** Compute OT transport plans between reference and target embryos. Saves feature dataset with QC.

**Command:**
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

"$PYTHON" scripts/s02_compute_ot_features.py \
    --reference-embryo-id 20251112_H04_e01 \
    --reference-frame-index 39 \
    --n-wt 10 \
    --n-mut 10 \
    --output-dir scripts/output/phase0_run_001 \
    --data-root /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground
```

**Outputs:**
- `scripts/output/phase0_run_001/feature_dataset/features.zarr` - OT feature maps
- `scripts/output/phase0_run_001/qc/` - QC diagnostics

**Time:** ~10-30 minutes depending on sample size

---

### Script 3: Visualize WT vs Mutant Differences

**Purpose:** Create heat maps showing phenotype differences along the A-P axis.

**Command:**
```bash
"$PYTHON" scripts/s03_visualize_differences.py \
    --feature-dir scripts/output/phase0_run_001/feature_dataset \
    --output-dir scripts/output/phase0_run_001
```

**Outputs:**
- `scripts/output/phase0_run_001/viz/s_map_ref.png` - S-coordinate map
- `scripts/output/phase0_run_001/viz/cost_density_*.png` - Cost density heat maps
- `scripts/output/phase0_run_001/viz/sbin_comparison.png` - WT vs mutant bar chart
- `scripts/output/phase0_run_001/features_sbins.parquet` - S-bin feature table

**Time:** ~1-2 minutes

---

### Script 4: Run Classification and Bootstrap Analysis

**Purpose:** Compute AUROC localization, select best ROI interval, run bootstrap/permutation tests.

**Command:**
```bash
"$PYTHON" scripts/s04_run_classification.py \
    --sbin-features scripts/output/phase0_run_001/features_sbins.parquet \
    --output-dir scripts/output/phase0_run_001
```

**Outputs:**
- `scripts/output/phase0_run_001/results/auroc_curve.png` - AUROC vs S-position
- `scripts/output/phase0_run_001/results/selected_interval.json` - Best ROI
- `scripts/output/phase0_run_001/results/bootstrap_stability.json` - Confidence intervals
- `scripts/output/phase0_run_001/results/permutation_nulls.json` - Statistical significance

**Time:** ~2-5 minutes

---

## Complete Workflow Example

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

# Step 2: Compute OT features (compute once)
"$PYTHON" scripts/s02_compute_ot_features.py \
    --reference-embryo-id 20251112_H04_e01 \
    --reference-frame-index 39 \
    --n-wt 10 --n-mut 10 \
    --output-dir scripts/output/phase0_run_001

# Step 3: Create heat maps (iterate on visualization)
"$PYTHON" scripts/s03_visualize_differences.py \
    --feature-dir scripts/output/phase0_run_001/feature_dataset

# Step 4: Run classification (analyze results)
"$PYTHON" scripts/s04_run_classification.py \
    --sbin-features scripts/output/phase0_run_001/features_sbins.parquet
```

---

## Pipeline Design

The workflow is **modular** to support iterative analysis:

1. **Step 2 (OT computation)**: Expensive (~30 min) - Run once, save features
2. **Step 3 (Visualization)**: Fast (~2 min) - Iterate on smoothing, binning parameters
3. **Step 4 (Classification)**: Fast (~5 min) - Iterate on interval selection, bootstrap settings

---

## Additional Scripts

### Script 1b: QC Mask Visualization

**Purpose:** Render canonical-grid masks for reference, WT, and mutants to check alignment/scale.

**Command:**
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" scripts/s01b_visualize_masks.py \
    --reference-embryo-id 20251112_H04_e01 \
    --reference-frame-index 39 \
    --n-wt 10 \
    --n-mut 10 \
    --output-dir scripts/output/mask_qc \
    --data-root /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground
```

**Outputs:**
- `scripts/output/mask_qc/reference_mask.png`
- `scripts/output/mask_qc/wt_masks.png`
- `scripts/output/mask_qc/mutant_masks.png`

---

### Script 2c: Outlier Alignment/Mask Diagnostics

**Purpose:** Plot only IQR-flagged outliers with overlays + alignment metrics to help distinguish alignment failures vs mask-quality defects.

**Command:**
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" scripts/s02c_plot_outlier_diagnostics.py \
    --feature-dir scripts/output/phase0_run_001/feature_dataset
```

**Outputs:**
- `scripts/output/phase0_run_001/qc_outlier_diagnostics/qc4_outlier_*.png`
- `scripts/output/phase0_run_001/qc_outlier_diagnostics/qc4_outlier_diagnostics.csv`

---

## Phase 0 Pipeline Components

The modular pipeline separates concerns:

**Script 2 (OT Computation):**
1. Load reference + target masks (with yolk for alignment)
2. Compute OT transport plans → feature maps X (N×H×W×C)
3. Run QC suite (outlier detection, cost distribution)
4. Save feature dataset (Zarr format)

**Script 3 (Visualization):**
1. Build S-coordinate map (A-P axis parameterization)
2. Create cost density heat maps (WT vs mutant)
3. Aggregate into S-bins (K=10 bins along body axis)
4. Plot phenotype differences

**Script 4 (Classification):**
1. Compute AUROC for each S-bin independently
2. Select best contiguous interval (parsimony-based)
3. Bootstrap resampling for confidence intervals
4. Permutation testing for statistical significance

**Gates:**
- Gate 2.1: QC passes (no catastrophic alignment failures)
- Gate 3.1: S-coordinate covers >95% of mask pixels
- Gate 4.1: At least one bin has AUROC > 0.65
- Gate 4.2: Permutation p < 0.05

---

## Key Parameters

### Script 2 Options
- `--n-wt`, `--n-mut`: Sample sizes (default: 10 each)
- `--stage-window`: Developmental stage (default: "47-49" hpf)
- `--seed`: Random seed for reproducibility (default: 42)
- QC outlier threshold: **IQR multiplier = 2.0x** on `total_cost_C` (updated from 1.5x)

### Script 3 Options
- `--sigma`: Gaussian smoothing for cost density (default: 2.0)
- `--k-bins`: Number of S-bins (default: 10)

### Script 4 Options
- `--n-bootstrap`: Bootstrap iterations (default: 100)
- `--n-permutations`: Permutation null size (default: 1000)

---

## Notes

### Canonical Grid
- All OT computations use the canonical grid (256×576 at 10 µm/px)
- Raw masks are automatically aligned by the UOT pipeline (`use_canonical_grid=True`)
- No manual alignment required — just provide raw RLE-decoded masks + physical resolution

### Genotype Labels
- WT: `genotype == "cep290_wildtype"`
- Mutant: `genotype == "cep290_homozygous"`
- Excludes `cep290_unknown` and heterozygous

### Yolk Masks
- Yolk masks are required for canonical alignment (`use_yolk=True`).
- They are loaded from Build02 segmentation outputs via `--data-root`.
- The reference workflow in [results/mcolon/20260213_stream_d_reference_embryo](results/mcolon/20260213_stream_d_reference_embryo) uses the same loader.

### Timing
- **Script 1** (Reference selection): ~1-2 minutes
- **Script 2** (OT computation): ~10-30 minutes (20 samples × ~30-60 sec/pair)
- **Script 3** (Visualization): ~1-2 minutes
- **Script 4** (Classification): ~2-5 minutes

### Workflow Strategy
Since OT computation is expensive:
1. Run Script 2 **once** with desired sample size
2. Iterate on Scripts 3-4 to explore visualization and analysis parameters
3. Re-run Script 2 only if you need different samples or stage windows

---

## Troubleshooting

**Import errors:**
- Both scripts add `src/` to `sys.path` automatically
- Ensure the segmentation_grounded_sam environment is active

**Missing data:**
- Verify CEP290 CSV exists at expected path
- Check that `embryo_id` exists in the CSV

**OT failures:**
- Check that masks decode correctly from RLE
- Verify physical dimensions (`Height (um)`, `Height (px)`) are present

**Low AUROC:**
- Check QC plots for alignment failures
- Verify that samples are from a narrow stage window (47-49 hpf)
- Consider increasing sample size or adjusting stage window
