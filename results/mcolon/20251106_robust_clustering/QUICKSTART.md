# Quick Start Guide: Clean Architecture Implementation

**Status:** Ready to run
**Date:** 2025-11-07

---

## What We Built

**Two clean, focused scripts:**

1. **`run_hierarchical_posterior_analysis.py`** (~340 lines)
   - Hierarchical clustering + bootstrap + posterior analysis
   - k=2-7 (consistent with k-medoids)
   - No plotting clutter
   - Saves comprehensive results

2. **`consensus_clustering_plotting.py`** (~490 lines)
   - All visualization types
   - Works for both hierarchical and k-medoids
   - Modular design (easy to add new plots)
   - Batch processing support

---

## Quick Start

### Step 1: Run Hierarchical Analysis (single genotype test)

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251106_robust_clustering

# Test on single genotype first
python run_hierarchical_posterior_analysis.py --genotype cep290_homozygous

# Expected runtime: ~15-20 minutes
```

**Output:**
```
output/data/hierarchical/cep290_homozygous_all_k.pkl
```

### Step 2: Generate Plots

```bash
# Generate all plots for the genotype you just analyzed
python consensus_clustering_plotting.py --genotype cep290_homozygous --method hierarchical

# Expected runtime: ~2-3 minutes
```

**Output:**
```
output/figures/hierarchical/cep290_homozygous/
├── posterior_heatmaps/heatmap_k{2-7}.png
├── posterior_scatters/scatter_k{2-7}.png
├── temporal_trends_posterior/trends_k{2-7}.png
└── temporal_trends_category/trends_k{2-7}.png
```

### Step 3: Run Full Pipeline (all genotypes)

```bash
# Run hierarchical analysis for all genotypes
python run_hierarchical_posterior_analysis.py

# Expected runtime: ~1-1.5 hours total
```

### Step 4: Generate All Plots

```bash
# Generate plots for all genotypes
python consensus_clustering_plotting.py

# Expected runtime: ~5-10 minutes
```

---

## Extend K-Medoids to k=2-7

Your existing k-medoids results only go up to k=5. To match hierarchical (k=2-7), run:

```bash
# Re-run k-medoids for k=6,7
python compare_methods_v2.py --k_min 2 --k_max 7

# Expected runtime: ~5 minutes
```

Then generate plots:

```bash
# Generate k-medoids plots
python consensus_clustering_plotting.py --method kmedoids --k_max 7
```

---

## File Structure

```
results/mcolon/20251106_robust_clustering/
├── run_hierarchical_posterior_analysis.py  # NEW: Clean hierarchical analysis
├── consensus_clustering_plotting.py        # NEW: Unified plotting module
│
├── bootstrap_posteriors.py                 # Existing: Posterior computation
├── adaptive_classification.py              # Existing: 2D gating
├── compare_methods_v2.py                   # Existing: K-medoids analysis
│
├── output/
│   ├── data/
│   │   ├── hierarchical/
│   │   │   ├── cep290_wildtype_all_k.pkl
│   │   │   ├── cep290_heterozygous_all_k.pkl
│   │   │   ├── cep290_homozygous_all_k.pkl
│   │   │   └── cep290_unknown_all_k.pkl
│   │   └── kmedoids/
│   │       └── [existing posteriors_k{2-5}.pkl]
│   │
│   └── figures/
│       ├── hierarchical/
│       │   └── {genotype}/
│       │       ├── posterior_heatmaps/
│       │       ├── posterior_scatters/
│       │       ├── temporal_trends_posterior/
│       │       └── temporal_trends_category/
│       └── kmedoids/
│           └── [similar structure]
```

---

## Key Features

### run_hierarchical_posterior_analysis.py

**What it does:**
- Loads curvature data for specified genotype
- Extracts and interpolates trajectories
- Computes DTW distance matrix
- For k=2-7:
  - Bootstrap hierarchical clustering (100 iterations)
  - Hungarian label alignment
  - Posterior probability calculation
  - 2D gating classification
- Saves comprehensive results

**Options:**
```bash
python run_hierarchical_posterior_analysis.py --help

Options:
  --experiment, -e    Experiment ID (default: 20251017_combined)
  --genotype, -g      Single genotype to process (default: all)
  --output_dir, -o    Output directory (default: output/data/hierarchical)
```

**Example uses:**
```bash
# Single genotype
python run_hierarchical_posterior_analysis.py -g cep290_homozygous

# Different experiment
python run_hierarchical_posterior_analysis.py -e 20250305 -g cep290_wildtype

# All genotypes (default)
python run_hierarchical_posterior_analysis.py
```

### consensus_clustering_plotting.py

**What it does:**
- Loads saved analysis results
- Generates 4 plot types per k:
  1. Posterior heatmap (p_i(c) matrix)
  2. 2D scatter (validates gating)
  3. Trajectory plot with continuous alpha (posterior-weighted)
  4. Trajectory plot with category colors (core/uncertain/outlier)

**Options:**
```bash
python consensus_clustering_plotting.py --help

Options:
  --genotype, -g      Genotype to plot (default: all)
  --method, -m        hierarchical/kmedoids/both (default: hierarchical)
  --k_min             Minimum k (default: 2)
  --k_max             Maximum k (default: 7)
  --output_dir, -o    Output directory (default: output/figures)
```

**Example uses:**
```bash
# Single genotype, hierarchical only
python consensus_clustering_plotting.py -g cep290_homozygous -m hierarchical

# All genotypes, both methods
python consensus_clustering_plotting.py -m both

# Specific k range
python consensus_clustering_plotting.py --k_min 3 --k_max 5
```

---

## Troubleshooting

**Error: "Curvature file not found"**
- Check that experiment ID exists in `morphseq_playground/metadata/body_axis/summary/`
- Try: `ls morphseq_playground/metadata/body_axis/summary/curvature_metrics_*.csv`

**Error: "Too few embryos for clustering"**
- Some genotypes may have insufficient data
- Check: `--genotype cep290_unknown` often has fewer samples
- Solution: Skip that genotype or lower `MIN_TIMEPOINTS` in script

**Error: "Module not found: bootstrap_posteriors"**
- Ensure you're running from the `20251106_robust_clustering` directory
- Solution: `cd /path/to/20251106_robust_clustering`

**Error: "plot_embryos_metric_over_time not found"**
- This function is in `src/analyze/utils/plotting.py`
- Solution: Check that `PROJECT_ROOT` path is correct (line 19 in both scripts)

**Plots look wrong / no trajectories**
- Check that `prepare_trajectory_dataframe()` is getting valid data
- Add debug prints: `print(df.head())` before plotting
- Verify `common_grid` and `trajectories` are not empty

---

## Next Steps

1. **Test on single genotype** (Step 1-2 above) - ~20 minutes
2. **Verify plots look correct** - check for:
   - Posterior heatmap shows block diagonal structure
   - 2D scatter shows distinct core/uncertain/outlier populations
   - Trajectory plots show reasonable patterns
3. **Run full pipeline** (Steps 3-4) - ~1.5 hours
4. **Extend k-medoids to k=6-7** (optional) - ~5 minutes
5. **Generate method comparison plots** (future addition to plotting module)

---

## Validation Checklist

After running:

- [ ] Hierarchical analysis completed for at least 1 genotype
- [ ] Output file exists: `output/data/hierarchical/{genotype}_all_k.pkl`
- [ ] Plots generated for all k=2-7
- [ ] Posterior heatmaps show clear block structure
- [ ] 2D scatter validates gating logic
- [ ] Trajectory plots show biological patterns
- [ ] Core membership rates > 50% for k=3-4
- [ ] Summary table printed at end shows reasonable values

---

## Performance Notes

**Runtime:**
- Single genotype analysis: ~15-20 minutes
  - DTW computation: ~5 min
  - Bootstrap (100 iterations): ~10 min
  - Posterior analysis: ~1 min
- Plotting (all k values): ~2-3 minutes per genotype

**Memory:**
- Peak usage: ~2-3 GB
- Main cost: DTW distance matrix storage (n² floats)
- For n=20 embryos: ~3.2 KB
- For n=50 embryos: ~20 KB

**Storage:**
- Per-genotype results file: ~500 KB - 2 MB
- Per-plot image (PNG, 300 DPI): ~500 KB
- Total for 4 genotypes × 4 methods × 6 k values × 4 plots: ~400 MB

---

## Contact

For issues or questions, check:
- `IMPLEMENTATION_STATUS.md` - Detailed implementation notes
- `cluster_assignment_quality.md` - Method documentation
- `README.md` - Original k-medoids implementation guide
