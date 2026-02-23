# TMEM67 Cluster Projection Analysis

Project TMEM67 embryo trajectories onto pre-existing CEP290 cluster assignments using DTW distance.

## Quick Start

```bash
# Activate conda environment (adjust environment name as needed)
conda activate morphseq  # or your environment name

# Run the main analysis
python run_projection.py
```

## What This Does

1. **Loads CEP290 reference data** with pre-computed cluster labels (Low_to_High, High_to_Low, Intermediate, Not Penetrant)
   - Uses ALL CEP290 data (not just spawn)
   - Filters to only CEP290 embryos with valid cluster assignments (removes embryos with NaN clusters)
2. **Loads TMEM67 data** from experiments 20250711 and 20251205
   - Uses ALL TMEM67 data (all pairs: spawn, pair_1, etc.)
3. **Computes cross-dataset DTW distances** between TMEM67 and CEP290 embryos (only valid clusters)
4. **Projects TMEM67 onto CEP290 clusters** using:
   - Nearest Neighbor (primary method)
   - K-NN with posteriors (k=5, for comparison)
5. **Compares cluster frequencies** between datasets (can filter by pair for comparison)
6. **Runs statistical tests** (Chi-square, Fisher's exact if applicable)

## Output Files

All results saved to `output/`:

- `cross_dtw_distance_matrix.npy` - Distance matrix (N_tmem67 × N_cep290)
- `tmem67_nn_projection.csv` - Nearest neighbor cluster assignments
- `tmem67_knn_projection.csv` - K-NN cluster assignments with posteriors
- `cluster_frequency_comparison_spawn.csv` - Frequency comparison table

## Key Parameters

In `run_projection.py`:

- `METRICS = ['baseline_deviation_normalized']` - Curvature trajectory metric for DTW
- `SAKOE_CHIBA_RADIUS = 3` - DTW warping window constraint
- `K_NN = 5` - Number of neighbors for K-NN method
- `TMEM67_EXPERIMENTS = ['20250711', '20251205']` - Experiments to load

## Files

- `run_projection.py` - Main analysis script
- `projection_utils.py` - Helper functions (cross-DTW, assignment methods)
- `PLAN.md` - Detailed implementation plan with lessons learned
- `README.md` - This file

## Future Improvements

1. **Bootstrap-based cluster assignment**: Use bootstrap utilities from `src/analyze/trajectory_analysis/` to:
   - Compute DTW distances with uncertainty via bootstrap resampling
   - Assign clusters with confidence intervals
   - Handle borderline cases more robustly than nearest-neighbor alone

## Next Steps

After running `run_projection.py`:

1. Check distance distribution for outliers
2. Create visualizations (bar charts, trajectory overlays)
3. Analyze TMEM67 pair_1 data separately if needed
4. Optionally export functions to `src/analyze/trajectory_analysis/cluster_projection.py` for reuse

## Two-Stage Hypothesis

### Background
TMEM67 embryos exhibit a similar curvature phenotype to CEP290, with evidence of developmental rescue (curvature improvement over time). This raises two related hypotheses about the mechanistic relationship between these ciliopathy genes.

### Stage 1: Shared Mechanistic Pathway
**H1a**: CEP290 and TMEM67 mutations affect the same underlying developmental pathway controlling body curvature.

**Prediction**: If this is true, TMEM67 embryos should exhibit the same temporal curvature trajectory patterns as CEP290 embryos when projected onto CEP290's phenotypic clusters.

**Test**: Project TMEM67 trajectories onto pre-defined CEP290 clusters using DTW distance. High-quality assignments (low DTW distances, high posterior confidence) would support a shared mechanistic basis.

### Stage 2: Differential Penetrance Through Cluster Distribution
**H1b**: TMEM67 shows a higher rate of developmental rescue compared to CEP290 due to a greater proportion of embryos following "High_to_Low" (rescue) trajectories.

**Prediction**: If H1a is supported (shared mechanism), then TMEM67's increased rescue phenotype should manifest as:
1. Higher proportion of TMEM67 embryos assigned to the "High_to_Low" cluster
2. Lower proportion in "Low_to_High" (worsening) and "Intermediate" (non-rescue) clusters
3. Similar or lower proportion in "Not Penetrant" cluster

**Test**: Compare cluster frequency distributions between CEP290 spawn and TMEM67 spawn using:
- Chi-square test for overall distribution differences
- Post-hoc proportion tests for specific cluster enrichments
- Stratify by genotype (homozygous, heterozygous, wildtype) to assess allelic effects

### Interpretation Framework

| Outcome | Stage 1 (Shared Mechanism) | Stage 2 (Differential Penetrance) | Biological Interpretation |
|---------|---------------------------|-----------------------------------|---------------------------|
| **Both supported** | TMEM67 trajectories map well to CEP290 clusters | TMEM67 enriched in High_to_Low | CEP290 and TMEM67 act on same pathway; TMEM67 has stronger rescue dynamics |
| **H1a only** | Good cluster mapping | No enrichment difference | Shared pathway but similar penetrance patterns |
| **H1b only** | Poor cluster mapping | Apparent enrichment | Different mechanisms that happen to produce similar endpoint phenotypes |
| **Neither** | Poor mapping | No enrichment | Different pathways and different penetrance; phenotypic similarity is superficial |

### Key Comparisons
1. **CEP290 spawn vs TMEM67 spawn**: Primary test of differential penetrance
2. **CEP290 all pairs vs TMEM67 all pairs**: Generalizability across genetic backgrounds
3. **Homozygous-only**: Maximum penetrance comparison (removes heterozygote variability)
