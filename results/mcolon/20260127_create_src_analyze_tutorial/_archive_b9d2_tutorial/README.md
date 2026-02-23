# B9D2 Tutorial Archive

This directory contains the original B9D2 phenotype analysis tutorial scripts. These scripts demonstrated the within-experiment clustering workflow but are no longer the canonical tutorial.

The canonical tutorial now uses CEP290 data to demonstrate cross-experiment projection and temporal coverage effects.

## Scripts

- **04_cluster_labeling.py** - Manual cluster annotation for B9D2 experiments
- **05_faceted_feature_plots.py** - Multi-feature visualization across genotypes
- **06_proportions.py** - Cluster distribution analysis and proportions
- **07_spline_per_cluster.py** - Trajectory modeling with spline fits per cluster
- **08_difference_detection.py** - Statistical testing for genotype differences
- **09_plot_results.py** - AUROC visualization for cluster discrimination

## Original Context

These scripts analyzed B9D2 experiments (20251121, 20251125) and demonstrated:
- Within-experiment DTW clustering
- Manual cluster labeling workflow
- Cluster-based trajectory analysis
- Statistical testing framework

## Why Archived

The B9D2 analysis was exploratory work that established the basic workflow. The CEP290 analysis (scripts 01-06 in the parent directory) better demonstrates the key morphseq capabilities:

- **Cross-experiment projection**: Projecting new experiments onto established reference clusters
- **Batch effect detection**: Quantifying temporal coverage impact on cluster assignments
- **Temporal coverage analysis**: Understanding how imaging windows affect penetrance trajectory classification

The CEP290 tutorial flow provides a more robust and generalizable analysis pattern suitable for production use.

## Data Sources

Original B9D2 scripts used:
- `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251121.csv`
- `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251125.csv`

Experiments:
- 20251121: B9D2 experiment 1
- 20251125: B9D2 experiment 2
