# Curvature vs. Embedding Regression (2025-11-14)

This directory stores artifacts for Goal 1: quantifying how well VAE embeddings
predict curvature-derived metrics with ridge regression.

Run via:

```bash
python scripts/run_curvature_embedding_regression.py 20251106 \\
  --target-metric mean_curvature_per_um baseline_deviation_um \\
  --output-dir results/mcolon/20251114_curvature_embedding_interplay
```

Each target metric gets its own subfolder with:

- Summary CSV of CV metrics
- Fold-level diagnostics
- Per-sample predictions with metadata
- Scatter plot of predicted vs. observed curvature
