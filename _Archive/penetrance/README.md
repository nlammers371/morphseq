# Penetrance Bin-Width Analysis Utilities

Reusable helpers for the bin-width sensitivity workflow (global WT IQR threshold, time-binning, penetrance curves, and plotting). These utilities were distilled from `results/mcolon/20251029_curvature_temporal_analysis/06f_global_iqr_binwidth_comparison.py` so that future notebooks and pipelines can reuse the same logic without copying the script.

## Available Functions

| Function | Description | Input → Output |
|----------|-------------|----------------|
| `ensure_metric_column(df, metric='baseline_deviation_normalized', alt_metric='normalized_baseline_deviation')` | Guarantees that the preferred metric column exists by copying from the fallback column if necessary. | **Input:** DataFrame with either column. **Output:** DataFrame containing `metric` column (copy-on-write). Raises `KeyError` if neither column exists. |
| `compute_global_iqr_bounds(wt_df, metric='baseline_deviation_normalized', k=1.5)` | Computes pooled WT statistics and returns a single IQR band (low/high/median/etc.). | **Input:** WT-only DataFrame. **Output:** `dict` with keys `low`, `high`, `median`, `mean`, `q1`, `q3`, `iqr`, `n_samples`, `k`. |
| `mark_penetrant_global(df, wt_bounds, metric='baseline_deviation_normalized')` | Adds a `penetrant` (0/1) column by testing each row against the WT band. | **Input:** DataFrame + bounds dict. **Output:** Copy of DataFrame including `penetrant` column. |
| `bin_data_by_time(df, bin_width=2.0, time_col='predicted_stage_hpf')` | Adds a `time_bin` column with uniform bin centers and returns those centers. | **Input:** DataFrame with `time_col`. **Output:** `(df_with_bins, bin_centers)` where `bin_centers` is a NumPy array. |
| `compute_penetrance_by_time(df_binned, time_bins, metric_col='penetrant')` | Collapses per-embryo penetrance inside each time bin. | **Input:** Binned DataFrame for a single genotype and the `bin_centers`. **Output:** `List[Dict]` items containing `time_bin`, `embryo_penetrance`, `n_embryos`, `n_penetrant`, `se`. |
| `compute_summary_stats(temporal_results, binwidths, genotype_order)` | Builds a tidy summary table for reporting. | **Input:** Nested dict (described below). **Output:** Pandas DataFrame with columns `binwidth_hpf`, `genotype`, `n_time_bins`, `mean_penetrance_%`, `std_penetrance_%`, `min_penetrance_%`, `max_penetrance_%`, `range_penetrance_%`, `mean_se_%`. |
| `plot_temporal_by_binwidth(...)` | One subplot per bin width, showing WT/Het/Homo penetrance curves with error bars. Returns the `matplotlib.figure.Figure`. |
| `plot_genotype_smoothing(...)` | One subplot per genotype; overlays curves for each bin width to show smoothing. Returns `Figure`. |
| `plot_wt_focus(...)` | Single-axis plot highlighting WT penetrance across bin widths. Returns `Figure`. |

### `temporal_results` Structure

All plotting/summary helpers expect:

```python
{
    binwidth: {
        'wt':  [ { 'time_bin': float, 'embryo_penetrance': float, 'n_embryos': int, 'n_penetrant': int, 'se': float }, ... ],
        'het': [...],
        'homo':[...],
    },
    ...
}
```

You can provide a custom `genotype_order` (list of `(key, label)` tuples) when calling the plotting or summary functions to reuse the same logic for other loci.

## Example Workflow

```python
from src.analyze.penetrance import binwidth

# 1. Load raw curvature dataframe
df, metadata = get_analysis_dataframe()
df = binwidth.ensure_metric_column(df)

# 2. Compute WT bounds
wt_df = df[df['genotype'] == 'cep290_wildtype']
wt_bounds = binwidth.compute_global_iqr_bounds(wt_df)

# 3. Mark penetrant frames
df_marked = binwidth.mark_penetrant_global(df, wt_bounds)

# 4. Iterate over bin widths
BINWIDTHS = [2.0, 5.0, 10.0]
genotypes = [('wt', 'cep290_wildtype'), ('het', 'cep290_heterozygous'), ('homo', 'cep290_homozygous')]

temporal_results = {}
for bw in BINWIDTHS:
    df_binned, time_bins = binwidth.bin_data_by_time(df_marked, bin_width=bw)
    temporal_results[bw] = {}
    for key, geno in genotypes:
        geno_df = df_binned[df_binned['genotype'] == geno]
        temporal_results[bw][key] = binwidth.compute_penetrance_by_time(geno_df, time_bins)

# 5. Summaries & plots
summary_df = binwidth.compute_summary_stats(temporal_results, BINWIDTHS, genotype_order=genotypes)
fig = binwidth.plot_temporal_by_binwidth(temporal_results, BINWIDTHS, genotype_order=genotypes)
fig.savefig('temporal_penetrance_by_binwidth.png', dpi=300)
```

This mirrors the original `06f_global_iqr_binwidth_comparison` analysis, but now lives in a reusable module that can be imported by notebooks, pipelines, or tests.
