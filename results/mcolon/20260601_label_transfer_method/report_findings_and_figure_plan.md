# Label-Transfer Conformal Benchmark: Findings and Figure Plan

Date: 2026-06-04

## Working Conclusion

The result is stronger than "KNN needs calibration."

Raw Euclidean KNN is the wrong evidence model for these trajectory labels. The failure is
not primarily a conformal-prediction failure, and it is not just a calibration problem.
The labels of interest are not locally majority labels in the raw feature space.

KNN remains useful as a diagnostic of local geometry and support. The stronger prediction
candidate is:

1. supervised multiclass probabilities as the image-level score vector `q`,
2. APS conformal prediction sets as the uncertainty layer,
3. KNN density/support diagnostics as a low-support gate,
4. image-to-embryo rollup downstream.

## Current Evidence

### Global Benchmark

Source: `q_conformal_benchmark_global_summary.csv`

| Method | Output | Accuracy | Balanced Acc | Macro F1 | Coverage | Mean Set Size | Singleton Rate | LtH->NP | NP->LtH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KNN-q | argmax | 0.695 | 0.483 | 0.477 | - | - | - | 0.681 | 0.058 |
| KNN-q | conformal | - | - | - | 0.901 | 2.328 | 0.006 | - | - |
| Multiclass-q | argmax | 0.663 | 0.529 | 0.519 | - | - | - | 0.406 | 0.150 |
| Multiclass-q | conformal | - | - | - | 0.899 | 2.303 | 0.130 | - | - |

Interpretation:

- KNN argmax has higher raw accuracy because it is very good at the majority
  `Not Penetrant` class.
- Multiclass argmax has better balanced accuracy and macro F1, and much lower
  LtH->NP collapse.
- Both conformal methods hit roughly 90% marginal coverage, but multiclass-q produces
  many more singleton sets while preserving coverage.
- KNN-q conformal reaches marginal coverage by emitting broad sets almost all the time.

### Rare-Class Coverage

Source: `q_conformal_benchmark_global_summary.csv`

| Method | Low_to_High Coverage | Intermediate Coverage | High_to_Low Coverage | NP Coverage |
|---|---:|---:|---:|---:|
| KNN-q conformal | 0.816 | 0.520 | 0.908 | 0.962 |
| Multiclass-q conformal | 0.793 | 0.781 | 0.925 | 0.934 |

Interpretation:

- The key failure is `Intermediate`.
- KNN-q achieves marginal coverage while badly under-covering `Intermediate`.
- Multiclass-q improves `Intermediate` coverage from 0.520 to 0.781.

### Neighbor Geometry

Source: `q_diagnostic_report.md`, `q_diagnostic_neighbor_geometry_summary.csv`

| True Label | First Same-Label Neighbor Rank | Same-Label Fraction Top15 | NP Fraction Top15 |
|---|---:|---:|---:|
| High_to_Low | 4.23 | 0.693 | 0.118 |
| Intermediate | 22.31 | 0.119 | 0.454 |
| Low_to_High | 11.93 | 0.216 | 0.567 |
| Not Penetrant | 1.69 | 0.821 | 0.821 |

Interpretation:

- For `Intermediate`, the average first same-label neighbor appears after rank 22.
  KNN-15 often cannot even include a same-label example in its evidence pool.
- For `Low_to_High`, same-label neighbors appear earlier but the top-15 neighborhood is
  still dominated by `Not Penetrant`.
- KNN is measuring local Euclidean majority honestly, but local Euclidean majority is
  not aligned with trajectory labels.

### True-Label Rank

Source: `q_diagnostic_true_label_rank_summary.csv`

| Method | True Label | Rank 1 Rate | Rank <=2 Rate | Mean True-Label Rank | Coverage |
|---|---|---:|---:|---:|---:|
| KNN-q | Intermediate | 0.133 | 0.414 | 2.643 | 0.520 |
| Multiclass-q | Intermediate | 0.272 | 0.625 | 2.256 | 0.781 |
| KNN-q | Low_to_High | 0.124 | 0.744 | 2.211 | 0.816 |
| Multiclass-q | Low_to_High | 0.311 | 0.706 | 2.072 | 0.793 |

Interpretation:

- Multiclass-q improves the rank of the true rare/mixed labels.
- Conformal prediction works better downstream because the score generator puts the
  true label higher in the ranking.

### Rescue Groups

Source: `q_diagnostic_rescue_group_summary.csv`, `q_diagnostic_report.md`

| True Label | KNN Wrong / Multiclass Right | KNN Right / Multiclass Wrong |
|---|---:|---:|
| Intermediate | 161 | 60 |
| Low_to_High | 613 | 185 |

Interpretation:

- Multiclass is not just exchanging errors with KNN.
- It is net-rescuing many `Intermediate` and `Low_to_High` images.

## Figure 1: Main Benchmark Comparison

Purpose: show the two main axes clearly:

1. q generator: KNN-q vs multiclass-q
2. output mode: argmax vs conformal

This should be the first report figure. It should establish that the method choice is
not "KNN versus conformal"; it is "which q generator produces useful argmax and useful
conformal sets?"

### Recommended Layout

Use a compact multi-panel figure:

Panel A: Argmax performance

- x-axis: q generator (`KNN-q`, `Multiclass-q`)
- bars or points: `accuracy`, `balanced_accuracy`, `macro_f1`
- optional fold-level points overlaid from `q_conformal_benchmark_summary.csv`

Panel B: Rare-class failure rates

- x-axis: q generator
- bars or points: `LtH->NP_collapse`, `NP->LtH_falsecall`
- this highlights that KNN's raw accuracy hides LtH collapse

Panel C: Conformal performance

- x-axis: q generator
- bars or points: `marginal_coverage`, target coverage line at 0.90,
  `mean_set_size`, and/or `singleton_rate`
- if combining metrics is visually crowded, use two small panels:
  coverage and singleton/set-size

Panel D: Per-class conformal coverage

- x-axis: true label
- color: q generator
- y-axis: conformal coverage
- horizontal line: target coverage 0.90
- this should make the `Intermediate` KNN under-coverage obvious

### Data Needed

Already available:

- `q_conformal_benchmark_global_summary.csv`
  - pooled values for the main bars
- `q_conformal_benchmark_summary.csv`
  - fold-level values for jittered points/error bars
- `q_conformal_benchmark_image_predictions.csv`
  - image-level values if we want bootstrap confidence intervals

Need to derive for plotting:

- A long-form metric table with columns:
  - `method`
  - `q_source`
  - `output_type`
  - `metric`
  - `value`
  - `heldout_experiment_id`
  - `label` where applicable
  - `metric_family` (`argmax`, `conformal`, `rare_failure`, `per_class_coverage`)
- For per-class conformal coverage, reshape columns like
  `coverage[Intermediate]` into rows:
  - `method = knn_q`
  - `output_type = conformal`
  - `label = Intermediate`
  - `metric = coverage`
  - `value = 0.520`

Recommended output table:

- `figure_01_main_benchmark_plot_data.csv`

### What Figure 1 Should Demonstrate

The reader should be able to see:

1. KNN-q has higher raw accuracy but lower macro F1 and balanced accuracy.
2. KNN-q has severe LtH->NP collapse.
3. Both conformal methods achieve near-target marginal coverage.
4. KNN-q conformal badly under-covers `Intermediate`.
5. Multiclass-q conformal gives more useful sets: similar coverage, slightly smaller
   mean set size, and much higher singleton rate.

## Figure 2: Why KNN Fails

Purpose: show that rare/mixed labels are not local Euclidean majorities.

Recommended panels:

- first same-label neighbor rank by true label
- same-label fraction in top 15 / 50 / 200 by true label
- NP fraction in top 15 by true label
- optionally split by HPF bin

Data:

- `q_diagnostic_neighbor_geometry_summary.csv`
- `q_diagnostic_neighbor_geometry.csv` for image-level distributions

Key visual:

- `Intermediate` first same-label rank around 22
- `Intermediate` top-15 same-label fraction around 0.12
- `Low_to_High` top-15 NP fraction around 0.57

## Figure 3: True-Label Rank Explains Conformal Coverage

Purpose: connect q-generator quality to conformal success.

Recommended panels:

- true-label rank distribution by method and true label
- true-label q score quantiles by method and true label
- conformal coverage by true-label rank

Data:

- `q_diagnostic_true_label_rank.csv`
- `q_diagnostic_true_label_rank_summary.csv`

Key visual:

- multiclass-q moves `Intermediate` from rank 3/4 toward rank 1/2
- conformal failures concentrate where the true label is rank 3 or 4

## Figure 4: Rescue Cases

Purpose: show that multiclass is net-rescuing rare/mixed images.

Recommended panels:

- stacked bars of rescue groups by true label
- HPF distribution for `KNN wrong / Multiclass right`
- q_true delta: multiclass minus KNN, especially for `Intermediate` and `Low_to_High`

Data:

- `q_diagnostic_rescue_groups.csv`
- `q_diagnostic_rescue_group_summary.csv`

Key visual:

- `Intermediate`: 161 KNN-wrong/multiclass-right vs 60 KNN-right/multiclass-wrong
- `Low_to_High`: 613 vs 185

## Next Plotting Step

Create `make_report_figures.py` with the first deliverable:

1. Read `q_conformal_benchmark_global_summary.csv` and
   `q_conformal_benchmark_summary.csv`.
2. Build `figure_01_main_benchmark_plot_data.csv`.
3. Render Figure 1 as:
   - `plots/report_figure_01_main_benchmark.png`
   - optionally `plots/report_figure_01_main_benchmark.svg`

Only after Figure 1 is clear should we move to the deeper KNN-failure figures.
