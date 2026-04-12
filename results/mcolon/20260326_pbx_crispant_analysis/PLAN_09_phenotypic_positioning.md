# Plan: 09 Phenotypic Positioning via Pairwise Classification Graphs

Date: 2026-03-28

## Goal

Embed each embryo in pairwise classification probability space — a 10D vector where
each dimension is the predicted probability from one of the 10 pairwise binary classifiers
(all pairs of 5 genotypes). Visualize genotype structure, within-genotype heterogeneity,
and temporal trajectory divergence using AlignedUMAP for temporally coherent embeddings.

## What Script 08 Does vs. What We Need

Script 08 runs all 10 pairwise classifications but only saves:
- Aggregated per-embryo labels (mean signed margin across all timebins)
- Per-pair AUROC summaries

**Missing:** raw per-embryo-per-timebin `pred_prob_group2` for all 10 pairs.
Script 09 must re-run (or load cached) to get this.

## Key Design Decisions

| Item | Choice | Reason |
|------|--------|--------|
| Probability vector | Raw `pred_prob_group2` ∈ [0,1]; NaN → 0.5 | 0.5 = uninformative neutral |
| Temporal embedding | AlignedUMAP (umap-learn 0.5.11) | Purpose-built for aligned slices with shared entities |
| 3D trajectories | Plotly via `plot_3d_scatter()` | Existing infra, interactive |
| AUROC viz | Ribbon (pair × time) + snapshot N×N heatmaps | Shows full time course + structure |
| Condition graph | networkx spring layout, pos warm-start | Simple, interpretable |
| Colors | `SPECIAL_GENOTYPE_COLORS` from `analyze.viz.styling` | Consistent with rest of project |

## AlignedUMAP API

```python
import umap
aligned = umap.AlignedUMAP(
    alignment_regularisation=1e-2,
    alignment_window_size=3,
    n_components=2,           # or 3 for 3D
    n_neighbors=15,
    min_dist=0.1,
).fit(slices, relations=relations)
# slices[t]: (n_embryos_at_t, 10) float array
# relations[t]: {row_idx_in_slice_t: row_idx_in_slice_t+1} for shared embryos
# aligned.embeddings_: list of T arrays, each (n_embryos_at_t, n_components)
```

Embryos appear/disappear across timebins — relations dicts only map embryos
present in both consecutive slices.

## Steps

1. **Load data** — same as script 08 (`load_dataframe`)
2. **Run or load all-pairs predictions** — cache to `all_pairs_predictions.csv` + `auc_bins.csv`
3. **Build 10D probability vectors** — pivot `pred_prob_group2` by `pair_id`, fill NaN=0.5
4. **AUROC ribbon** — pair × time heatmap (matplotlib imshow)
5. **AUROC snapshots** — 5×5 heatmaps at 6 timepoints
6. **2D AlignedUMAP** — temporally coherent 2D snapshots at ~25 and ~79 hpf
7. **3D AlignedUMAP** — trajectories via `plot_3d_scatter()`, both genotype- and time-colored
8. **Condition graph** — networkx per timepoint, spring layout with warm start

## Output Directories

- Results: `results/misclassification/embedding/phenotypic_positioning/`
- Figures: `figures/misclassification/embedding/phenotypic_positioning/`

## Output Files

**Results (CSV):**
- `all_pairs_predictions.csv` — raw per-embryo-per-timebin predictions for all 10 pairs
- `probability_vectors.csv` — 10D probability vectors per embryo-timebin
- `aligned_umap_2d_coordinates.csv`
- `aligned_umap_3d_coordinates.csv`

**Figures (PNG/HTML):**
- `auroc_ribbon.png`
- `auroc_heatmap_snapshots.png`
- `snapshot_umap_2d_{time}hpf.png`
- `joint_3d_umap_genotype.html` + `.png`
- `joint_3d_umap_time.html` + `.png`
- `condition_graph_over_time.png`

## Verification

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260326_pbx_crispant_analysis/scripts/09_phenotypic_positioning.py \
  --skip-classification  # reuse cached predictions if available
```

Expected:
- `probability_vectors.csv` ~3000 rows × 10 pair columns
- 2D snapshots show genotype clusters (pbx1b_pbx4 furthest from controls)
- 3D HTML loads with smooth per-embryo trajectory lines
- Condition graph: inj_ctrl and wik_ab close, pbx1b_pbx4 distant

## Concrete Data Contracts

### Input dataframe requirements

Script 09 assumes the merged build06 tables contain:

- `embryo_id`
- `genotype`
- `predicted_stage_hpf`
- `use_embryo_flag` (optional but used if present)
- latent columns with prefix `z_mu_b`

It reads:

- `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20260304.csv`
- `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20260306.csv`

### Cached classifier outputs

`all_pairs_predictions.csv` must contain at minimum:

- `embryo_id`
- `time_bin_center`
- `pair_id`
- `group1`
- `group2`
- `true_label`
- `pred_prob_group2`

`auc_bins.csv` must contain at minimum:

- `time_bin_center`
- `pair_id`
- `group1`
- `group2`
- `auroc_obs`
- `n_positive`
- `n_negative`

### Probability vector table

`probability_vectors.csv` is the core intermediate. Each row is one embryo at one
time bin. Required columns:

- `embryo_id`
- `time_bin_center`
- `genotype`
- 10 pairwise probability columns, one per `pair_id`

Interpretation:

- each pair column stores `P(group2)` from that binary classifier
- missing pair/time combinations are filled with `0.5`
- no sign-flipping is applied; orientation is whatever `group1/group2` ordering produced

## Planned Script Structure

The current implementation already maps cleanly onto the following stages:

1. `load_dataframe()` loads and filters the PBX/control subset across the two experiments.
2. `embedding_features()` discovers latent dimensions from the `z_mu_b*` prefix.
3. `run_or_load_all_pairs()` loops over all 10 genotype pairs and caches predictions/AUROC bins.
4. `build_probability_vectors()` pivots long-form predictions into embryo-timebin vectors.
5. `plot_auroc_ribbon()` renders the pair-by-time separability heatmap.
6. `plot_auroc_heatmap_snapshots()` renders condition-by-condition AUROC matrices at selected times.
7. `_build_aligned_umap_inputs()` constructs per-time slices plus consecutive-time embryo relations.
8. `run_aligned_umap()` fits 2D and 3D aligned embeddings and returns tidy coordinates.
9. `plot_2d_snapshot()` renders selected 2D time slices.
10. `plot_condition_graphs()` converts AUROC into a similarity graph with warm-started layouts.
11. `plot_3d_scatter()` handles genotype-colored and time-colored 3D exports.

## Parameterization To Keep Stable

These defaults are reasonable and should stay fixed unless a specific failure mode appears:

- `bin_width = 2.0`
- `n_splits = 5`
- `n_permutations = 0`
- `random_state = 42`
- `snapshot_times = [25, 55, 79]`
- `auroc_snapshot_bins = [25, 35, 45, 55, 65, 75]`
- `umap_alignment_regularisation = 1e-2`
- `umap_alignment_window = 3`
- `umap_n_neighbors = 15`
- `umap_min_dist = 0.1`

Rationale:

- keep classifier settings consistent with the rest of the PBX analysis
- favor a modest amount of temporal coupling in AlignedUMAP
- keep snapshots sparse enough to read but broad enough to show developmental progression

## Known Methodological Caveats

### 1. Pair orientation is arbitrary

Each vector dimension is `P(group2)`, not a symmetric distance. That means two columns
can be harder to compare directly than a standardized signed score would be.

Why acceptable for now:

- the classifier output still preserves informative relative positioning
- the same orientation is used consistently across all embryo-timebin rows
- neutral fill remains centered at `0.5`

### 2. Missingness is encoded as neutrality

Filling NaN with `0.5` is pragmatic, but it conflates:

- truly ambiguous classification
- no model output for that embryo/time/pair

This is acceptable for a first-pass embedding, but if sparse bins dominate, we may want:

- a missingness mask export
- pair coverage counts per embryo-timebin
- sensitivity analysis with sparse rows dropped

### 3. AUROC is being reused as a graph similarity proxy

The condition graph uses `similarity = 1 - AUROC`.
That is intuitive but not a metric, and values near `0.5` are noisy when sample counts are low.

Possible refinement if the graph looks unstable:

- shrink AUROC toward `0.5` using sample count
- threshold weak edges
- smooth AUROC over adjacent time bins before graph construction

### 4. AlignedUMAP is descriptive, not inferential

The 2D/3D layouts are useful summaries, but exact geometry should not be over-interpreted.
The main trustworthy signals are:

- stable genotype neighborhood structure
- broad trajectory separation/reconvergence patterns
- within-genotype spread and heterogeneity

## Failure Modes And Triage

### If classification rerun is too slow

- run once without `--skip-classification`
- reuse cached `all_pairs_predictions.csv` and `auc_bins.csv` thereafter

### If AlignedUMAP throws relation or shape errors

Check:

- every slice uses the same pair column order
- `relations[t]` only maps embryos present in both adjacent slices
- no slice is empty

### If 3D plots are too cluttered

Reduce visual density by:

- increasing `min_points_per_line`
- lowering point opacity
- disabling trajectory lines in exploratory runs

### If condition graph jumps between timepoints

Likely causes:

- AUROC instability at sparse bins
- spring layout sensitivity

First fixes:

- reduce to fewer snapshot bins
- smooth AUROC over time before plotting
- preserve prior positions more strongly

## Recommended Additions Before Calling This Final

1. Export per-row pair coverage count in `probability_vectors.csv`.
2. Add a QC plot showing number of embryos per time bin after pivoting.
3. Add one summary table of genotype centroids in 3D UMAP by time bin.
4. Save the exact CLI invocation and parameters to a small run manifest text file.
5. Optionally add a second graph view using thresholded edges only.

## Minimal Acceptance Checklist

Before this analysis is considered complete, verify:

- all 10 genotype pairs are present in both caches
- `probability_vectors.csv` has one row per embryo-timebin and exactly 10 pair columns
- 2D snapshots show consistent broad ordering across nearby time bins
- 3D genotype plot shows coherent embryo trajectories rather than frame-to-frame scrambling
- AUROC ribbon reproduces the known negative control weakness for `inj_ctrl vs wik_ab`
- condition graph places `pbx1b_pbx4_crispant` furthest from controls at late stages

## Next Interpretation Targets

Once the outputs are regenerated cleanly, the immediate biological questions should be:

1. Does `pbx4_crispant` occupy a bridge region between controls and `pbx1b_pbx4_crispant`?
2. Is `pbx1b_crispant` closer to controls early and then displaced later, or stable throughout?
3. Are there persistent outlier embryo trajectories within `pbx4_crispant` that match the previously identified wildtype-like subclass?
4. Do late-stage graph layouts support a continuum model or a sharper bifurcation among PBX perturbations?
