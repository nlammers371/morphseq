# Stream D: Reference Embryo — Decisions

## Status
- Implemented end-to-end with modular scripts and persisted outputs.
- Executed on cohort:
  - `3` WT reference embryos
  - `3` WT held-out embryos
  - `20` mutants
  - 2-hour bins from `24` to `48` hpf (`12` transitions per embryo)

## Modules Added
- `01_build_cohort_manifest.py`:
  - Selects embryos by **coverage first** (2h bins in 24-48), **curvature second**.
  - Writes manifest + QC plots under `cohort_selection/`.
- `02_run_batch_ot_export.py`:
  - Runs batch OT only (resume-safe), saves metrics/features/artifacts.
- `03_build_reference_fields.py`:
  - Builds per-bin WT reference fields from selected reference embryos.
- `04_compute_deviations.py`:
  - Computes deviations vs reference and writes cohort trend plots.
- `05_pca_raw_vector_fields.py`:
  - PCA on raw velocity fields using WT-reference support union mask.
- `06_difference_classification_clustering.py`:
  - Difference testing, grouped classification, and clustering on PCA embeddings.

## Selection Decisions
- WT reference and held-out sets use `genotype == cep290_wildtype`.
- Mutant set uses `genotype == cep290_homozygous`.
- Coverage requirement: nearest frame per target 2h bin with tolerance `<=1.25 hpf`.
- Control sanity pair included in transitions manifest:
  - `20251113_A05_e01 f0014 -> 20251113_E04_e01 f0014`
  - marked `is_control_pair=True` and excluded from reference/deviation stats.

## Selected Embryos (coverage=1.0 for all selected)
- Reference WT:
  - `20251212_H07_e01`
  - `20251112_H05_e01`
  - `20251112_F08_e01`
- Held-out WT:
  - `20251112_D02_e01`
  - `20251212_B06_e01`
  - `20251205_G08_e01`
- Mutants:
  - `20` embryos listed in `cohort_selection/cohort_selected_embryos.csv`

## Batch OT Export Results
- Run ID: `phase2_24_48_ott_v1`
- Backend: `OTT`, `epsilon=1e-4`, `reg_m=10`, `max_support_points=3000`
- Transition rows processed: `313`
- Successes: `313`
- Failures: `0`
- Mean runtime: `~1.76s` per transition
- Median runtime: `~1.71s` per transition
- Output root: `ot_24_48_exports/` (kept local, ignored in git)

## Reference Field Outputs
- Built references for all 12 bins:
  - `24->26`, `26->28`, ..., `46->48`
- Each bin uses exactly `3` WT reference embryos.
- Summary CSV:
  - `reference_fields/reference_summary_phase2_24_48_ott_v1.csv`

## Deviation Results (vs WT reference)
- Output:
  - `deviation_plots/deviation_metrics_phase2_24_48_ott_v1.csv`
  - `deviation_plots/deviation_group_trends.png`
  - `deviation_plots/deviation_rmse_boxplot.png`
- Mean metrics by cohort:
  - `reference_wt`: RMSE `69.59`, cosine `0.838`
  - `heldout_wt`: RMSE `86.50`, cosine `0.078`
  - `mutant`: RMSE `75.84`, cosine `0.146`
- Interpretation:
  - WT reference self-consistency is strongest (as expected).
  - Held-out WT and mutant groups show larger deviation from the tight 3-embryo reference.
  - Distribution overlap is substantial; separation is partial, not cleanly binary.

## Raw-Field PCA Results
- Union support pixels used: `16867`
- Variance explained:
  - PC1: `20.8%`
  - cumulative PC3: `33.1%`
  - cumulative PC10: `62.2%`
- Outputs:
  - `pca/raw_velocity_pca_embeddings_phase2_24_48_ott_v1.csv`
  - `pca/raw_velocity_pca_pc1_pc2_phase2_24_48_ott_v1.png`
  - `pca/raw_velocity_pca_group_trajectories_phase2_24_48_ott_v1.png`

## Difference/Classification/Clustering Results
- Summary JSON:
  - `difference_detection/difference_classification_summary_phase2_24_48_ott_v1.json`
- Group-CV logistic regression (mutant vs held-out WT, PCs 1-6):
  - AUROC `0.585`
  - Accuracy `0.583`
  - Permutation p-value `0.0778` (not significant at 0.05)
- Time-resolved classification:
  - Earliest significant bin center: `32 hpf`
  - Max AUROC: `0.95` at `38 hpf`
  - Significant bins: `2`
- KMeans(k=2) clustering:
  - Silhouette `0.886`
  - Cluster composition is dominated by one large mixed cluster.

## Operational Notes
- Heavy cached artifacts are intentionally ignored via `.gitignore`:
  - `ot_24_48_exports/`
  - `reference_fields/*.npz`
  - `difference_detection/*.npy`
- Lightweight summaries/plots/manifests are kept for review.
