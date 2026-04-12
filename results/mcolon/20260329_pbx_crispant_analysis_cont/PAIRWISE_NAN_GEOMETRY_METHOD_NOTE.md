# Pairwise NaN Geometry Method Note

## Initial issue

The pairwise contrast-coordinate pipeline was producing a spurious separation between `inj_ctrl` and `wik_ab` within matched time bins, even though this direct negative-control comparison should be near-null.

The strongest symptom was that the artifact was already visible at the pairwise AlignedUMAP initialization stage, which made the downstream cosmology geometry untrustworthy for null-control diagnostics and for later phylogeny-style interpretation.

## What the bug turned out to be

The core failure mode was off-support zero-imputation.

Pairwise classification outputs already preserved off-support probe entries as `NaN` in `raw_coordinates` and `shrunk_coordinates`. The artifact entered downstream when sparse pairwise coordinates were exported or consumed as dense vectors with `0.0` substituted for those `NaN` entries.

In this representation:
- `0.0` is a real signed-margin value
- `NaN` means "not comparable / off-support"

Treating off-support entries as `0.0` created synthetic geometry. For example, one class could have a real value on one probe while the other class had `0.0` only because that probe was undefined for that class. That turned missingness into apparent biological structure.

## Files showing the problem

### Direct diagnosis note
- [PAIRWISE_NAN_GEOMETRY_NOTE.md](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/PAIRWISE_NAN_GEOMETRY_NOTE.md)

### Null-control artifact before the fix
- [WITHIN_BIN_NULL_SUMMARY.md](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_within_bin_null_audit_bin4_perm10/WITHIN_BIN_NULL_SUMMARY.md)
- [anchor_bin_null_auroc_summary.csv](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_within_bin_null_audit_bin4_perm10/anchor_bin_null_auroc_summary.csv)

Key bad pre-fix behavior:
- `26 hpf`: `pairwise_raw_aligned_umap_init ~ 0.92`
- `54 hpf`: `pairwise_raw_aligned_umap_init ~ 0.95`
- `78 hpf`: `pairwise_raw_aligned_umap_init ~ 1.00`

### Probe-level evidence that missingness was masquerading as signal
- [FOCAL_PROBE_ZOOM_SUMMARY.md](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_focal_probe_zoom/FOCAL_PROBE_ZOOM_SUMMARY.md)
- [raw_focal_probe_summary.csv](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_focal_probe_zoom/raw_focal_probe_summary.csv)

That zoom showed a one-sided pattern where `inj_ctrl` was nonzero on one external-reference probe and `wik_ab` was nonzero on another, which was the red flag that off-support axes were being interpreted as real zeros.

## Files showing the fix

### NaN-aware null-control audit after the fix
- [WITHIN_BIN_NULL_SUMMARY.md](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_within_bin_null_audit_nanfix/WITHIN_BIN_NULL_SUMMARY.md)
- [anchor_bin_null_auroc_summary.csv](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_within_bin_null_audit_nanfix/anchor_bin_null_auroc_summary.csv)

Key post-fix behavior:
- `26 hpf`: `pairwise_raw_aligned_umap_init = 0.231`, `pairwise_shrunk_aligned_umap_init = 0.366`
- `54 hpf`: `pairwise_raw_aligned_umap_init = 0.333`, `pairwise_shrunk_aligned_umap_init = 0.267`
- `78 hpf`: `pairwise_raw_aligned_umap_init = 0.417`, `pairwise_shrunk_aligned_umap_init = 0.308`

These are back near null rather than near-perfect separation.

### Representation ablation after the fix
- [REPRESENTATION_ABLATION_SUMMARY.md](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_representation_ablation_audit_nanfix/REPRESENTATION_ABLATION_SUMMARY.md)
- [representation_ablation_anchor_auroc.csv](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/wikab_injctrl_representation_ablation_audit_nanfix/representation_ablation_anchor_auroc.csv)

Important post-fix interpretation:
- `all_pairs` no longer blows up at init just because sparse probe support was densified incorrectly
- `focal_vs_pbx1b_pbx4` is now reported as unsupported (`n_supported_features = 0`) rather than spuriously separated

## Implementation files

### Classification-side support metadata
- [contrast_support.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/engine/contrast_support.py)
- [contrast_coordinates.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/engine/contrast_coordinates.py)
- [loop.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/engine/loop.py)
- [run_classification.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/run_classification.py)
- [analysis.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/engine/analysis.py)
- [test_run_classification.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/analyze/classification/tests/test_run_classification.py)

Support states now tracked:
- `supported`
- `unsupported_id`
- `unsupported_group`

Artifacts now tracked:
- `contrast_support_long`
- support columns on `contrast_specificity_by_timebin`

### Pairwise geometry path
- [08_pairwise_probe_fingerprint.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/08_pairwise_probe_fingerprint.py)
- [schema.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/trajectory_cosmology/schema.py)
- [init_embedding.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/trajectory_cosmology/init_embedding.py)
- [05_pbx_condensation.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/05_pbx_condensation.py)

Primary geometry change:
- preserve sparse pairwise probe matrices with `NaN`
- compute pairwise distances only on shared supported probes
- initialize per time bin with `UMAP(metric="precomputed")`
- align consecutive bins with orthogonal Procrustes

## Hyperparameters / method settings currently used

### Pairwise classification / coordinate generation
Source run:
- [phenotypic_positioning_pairwise_bin4_perm500](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_pairwise_bin4_perm500)

Main settings:
- comparisons: `all_pairs`
- features: `z_mu_b*` via `{"vae": "z_mu_b"}`
- bin width: `4.0 hpf`
- permutations for specificity/shrinkage: `500`
- shrinkage weight: `w = clip((auroc_obs - auroc_null_mean) / 0.5, 0, 1)`

### NaN-aware pairwise condensation validation runs
Raw:
- [pairwise_raw_condensation_nanfix_iter50](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_raw_condensation_nanfix_iter50)

Shrunk:
- [pairwise_shrunk_condensation_nanfix_iter50](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_shrunk_condensation_nanfix_iter50)

Condensation settings used for validation:
- iterations: `50`
- save every: `10`
- learning rate: `1e-4`
- random seed for init: `42`

Sparse init settings in the current implementation:
- metric: precomputed NaN-aware Euclidean distance
- neighbor count per bin: `min(15, n_bin - 1)` with lower safeguard when very small
- consecutive-bin alignment: orthogonal Procrustes

### Audit settings used for fix confirmation
Within-bin null audit:
- [12_wikab_injctrl_within_bin_null_audit.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/12_wikab_injctrl_within_bin_null_audit.py)
- anchor bins: `26`, `54`, `78 hpf`
- permutations in quick confirmation run: `10`

Representation ablation audit:
- [14_wikab_injctrl_representation_ablation_audit.py](/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260329_pbx_crispant_analysis_cont/14_wikab_injctrl_representation_ablation_audit.py)
- variants: `direct`, `focal_vs_pbx1b_pbx4`, `all_pairs`
- support-aware feature selection: only probes with support in both classes in the anchor bin
- vector-space model: logistic regression with train-fold mean imputation and standardization

## Current method conclusion

The major null-control failure was not the direct `inj_ctrl__vs__wik_ab` classifier. It was downstream geometry built from sparse pairwise coordinates after off-support entries were treated as zeros.

The NaN-aware geometry path materially fixes that issue and should now be treated as the correct default for pairwise trajectory/cosmology work.
