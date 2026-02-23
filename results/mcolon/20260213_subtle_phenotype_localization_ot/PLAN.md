# Subtle-Phenotype Localization via WT-Referenced OT + Rostral–Caudal Coordinates

**Pilot: cep290 (simple curvature phenotype)**

## Goal

- Build a time-consistent, interpretable "where along the embryo" signal by mapping mutant masks onto a WT reference mask within coarse stage bins.
- Produce stable visualizations + per-region discriminability (AUROC) over S (head→tail), then generalize to automated ROI discovery (patch search → sparse mask).

## Core constraints / design principles

- **Time consistency:** define regions in embryo-intrinsic coordinates (rostral→caudal spline parameter S ∈ [0,1]) rather than raw pixels.
- **Stability first:** prioritize robust mapping + outlier filtering + cross-bin consistency before any "fancy" explanation methods.
- **Start abstract** (masks + OT fields) and later extend to real images once the pipeline is stable.

## Data + scope

- **Dataset:** cep290 mutants + WT controls.
  - **WT reference (n=1):** OT target/template for alignment (defines coordinate system)
  - **WT controls (n≥10):** Mapped to reference, used for statistical comparison vs mutants
  - **Mutants (n≥20):** Mapped to reference, compared to WT controls
- **Representation for pilot:** binary masks (no raw intensity images yet).
- **First implementation:** one 2 hpf time window only. Once the pipeline works end-to-end within a single window, extending to multiple windows is straightforward — but that is not needed for the first implementation.
- **Temporal window policy (locked for pilot):**
  - Primary: single 2 hpf bin centered at 48 hpf, using `tolerance_hpf=1.25` for frame matching.
  - Optional robustness run: 4 hpf window `[46, 50]`.
  - If 4 hpf window is used, collapse to one row per embryo before stats by taking the per-embryo median of each feature across included frames (to avoid pseudo-replication).
- **Reference:** pre-selected WT reference mask (already chosen).
- **OT method:** unbalanced OT with parameters already tuned — treat as fixed for this pilot.
- **Sample size:** can always scale up cep290 embryos for a given time window, especially once AUROC analysis begins. Plan to scale up as needed.
- **Key distinction:** WT reference provides the spatial coordinate system (template space) but is NOT used in statistical comparisons. Statistical tests compare WT controls vs mutants after both are mapped to the reference.

## Smoothing policy (applies to all sections)

- **All AUROCs and statistical tests are computed on unsmoothed per-bin features; Gaussian kernel smoothing is for visualization only.**
- Gaussian kernel smoothing is applied in three contexts:
  - 2D cost density maps in template space (Section 1)
  - 1D along-S profiles (Section 2)
  - 1D AUROC-along-S profiles (Section 4)
- This avoids artificially inflating spatial coherence in the statistics while still producing clean, interpretable plots.

-----

### 1.1 Reference mask (pre-selected)

- A well-chosen WT reference mask is already available from Stream D cohort selection.
- **Source location** (as of 2026-02-13): `results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/`
  - Cohort manifest: `cohort_selected_embryos.csv` (set_type='reference_wt', rank=1)
  - Bin-frame mapping: `cohort_bin_frame_manifest.csv`
  - Selection script: `pipeline/01_build_cohort_manifest.py`
- **Locked contract for this pilot:** `results/mcolon/20260213_subtle_phenotype_localization_ot/data/cohort_contract_48hpf.json`
- **Selection criteria**: Maximizes 24-48 hpf bin coverage, minimizes curvature (straighter embryos preferred).
- **Note**: If location changed due to migration, search for `cohort_selected_embryos.csv` in `results/mcolon/2026*` directories.
- Store/reference its canonical grid coordinates.

### 1.2 Project masks onto reference (OT mapping)

- For each embryo mask in the bin (WT + cep290):
  - Compute unbalanced OT mapping to the WT reference mask on canonical grid.
  - (OT parameters already tuned — use as-is.)
  - Export:
    - Displacement vector field d(x) = T(x) - x (template coordinates)
    - Cost density map c(x) (scalar per pixel/point, consistent definition)
    - Total cost C = sum_x c(x) (scalar summary)
    - Mass delta map Δm(x) from unbalanced OT (and scalar summary)

### 1.3 Outlier detection + filtering (must-have)

**Problem:**

- Occasionally the aligner fails (bad alignment → absurd cost), producing clear outliers that should be removed.

**Plan:**

- Compute total cost C per embryo.
- Filter outliers using IQR rule:
  - Remove if C < Q1 - 1.5×IQR or C > Q3 + 1.5×IQR.
- Log/track removed embryos with:
  - embryo_id, snip_id, C value, reason.
- This filtering becomes especially important when scaling up sample sizes for AUROC analysis — weird embryos must not pollute the discriminability signal.
- (Optional) Add secondary sanity checks:
  - abnormal displacement magnitude statistics
  - obvious centroid mismatch
  - failure codes from aligner, if available

### Deliverables (Section 1)

- **Figure A:** average cost density heatmap in template space
  - Compute mean(c(x)) for WT and mutants separately (post-filtering)
  - Show mutant-minus-WT difference heatmap
- **Figure A':** Gaussian-kernel-smoothed versions of the above
  - Apply Gaussian kernel (diffusion smoothing) to cost density maps before averaging or to the difference map
  - These provide noise-robust, visually striking representations of where cost concentrates
  - Explore a few kernel bandwidths (σ) to find a good balance of smoothness vs detail
- **Figure B:** average displacement vector field in template space
  - Mean(d(x)) for mutants and WT
  - Show difference vector field (mutant minus WT)
- **QC panel:** histogram/violin of total cost C (before/after filtering)

### Success criteria to move on

- Costs look stable after outlier removal.
- Mean vector fields are not dominated by obvious alignment failures.
- Gaussian-kernel-smoothed maps highlight coherent spatial structure (not just noise).

-----

## SECTION 2 — Time-consistent regions: define S and compute along-axis profiles

### 2.1 Spline / centerline definition (use consistent method)

- Use the same spline-fitting method used elsewhere in MorphSeq for consistency.
- **Orientation convention (fixed):** S=0 is always head (rostral), S=1 is always tail (caudal).
- For the WT reference:
  - Fit centerline spline and define S ∈ [0,1] (head→tail).
- For each mapped embryo (after OT):
  - Use template-space spline to assign each template pixel/point to an S value.

### 2.2 Define regions by S bins

- Choose K bins along S (start with K = 10; later try K = 20 for resolution).
- Region k corresponds to S ∈ [k/K, (k+1)/K).
- This yields consistent regions across time because S is normalized.

### 2.3 Compute simple along-S curves (per embryo)

For each embryo, per bin region k:

- Mean cost density:      c̄_k = mean_{x in bin k} c(x)
- Mean |displacement|:    |d̄|*k = mean*{x in bin k} ||d(x)||
- Mean axial displacement: d̄∥_k = mean_{x in bin k} (d(x) · e∥(x))
- Mean perpendicular disp: d̄⊥_k = mean_{x in bin k} (d(x) · e⊥(x))
- Mean divergence:        div̄_k = mean_{x in bin k} ∇·d(x)  (finite diff in template grid)
- Mass delta (if any):    Δm̄_k = mean_{x in bin k} Δm(x)

**Notes:**

- e∥(x): local tangent direction of the spline at S(x)
- e⊥(x): local normal direction (perpendicular to tangent)

### Deliverables (Section 2)

- **Plot 1:** mean cost along S (WT vs mutant; with confidence bands)
- **Plot 1':** Gaussian-kernel-smoothed cost along S (smoothing over the S axis to reduce bin noise)
- **Plot 2:** mean |displacement| along S (WT vs mutant)
- **Plot 3:** mean axial vs perpendicular displacement along S (WT vs mutant)
- **Plot 4:** mean divergence and mass delta along S

### Success criteria to move on

- Along-S profiles show sensible, smooth structure.
- Profiles are robust to K (10 vs 20) and to minor bin shifts.
- Gaussian kernel smoothing produces clean, interpretable curves.
- Curvature phenotype yields expected head/tail emphasis (qualitative sanity check).

-----

## SECTION 3 — Feature map set: finalize "what we measure" per region

### 3.1 Minimal feature sets (start simple)

**Feature Set A (scalar-only MVP):**

- c̄_k (mean cost density per S bin)

**Feature Set B (vector-informed):**

- |d̄|_k
- d̄∥_k
- d̄⊥_k

**Feature Set C (local deformation + unbalanced OT):**

- div̄_k
- Δm̄_k (always available since we use unbalanced OT)

### 3.2 Organize outputs

- For each embryo/timepoint:
  - A feature vector per S bin (K-dimensional per feature)
  - A stacked feature matrix for downstream AUROC/classification

### Deliverable (Section 3)

- A standardized feature table schema:
  - columns: embryo_id, snip_id, stage_bin, label, k_bin, c̄, |d̄|, d̄∥, d̄⊥, div̄, Δm̄
- Saved intermediate maps (optional): c(x), d(x), div(x), Δm(x)

-----

## SECTION 4 — Discriminability: AUROC-by-region over time (the interpretability core)

### 4.1 AUROC per S bin (per-feature)

- For each S bin k independently:
  - **Scalar features** (univariate AUROC, equivalent to the Mann–Whitney U statistic up to scaling, computed via rank statistics):
    - c̄_k (cost density — i.e. the mass transport cost)
    - Δm̄_k (mass delta — mass created/destroyed by unbalanced OT)
    - |d̄|_k (displacement magnitude)
    - div̄_k (divergence)
  - **Directional feature** (2D vector, requires a small classifier):
    - Combine (d̄∥_k, d̄⊥_k) into a single 2D displacement direction vector per S bin.
    - The question: "is the directionality at this position discriminative on average?"
    - Use a simple classifier (e.g. logistic regression) with CV-by-embryo to get AUROC for this 2D feature.
- **Key point:** each S bin is evaluated independently. The question is "how discriminative is position k alone?" — not a joint model across S bins.
- This yields an AUROC(S_bin) vector per feature (single timepoint for now; extends to AUROC(stage_bin, S_bin) matrix when multiple timepoints are added).

### 4.1b Statistical significance: embryo-level label permutation

- **Null hypothesis:** For S bin k, WT controls and mutants have the same feature distribution (AUROC = 0.5).
- **Null distribution:** Shuffle genotype labels at the embryo level (not frame/snip level):
  - Collect all embryos: N WT controls + M mutants
  - Each embryo retains its features (c̄_k, |d̄|_k, etc.) but genotype labels are randomly reassigned
  - Recompute AUROC for the shuffled labels
  - Repeat for n_permutations (e.g., 999)
- **P-value:** Fraction of permuted AUROCs ≥ observed AUROC
- **Multiple testing correction:** Apply FDR correction (Benjamini-Hochberg) across S bins
- **Pattern from UOT MVP:** Use embryo-level label shuffling as in `results/mcolon/20260213_stream_d_reference_embryo/pipeline/06_difference_classification_clustering.py::_embryo_label_shuffle_pvalue()`

### 4.2 Future: layered noise-injection ablation (DEFERRED)

**Status:** Defer until Sections 1-5 validated. Permutation tests are sufficient for initial discriminability.

- Once AUROC profiles are established, importance of each feature layer can be tested by **adding varying amounts of noise** to specific features and measuring prediction degradation.
- Noise injection tests feature reliance given a trained model; **label-permutation remains the primary null for discriminability**. These answer different questions — noise injection measures "how much does this feature matter to the model," while label permutation measures "is there any real signal at all."
- Natural ordering for ablation layers: cost → mass delta → direction. This reveals whether directionality adds information beyond scalar cost/mass signals.
- Varying noise magnitude traces out a degradation curve per feature, giving a continuous measure of importance rather than a binary in/out test.
- **Guardrails:** all noise-injection results must be validated with bootstrap stability and cross-embryo/batch replication before interpreting.

### 4.3 Plotting deliverables

- **Bar/line plot:** AUROC along S (one curve per feature), single timepoint
  - Separate panels or overlaid for:
    - cost features (c̄_k)
    - vector magnitude (|d̄|_k)
    - axial displacement (d̄∥_k)
    - divergence / mass delta
  - **Add significance markers** (stars/shading) for bins passing FDR-corrected p < 0.05
- **Gaussian-kernel-smoothed AUROC along S:** smooth the AUROC(S) profile to highlight broad regions of discriminability rather than noisy per-bin values (visualization only, stats on unsmoothed)
- *(Future, multi-timepoint):* **Heatmap:** AUROC over time (stage bins) × position (S bins), with Gaussian kernel smoothing applied in both dimensions

### Success criteria

- AUROC localizes to interpretable positions along S.
- Gaussian-kernel-smoothed profiles show clear peaks rather than flat/noisy signal.
- Patterns are consistent under resampling/bootstrap.
- **Significant bins (FDR < 0.05) align with known phenotype** (e.g., tail for cep290 curvature).
- Scale up sample size if per-bin AUROCs are too noisy to interpret.

-----

## SECTION 5 — Automated ROI discovery in 1D first: "find the best S patch"

**Motivation:**

- Before full 2D sparse masks, validate the "find-region" concept in a constrained space where the answer is expected (curvature should localize to tail/head, likely tail).

### 5.1 Patch search on S (parameterized ROI)

- **Scope:** pooled across all embryos within the single 2 hpf time window. *(When extended to multiple windows, patch search can be run per-window to see if the best interval shifts over developmental time.)*
- If optional 4 hpf window `[46, 50]` is used, collapse to one row per embryo by per-embryo median feature aggregation before running patch search and permutation tests.
- Define a contiguous interval I = [a, b] over S bins.
- For each candidate interval:
  - Use only features inside I (e.g., c̄_k for k in I) to classify mutants vs WT.
  - Score by cross-validated AUROC (GroupKFold by embryo_id).
- Select interval(s) maximizing AUROC (with penalty on interval length if desired).

### 5.2 Patch ablation sanity check

- Train classifier on all S bins.
- For each interval I:
  - Remove/zero features in I and measure performance drop.
- Interval with biggest drop indicates most "important" S region.

### 5.3 Statistical significance for selected interval

- **Null hypothesis:** Selected interval's AUROC is no better than random intervals.
- **Null distribution:** Embryo-level label permutation:
  - Shuffle genotype labels across all embryos
  - Re-run patch search to find best interval on shuffled data
  - Record AUROC of best interval in null
  - Repeat for n_permutations
- **P-value:** Fraction of null best-interval AUROCs ≥ observed best-interval AUROC
- **Bootstrap stability:** Resample embryos (with replacement), re-run patch search, measure interval overlap (Jaccard similarity)

### Deliverables (Section 5)

- **Plot:** best AUROC vs interval length (tradeoff curve)
- **Plot:** selected interval(s) highlighted on S axis
- **Plot:** permutation null histogram for best interval AUROC
- **Report:** p-value for selected interval
- **Report:** stability of selected interval under bootstrap (mean Jaccard overlap)

### Success criteria

- Selected interval aligns with known curvature phenotype (qualitative).
- Selected interval is statistically significant (p < 0.05).
- Result stable across bootstrap resamples (Jaccard > 0.7).
- *(When extended to multiple timepoints: stable across stage bins.)*

-----

## SECTION 6 — Generalize to 2D sparse mask learning (L1 + TV), optional final step

**Motivation:**

- Move from 1D (S-only) region discovery to full 2D template-space localization.

### 6.1 Learn mask m(x) in template space

**Objective:**

- Classification loss + λ × L1(m) + μ × TV(m)

Where:

- L1 encourages small area (sparsity)
- TV encourages contiguous regions (few smooth blobs)

### 6.2 Tuning λ, μ

- Sweep (λ, μ) and select via Pareto:
  - maximize CV AUROC while minimizing:
    - mask area fraction
    - number of connected components
- **Practical note:** tuning λ and μ simultaneously via Pareto front can be computationally brutal if the template grid is dense. Aggressively downsample the template grid first to find the approximate hyperparameter basin, then refine at full resolution only for the final selected (λ, μ) region.
- Validate with:
  - patch ablation in learned mask area
  - bootstrap stability (mask overlap metrics)

### 6.3 Relationship to S-bin framework

- The current pipeline uses S bins as the spatial unit throughout. The 2D sparse mask learning generalizes this — instead of 1D intervals along S, the mask m(x) can select arbitrary 2D regions in template space.
- **Looking ahead:** S bins are a discretization of the body axis. In the future, this framework abstracts naturally to actual anatomical regions defined by segmentation (e.g., head, yolk, trunk, tail fin). The S-bin approach is the right first pass and is already highly informative; segmented-region analysis is a natural extension once region definitions are available.

### Deliverables (Section 6)

- Learned mask overlays on template
- Performance vs mask complexity plots
- Stability summary across bootstraps

-----

## Global QC / hygiene (applies everywhere)

- Always split CV by embryo_id (avoid leakage across snips).
- When extending to multiple timepoints: track stage bins explicitly and never mix bins silently.
- If multiple frames per embryo are included in a wider stage window (e.g., 4 hpf), collapse to one row per embryo per feature (median) before WT-vs-mutant testing.
- **Scaling strategy:** start with available embryos; if AUROC profiles are noisy, collect more cep290 embryos in the target time window before drawing conclusions.
- Maintain a per-run manifest:
  - reference mask used, K bins, OT parameters, filtering thresholds, Gaussian kernel bandwidth(s)
- For later extension to raw images:
  - intensity normalization inside embryo mask (robust scaling + stage-wise histogram matching)
  - but defer until mask-only pipeline is validated

-----

## Immediate next actions (implementation order)

1. Load pre-selected WT reference mask + implement unbalanced OT mapping export (c(x), d(x), Δm(x), C) — **single timepoint only**.
1. Add IQR-based outlier filtering on total cost C with logging.
1. Generate Section 1 figures: cost heatmaps, Gaussian-kernel-smoothed heatmaps, mean vector fields.
1. Implement spline-based S assignment in template space (head=0, tail=1) and S binning (K=10 first).
1. Compute along-S profiles for cost and displacement magnitude; generate plots + Gaussian-kernel-smoothed versions.
1. Build AUROC(S) profile for cost (univariate per S bin); then add displacement features. Scale up embryo count if needed.
1. Implement 1D contiguous-interval patch search + ablation on S.
1. Extend to multiple timepoints (stage bins) and produce AUROC(stage, S) heatmaps.
1. (Optional) 2D sparse mask learning with L1 + TV once 1D is stable.

-----

## Definition of "done" for this pilot

A stable pipeline that:

1. Maps cep290/WT masks to WT reference within a single 2 hpf window (or optional 4 hpf robustness window with per-embryo feature collapse)
1. Filters alignment failures robustly
1. Produces interpretable spatial summaries (cost + vectors)
1. Yields AUROC-by-S plots that localize the phenotype
1. Supports automated region discovery in S (and optionally 2D)

-----

## Status / Notes

*(space for tracking progress, decisions, and revisions)*
