MorphSeq Subtle Phenotyping (v3.5)
==================================
Purpose: A methods-first, interpretable, cluster-number-agnostic framework for discovering subtle
morphological phenotypes in embryo time series. We treat embryos over time as signals and analyze
them through multiple representations (distance, neighborhood/cohort persistence, event timing),
requiring convergence across representations for high-confidence claims.

This v3.5 is a strict refinement of v3:
  - No information removed (only clarified, re-ordered, or upgraded).
  - Abstract graph summary features are de-emphasized further in favor of embryo-grounded signals.
  - Explicit split between “Discovery” distances (directional/multivariate) and “Scoring” signals
    (severity/onset magnitude).
  - AB reference upgraded to geometric median.
  - Bootstrapping/permutation infrastructure explicitly reuses existing utilities:
      * run_classification_test (already includes null distribution)
      * plot_features_over_time (faceted plotting already exists)
    We extract a shared bootstrap/permutation utility from run_classification_test and reuse it
    throughout the pipeline.

Key constraints / priorities
----------------------------
1) Interpretability first:
   - Outputs must map back to embryo stories (neighbors, cohorts, deviation curves, onset windows,
     effect sizes, exemplar embryos).

2) Cluster-number agnostic:
   - Avoid specifying #clusters.
   - Use persistence/co-association, stability plateaus, and bootstrap consensus.

3) Validation gates:
   - All tuning/validation on cep290 and b9d2 where we have:
       (a) phenotype_label (prior phenotype classifications),
       (b) expected onset windows:
           - b9d2: CE phenotype emerges ~18 hpf
           - cep290: phenotype emerges ~26–30 hpf
   - Only after passing gates do we deploy to TFAP2 CRISPR panel.

4) Windows:
   - Default time-local unit = every 2 hpf window.
   - Optionally slide by 1 hpf for smoother visualization; core analyses use 2 hpf bins.

5) Inference is always relative to a null:
   - Wherever we claim “signal,” we must show it exceeds null via permutation/bootstrap.
   - We avoid duplicating null-generation logic by extracting a shared utility from
     run_classification_test (already has null distribution baked in).

What we are trying to learn (goals)
-----------------------------------
G0. Is there genotype-associated structure at all? (global signal)
G1. Are there phenotype cohorts (embryos moving together) that are genotype-agnostic? (embryo-first)
G2. Is the effect subtle but heterogeneous (penetrance/dispersion/outliers)?
G3. When does structure emerge or reorganize in development? (onset windows)
G4. What explains structure (main effects + epistasis)? (stable effect sizes)

Data objects and representations (interpretable)
------------------------------------------------
Per embryo i:
  - embedding trajectory Z_i(t) in R^d (VAE metric embedding / biological embedding)
  - derived PC trajectory PC_i(t) in R^m for dynamic analyses (m small, e.g., 10)
  - time axis: predicted_stage_hpf (preferred), resampled to common grid
  - phenotype_label (validation datasets only)

Core primitives:
A) Robust AB reference + deviation magnitude (severity/onset)
   - mu_AB(t): geometric median of AB embryos at each time point in PC space
   - d_i(t) = ||PC_i(t) - mu_AB(t)||_2
   - summaries: AUC_i, max_i, (optional) threshold crossing times
   - “AB tube”: quantile bands of ||PC_AB(t) - mu_AB(t)|| (e.g., 50/90/95%)

B) Distance matrices (role-split; crucial)
   Discovery / Cohorting (phenotype identity; directional):
     - D_dtw_pc: multivariate DTW on PC_i(t) in R^m  [CUSTOM IMPLEMENTATION AVAILABLE]
   Scoring / Timing (phenotype severity profile; magnitude):
     - D_dtw_devAB: DTW on 1D deviation curves d_i(t)
   Baseline:
     - D_static: Euclidean on summary vectors

C) Dynamic neighbor structure (co-movement)
   - For each 2 hpf window w: compute distances D(w) in PC space
   - Build weighted directed kNN graph with local scale; symmetrize by soft mutuality
   - Build persistence/co-association matrices:
       * global P_mean (overall co-movement)
       * global P_topq (late-onset friendly pooling)
       * epoch-specific P^(e) (cohorts per developmental epoch)

Existing utilities (must be used)
---------------------------------
- run_classification_test:
    * already implements a bootstrap/permutation/null distribution pattern
    * use it to verify “real signal” and to extract reusable null/bootstrap utility functions
- plot_features_over_time:
    * already exists; use it for faceted time plots (genotype/cohort/phenotype_label/batch)

Plotting outputs to standardize early (faceted)
-----------------------------------------------
(Use plot_features_over_time + faceting; these are required artifacts.)
1) d_i(t) curves: individual + group mean + CI + AB tube
2) drift/gain/loss over time (per embryo and aggregated by cohort/genotype)
3) cohort cohesion over time (within-cohort vs between-cohort similarity)
4) MDS/PCoA embeddings colored by genotype/cohort/phenotype_label/batch
5) exemplar embryo panels (images at key times) for each cohort + outliers

Packages (Python)
-----------------
Core: numpy, pandas, scipy
ML: scikit-learn (PCA, pairwise distances, MDS)
Stats: scikit-bio (PERMANOVA), statsmodels (FDR)
Graphs: igraph + leidenalg, networkx
CPD: ruptures
Matrix profile (extension): stumpy
DTW: CUSTOM multivariate DTW implementation (NA-aware, distance-similarity aware)

Part 0: Policies (once per dataset)
-----------------------------------
0.1 Missingness
  - Global distances (DTW): restrict to a common “core interval” with high coverage OR exclude embryos
    for that analysis.
  - Time-local windows: analyze only windows where embryo has valid data; do not force global truncation.
  - NaNs: interpolate missing frames in embedding/PC space before PCA/DTW/kNN.

0.2 Batch/confounds
  - Always visualize batch early.
  - Use restricted permutations within batch strata for hypothesis tests if needed.
  - Prefer restricted nulls to explicit batch correction unless batch dominates geometry.

0.3 Defaults
  - PCs: m = 10 (range 5–15)
  - kNN: k = 15 (range 10–25)
  - Windows: 2 hpf bins (core)
  - Permutations/bootstraps: dev 1000; final 5000
  - Community resolution: scan small grid; select stability plateaus via bootstrap consensus

Part 1: Preflight QC + robust AB reference (Layer 0)
----------------------------------------------------
1.1 Trajectory sanity
  - Plot PC1–PC3 trajectories vs time (subset) to detect discontinuities/seg failures.

1.2 Batch visualization
  - Quick global embedding (MDS/PCoA on D_static or D_dtw_pc) colored by batch.

1.3 Robust mu_AB(t) and deviation curves (core)
  - mu_AB(t) = geometric median of AB embryos at each time point (in PC space).
  - d_i(t) = ||PC_i(t) - mu_AB(t)||.
  - AB tube: quantile bands of ||PC_AB(t) - mu_AB(t)|| for interpretability.
  - Store AUC_i, max_i, optional threshold crossing times.

Outputs:
  - mu_AB(t), AB tube bands, d_i(t), AUC_i, max_i
  - QC exclusions table

Part 2: Distance toolbox (Layer 0) — role split enforced
--------------------------------------------------------
2.1 D_dtw_pc (DISCOVERY distance; cohorting)
  - multivariate DTW on PC trajectories using CUSTOM DTW.
  - This is the default global distance for any “what phenotype group is this?” step.

2.2 D_dtw_devAB (SCORING distance; severity/onset profile)
  - DTW on 1D d_i(t).
  - Use for severity-profile similarity, penetrance scoring, and onset timing comparisons.
  - Do NOT use as the primary cohort discovery distance.

2.3 D_static (baseline)
  - Euclidean on summary vectors (mean/var/range PCs + AUC_i + max_i).

Outputs:
  - D_dtw_pc.npy, D_dtw_devAB.npy, D_static.npy

Part 3: Genotype-first global tests (Layer 1)
---------------------------------------------
3.1 Global association
  - Primary: PERMANOVA on D_dtw_pc (genotype factor)
  - Secondary: PERMANOVA on D_dtw_devAB and D_static (sanity and severity signal)
  - Restricted permutations if batch strata exist
  - Report: R^2 + p-value

3.2 Heterogeneity / penetrance proxy
  - Within-genotype dispersion in distance space (use D_dtw_pc as default; optionally compare to devAB)
  - Report dispersion ratios (disp_g / disp_AB) with permutation p-values
  - Use d_i(t) summaries (AUC_i, max_i) as interpretable severity metrics.

3.3 Optional: Hamming trend (genetic distance vs morphology distance)
  - Compute D_gen (Hamming) vs D_dtw_pc (or devAB) with permutation test.
  - Interpret cautiously; epistasis/subset effects can break monotonicity.

Outputs:
  - global genotype signal report (PERMANOVA + dispersion + severity summaries)

Part 4: Embryo-first cohorts via persistence (Layer 1)
------------------------------------------------------
Goal: find embryos moving together without labels; then annotate/enrich.

4.1 Dynamic distances per 2 hpf window
  - For each 2 hpf window w:
      compute Euclidean distances in PC space -> D(w)

4.2 Weighted directed kNN (density adaptive)
  - For each embryo i:
      kNN in D(w)
      sigma_i(w) = distance to kth neighbor
      w_{i->j}(w) = exp(-D_ij(w)^2 / sigma_i(w)^2)

4.3 Soft mutuality symmetrization
  - w_ij(w) = sqrt(w_{i->j}(w) * w_{j->i}(w))

4.4 Persistence pooling (two variants; interpretably motivated)
  - P_mean_ij = mean_w [ w_ij(w) ]         (overall co-movement)
  - P_topq_ij = mean of top q% windows     (late-onset friendly; default q=25%)
  Notes:
    * P_mean is the default global summary.
    * P_topq is used when onset is late or when epoch analysis suggests regime changes.

4.5 Cohort discovery
  - Run Leiden on P_mean (primary).
  - Use bootstrap consensus to identify stability plateaus across resolution.
  - Optionally compare cohorts from P_topq as a sensitivity check (late-onset recovery).

4.6 Embryo-level neighborhood dynamics (interpretable)
  - drift_w(i) = 1 - Jaccard(N_w(i), N_{w+1}(i))
  - gain_w(i), loss_w(i) as in v3
  - taxonomy: stable / transitioner / joiner / expelled
  - Produce “drift vs max deviation” plot for artifact triage:
      X = stability (1 - drift), Y = max_i

4.7 Cohort annotation (after discovery)
  - Enrichment of genotype and bit-features (has A, has C, A&C, etc.)
  - Fisher exact and/or permutation null; report odds ratios + FDR p-values
  - Exemplar embryos per cohort (high stability) + image panels at key times

Outputs:
  - P_mean, P_topq, cohort labels, drift tables, enrichment, exemplar lists/panels

Part 5: “When” signal (Layer 2) — interpretable, low-abstraction
----------------------------------------------------------------
Engine A: CPD on deviation curves (primary timing)
  5A.1 Per-embryo CPD on d_i(t)
      - run CPD on d_i(t); output breakpoint distributions per genotype/cohort
      - report bootstrap CIs on onset times and breakpoint times
  5A.2 Group-level CPD
      - CPD on mean d(t) per genotype and per cohort

Engine B: Epoch-specific persistence (default follow-up when timing differs)
  5B.1 Define epochs
      - Validation: ensure epochs include ~18 hpf (b9d2) and ~26–30 hpf (cep290)
      - General: define epochs using CPD outputs (above)
  5B.2 Compute P^(e) per epoch
      - P^(e)_ij = mean_{w in epoch e} w_ij(w)
  5B.3 Cohorts per epoch
      - Leiden on P^(e), track cohort birth/death/split/merge across epochs
      - Interpret in embryo terms: who moves together in each epoch?

Engine C (optional): CPD on population drift (interpretable “reorganization”)
  - Delta(w) = mean_i drift_w(i)
  - CPD on Delta(w) (and optionally mean gain/loss)

Outputs:
  - onset/breakpoint distributions + CIs
  - epoch persistence matrices + epoch cohorts + event timelines

Part 6: Explanatory genotype modeling (Layer 2)
-----------------------------------------------
6.1 Embed distances
  - MDS/PCoA on D_dtw_pc (primary) to obtain axes; check artifacts.

6.2 Fit genotype feature model (main + pairwise interactions)
  - Axis-wise regression; compute ΔR^2 per term; report stable effect sizes.

6.3 Regularization fallback
  - If rank/conditioning unstable: ElasticNet fallback (stable term selection).
  - Report bootstrap stability of selected terms.

6.4 Time-local epistasis (optional)
  - Repeat modeling within epochs from Part 5.

Outputs:
  - ranked effects (main + interactions) with stable effect sizes + uncertainty

Part 7: Shared bootstrap/permutation utility (required infrastructure)
----------------------------------------------------------------------
7.1 Extract from run_classification_test
  - Pull the core machinery that generates:
      * permuted label sets / restricted permutations
      * bootstrap resampling
      * null distributions for metrics
  - Wrap into a reusable module (single interface) used by Parts 3–6.

7.2 Standardize outputs everywhere
  - effect size + CI
  - null p-value
  - stability metrics (ARI/NMI when reference labels exist)
  - membership probabilities / entropy (cohort confidence)
  - caching of permuted labels and bootstrap draws

7.3 Use cases
  - PERMANOVA + dispersion + enrichment
  - cohort stability across Leiden resolution (plateaus)
  - DTW clustering stability
  - onset timing stability (CPD breakpoints)
  - “is there signal?” verification via run_classification_test framework

Validation gates (cep290 + b9d2) — REQUIRED
-------------------------------------------
V1) Cohort recovery vs phenotype_label
  - Compare cohorts from:
      - DTW clustering on D_dtw_pc
      - persistence cohorts (P_mean; epoch P^(e))
  - ARI/NMI vs phenotype_label with bootstrap CI

V2) Timing expectations
  - b9d2: detect major onset/reorganization near ~18 hpf
  - cep290: detect near ~26–30 hpf
  - report distributions and effect magnitudes

V3) Null validation
  - Under permutation: enrichment and timing alignment should vanish/degrade
  - Under bootstrap: key cohorts and onset claims should persist

Execution plan (simple -> complex)
----------------------------------
Phase 1 (MVP signal + cohorts; cep290/b9d2)
  1) Part 1 QC + geometric-median mu_AB + d_i(t) + AB tube plots
  2) Part 2 distances (D_dtw_pc primary; devAB for severity/timing)
  3) Part 3 PERMANOVA + dispersion + severity summaries
  4) Part 4 persistence cohorts (P_mean + optional P_topq) + drift taxonomy
  5) Validation vs phenotype_label + null checks using extracted utility

Phase 2 (when; cep290/b9d2)
  6) Part 5 CPD on d_i(t) + epoch persistence P^(e) + epoch cohorts

Phase 3 (explain; cep290/b9d2)
  7) Part 6 epistasis decomposition + bootstrap stability

Phase 4 (deploy; TFAP2 panel)
  8) Repeat Phases 1–3 (phenotype_label absent; rely on convergence + stability)

High-impact extensions (after core stable)
------------------------------------------
E1) Matrix Profile (stumpy) on PC1(t) and/or d_i(t)
  - motifs: stereotyped subsequences
  - discords: anomalies / segmentation failures
  - use to prioritize window inspection and restricted analyses

E2) Minimal dynamical features (optional)
  - catch22 on d_i(t) or top PCs as descriptive feature ranking (not main engine)

Definition of success
---------------------
- Stable embryo-first cohorts that map to interpretable exemplars and validated phenotype_label
- Robust onset timing near known windows (18 hpf b9d2; 26–30 hpf cep290)
- Clear separation between discovery (multivariate) and scoring (magnitude)
- Shared null/bootstrap utility used everywhere (no ad hoc permutation code)
- Strong evidence above null: effect sizes + uncertainty, not only p-values

End of document
---------------
Immediate next steps:
  - Implement Phase 1 using existing run_classification_test + plot_features_over_time,
    extract the shared bootstrap/permutation utility first, then re-run Parts 3–4 with consistent nulls.