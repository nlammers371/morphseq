ADDENDUM: Phase 2 Occlusion Validation Plan
Incorporating “ASSESSMENT: Phase 2 Counterfactual Occlusion Validation Plan vs PR #9”
Date: 2026-02-15

This addendum updates the Phase 2 plan with concrete integration fixes and risk mitigations identified in the assessment. It is still Phase 2.0-only (validation, no mask learning) and remains additive to PR #9.

============================================================
A) REQUIRED INTEGRATION FIXES (Phase 1 / PR #9 alignment)
============================================================

A1) Fix driver import bug (PR #9)
- Problem: run_roi_discovery.py imports ROIRunConfig from roi_config, but ROIRunConfig is commented out → ImportError.
- Action:
  [ ] Either (preferred) re-enable ROIRunConfig in roi_config.py and keep it as the single config surface
  [ ] OR remove the import and use explicit args until ROIRunConfig is wired

A2) Add channel ordering / names to TrainResult (required for Phase 2)
- Problem: Phase 2 needs channel_names/channel_schema to apply baseline policies deterministically.
- Action:
  [ ] Add channel_names (tuple[str]) to TrainResult.config (or TrainResult as a field)
  [ ] Source of truth: FeatureDataset manifest channel_schema
  [ ] Pass through from loader → trainer → TrainResult

A3) Extract shared compute_logits() utility (avoid drift)
- Problem: logit computation duplicated in roi_sweep.py; Phase 2 would add another copy.
- Action:
  [ ] Create compute_logits(X, w_full, b) -> (N,) in roi_trainer.py or a small roi_utils.py
  [ ] Replace all inlined logits computations with this utility (Phase 1 + Phase 2)

A4) Bootstrap resampling helper (reduce duplication)
- Phase 1 bootstrap loop differs from Phase 2 OOB bootstrap loop; both are group-aware and similar.
- Action:
  [ ] Create a group-aware resampling helper (e.g., iter_bootstrap_groups(groups, y, stratify=True)) that yields:
      - inbag_group_ids
      - oob_group_ids
      - flags for empty/degenerate OOB
  [ ] Phase 2 uses it for OOB evaluation; Phase 1 can optionally keep its current loop

============================================================
B) PHASE 2.0 COMMITMENTS (tightened)
============================================================

B1) Bootstrap strategy: FIXED (λ*, μ*) + true OOB evaluation
- Commit: Phase 2 bootstrap_occlusion trains at fixed selected (λ*,μ*) and evaluates on OOB embryos only.
- This answers “is THIS ROI stable?” and is cheaper and more interpretable than a full-sweep bootstrap.

B2) Baseline fold-safety
- Baselines MUST be computed from train/inbag data only for any reported metric.
- “Full dataset sanity check” is allowed only as a diagnostic:
  - Must log WARNING: baseline computed from full dataset (leakage) and results are non-inferential.

B3) Primary metric: logit-gap difference (bootstrap target)
- Primary reported statistic per replicate:
  - observed_gap_b = mean_i[(z_orig - z_delete) - (z_orig - z_preserve)] on OOB
- Secondary (report-only unless later requested): AUROC deltas

B4) Phase 2.0 baseline policy (avoid scope creep)
- Implement baseline_policy plumbing, but default ALL channels to "spatial" for Phase 2.0.
- Defer per-channel customization to Phase 2.1 unless a concrete failure mode appears.

============================================================
C) EDGE CASES (must be handled explicitly)
============================================================

C1) OOB set degeneracy
- In some bootstrap replicates, OOB may be empty OR contain only one class.
- Rules:
  - If OOB empty → skip replicate, log
  - If OOB single-class:
      - logit gaps still computed and used for observed_gap_b
      - AUROC metrics are undefined → set NaN, log

C2) ROI threshold sensitivity (cheap but important)
- Phase 2 results depend on quantile threshold q used in extract_roi().
- Requirement:
  - Run Phase 2.0 at Phase 1 default q first
  - Add a quick sensitivity report for q ∈ {0.85, 0.90, 0.95}
  - This requires no retraining if ROI is re-thresholded from w_full, but in Phase 2 bootstrap the ROI is extracted per replicate anyway:
      - implement threshold sweep inside evaluation step (cheap), or run 3 passes of evaluation using same bootstraps

C3) Small-N power expectation
- Add a pre-flight diagnostic:
  - estimate “expected gap scale” from Phase 1 weights inside vs outside ROI on training set
  - if predicted gap is comparable to noise, mark Phase 2 as potentially inconclusive a priori
- This is a warning flag, not a blocker.

============================================================
D) UPDATED PHASE 2.0 IMPLEMENTATION SEQUENCE
============================================================

Week 0 (pre-Phase 2 coding; must do)
  [ ] Fix ROIRunConfig import bug in run_roi_discovery.py
  [ ] Extract compute_logits() utility and use it in Phase 1 call sites
  [ ] Ensure TrainResult carries channel_names (from FeatureDataset manifest)

Week 1: roi_perturbation.py (NEW)
  - Baseline computation:
      constant spatial baseline from TRAIN only (WT-only default)
  - Perturbation operator:
      P(X,m) = m*X + (1-m)*B
  - Strict shape/mask checks + unit tests
  - (optional) group-resampling helper added here if convenient

Week 2: roi_occlusion.py (NEW)
  - evaluate_occlusion():
      compute z_orig, z_delete, z_preserve via compute_logits()
      output per-sample gaps + observed_gap
      compute AUROC deltas if both classes present (else NaN)
  - bootstrap_occlusion():
      OOB bootstrap loop at fixed (λ*, μ*)
      baseline fit on inbag only
      OOB evaluation only
      handle empty/single-class OOB cases
      store bootstrap_gaps, CI, frac_positive

Week 3: integration + config surface
  - Prefer enabling ROIRunConfig (single config object) to avoid fit() kwargs explosion
  - Add occlusion section to run artifacts:
      occlusion_config.json
      occlusion_summary.json
      bootstrap_gaps.npy
      plots/
  - Add baseline leakage warning for optional diagnostic full-dataset sanity eval

Week 4: validation
  - Synthetic planted-ROI smoke test (required before real data):
      create a toy dataset with a known ROI → verify Δ_delete > Δ_preserve
  - Real cep290 run:
      run Phase 2.0 default q
      run threshold sensitivity q ∈ {0.85, 0.90, 0.95}
  - If displacement channels look weak under constant baseline:
      queue Phase 2.1 blur baseline (not required for Phase 2.0 completion)

============================================================
E) PHASE 2.0 “DONE” DEFINITION (unchanged, but sharper)
============================================================

Phase 2.0 is DONE when:
- Occlusion preserve/delete evaluation works end-to-end
- Primary metric (logit-gap difference) has OOB bootstrap CI
- Degenerate OOB cases are handled and logged
- Artifacts saved in standard run directory
- Threshold sensitivity report is generated (q ∈ {0.85,0.90,0.95})
- No API/config sprawl: configuration is centralized (prefer ROIRunConfig)

============================================================
F) NOTES ON “COMPUTATIONAL STABILITY” VS “BIOLOGY-DEPENDENT” TUNING
============================================================

We explicitly do NOT claim a universal optimal (λ,μ,q).
Instead, we will:
- maintain “computationally stable defaults” (learn_res=128, modest λ/μ grid, default q)
- log sweep behavior and runtime scaling as part of run artifacts
- treat parameter sweeps as biology-dependent discovery, documented per dataset/phenotype
