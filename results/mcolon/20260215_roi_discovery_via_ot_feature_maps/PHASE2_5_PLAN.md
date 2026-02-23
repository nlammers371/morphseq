PHASE 2.5 (Research / If Needed) — Learn ROI Masks by Perturbation (Fong/TorchRay-style)
Goal: Integrate perturbation logic into training (learn m), in a way that is testable, selection-aware, and scalable.

References (inspiration + implementation patterns)
- Fong & Vedaldi 2017 “Meaningful Perturbations”: optimize a mask to affect a model output under perturbations, with L1/TV + jitter/smoothing to avoid artifacts.  [oai_citation:0‡arXiv](https://arxiv.org/abs/1704.03296?utm_source=chatgpt.com)
- Fong et al. 2019 “Extremal Perturbations”: smooth masks, area-controlled solutions; TorchRay provides a practical implementation with preserve/delete/dual variants and perturbation operators/pyramid.  [oai_citation:1‡arXiv](https://arxiv.org/pdf/1910.08485?utm_source=chatgpt.com)
- Fong’s GitHub (older) and TorchRay’s extremal_perturbation.py show concrete design ideas (mask parameterization, perturbation operators, optimization loops).  [oai_citation:2‡GitHub](https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/extremal_perturbation.py?utm_source=chatgpt.com)

=====================================================================
0) Why Phase 2.5 is different (and how they did it)
=====================================================================
Fong-style methods typically:
- fix the model and optimize a mask per-example via gradient descent, using a perturbation (blur/constant) and regularizers (L1/TV).  [oai_citation:3‡arXiv](https://arxiv.org/abs/1704.03296?utm_source=chatgpt.com)
TorchRay extremal perturbations:
- optimize a smooth mask (often via a multi-scale/pyramid schedule) and support “preserve”/“delete”/“dual” objective variants with differentiable perturbations.  [oai_citation:4‡GitHub](https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/extremal_perturbation.py?utm_source=chatgpt.com)

MorphSeq adaptation:
- we do NOT want per-image masks (too expensive, brittle).
- we want a dataset-level ROI (per genotype/time-bin/feature-set): one mask m shared across many embryos.

This is the key scaling choice.

=====================================================================
1) Phase 2.5 scope decision (critical, pick ONE for Phase 2.5a)
=====================================================================
We will implement ONE of these first:

(Option A, recommended first) Learn a GLOBAL mask m for a fixed classifier
- Freeze trained Phase 1 classifier (w,b) (or retrain once, then freeze).
- Optimize only m to maximize a perturbation objective.
- Pros: clean, safe, easy to test; isolates “mask learning” as a module.
- Cons: m depends on a fixed classifier; might miss alternate decision boundaries.

(Option B) Jointly learn classifier + mask (w,b,m) end-to-end
- Pros: can find a better combined solution.
- Cons: more failure modes; easier to “cheat” (co-adaptation); harder to interpret.

Phase 2.5a should be Option A.

=====================================================================
2) Mask parameterization (JAX-friendly and stable)
=====================================================================
We learn m_low at low resolution (learn_res=128), then upsample to 512.
- m_param ∈ R^(learn_res, learn_res), unconstrained
- m = sigmoid(m_param / τ) ∈ (0,1)
- m_full = upsample(m → 512×512), then multiply by mask_ref

Why:
- TorchRay and Fong both lean on smoothness/low-res to avoid brittle pixel artifacts.  [oai_citation:5‡arXiv](https://arxiv.org/pdf/1910.08485?utm_source=chatgpt.com)
- Low-res mask is also a compute win (huge for nulls / bootstraps).

Binary masks are deferred; we keep soft masks in training for differentiability.

=====================================================================
3) Perturbation operator (must be differentiable)
=====================================================================
Use the same semantics you already defined (preserve mask):
- P(X, m) = m*X + (1-m)*B

Baseline B:
- constant WT-only spatial mean baseline (fold-safe) from Phase 2.0
- Phase 2.1 blur baseline remains useful later, but Phase 2.5a uses constant first

TorchRay uses blur perturbations frequently for “naturalistic” occlusion; we can add later.  [oai_citation:6‡Facebook Research](https://facebookresearch.github.io/TorchRay/attribution.html?utm_source=chatgpt.com)

=====================================================================
4) Objective (what we optimize for)
=====================================================================
We define a scalar “score” S(X) to optimize under perturbation.
In your setting, S can be:
- mean logit for mutant class (or logit gap mutant-WT), consistent with Phase 2.0

Two canonical games:
Deletion game (learn necessary region)
- Want deletion to hurt: maximize drop in score when ROI is deleted
- Equivalent: maximize S(X) - S(P(X, 1-m))
Preservation game (learn sufficient region)
- Want preserved ROI to keep score: maximize S(P(X, m))

Dual game (recommended; TorchRay supports variants like this conceptually)
- maximize: S(P(X, m)) - S(P(X, 1-m))
This directly matches your “inside better than outside” principle.

Regularization on m (same spirit as Phase 1)
- λ * ||m||_1  (area control)
- μ * TV(m)    (contiguity)
Optionally add:
- entropy penalty to avoid mushy masks early: η * mean(m*(1-m))

This is directly in the Fong lineage: perturbation objective + L1/TV to get compact smooth regions.  [oai_citation:7‡arXiv](https://arxiv.org/abs/1704.03296?utm_source=chatgpt.com)

=====================================================================
5) Safety rails (avoid “mask cheating” and unstable artifacts)
=====================================================================
Borrowed from the Fong concerns about masks exploiting model artifacts:
- Jitter: random small spatial shifts of the mask (or perturbation alignment) each step so mask cannot overfit exact pixel grid quirks.  [oai_citation:8‡openaccess.thecvf.com](https://openaccess.thecvf.com/content_ICCV_2017/papers/Fong_Interpretable_Explanations_of_ICCV_2017_paper.pdf?utm_source=chatgpt.com)
- Smooth mask: enforce TV and/or learn at low-res then upsample (already planned).
- Fold-safety: baseline computed only from training fold, mask trained only on training fold.
- Selection-aware inference: treat mask hyperparameters (λ,μ,τ) as selected; nulls must repeat selection.

All of these are about making the perturbation meaningful rather than adversarial.

=====================================================================
6) Training loop architecture (DRY/KISS in MorphSeq)
=====================================================================
Modules (new in Phase 2.5; Phase 2.0 code remains additive)
- roi_mask_param.py
    - MaskParam (m_param → m_low → m_full)
    - tv(m_low) with mask_ref downsampled to learn_res
    - mask jitter utility
- roi_mask_objective.py
    - compute_score(logits or logit-gap)
    - apply_perturbation(X, m_full, baseline)
    - objective variants: delete/preserve/dual
- roi_mask_trainer.py
    - train_mask_fixed_model(...)
    - optional: train_mask_and_model_joint(...), deferred

Key reuse:
- compute_logits(X, w_full, b) shared utility (Phase 1/2 already needed)
- baseline computation (roi_perturbation.py) reused

=====================================================================
7) Scaling plan (what will actually scale)
=====================================================================
We train ONE mask per “unit of analysis”:
- per genotype comparison + per feature-set + per stage-bin (if you later stage-bin)
Not per embryo, not per snip.

Compute:
- Each step requires computing logits on:
  - X_orig (maybe optional)
  - X_delete
  - X_preserve
You can avoid X_orig in dual objective.

Batching:
- minibatch over embryos/snips with group-aware sampling
- low-res mask makes gradients cheap; upsampling is cheap
- keep X on-disk chunked by N; stream batches (Zarr)

Backends:
- JAX jit for the step, run sequential batches
- GPU helps for large N, but CPU jit can still be fine; treat as engineering detail

=====================================================================
8) Testing strategy (must be done before real data)
=====================================================================
T0: Synthetic planted-ROI test (required)
- Generate X with a known ROI where only that region carries signal
- Verify learned mask recovers ROI and that dual objective improves

T1: Regression test vs Phase 2.0 occlusion
- Using Phase 1 ROI, Phase 2.0 should show Δ_delete > Δ_preserve
- Phase 2.5 learned mask should show stronger (or at least consistent) gap

T2: Stability tests
- Fixed hyperparams bootstrap (OOB) for learned mask: IoU + gap CI
- Sensitivity to quantile thresholding (if you binarize for reporting)

=====================================================================
9) Inference / Nulls for Phase 2.5 (selection-aware by design)
=====================================================================
Because we are learning m, hyperparameter selection matters even more.
Minimum required:
- Embryo-level label permutation null:
  - rerun mask training with permuted labels (or permuted score assignment)
  - record best achieved dual-gap under same training budget
This is expensive; mitigation:
- learn at 128²
- cap steps (early stopping)
- small hyperparam grid in Phase 2.5a

Bootstrap:
- OOB bootstrap fixed hyperparams:
  - fit mask on inbag
  - evaluate gap on OOB

=====================================================================
10) Phase 2.5a “Done” definition
=====================================================================
- Implement Option A: learn mask m with fixed classifier
- Dual objective + L1/TV + low-res + jitter
- Synthetic planted-ROI test passes
- On cep290 pilot: learned m produces Δ_delete > Δ_preserve with OOB bootstrap CI
- Artifacts saved (mask maps, training curves, summary JSON, plots)

=====================================================================
11) Practical staging (what to do first)
=====================================================================
Phase 2.5a (Week 1–2 equivalent)
1) Implement MaskParam + TV + jitter at learn_res=128
2) Implement dual objective with fixed (w,b), constant baseline
3) Add trainer with minibatch streaming
4) Add synthetic planted-ROI test + one real small cep290 run
5) Only then consider adding blur baseline (Phase 2.1) into objective

Deferred:
- joint learning of (w,m)
- multi-scale / pyramid schedules (TorchRay-style) unless needed
- blur/noise perturbations as defaults (keep constant first)
