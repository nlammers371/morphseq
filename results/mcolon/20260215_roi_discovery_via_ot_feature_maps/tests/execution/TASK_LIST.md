# Simple Task List — ROI Discovery Test Execution

## Phase 0
- [ ] 0.1 OT map generation
- [ ] 0.2 Outlier detection + QC gate
- [ ] 0.3 S coordinate construction
- [ ] 0.4 S-bin feature aggregation
- [ ] 0.5 AUROC localization by bin
- [ ] 0.6 Contiguous interval search
- [ ] 0.7 Selection-aware nulls + stability
- [ ] 0.8 Phase 0 review + handoff

## Phase 1
- [ ] 1.1 Config construction + validation
- [ ] 1.2 FeatureDataset build + validate
- [ ] 1.3 Loader + grouped CV splits
- [ ] 1.4 TV edge construction + computation
- [ ] 1.5 Trainer convergence + logging
- [ ] 1.6 ROI extraction + tail concentration inspection
- [ ] 1.7 Lambda/mu sweep + deterministic selection
- [ ] 1.8 Permutation null (selection-aware)
- [ ] 1.9 Bootstrap stability (fixed lambda/mu)
- [ ] 1.10 API integration smoke test

## Phase 2.0
- [ ] 2.1 Perturbation + train-fold-safe baseline
- [ ] 2.2 Occlusion evaluation metrics
- [ ] 2.3 OOB bootstrap occlusion + threshold sensitivity
- [ ] 2.4 Group-aware resampling helpers
- [ ] 2.5 Integration fixes from addendum

## Phase 2.5
- [ ] 2.5.1 Mask parameterization primitives
- [ ] 2.5.2 Dual objective behavior
- [ ] 2.5.3 Fixed-model mask trainer + planted ROI recovery

## Cross-phase inspection checkpoints
- [ ] Tail-localization sanity check logged for every ROI/mask result.
- [ ] Selection-aware inference used whenever hyperparameters are selected.
- [ ] OOB/holdout-only reporting used for inferential metrics.
