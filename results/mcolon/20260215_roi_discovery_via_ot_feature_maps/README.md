# ROI Discovery via WT-Referenced OT Feature Maps

This directory contains the Phase 1 implementation of ROI (Region of Interest)
discovery using optimal transport feature maps on the 512x512 canonical embryo grid.

## Overview

The pipeline learns contiguous, interpretable spatial regions that explain
genotype discriminability (WT vs cep290) using OT-derived template-space
feature maps (cost, displacement, divergence, mass change).

**Core model:** Weight-map regularized logistic regression with L1 + TV penalty,
trained at low resolution (128x128) and upsampled to canonical 512x512.

## Directory Structure

```
20260215_roi_discovery_via_ot_feature_maps/
├── PLAN.md                          # Full implementation plan
├── README.md                        # This file
├── PHASE2_ADDENDUM_PLAN.md          # Phase 2.0 occlusion validation addendum
├── PHASE2_5_PLAN.md                 # Phase 2.5 learned-mask plan
├── roi_config.py                    # Configuration dataclasses
├── roi_feature_dataset.py           # FeatureDataset builder + validator
├── roi_loader.py                    # Streaming loader with grouped CV splits
├── roi_tv.py                        # Total Variation with mask-aware boundaries
├── roi_trainer.py                   # JAX trainer (logistic + L1 + TV)
├── roi_resampling.py                # Group-aware bootstrap/OOB helpers
├── roi_perturbation.py              # Phase 2 perturbation + fold-safe baseline
├── roi_occlusion.py                 # Phase 2 OOB occlusion evaluation
├── roi_mask_param.py                # Phase 2.5 mask parameterization utilities
├── roi_mask_objective.py            # Phase 2.5 perturbation objectives
├── roi_mask_trainer.py              # Phase 2.5 fixed-model mask trainer
├── roi_sweep.py                     # λ/μ sweep + deterministic selection
├── roi_nulls.py                     # Permutation null + bootstrap stability
├── roi_viz.py                       # Visualization (weight maps, ROI overlays, null plots)
├── roi_api.py                       # Biologist-facing API (fit / plot / report)
└── run_roi_discovery.py             # Example driver script
```

## Quick Start

```bash
# From morphseq root
cd results/mcolon/20260215_roi_discovery_via_ot_feature_maps

# 1) Build a FeatureDataset from existing OT results
python roi_feature_dataset.py --build \
    --ot-results-dir <path_to_ot_pair_outputs> \
    --out-dir roi_feature_dataset_cep290

# 2) Run ROI discovery with defaults
python run_roi_discovery.py

# 3) Or use the API from a notebook/script:
#    from roi_api import fit, plot, report
#    result = fit(genotype="cep290", features="cost")
#    plot(result.run_id)
#    report(result.run_id)
```

## Integration with Existing MorphSeq Code

- **OT backend:** Uses `src/analyze/utils/optimal_transport/` (UOTConfig, transport_maps)
- **Classification patterns:** Mirrors `src/analyze/difference_detection/` (balanced LogisticRegression, permutation testing, PermutationResult)
- **Canonical grid:** Outputs live on the 512x512 canonical grid from `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`
- **Class balancing:** Uses sklearn `class_weight='balanced'` consistent with existing classification code
- **Parquet storage:** Follows the schema patterns in `feature_compaction/storage.py`

## Dependencies

- JAX + Optax (for GPU-accelerated training)
- zarr (for chunked feature storage)
- pandas, pyarrow (for metadata/Parquet)
- scikit-learn (for class weights, AUROC)
- scipy (for spatial operations)
- matplotlib (for visualization)

## References

- See PLAN.md for the full implementation plan
- See `src/analyze/utils/optimal_transport/` for OT infrastructure
- See `src/analyze/difference_detection/` for classification/permutation patterns
- See `results/mcolon/20260121_uot-mvp/` for UOT MVP that produces input data
