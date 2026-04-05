# Force Separation Benchmark Note

## Outcome

The bifurcating-trunk benchmark was used to test whether attraction amplitude and temporal-coherence amplitude should remain explicitly separated in the PBX condensation solver. The answer from the current benchmark runs is no: keeping those amplitudes separate gave less stable structures than the original multiplicative attraction-by-coherence interaction.

What we learned:
- temporal coherence mainly smooths trajectories across time
- canonical attraction mainly drives aggregation / bundle tightening
- the strongest and most robust behavior still comes from multiplying those effects through the original attraction kernel gated by coherence
- increasing the two amplitudes separately or even in matched `10/10` and `50/50` runs did not recover the older behavior as robustly as the original multiplicative formulation

## Files To Inspect

Primary evidence folders:
- `results/bifurcating_trunk_refactored_scales_v3/`
- `results/bifurcating_trunk_refactored_weights_v2/`
- `results/bifurcating_trunk_refactored_weights_v3_extreme/`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/`

Most useful outputs:
- `results/bifurcating_trunk_refactored_scales_v3/refactored_force_sweeps.png`
- `results/bifurcating_trunk_refactored_scales_v3/refactored_force_sweeps_summary.csv`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/combined_weight_summary.csv`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/combined_weights_10_10/3d_before_after.gif`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/combined_weights_50_50/3d_before_after.gif`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/combined_weights_10_10/trunk_3d_before_after.png`
- `results/bifurcating_trunk_refactored_weight_synergy_v1/combined_weights_50_50/trunk_3d_before_after.png`

## Decision

Keep the public naming cleanup in the condensation config, especially:
- `attract_*`
- `temporal_cohere_*`
- `solver_*`

Do not keep the separated temporal-coherence amplitude behavior in the solver. Revert to the original multiplicative attraction × coherence behavior for production runs.
