# Force Internals

This document collects the solver-side force terms in one place so the main algorithm note can stay readable. The package-level docs are still small enough that a separate `docs/` directory would be unnecessary overhead.

## Reading Order

- [condensation/forces/attraction.py](condensation/forces/attraction.py)
- [condensation/forces/elasticity.py](condensation/forces/elasticity.py)
- [condensation/forces/fidelity.py](condensation/forces/fidelity.py)
- [condensation/forces/local_scale.py](condensation/forces/local_scale.py)
- [condensation/forces/repulsion.py](condensation/forces/repulsion.py)
- [condensation/forces/slice_outlier.py](condensation/forces/slice_outlier.py)
- [condensation/forces/void.py](condensation/forces/void.py)
- [condensation/forces/anisotropy.py](condensation/forces/anisotropy.py)
- [condensation/forces/total.py](condensation/forces/total.py)

## Force Model

The solver is easiest to think about as a gated energy system:

- temporal coherence builds a persistence field over embryos and time
- attraction uses that field to decide which pairs should pull together
- elasticity keeps each embryo trajectory smooth over time
- fidelity anchors the run to the initialization and then decays
- repulsion, void, local-scale, and outlier terms correct failure modes

The key detail is that attraction is not unconditional. It is gated by coherence, so spatial pulling is stronger for embryo pairs that have persistently co-traveled in time.

## Scale Determination

Most force scales are not hand-tuned in raw coordinate units. They are resolved from geometry calibration in `condensation/engine/run.py` using the layout-dependent references built in `condensation/geometry_refs.py`.

- `sigma_att` is derived from the calibrated attraction scale, optionally multiplied by `attract_bandwidth_mult`
- `sigma_coh` is derived from the calibrated coherence scale, optionally multiplied by `temporal_cohere_bandwidth_mult`
- `lambda_stretch` and `lambda_bend` are resolved from `elastic_strength` plus `elastic_mix` against the step and bend references
- `epsilon_r` comes from the calibrated repulsion scale
- `void_bandwidth` defaults to the attraction scale when not set explicitly
- `fidelity_init_strength` becomes a decaying `mu` through `fidelity_half_life`
- local-scale and outlier terms use their own geometry references from the initialization layout

The intent is that one strength knob means approximately the same thing across datasets after calibration, even though the raw coordinate units may differ.

### Core Terms

- `attraction.py`: coherence-weighted pairwise attraction, optionally restricted to a symmetric kNN graph per time slice.
- `elasticity.py`: stretch and bend regularization over each embryo's time chain.
- `fidelity.py`: decaying penalty for moving away from `x0`.
- `local_scale.py`: preserves the mean distance to fixed initial local neighbors.
- `repulsion.py`: short-range exclusion that prevents collapse.
- `slice_outlier.py`: suppresses slice-relative detachments from the baseline manifold.
- `void.py`: broad Gaussian density-field repulsion that separates bundles globally.
- `anisotropy.py`: placeholder for directional shaping terms; currently a stub.
- `total.py`: assembles all active terms into one energy and gradient.

## API Nomenclature

Public-facing configuration should be read as strength-based knobs:

- attraction strength
- temporal coherence strength
- elasticity strength
- fidelity strength
- local-scale strength
- outlier strength
- void strength

In the current code, some fields still carry legacy `*_weight` names for compatibility. The documentation should treat those as strength parameters conceptually, not as a separate naming scheme.

Internal solver assembly also uses calibration parameters and resolved coefficients:

- `sigma` / `sigma_coh`: attraction and coherence length scales
- `epsilon_r`: repulsion core scale
- `lambda_stretch` / `lambda_bend`: resolved elasticity coefficients
- `mu`: decaying fidelity coefficient

## Strength vs Mult

Use this rule:

- `strength` controls amplitude
- `mult` controls geometric distance or bandwidth

Concrete example:

- `attract_bandwidth_mult = 2.0` means the attraction kernel reaches twice the calibrated reference distance
- `void_strength = 2.0` means the void term contributes twice as much energy and gradient as the same term at `1.0`

Those are different knobs. The first changes the spatial footprint of the force; the second changes the force magnitude at that footprint.

## Equation Sketch

For a single solver state, the total objective can be written schematically as

```text
E_total = E_attr + E_rep + E_void + E_elastic + E_fidelity + E_scale + E_outlier
```

where

```text
E_attr = - sum_t sum_{i<j} C_ijt * K_ijt * G_ijt
```

and:

- `C_ijt` is temporal coherence between embryos `i` and `j` at time `t`
- `K_ijt` is the spatial Gaussian kernel
- `G_ijt` is the optional kNN gate

The remaining terms act as regularizers or failure-mode corrections rather than the primary organizing signal.

## Term Notes

### Attraction

The attraction kernel uses the current positions within each time slice and multiplies them by coherence. That means spatial pulling is only strong when coherence says the pair has a temporal relationship worth preserving.

### Elasticity

Elasticity is a per-trajectory regularizer. The stretch term penalizes large step-to-step jumps; the bend term penalizes curvature through second differences.

### Fidelity

Fidelity anchors optimization to the initial layout. The anchor decays with iteration, so the initialization matters most early in the solve.

### Local Scale

Local-scale preservation keeps dense and sparse neighborhoods from being treated identically. It compares current neighborhood radii against fixed radii measured from the initialization.

### Repulsion and Void

Repulsion is short-range and prevents point collapse. Void is broader and acts more like a density-field pressure that helps separate bundles at the population scale.

### Slice Outlier

Slice outlier handling is slice-relative, not global. It catches observations that drift far from the baseline manifold for their time slice even if their trajectory stays smooth.

### Anisotropy

Anisotropy is intentionally not a core organizing force yet. The file exists as a placeholder for directional regularization once the required behavior is better specified.

## Solver Interface

The solver consumes the resolved force parameters and aggregates all force contributions in `total.py`. That is the narrow point where the configuration, geometry calibration, and force modules meet.
