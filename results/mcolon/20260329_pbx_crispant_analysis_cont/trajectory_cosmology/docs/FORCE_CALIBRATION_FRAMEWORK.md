# Force Calibration Framework

**Date**: 2026-03-30
**Status**: Design spec — ready for implementation
**Context**: Derived from repulsion auto-calibration lesson and bifurcating trunk diagnostic.

---

## The Core Lesson

The repulsion episode taught this:

> The force itself was not necessarily wrong.
> The scale it was calibrated to was wrong.

Raw numeric coefficients are not portable across datasets. The same value of `epsilon_r` that
worked on one geometry was 25× too strong on another, because the underlying spacing was different.
The fix — `epsilon_r = lambda_rep * s_local²` — made the knob dimensionless and the behaviour
portable. This principle should be applied systematically to every force term.

**Rule**: Every force must be calibrated against the natural reference scale for the geometric
feature it regulates.

---

## The Coherence Principle (Separate but Related)

Before the force table: coherence has a different job.

Coherence should be an **information signal**, not a force-strength knob.

- `C_ij(t) ∈ [0,1]` — dimensionless persistence score
- It answers: "have these two embryos been persistently close over the past δ time steps?"
- It gates attraction (who deserves to attract) — it does not set how strongly

The scale problem applies to coherence too, but on the **input** side only:

```
Current (scale-dependent):
    K(x_i, x_j, t) = exp(-||x_i - x_j||² / 2σ²)
    σ is calibrated to inter-bundle scale via sigma_frac × scale_ref

Desired (scale-independent output):
    σ_coh = coherence_scale_mult × s_local
    where s_local is fixed from initialization
    → coherence still outputs C_ij ∈ [0,1], but "close" is defined
      relative to local spacing, not global geometry
```

**Coherence knobs should be structural, not force magnitudes**:
- `delta` — temporal memory length (how many past steps count)
- `coherence_scale_mult` — dimensionless, sets what "close" means relative to `s_local`

**Not** a coherence strength multiplier — that would conflate "who deserves attraction" with
"how strongly they attract."

---

## Force Calibration Table

| Force | Regulates | Reference scale | Definition | Config knob | Derived coefficient |
|-------|-----------|-----------------|------------|-------------|---------------------|
| **Repulsion** | Local point spacing | Initial k-NN spacing `s_local` | `estimate_local_spacing_ref(x0, mask, k=5)` | `repulsion_strength_mult` (λ_rep) | `ε_r = λ_rep × s_local²` |
| **Fidelity** | Displacement from anchor | Local spacing `s_local` (tight) or global scale `s_global` (loose) | Same as repulsion ref or `radial_spread` | `fidelity_strength_mult` (λ_fid) | `μ = λ_fid / s_local²` |
| **Stretch** | Per-step trajectory motion | Typical step size in initialization | `estimate_step_scale_ref(x0, mask)` = median ‖x_i(t+1)−x_i(t)‖ | `stretch_strength_mult` (λ_str) | `λ_stretch = λ_str / s_step²` |
| **Bend** | Trajectory curvature | Typical second-difference magnitude | `estimate_bend_scale_ref(x0, mask)` = median ‖x_i(t+1)−2x_i(t)+x_i(t−1)‖ | `bend_strength_mult` (λ_bnd) | `λ_bend = λ_bnd / s_bend²` |
| **Void** | Bundle-centre spacing in domain | Global domain scale `s_global` | Grid-based: σ_void = σ_void_frac × s_global | `void_strength_mult` (λ_void) | Grid occupancy penalty weight |
| **Anisotropy** | Directional compression ratio | Intrinsically dimensionless | — | `anisotropy_ratio` (β) | β = compression⊥ / compression∥ |
| **Coherence** | Persistence of proximity | Local spacing `s_local` (input side only) | `σ_coh = coherence_scale_mult × s_local` | `coherence_scale_mult`, `delta` | σ_coh (for kernel inside C computation) |

---

## Reference Scale Estimators Needed

Currently implemented:
- `estimate_local_spacing_ref(x0, mask, k=5)` → `s_local` ✓  (`forces.py`)

Need to add to `forces.py`:

```python
def estimate_step_scale_ref(x0: np.ndarray, mask: np.ndarray) -> float:
    """Median per-step displacement ||x_i(t+1) - x_i(t)|| across all valid transitions.
    Reference scale for stretch penalty calibration."""

def estimate_bend_scale_ref(x0: np.ndarray, mask: np.ndarray) -> float:
    """Median second-difference ||x_i(t+1) - 2*x_i(t) + x_i(t-1)|| across all valid triples.
    Reference scale for bend penalty calibration."""
```

`s_global` is already computed in `temporal_sandbox.py` as `scale_ref = mean(radial_spread per slice)`.

---

## Config Pattern

Every force in `TemporalRunConfig` should expose a dimensionless multiplier:

```python
@dataclass
class TemporalRunConfig:
    # --- Dimensionless force multipliers ---
    repulsion_strength_mult:  float = 0.005   # ✓ already done
    fidelity_strength_mult:   float = 0.0     # TODO: replaces mu0
    stretch_strength_mult:    float = 0.0     # TODO: replaces lambda_stretch
    bend_strength_mult:       float = 0.0     # TODO: replaces lambda_bend
    void_strength_mult:       float = 0.0     # TODO: replaces epsilon_void
    anisotropy_ratio:         float = 1.0     # TODO: β = 1.0 = isotropic

    # --- Structural / informational knobs (not force magnitudes) ---
    delta:                    int   = 3        # coherence temporal window
    coherence_scale_mult:     float = 1.0     # TODO: σ_coh = mult × s_local
    k_attract:                int   = 20
    sigma_frac:               float = 0.5     # inter-bundle attraction bandwidth

    # --- Optimization ---
    lr:    float = 1e-4
    n_iter: int  = 300
    gamma: float = 0.999   # fidelity decay
    alpha: float = 0.9     # momentum
```

Scale derivation in `run_temporal()` becomes:

```python
s_local  = estimate_local_spacing_ref(positions, mask, k=5)
s_step   = estimate_step_scale_ref(positions, mask)
s_bend   = estimate_bend_scale_ref(positions, mask)
s_global = mean(radial_spread per slice)

epsilon_r     = config.repulsion_strength_mult * s_local**2
mu0           = config.fidelity_strength_mult  / (s_local**2 + 1e-16)
lambda_stretch = config.stretch_strength_mult  / (s_step**2  + 1e-16)
lambda_bend    = config.bend_strength_mult     / (s_bend**2  + 1e-16)
sigma_coh      = config.coherence_scale_mult   * s_local
```

---

## Current State (2026-03-30)

### What is done

| Item | Status |
|------|--------|
| Repulsion calibration (`epsilon_r = λ_rep × s_local²`) | ✓ Done, validated |
| `estimate_local_spacing_ref()` in `forces.py` | ✓ Done |
| Bifurcating trunk synthetic dataset (`make_bifurcating_trunk()`) | ✓ Done |
| Force comparison baseline (A/B/C/D conditions, shared init) | ✓ Done |
| Grid-based void term validated in `void_sandbox.py` | ✓ Done (not yet in core stack) |

### Key finding from bifurcating trunk comparison

All four currently implemented forces (isotropic, fidelity, pairwise-void-proxy, elasticity)
produce **identical results** on the Y-trunk dataset. The metrics are:

| Condition | trunk_lin_early | branch_sep_late | spread_ratio | coherence_sel |
|-----------|-----------------|-----------------|--------------|---------------|
| A: isotropic | 0.94 | 1.44 | 0.93 | 1.19 |
| B: + fidelity | 0.94 | 1.45 | 0.94 | 1.20 |
| C: pairwise void proxy | 0.94 | 1.44 | 0.93 | 1.19 |
| D: + elasticity | 0.94 | 1.44 | 0.93 | 1.19 |

Ground truth target: `branch_sep_late > 2.0`.

**Interpretation**: all currently implemented forces are isotropic in the spatial plane. They can
compact blobs and separate blobs, but they cannot distinguish "these points should form a line"
from "these points should form a blob." The Y-trunk collapses because coherence has no information
about which spatial direction to compress.

This is the gap that anisotropy fills — and the test is now sharp enough to detect it.

### What is next

In priority order:

1. **Coherence scale normalization** — compute `σ_coh = coherence_scale_mult × s_local`
   inside `compute_coherence()`. Currently `sigma` is calibrated to inter-bundle scale, which
   may make coherence too weak at within-bundle distances. Low-risk change; high interpretability
   gain.

2. **Fidelity / stretch / bend calibration** — implement `estimate_step_scale_ref()`,
   `estimate_bend_scale_ref()`, replace raw `mu0`/`lambda_stretch`/`lambda_bend` with
   dimensionless multipliers in `TemporalRunConfig`. No new forces — just scale normalization.

3. **Anisotropic force term** — the first genuinely new force. Uses local 3D covariance of
   stacked (x, y, α·t) neighborhoods to estimate trunk direction; applies stronger attraction
   perpendicular to trunk than along it. Bifurcating trunk dataset is the primary validation test.

4. **Grid void integration into core stack** — currently in `void_sandbox.py` only. Wire into
   `CondensationConfig` and `forces.py` once the design is finalized.

---

## Anisotropy Design (Sketch)

For reference when implementation begins:

```
1. For each point i, collect k-NN in stacked 3D space (x, y, α·t)
   where α converts time to spatial units (tune: α = s_local / median_step)

2. Compute 3×3 local covariance Σ_i = Σ_j w_ij (p_j - p̄_i)(p_j - p̄_i)ᵀ

3. Eigendecompose: leading eigenvector q1 = local trunk direction in 3D

4. Project to spatial plane: u_i = q1_xy / ||q1_xy||  (drop time component)

5. Build projectors:
     P∥ = u_i u_iᵀ          (along trunk)
     P⊥ = I - P∥             (across trunk)

6. Anisotropic attraction gradient for pair (i,j):
     F_ij = -w_ij [ β⊥ · P⊥ d_ij  +  β∥ · P∥ d_ij ]
   where β⊥ > β∥ (stronger pull across trunk)

   Dimensionless knob: anisotropy_ratio = β⊥ / β∥
   β∥ = 1 = isotropic base; β⊥ = anisotropy_ratio

7. Time coordinate is NEVER updated — only xy moves
```

Expected bifurcating trunk result with anisotropy working:
- Early t: trunk compresses across its axis → tighter, more linear
- Late t: two distinct branch strands emerge
- `branch_sep_late` should exceed 2.0
- `trunk_linearity_early` should increase toward 1.0
