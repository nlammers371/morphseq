# Force Calibration Summary

**Solver calibration**: `lr=1e-4`, `n_iter=500`
**Benchmark**: well-initialized Y-trunk (bifurcating_trunk_sandbox.py)
**Key note**: all thresholds measured on a WELL-INITIALIZED geometry. On poorly-initialized
or real data, forces activate at LOWER multiplier values — defaults are intentionally set
just below the no-change threshold so they remain gentle under good initialization but
provide structure-preserving pressure when the embedding needs help.

See figures:
- `force_sweeps/force_sweeps.png` — 2×3 sweep grid, all families
- `solver_tempo_selection.png` — lr × n_iter selection rationale
- `regime_gifs/` — 3D rotating GIFs of baseline / no_change / moderate / extreme per family

---

## Solver Tempo

| Parameter | Chosen value | Rationale |
|-----------|-------------|-----------|
| `lr`      | `1e-4`      | Step-size safe (disp_rms_step1 ≪ 5× min). lr=1e-3 is destructive; lr=1e-5 under-integrates even at 800 iters |
| `n_iter`  | `500`       | Metrics plateau by 300–400 iters at lr=1e-4. 500 gives margin and distinguishability for weak forces |

Goal: a slow, stable solver regime where weak forces have "bandwidth" to interact and accumulate
measurable effect before convergence cuts them off.

---

## Force 1: Repulsion (`repulsion_strength_mult`)

**Role**: Prevents particle collapse. The isotropic repulsion spreads the embedding into space and
maintains local neighborhood structure. This is always ON — it is the baseline force.

**Default**: `0.005`

**Thresholds** (well-initialized Y-benchmark):
| Regime | Value | Effect |
|--------|-------|--------|
| no_change | ≤ 0.0071 | Δbranch_sep < 2% vs baseline |
| moderate  | ~0.0205 | +3.3% branch_sep, geometry intact, mild dilation |
| extreme   | ~0.172  | +29% within-branch spread, Y visibly distorted |

**Good range**: 0.002–0.01 (keeps local structure, does not inflate)
**Bad** (too low): < 0.001 → particles collapse toward coherence attractors
**Bad** (too high): > 0.05 → over-inflation, branches bloat, separation misleading

---

## Force 2: Fidelity (`fidelity_strength_mult`)

**Role**: Anchors each trajectory point to its initial position (decaying over time via
`fidelity_half_life_iters`). Prevents drift from global structure. Useful under poor
initialization to resist spurious rearrangement. With fast decay (small half_life_iters),
acts mainly at early time steps.

**Default**: `fidelity_strength_mult=0.25`, `fidelity_half_life_iters=70`

**Thresholds** (well-initialized):
| Regime | Value | Effect |
|--------|-------|--------|
| no_change | ≤ 0.334 | Δbranch_sep < 0.1% |
| moderate  | ~0.334  | +3.7% sep, slight anchoring visible |
| extreme   | 5.0     | Convergence triggered at iter ~71; frozen at init |

**Good range** (strength_mult): 0.1–0.5 with half_life_iters=50–100
**Bad** (too high strength_mult): > 2.0 → over-anchored, dynamics freeze
**Bad** (half_life_iters too small): < 5 → anchor vanishes before solver equilibrates
**Bad** (half_life_iters too large / None): → perpetual anchoring, blocks late adaptation

---

## Force 3: Fidelity Decay (`fidelity_half_life_iters`)

**Role**: Controls how quickly the fidelity anchor decays. γ = 2^(-1/h) where h is the number
of solver iterations for the anchor weight to halve. Large h = slow decay (persistent anchor).
Small h = fast decay (anchor active only at start).

**Default**: `70` (≈ old gamma=0.99)

**Thresholds**: no discrete no_change threshold — continuous parameter.

**Good range**: 50–200 for gentle anchoring; 5–20 for "early pressure only"
**Bad** (too small, < 3): anchor halved in 3 iters — effectively no anchoring
**Bad** (None / very large): permanent anchor — embedding cannot evolve

---

## Force 4: Void Proxy (`epsilon_void`)

**Role**: Pairwise Gaussian void force — repels particles that are too close in time but
far in embedding space. Acts as a soft "personal space" preventing temporal overlaps.
Controlled jointly with `sigma_void_frac` (spatial width of the Gaussian).

**Default**: `epsilon_void=0.014`, `sigma_void_frac=5.0`

**Thresholds** (well-initialized):
| Regime | Value | Effect |
|--------|-------|--------|
| no_change | ≤ 0.018 | Δ < 0.1% |
| moderate  | ~0.0178 | −1% within-branch spread (mild temporal compression) |
| extreme   | 0.1     | −5.8% spread, bundles compressed, temporally close points pushed apart |

**Good range**: 0.005–0.02 with sigma_void_frac=5.0
**Bad** (too high): > 0.05 → strong temporal compression competes with coherence
**Bad** (sigma_void_frac too small): void activates only at very close distances → nearly inert

---

## Force 5: Elasticity Stretch (`stretch_strength_mult`)

**Role**: Resists changes in inter-particle distances (spring-like). Penalizes stretching or
compression of the local neighborhood. On poorly-initialized embeddings this helps maintain
locally coherent structure.

**Default**: `0.04`

**Thresholds** (well-initialized):
| Regime | Value | Effect |
|--------|-------|--------|
| no_change | ≤ 0.049 | Δ < 0.1% |
| moderate  | ~0.0488 | sep=1.392, lin=0.942 — trajectories slightly tighter |
| extreme   | 2.0     | sep=1.033, lin=0.994 — branches near-line, Y-shape destroyed |

**Good range**: 0.01–0.05
**Bad** (too high): > 0.2 → branch separation collapses (elasticity overcomes coherence)
**Bad** (too low, < 0.005): effectively inert even on poor initialization

---

## Force 6: Elasticity Bend (`bend_strength_mult`)

**Role**: Resists bending / angular changes in trajectories. Encourages smooth curvature.
Acts on the angle between successive displacement vectors. Useful for preventing sharp kinks
under poor initialization.

**Default**: `0.04`

**Thresholds** (well-initialized):
| Regime | Value | Effect |
|--------|-------|--------|
| no_change | ≤ 0.049 | Δ < 0.1% |
| moderate  | ~0.0488 | sep=1.400, lin=0.940 — marginally smoother |
| extreme   | 2.0     | sep=1.280, lin=0.982 — trunk over-straightened, branching reduced |

**Good range**: 0.01–0.05 (matched to stretch for balanced elasticity)
**Bad** (too high): > 0.2 → excessive trunk straightening, branch diversity suppressed
**Bad** (too low, < 0.005): effectively inert on well-initialized data

---

## Summary Table

| Force | Default | no_change (well-init) | Moderate | Extreme |
|-------|---------|----------------------|----------|---------|
| repulsion_strength_mult | 0.005 | 0.0071 | 0.0205 | 0.172 |
| fidelity_strength_mult  | 0.25  | 0.334  | 0.334  | 5.0   |
| fidelity_half_life_iters| 70    | —      | 20–50  | <5    |
| epsilon_void            | 0.014 | 0.018  | 0.0178 | 0.1   |
| stretch_strength_mult   | 0.04  | 0.049  | 0.0488 | 2.0   |
| bend_strength_mult      | 0.04  | 0.049  | 0.0488 | 2.0   |

**All defaults are intentionally just below the no_change threshold on the well-initialized
benchmark.** Under poor initialization (real data), forces will activate at these default
values — providing gentle structure-preserving pressure without overpowering the dynamics.
