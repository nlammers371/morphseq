# PBX Additivity Decomposition — Methods

## Overview

This analysis validates whether the pbx1b/pbx4 single-gene axes form a
biologically meaningful and specific coordinate frame in VAE latent space,
before using that frame to interpret the pbx1b×pbx4 double mutant.

All calculations are performed within time bins (4 hpf width), using
per-embryo mean VAE latent vectors (`z_mu_b*` columns).

---

## Data and notation

**Input:** VAE latent space embeddings from three experiments:
`20251207_pbx`, `20260304`, `20260306`.

**Genotypes:** `wik_ab` (non-injected), `inj_ctrl`, `pbx1b_crispant`,
`pbx4_crispant`, `pbx1b_pbx4_crispant`.

**Time bins:** `floor(stage_hpf / 4) × 4`, width = 4 hpf.
Bin center plotted = `bin_left + 2` hpf.

**Unit of analysis:** embryo-level mean within each time bin, in latent space.

**Within each valid time bin** (≥3 embryos per required genotype):

```
mu_non  = mean(wik_ab latent vectors)
mu_1    = mean(pbx1b_crispant latent vectors)
mu_4    = mean(pbx4_crispant latent vectors)
mu_inj  = mean(inj_ctrl latent vectors)

v_1 = mu_1   − mu_non     (pbx1b single-gene effect vector)
v_4 = mu_4   − mu_non     (pbx4  single-gene effect vector)

V = [v_1 | v_4]           (design matrix, n_features × 2)
```

**OLS decomposition** of any target vector `z`:

```
z ≈ V·c + r
c = argmin ||z − V·c||²   (via numpy.linalg.lstsq)
[alpha, beta] = c
R²_span = 1 − ||r||² / ||z||²
```

`alpha` = loading on the pbx1b axis.
`beta`  = loading on the pbx4 axis.
`R²_span` = fraction of target variance explained by the pbx1b/pbx4 span.

---

## Q1 — Control-span symmetry  (`04_q1_control_span_symmetry.py`)

**Question:** When a target contains no real pbx signal, does the
decomposition project it symmetrically (not biased toward one axis)?

**Bootstrap (500 reps per bin):**

1. Randomly split `wik_ab` → `non_A` / `non_B`.
2. Build span from full-group means (fixed per bin):
   `V = [mu_pbx1b − mu_non_A | mu_pbx4 − mu_non_A]`
3. Fit **three targets** in the same span:
   - `z_ctrl  = mu_non_B − mu_non_A`   ← control noise (what we're testing)
   - `z_pbx1b = mu_pbx1b − mu_non_A`  ← real pbx1b signal (reference upper bound)
   - `z_pbx4  = mu_pbx4  − mu_non_A`  ← real pbx4 signal (reference upper bound)

**Expected outcome:**

| Target   | alpha  | beta   | R²_span |
|----------|--------|--------|---------|
| z_ctrl   | ≈ 0    | ≈ 0    | low     |
| z_pbx1b  | ≈ 1    | ≈ 0    | ≈ 1     |
| z_pbx4   | ≈ 0    | ≈ 1    | ≈ 1     |

**Observed result (good):** Control noise (gray) stays near alpha ≈ 0 across all time bins.
The pbx1b reference signal loads primarily onto the pbx1b axis (alpha ≈ ±1, near zero beta),
confirming the decomposition cleanly separates real signal from noise. Both pbx phenotypes
are subtle — so this clean separation despite low effect size is an encouraging validation
that the span is not projecting noise onto the biological axes.

**Figures:** `figures/q1_control_span_symmetry/`
- `q1_alpha_control_vs_real.png` — alpha over time, 3 targets
- `q1_beta_control_vs_real.png`  — beta over time, 3 targets
- `q1_r2_control_vs_real.png`   — R²_span over time, 3 targets

---

## Q2 — Axis reality  (`05_q2_axis_reality.py`)

**Question:** Are the pbx1b/pbx4 axes structurally stronger than
axes built from control-only sampling noise?

**Q2A — Real axes (no bootstrap, one value per bin):**

```
||v_1||,  ||v_4||                          (effect magnitudes)
cos(v_1, v_4) = (v_1·v_4) / (||v_1|| ||v_4||)   (directional overlap)
kappa(V) = sigma_max(V) / sigma_min(V)     (condition number; lower = more separable)
```

**Q2B — Fake-axis null (500 reps per bin):**

1. Split `inj_ctrl` → `inj_A` / `inj_B`.
2. Split `wik_ab`   → `non_A` / `non_B`.
3. Build two fake axes from control splits:
   `v_fake1 = mu_inj_A − mu_non_A`
   `v_fake2 = mu_inj_B − mu_non_B`
4. Compute same three metrics for `[v_fake1 | v_fake2]`.

The fake-axis distribution is the noise floor.

**Expected outcome:**

- `||v_1||`, `||v_4||` above fake-axis null → real signal is larger than noise.
- `cos(v_1, v_4)` below fake-axis null → real axes more independent than noise.
- `kappa(V)` below fake-axis null → real decomposition more stable than noise.

**Figures:** `figures/q2_axis_reality/`
- `q2_norm_comparison.png`   — axis norms over time (real vs fake null ribbon)
- `q2_cosine_comparison.png` — cosine similarity over time
- `q2_cond_comparison.png`   — condition number over time

---

## Q3 — Span specificity  (`06_q3_span_specificity.py`)

**Question:** Does the real span reconstruct held-out single-gene targets
specifically, and reject control-derived targets?

**Bootstrap (500 reps per bin):**

1. Split each genotype randomly into A / B halves:
   `pbx1b_A / pbx1b_B`, `pbx4_A / pbx4_B`, `non_A / non_B`, `inj_A / inj_B`.

2. Build **real span** from A halves:
   `V_real = [mu_pbx1b_A − mu_non_A | mu_pbx4_A − mu_non_A]`

3. Build **fake span** from injection control (matched baseline):
   `V_fake = [mu_inj_A − mu_non_A | mu_inj_B − mu_non_B]`

4. Fit three **held-out B-half targets** in **both** spans:
   - `z_pbx1b = mu_pbx1b_B − mu_non_B`
   - `z_pbx4  = mu_pbx4_B  − mu_non_B`
   - `z_inj   = mu_inj_B   − mu_non_B`

5. Record `R²_real`, `R²_fake`, and `ΔR² = R²_real − R²_fake` per target.

**Expected outcome:**

| Target         | R²_real | R²_fake | ΔR²      |
|----------------|---------|---------|----------|
| pbx1b_heldout  | high    | low     | positive |
| pbx4_heldout   | high    | low     | positive |
| inj_ctrl       | low     | low     | ≈ 0      |

**Figures:** `figures/q3_span_specificity/`
- `q3_r2_real_vs_fake.png`   — R²_span in real (solid) vs fake (dashed) span per target
- `q3_delta_r2.png`          — ΔR²_span over time per target
- `q3_alpha_real_span.png`   — alpha in real span per target
- `q3_beta_real_span.png`    — beta in real span per target

---

## Interpretation priority

1. **Q2 norm**: do real axes exceed the control noise floor?
2. **Q3 ΔR²**: does the real span specifically reconstruct held-out single-gene targets?
3. **Q1 symmetry**: does the machinery project control noise symmetrically?
4. **Q2 cosine / condition number**: annotation for how stable coefficient attribution is.

If Q2 norms are marginal and Q2 cosine is high (near 1), the two pbx axes
are nearly collinear — coefficient attribution (alpha vs beta) becomes
unreliable even if R² is high.
