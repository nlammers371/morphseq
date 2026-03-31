# Hyperparameter Tuning Notes
## Trajectory Cosmology — PBX Multiclass Condensation

These notes document the empirical progression toward a well-balanced force schedule.
The goal is to inform a future automatic tuning strategy.

---

## Diagnostic checklist (what to measure, not just energy totals)

Energy magnitude alone is insufficient. The quantities that actually govern motion are:
- **gradient scale per term** (not raw energy)
- **disp_rms_rel trajectory**: is it monotonically decreasing, oscillating, or plateaued?
- **energy component fractions over time**: which terms dominate at each phase?
- **first-step displacement**: abs(x1 - x0) relative to spatial_scale_ref — must be << 1

---

## Run 1: lr=0.01, gamma=0.97, lambda_stretch=0.1, lambda_bend=0.05, n=100

**Symptom**: disp_rms_rel starts at 524, ends at 4.98. System clearly not converged.
**Root cause**: First step displacement = ~1.9× reference scale (catastrophic).
- scale_ref = 0.25, repulsion gradient per embryo ≈ 48× scale
- lr=0.01 × grad ≈ 0.48 per step → step/scale_ref ≈ 1.9
- The system explodes on iter 1 (energy jumps 278k → 60M, mostly elastic+fidelity)
- Remaining 99 iterations spent recovering from this explosion

**Lesson**: lr must be tuned so that first-step disp_rms_rel << 1. A reasonable target is <0.1.
**Lesson**: The initial repulsion configuration (tight PCA cloud) can dominate. Large initial repulsion = high-magnitude first gradient.

---

## Run 2: lr=0.001, gamma=0.97, lambda_stretch=0.1, lambda_bend=0.05, n=200

**Symptom**: disp_rms_rel ends at 0.33. Better, but still not converged.
**New visible problem**: Fidelity decays from mu=1.0 to mu≈0.002 by iter 200 (gamma=0.97^200).
- By iter 200: elastic=95% of total energy, fidelity=5%, attraction=0.01%
- The system is being driven toward minimum-stretch/bend config, not toward biologically meaningful grouping

**What is solid**:
- gamma=0.97 kills fidelity too fast — confirmed
- By iter 200, mu is effectively gone
- Without fidelity anchoring, system drifts toward elastically smooth (but biologically arbitrary) config

**What is not yet proven**:
- "lambda_stretch too large" — energy dominance ≠ gradient dominance; need gradient-scale comparison to confirm
- The "structural" framing may overstate certainty; this is a second problem exposed by fixing the first

**Lesson**: gamma must be much closer to 1.0. At gamma=0.97, half-life of mu is ~23 iters (ln(0.5)/ln(0.97)=22.8).
**Lesson**: For a 500-iter run, gamma=0.999 gives mu≈0.61 at iter 500 — still substantial.
**Heuristic**: Choose gamma so that mu > 0.5 for at least the first N/2 iterations, where N is total iters.
  - gamma = 1 - ln(2)/half_life_target
  - For half_life = N/2: gamma = exp(-ln(2) / (N/2)) ≈ 1 - ln(2)/N for large N

---

## Planned Run 3: lr=0.001, gamma=0.999, lambda_stretch=0.01, n=500

Rationale:
- lr unchanged: 10× reduction from run 1 was directionally correct
- gamma=0.999: mu=0.61 at iter 500, 0.37 at iter 1000 — slow enough to keep fidelity active
- lambda_stretch=0.01 (10× reduction): if elastic still dominates with gamma=0.999, this is the next lever
- lambda_bend unchanged at 0.05 (don't tune two things at once unless forced)

Expected diagnostics to check:
- First-step disp_rms_rel should be ~5 (since lr=0.001 was 10× smaller, and run 2 started at 47 → expect ~5)
- Energy fractions mid-run: fidelity should still be substantial (>20%) at iter 250
- disp_rms_rel should monotonically decrease, not oscillate

---

## Force balance intuition (for future auto-tuning)

A well-balanced system should have all four forces contributing comparably to gradient magnitude:
- Attraction: gates on coherence; only matters when embryos are nearby (sigma bandwidth)
- Repulsion: strong only at short range; primary role is preventing collapse at early times
- Elasticity: proportional to trajectory roughness; enforces smooth tracks
- Fidelity: anchor; decays as annealing proceeds; keeps system near initial embedding

Force balance at steady state (late iterations):
- Repulsion + attraction ≈ balanced (set the spatial arrangement)
- Elastic gradient ≈ fidelity gradient (set how much the trajectory can deviate)

Auto-tuning targets:
1. Ensure first-step disp_rms_rel < 0.1 → tune lr given sigma, epsilon_r, scale_ref
2. Ensure mu > 0.5 for first half of run → derive gamma from n_iter
3. Check energy fractions at iter N/2: each term should contribute > 5% of total (if a term is <5%, it's inactive)
4. Check energy fractions at iter N: attraction should be growing relative to elastic

---

## Run 3: lr=0.001, gamma=0.999, lambda_stretch=0.01, n=500

**Result**: Two distinct phases observed.

**Phase 1 (iters 0–183): normal convergence**
- Energy falls from 278k → 3777 monotonically
- Attraction growing (-10.7k), disp_rms_rel decreasing to ~0.26
- System appears to be working correctly during this phase

**Phase 2 (iters 183+): attraction overcollapse**
- At iter 183: sudden jump in disp_rms from 0.26 → 1.16
- Attraction magnitude reaches -11k, repulsion +4k, elastic starts growing explosively
- Embryos are pulled so close together that repulsion cannot stabilize them

**Root cause**: sigma=0.5 is 2× scale_ref=0.25. Attraction-repulsion equilibrium analysis:
- Balance point: attraction = repulsion only at d≈0.10 (and d≈1.68)
- Between d=0.10 and d≈1.5, attraction dominates with no stabilizing repulsion
- epsilon_r=0.01 is too small by ~15× to hold back the attraction force at the sigma scale
- Once pairs get within d<sigma, the whole cloud collapses inward together

**Conclusion**: sigma must be set relative to scale_ref, not as an absolute value.

---

## Sigma and epsilon_r scaling rules (derived from balance analysis)

For attraction-repulsion balance to hold near the desired cluster spacing d_target:
  - At d=sigma, K(sigma)=exp(-0.5)≈0.607
  - epsilon_r needed to prevent collapse: epsilon_r ≈ 0.607 × sigma^2

| sigma (× scale_ref) | epsilon_r needed | d range of attraction |
|--------------------|------------------|-----------------------|
| 0.20×              | 0.002            | 0–0.43 × scale_ref    |
| 0.40×              | 0.006            | 0–0.86 × scale_ref    |
| 0.60×              | 0.014            | 0–1.29 × scale_ref    |
| 1.0× (= scale_ref) | 0.038            | 0–2.15 × scale_ref    |
| 2.0× (run 3)       | 0.152            | 0–4.3 × scale_ref     |

**Practical rule**: set sigma = 0.3–0.5 × scale_ref, and epsilon_r ≈ 0.6 × sigma^2.
This ensures attraction only activates for nearby pairs (<1× scale_ref), leaving room for biological separation.

**Key insight for auto-tuning**: sigma must be computed from scale_ref at runtime, not set as a fixed constant.
Proposed default: sigma = 0.3 × scale_ref, epsilon_r = 0.6 × sigma^2 = 0.054 × scale_ref^2.

---

## Run 4: sigma=0.3×scale_ref=0.075, epsilon_r=0.0034, lr=0.001, gamma=0.999, lambda_stretch=0.01, n=500

**Result**: No collapse (good), but attraction is completely negligible.
- disp_rms_rel=0.004 at iter 500 — excellent convergence on displacement
- Energy fractions at convergence: repulsion=52%, fidelity=45%, elastic=2.5%, **attraction=0.015%**
- Repulsion/attraction ratio: 3500× — attraction effectively turned off

**Root cause**: sigma=0.075 is too narrow for the PCA point cloud geometry.
- Median pairwise distance = 0.207 (0.83× scale_ref)
- K(median dist) = exp(-0.207² / 2×0.075²) = 0.023 — kernel activates for only 2% of the median pair
- 39.7% of pairs have K>0.1, but **coherence C_ij(t) also near-zero in early iterations** when positions are scattered, so no coherence signal ever builds up
- The system finds a fidelity+repulsion minimum (good shape) but never clusters

**Lesson**: sigma must be large enough to see neighbors at the *initial* pairwise spacing, not just the final one.
- At sigma=0.3×scale_ref: K(median)=0.023 → attraction inactive
- At sigma=0.5×scale_ref: K(median)=0.256 → meaningful activation
- At sigma=1.0×scale_ref (run 3): K(median)=0.588 → over-activation, collapse

**Lesson**: The coherence-gating makes attraction self-suppressing at early iterations. C_ij(t) depends on spatial kernel, so if sigma is too small, C≈0 → no attraction → no coherence growth. The signal never bootstraps.

**Target range**: sigma = 0.4–0.6× scale_ref, with epsilon_r ≈ 0.6×sigma² to set equilibrium spacing at ~sigma.
At sigma=0.5×scale_ref=0.125: K(median)=0.256, d_eq=0.49×scale_ref — pairs equilibrate within the cloud.

---

## Planned Run 5: sigma=0.5×scale_ref=0.125, epsilon_r=0.009, lr=0.001, gamma=0.999, lambda_stretch=0.01, n=500

---

## Open questions

- Should lr adapt during training (line search or schedule)?
- Should sigma scale automatically with scale_ref? Yes — propose sigma=0.3×scale_ref as default.
- Does coherence C_ij(t) activate meaningfully? With sigma=0.5>>scale_ref, coherence could be saturated (all pairs get C≈1.0) from the start, which defeats the persistence-gating purpose.
  - With sigma=0.075 (0.3×scale_ref), coherence is much more selective — only nearby pairs accumulate C>0.1.
  - This is probably the intended behavior: coherence should discriminate between co-traveling and diverging pairs.

---

## Summary table (full trajectory runs)

| Run | lr | gamma | λ_stretch | n_iter | final disp_rms_rel | diagnosis |
|-----|----|-------|-----------|--------|---------------------|-----------|
| 1 | 0.01 | 0.97 | 0.1 | 100 | 4.98 | First-step explosion (step/scale=1.9) |
| 2 | 0.001 | 0.97 | 0.1 | 200 | 0.33 | Fidelity decayed too fast; elastic dominates late |
| 3 | 0.001 | 0.999 | 0.01 | 500 | pending | — |

---

## Slice Sandbox Experiments (2D kNN Force Tuning)

Starting 2026-03-30, we built `slice_sandbox.py` — a clean synthetic 2D slice sandbox to diagnose the force law before adding temporal coupling. Key findings:

### Motivation for the sandbox

The full trajectory model showed universal collapse. Rather than blindly sweeping full 5D configurations, the sandbox isolates a single 2D slice with synthetic 2-cluster data to ask:

> Can this force law (kNN-local attraction + repulsion) preserve cluster separation on simple geometry?

This is a more fundamental question than parameter tuning — it asks whether the mechanism is *capable* of working at all.

---

## Sandbox Experiment 1: Initial repulsion balance sweep

**Setup**:
- Synthetic 2-cluster dataset (separated variant: N=120, two well-separated Gaussians)
- Single 2D slice (T=1)
- 300 iterations, lr=5e-4 (default), momentum α=0.9
- Varied: k ∈ {5,10,20}, eps_mult ∈ {0.005, 0.01, 0.02, 0.05, 0.1}
- All combos with subtract_mean ∈ {False, True} × coherence ∈ {uniform, oracle}

**Key result**: The system shifts from collapse (prev rep 1) to *explosion* — but this is **progress**.

| eps_mult | k=20 final sep_ratio | collapse_score | sep_ratio_best | iter_best |
|----------|---------------------|----------------|-----------------|----------|
| 0.005 | **6.97** | 0.96 | 7.29 | iter 78 |
| 0.010 | 5.41 | 0.99 | 5.49 | iter 237 |
| 0.020 | 2.48 | 1.21 | 3.38 | iter 85 |
| 0.050 | 1.95 | 1.44 | 3.37 | iter 46 |
| 0.100 | 0.15 | 13.4 | 2.07 | iter 0 |

**Interpretation**:
- `eps_mult=0.005` (epsilon_r=0.0000515): Nearly perfect separation preservation! sep_ratio improves from 4.72 → 6.97. collapse_score ≈ 0.96 means cloud stays compact.
- `eps_mult=0.01`: Still good, but sep_ratio_best @ iter 237 suggests the system slowly merges.
- `eps_mult≥0.02`: Attraction turns into global collapse (sep_ratio falls fast, best at early iters).

**Critical insight**: The old heuristic `epsilon_r = 0.6 × sigma²` was *too strong*. At sigma=0.5×scale, we need `epsilon_r ≈ 0.005–0.01 × scale²`, not 0.6.

**Metric that saved us**: `sep_ratio_best` (best at iter 78) vs `sep_ratio_final` (at iter 300). The best structure appears *mid-run*, not at convergence. This is exactly analogous to diffusion-condensation: the optimal scale is not always the endpoint.

---

## Sandbox Experiment 2: Learning rate sensitivity (hold repulsion fixed)

**Setup**: Same as above, but fixed `eps_mults = {0.005, 0.01}`, varied `lr ∈ {1e-4, 5e-4 [default]}`.

**Result at lr=1e-4 (slower integrator)**:

| eps_mult | sep_ratio_final | collapse_score | sep_ratio_best | iter_best |
|----------|-----------------|----------------|-----------------|----------|
| 0.005 | **7.13** | 0.96 | 7.13 | iter 299 |
| 0.010 | 6.30 | 0.97 | 6.30 | iter 299 |

**Key change**: `sep_ratio_best` shifts from **iter 78 → iter 299** (the final iteration). The slower integrator doesn't overshoot; it stabilizes at the good config.

**Interpretation**:
- At lr=5e-4, the system reaches the good separation at iter ~80, then oscillates and degrades (sep_ratio_best ≠ sep_ratio_final).
- At lr=1e-4, the system reaches the same good config more slowly and *holds it*. No degradation.
- This suggests the slower lr avoids numerical overshoot in the balanced regime.

**Lesson**: For this force law class, a **slower integrator is more stable** than faster. The "sweet spot" force balance is narrow — the system can overshoot into the collapsed regime if lr is too large.

---

## Sandbox Experiment 3: Coherence mode comparison (oracle vs. uniform)

Both `oracle` (only same-label pairs attract) and `uniform` (all pairs attract) produce nearly identical results. This tells us:

**The force balance problem is primary, not the label information.**

Even with oracle coherence (the theoretical upper bound on label-aware separation), the system still oscillates and collapses if eps_mult is large enough. This confirms that the issue is not "I need better label information" but rather "the global all-pairs attraction is too strong."

**Important**: kNN restriction (k=20) + oracle coherence means at most k=20 label-aware neighbors attract each point. This is *much* weaker than the old all-pairs baseline.

---

## Sandbox key findings summary

1. **Repulsion is critical but was set too strong** (old heuristic ε_r = 0.6σ² is ~100× too large for this sigma scale).
   - True heuristic: ε_r ≈ 0.005–0.01 × scale_ref² when σ=0.5 × scale_ref

2. **Learning rate must be small** (lr=1e-4 is safer than lr=5e-4 for stable trajectory).
   - At larger lr, even the good force balance overshoots and oscillates.

3. **Best separation often appears mid-run** (iter 78 vs final 300), not at convergence.
   - Diffusion-condensation intuition: there is an optimal iteration number for each tradeoff between tightening and overcompression.

4. **collapse_score ≈ 0.96–0.99** is the signature of a working force law (cloud stays compact, structure preserved).
   - collapse_score > 1.2 → expansion/explosion
   - collapse_score < 0.8 → contraction/collapse

5. **subtract_mean has negligible effect** on 2D slices (all-pairs, oracle, uniform gave same results). Not a primary lever.

6. **k-nearest-neighbor restriction is essential** (k=20 with N=120 is 1/6 of all pairs, enough to preserve global separation while preventing global collapse).

---

## Revised defaults for the 2D sandbox

```python
sigma_frac = 0.5          # × scale_ref
eps_mult = 0.005          # × (0.6 × sigma²), i.e., total ε_r = 0.0015 × scale_ref²
k_attract = 20            # local neighborhood size
lr = 1e-4                 # conservative step size
alpha = 0.9               # standard momentum
n_iter = 300              # enough to see mid-run structure
```

These settings reliably preserve sep_ratio ≈ 7 (improvement from 4.7) with collapse_score near 0.96.

---

## Sandbox Experiment 4: Robustness across synthetic variants

**Setup**: Same optimal params (k=20, eps_mult={0.005, 0.01}, lr=1e-4) on all four synthetic variants.

**Results**:

| Variant | init sep_ratio | final sep_ratio (eps=0.005) | collapse_score | interpretation |
|---------|-----------------|---------------------------|-----------------|-----------------|
| separated | 4.72 | **7.13** | 0.96 | Clusters separate further, cloud stays compact ✓ |
| moderate | 1.97 | **2.55** | 0.92 | Modest improvement; harder case ✓ |
| overlapping | 0.75 | **0.99** | 0.80 | Slightly improved, cloud contracts (expected for overlap) ✓ |
| elongated | 3.63 | **4.52** | 0.96 | Preserves anisotropic shape ✓ |

**Interpretation**:
- **All four variants show sep_ratio improvement or preservation** — the force law is not specific to one geometry.
- **collapse_score ≈ 0.96 across variants** — the system doesn't implode or explode.
- **Overlapping variant shows smaller collapse_score (0.80)** — the cloud contracts in the overlap case, which makes sense (it's harder to maintain large spread when clusters overlap).
- **Elongated variant sep_ratio improves** (3.63 → 4.52) — the anisotropy is not destroyed.

**Conclusion**: The kNN + low-repulsion force law is **stable and geometry-agnostic**. It works across easy (separated), hard (overlapping), and structured (elongated) synthetic problems.

---

## Summary: What we learned from the sandbox

### The old problem (pre-sandbox)
- All-pairs attraction dominated; repulsion only emergency brake
- Global collapse to single blob across all parameters
- Unclear which lever to turn

### The new understanding (post-sandbox)
1. **Collapse was not a parameter-tuning problem — it was a topology problem.**
   - All-pairs interaction is intrinsically prone to collapse when repulsion is weak.
   - The system needs a fundamentally different interaction structure.

2. **kNN local attraction is the solution.**
   - Restricts attraction to k-nearest neighbors (k≈20 for N≈100).
   - Each point only pulls on nearby coherent partners, not the global centroid.
   - Preservation of cluster separation follows naturally.

3. **The repulsion strength must be re-tuned for the new topology.**
   - Old heuristic: ε_r = 0.6 × σ² (derived for all-pairs).
   - New heuristic: ε_r ≈ 0.005–0.01 × scale_ref² (100× weaker!) with σ = 0.5 × scale_ref.
   - This reflects that kNN-local attraction is much weaker than all-pairs attraction.

4. **Learning rate matters more than force balance.**
   - Even with correct forces, overshooting (lr too large) can still collapse.
   - lr = 1e-4 is safer than lr = 5e-4.
   - The "optimal scale" appears mid-run (iter ~80–240), not at convergence (iter 300).

5. **The force law is remarkably robust to initial geometry.**
   - Works across separated, moderate, overlapping, and elongated clusters.
   - Oracle vs. uniform coherence makes little difference (primary issue was topology, not labels).

---

## Recommended next action

**Do NOT tune any more parameters in the sandbox.** The mechanism has been validated. Now:

1. **Apply to full trajectory**: Use k=20, eps_mult=0.005, lr=1e-4 in `run_condensation_experiment.py` on real PBX multiclass data.
2. **Expect**: Energy terms to balance, structure preservation visible in trajectory plots, no numerical issues.
3. **If success**: Move to Phase 2 (add temporal elasticity and coherence decay).
4. **If issues**: Debug on the full model with metrics (not back to 2D sandbox).

The sandbox's job is done: it proved the force law class works on simple geometry. Now we scale to the full trajectory problem.
 Ready to code?                                                       
                                                                      
 Here is Claude's plan:                                               
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Fix Collapse Pathology in Trajectory Cosmology Force Model     
                                                                      
 Context                                                              
                                                        
 After 5 runs of hyperparameter tuning and a 48-combination slice
 diagnostic sweep, the
 trajectory cosmology condensation produces universal collapse to a
 single blob for every
 parameter combination. Even with oracle coherence (C=1 only for
 same-genotype pairs),
 all structure is destroyed.

 The root cause: the current attraction is dense, coherent, and
 all-to-all. Each embryo
 feels O(N_group) attraction pulling toward the group centroid.
 Repulsion is short-range and
 only activates once collapse is already underway — there is no
 stable intermediate morphology.
 The system has a broad pull inward with only a tiny emergency brake
 at close distances.

 Primary fix: kNN-local attraction — the most principled change. It
 alters the interaction
 topology itself, so attraction only acts on nearby coherent
 neighbors. This is both more
 physically correct (filaments are local structures) and biologically
  sensible.

 Optional stabilizer: mean-field subtraction — removes the global
 center-of-mass drift mode
 from the attraction gradient. Useful as a practical control knob,
 but more of a "control the
 system" move than a "derive the right physics" move. Default off;
 enable deliberately.

 Stretch goal (deferred): Fix A (anisotropic/tidal attraction via
 local PCA).

 ---
 Phase 1: kNN-local attraction (primary fix)

 What changes mathematically

 Currently attraction sums over all j ≠ i. Replace with a kNN mask:

 For each time t and observed embryo i:
 1. Compute pairwise distances d_ij = ||x_i - x_j||
 2. Find k nearest neighbors: N_k(i) = argsort(d_ij)[:k]
 3. Build binary mask M_knn[i,j] = 1 if j ∈ N_k(i)
 4. Symmetrize: M_knn = max(M_knn, M_knn.T)  (either-neighbor rule)
 5. Gate: W[i,j] = K(x_i, x_j) · C_ij(t) · M_knn[i,j]

 Energy becomes:
 E_attract = -Σ_t Σ_{i,j: j ∈ N_k(i)} K_s(x_i, x_j) · C_ij(t)

 Gradient is unchanged in form, just restricted to kNN set.

 Files to change

 condensation/state.py — add to CondensationConfig:
 k_attract: int | None = 15
 (None = all-pairs for backward compatibility. Default 15 for
 N~100-300.)

 condensation/forces.py — modify attraction():
 - Add k_attract: int | None = None parameter
 - Refactor inner loop: compute diff and sq_dist once (currently diff
  is at line 49,
 gaussian_kernel recomputes sq_dist internally)
 - Before applying coherence, build kNN mask from sq_dist:
 if k_attract is not None and k_attract < n_obs - 1:
     sq_for_knn = sq_dist.copy()
     np.fill_diagonal(sq_for_knn, np.inf)
     knn_idx = np.argpartition(sq_for_knn, k_attract, axis=1)[:,
 :k_attract]
     knn_mask = np.zeros((n_obs, n_obs))
     knn_mask[np.arange(n_obs)[:, None], knn_idx] = 1.0
     knn_mask = np.maximum(knn_mask, knn_mask.T)
 else:
     knn_mask = 1.0  # scalar broadcasts
 W = K * C * knn_mask
 - Modify total_energy_and_grad() to accept and pass k_attract

 condensation/dynamics.py — pass config.k_attract through.

 slice_diagnostic.py — add k_attract parameter to run_slice() and
 attraction_grad(),
 sweep k_attract_values = [5, 10, 20, None].

 Verify

 Run slice_diagnostic.py with kNN alone (no mean subtraction).
 Expected:
 - Local neighborhoods condense (attraction works on nearby pairs)
 - Global cloud structure at least partially preserved (each embryo
 only pulls on ~15 neighbors)
 - sep_after closer to sep_before than the all-pairs baseline

 ---
 Phase 2: Mean-field subtraction (optional stabilizer)

 What changes mathematically

 After computing the attraction gradient for all observed embryos at
 time t, subtract the
 per-slice mean:

 g_i_centered = g_i - mean_i(g_i)

 This removes the center-of-mass drift mode from the attraction
 gradient. It is a gradient-level
 modification (not energy-level), making the system more explicitly
 non-variational — but the
 system is already non-variational due to coherence re-estimation.

 This is a stabilization device, not a deep physical analog. Default
 off (False).

 Files to change

 Same files as Phase 1 (already wired). The subtract_mean parameter
 is added in Phase 1 but
 defaulted to False. To test, flip to True in the slice diagnostic
 sweep.

 Verify

 Run slice diagnostic with 3 conditions:
 1. all-pairs + no mean subtraction (baseline — should still
 collapse)
 2. kNN + no mean subtraction (Phase 1 — primary fix)
 3. kNN + mean subtraction (Phase 1 + 2 — optional stabilizer)

 This tells us whether kNN alone cures most of the disease.

 ---
 Phase 3: New diagnostic metric — per-slice radial spread

 Track per-slice second moment (cloud spread):

 R²_t = (1/n_t) Σ_i ||x_i(t) - x̄(t)||

 Average over time slices. This directly quantifies whether the slice
  cloud is shrinking to
 dust vs. preserving structure. Add to slice_diagnostic.py summary
 output for easy comparison.

 ---
 Phase 4: Validate on full pipeline

 Run run_condensation_experiment.py with new config (k_attract=15,
 possibly
 subtract_mean_attraction=True if Phase 2 showed benefit). Check:
 - Energy attraction term is nonzero and growing
 - disp_rms_rel converges to < 0.01
 - No explosion or collapse
 - Trajectory plot shows biologically interpretable structure

 ---
 Deferred: Anisotropic attraction (Fix A)

 Only if kNN + optional mean subtraction still fail to produce
 ridge-like structure.
 Compute local covariance of kNN neighbor positions, eigendecompose
 2×2, project attraction
 gradient mostly along dominant eigenvector.

 ---
 Testing strategy

 Regression: Run test_finite_difference_gradient with
 subtract_mean=False, k_attract=None
 to confirm the raw gradient math is unchanged.

 kNN behavior: New test — 2 well-separated clusters of 10 points
 each. With k_attract=5,
 verify attraction gradient is zero between clusters (no
 cross-cluster pull).

 Mean subtraction: New test — verify that after mean subtraction, the
  per-slice sum of
 attraction gradients is zero (to machine precision).

 Radial spread: New test — run slice diagnostic toy case, verify R²
 does not collapse to
 near-zero with kNN enabled.

 Slice diagnostic sweep (integration):
 - Stage 1: kNN only (k_attract=10,15,20, subtract_mean=False)
 - Stage 2: kNN + mean subtraction (subtract_mean=True)
 - Stage 3: compare all against baseline (k_attract=None,
 subtract_mean=False)

 ---
 Critical files

 File: trajectory_cosmology/condensation/state.py
 Action: Add `k_attract: int
 ────────────────────────────────────────
 File: trajectory_cosmology/condensation/forces.py
 Action: kNN mask + optional mean subtraction in attraction(),
   pass-through in total_energy_and_grad()
 ────────────────────────────────────────
 File: trajectory_cosmology/condensation/dynamics.py
 Action: Pass new config fields to total_energy_and_grad()
 ────────────────────────────────────────
 File: slice_diagnostic.py
 Action: Add new params to sweep grid, add radial spread metric,
   update attraction_grad() and run_slice()
 ────────────────────────────────────────
 File: test_trajectory_cosmology_smoke.py
 Action: Regression test with old params, new tests for kNN and mean
   subtraction behavior

 All files under results/mcolon/20260329_pbx_crispant_analysis_cont/.
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
---

## Temporal Sandbox Experiments (multi-slice force law, 2026-03-30)

### Context: Scale mismatch diagnosis (decompaction pathology)

After moving from the 2D slice sandbox to the full temporal sandbox (8 time slices, N=30 per cluster, 3 clusters), we observed a new pathology: **decompaction** — already-compact bundles inflate and develop outliers after running the force law.

Root cause (diagnosed):

- `sigma` (and thus `epsilon_r`) were calibrated to the **inter-bundle scale** (mean per-slice radial spread ≈ 3.3 units, scale_ref ≈ 1.65 units)
- Within-bundle spacing is ~0.14 units — **10–25× smaller than sigma**
- Attraction kernel `K = exp(-r²/2σ²)` is nearly flat inside compact bundles (K ≈ 0.98–1.0 for all within-bundle pairs) → attraction gradient is near zero inside bundles
- Repulsion `ε_r / (r² + η)` still has a meaningful gradient at r=0.14, so it continues pushing within-bundle points apart
- Result: bundles inflate to a scale set by repulsion balance, not initial compactness

**Key insight**: `eps_mult = 0.005` was validated in the 2D sandbox where `sigma ≈ scale_ref ≈ 1`. In the temporal sandbox, `sigma = 1.65` but within-bundle spacing = 0.14 — `epsilon_r` is calibrated to inter-bundle scale but acts at within-bundle scale. It is **~25× too strong** for within-bundle geometry.

---

### Temporal Experiment 1: Repulsion sweep

**Conditions tested** (8 total, on crossing_bundles and stable_bundles synthetic datasets, 300 iters):

| Condition | eps_mult | r_cut | Description |
|-----------|----------|-------|-------------|
| softcore_1x | 0.005 | 0 | Baseline (old 2D-sandbox default) |
| softcore_0.5x | 0.0025 | 0 | Half repulsion |
| softcore_0.25x | 0.00125 | 0 | Quarter repulsion |
| **softcore_0.1x** | **0.0005** | 0 | **10× reduction — sweet spot** |
| bump_rcut_0.10 | 0.005 | 0.10×median_5nn | Truncated bump, very small exclusion zone |
| bump_rcut_0.25 | 0.005 | 0.25×median_5nn | Truncated bump, small exclusion zone |
| bump_rcut_0.50 | 0.005 | 0.50×median_5nn | Truncated bump, moderate exclusion zone |
| bump_rcut_0.25_half | 0.0025 | 0.25×median_5nn | Bump + halved strength |

**Key results** (crossing_bundles, 300 iter):

| Condition | within_bundle_spread_ratio | sep_ratio_mean | local_radius_ratio_p95 |
|-----------|---------------------------|----------------|------------------------|
| softcore_1x | ~3–5× (decompaction) | ~5 | ~2.0 |
| **softcore_0.1x** | **≈1.0** | **~20** | **≈1.19** |
| bump conditions (50 iter) | ≈0.984 | — | — |
| bump conditions (300 iter) | collapsed | 71–79 | 0.10 |

**Winner**: `softcore_0.1x` (eps_mult=0.0005).

**Bump repulsion failure mode**: At 50 iterations, bump conditions looked good. By 300 iterations, they **collapsed** (sep_ratio jumped to 71–79, local_radius_median=0.10 = 10% of initial size). Root cause: when r_cut << healthy bundle spacing, attraction dominates with no balancing repulsion → bundles compress to the exclusion zone radius over many iterations.

**Lesson**: Bump repulsion is a **collision-prevention brake only**, not a long-range stabilizer. The `r_cut` must be set smaller than healthy bundle spacing (exclusion zone only), but this means it provides no equilibrium restoring force at the healthy bundle scale. For stable equilibrium, gradually-decaying soft-core repulsion tuned to the correct strength is more appropriate.

**Practical rule for temporal multi-slice setting**:
- `eps_mult = 0.0005` (not 0.005 as in the 2D sandbox — 10× weaker)
- This reflects that `sigma ≈ inter-bundle scale >> within-bundle scale` in the temporal case
- The 2D sandbox validated eps_mult=0.005 where sigma ≈ scale_ref ≈ 1; in temporal, sigma/within_scale ≈ 10×, so epsilon must scale down proportionally

---

### New metric: `local_radius_ratio_p95`

A **label-free** compactness metric that extrapolates from sandbox to real data (where bundle labels are unknown):

```
local_radius_ratio_p95 = 95th percentile of (current_median_kNN_radius / initial_median_kNN_radius)
```

- Value 1.0 = perfect compactness preservation
- Value > 1.2 = bundles have inflated meaningfully
- Value < 0.8 = compression toward collapse
- **Does not require ground-truth cluster labels** — usable on real embryo data

---

### Temporal Experiment 2: Local sigma (sigma_attract_local) — FAILED

Tested `sigma_local_frac=0.5` → `sigma_attract_local ≈ 0.075` (half of median_5nn ≈ 0.14).

**Result**: Made decompaction worse (local_radius_ratio_p95 jumped from 2.0 to 3.4).

Root cause: at sigma_attract_local=0.075, K_local ≈ 0 for nearly all pairs. Attraction gradient collapsed to near zero everywhere → effectively repulsion-only → bundles inflated more than baseline. A local sigma must be comparable to within-bundle spacing, not half of it, to provide a meaningful restoring force.

**Status**: `sigma_attract_local` parameter is wired in `forces.py` and `state.py` but not recommended as the primary fix. Kept for future experimentation with sigma fractions ≥ 1.0× median_5nn.

---

## Local Neighborhood Scale Preservation (2026-03-30)

### Design rationale

The validated fix (`eps_mult=0.0005`) is a **global parameter** — it weakens repulsion everywhere. A more principled approach anchors local density to the initialization:

```
E_scale = λ_scale × Σ_i (r_i^(n) - r_i^(0))²
```

Where:
- `r_i^(0)` = mean distance to k fixed initial neighbors (computed once from x0, **never updated**)
- `r_i^(n)` = current mean distance to the same fixed neighbor indices
- `λ_scale` = soft regularizer weight

**Key design choices**:
1. Anchored to **initial positions**, not current state — prevents drift from initialization while allowing mesoscopic rearrangement
2. **Soft regularizer, not hard constraint** — lambda starts small (0.1–1.0), not a rigid leash
3. Neighbor indices are **fixed** — determined from x0, same throughout the run

### Three-scale architecture (design decision)

Three forces serve three distinct roles at three spatial scales:

| Force | Parameter | Scale | Job |
|-------|-----------|-------|-----|
| Attraction | sigma, k_attract | Inter-bundle | Clusters co-traveling embryos across time |
| Short-range exclusion | epsilon_r, (r_cut) | Sub-bundle | Collision avoidance only |
| Local density preservation | lambda_scale, k_local_scale | Within-bundle | Anchors compactness to initialization |

This replaces the two-force design where repulsion had to simultaneously prevent collapse AND prevent decompaction — conflicting requirements at different spatial scales.

### Cautions (from design review)

- **Fixed neighbor identity can become too rigid**: if optimization changes topology substantially, some initial neighbors may be inappropriate. Start small and observe.
- **Log separately**: `energy_scale`, `scale_residual_mean`, `scale_residual_p95` tracked per iteration.
- **Do not replace eps_mult tuning**: scale preservation supplements the validated eps_mult=0.0005, it doesn't replace it.

### Implementation summary

- `build_neighborhood_info(x0, mask, k_local=5)` — called once before the loop; returns fixed neighbor indices and `r0` per time slice
- `local_scale_preservation(positions, mask, neighborhood_info, lambda_scale)` — energy + gradient
- New `CondensationConfig` fields: `lambda_scale=0.0` (off by default), `k_local_scale=5`
- `dynamics.py`: `neighborhood_info` precomputed before the loop, passed to `total_energy_and_grad` at every iteration

### Planned scale sweep

| Condition | lambda_scale | eps_mult |
|-----------|-------------|----------|
| baseline | 0.0 | 0.0005 |
| small_lambda | 0.1 | 0.0005 |
| moderate_lambda | 1.0 | 0.0005 |

Primary metric: `within_bundle_spread_ratio` ≤ 1.1. Secondary: `local_radius_ratio_p95`, `sep_ratio_mean`, `scale_residual_p95`.

---

## Void Term Design (2026-03-30)

### Job description

The void term has a specific, bounded job:

**Input**: compact bundles crowded unevenly within a bounded domain  
**Output**: bundle centers redistributed more evenly through that domain  
**Constraint**: local bundle density unchanged

It is not a second repulsion term. It acts at **bundle-centroid scale** (`sigma_void >> sigma`), not at point-to-point spacing. The kernel bandwidth must span multiple bundles — otherwise it degenerates into generic pairwise repulsion and is redundant with the existing term.

### Why a fixed domain is required

Without a fixed reference domain, "spread out more evenly" has no meaning. The system would simply drift outward indefinitely. Two anchoring choices:
- **Sandbox**: fixed `[-5,5]×[-5,5]` square — clean, unambiguous, used in `void_sandbox.py`
- **Real data**: padded initial bounding box per slice — revisit when porting to trajectory model

Confinement is sandbox scaffolding only (`lambda_conf` parameter). It is NOT proposed as a permanent force in the trajectory model — that decision is deferred until after the void term is validated.

### Implementation (Option A — first to test)

Broad Gaussian density-field repulsion:

```
rho_i = sum_{j≠i} exp(-||x_i - x_j||^2 / 2 sigma_void^2)
E_void = epsilon_void * sum_i rho_i
grad_i = epsilon_void * sum_j (x_i - x_j)/sigma_void^2 * exp(-r^2/2 sigma_void^2)
```

Parameters:
- `epsilon_void`: strength (default 0 = off)  
- `sigma_void_frac`: `sigma_void = sigma_void_frac × scale_ref`, default 3.0

Option B (grid-based occupancy equalization) is second-generation — only if Option A shows promise but drifts too much or requires careful tuning.

### Force architecture: three scales

| Force | Parameter | Spatial scale | Job |
|-------|-----------|---------------|-----|
| Attraction | sigma, k_attract | Inter-bundle | Who travels together |
| Repulsion | repulsion_strength_mult, s_local | Within-bundle | Collision avoidance |
| Void | epsilon_void, sigma_void | Multi-bundle | Global spacing equalization |

### Validation tests (`void_sandbox.py`)

Four synthetic cases, fixed `[-5,5]×[-5,5]` domain:

| Test | Setup | Success criterion |
|------|-------|-------------------|
| `crowded_one_side` | 4 bundles, left half only | `centroid_dist_improvement > 1`, `spread_ratio ≈ 1` |
| `well_spaced` | 4 bundles, already uniform | Nothing changes much |
| `mixed_sizes` | 3 bundles N=10/30/60 | Centers spread, size differences preserved |
| `line_crowded` | 4 bundles along line, too close | Spacing increases, topology preserved |

**Primary metrics**:
- Local (must stay near 1.0): `within_bundle_spread_ratio`, `local_radius_ratio_p95`
- Global (should improve for crowded cases): `mean_centroid_dist`, `centroid_spacing_cv`
- Stability: `collapse_score`, `domain_escape_frac`

### What to look for in sweep results

A void term is working correctly if:
1. `crowded_one_side`: `centroid_dist_improvement > 1` and `centroid_spacing_cv` decreases
2. `well_spaced`: both metrics barely change (< 5% deviation from initial)
3. All cases: `within_bundle_spread_ratio ≈ 1.0`, `local_radius_ratio_p95 ≈ 1.0` (local damage < 10%)
4. `domain_escape_frac ≈ 0` (confinement holding)

A void term is failing if it improves global spacing at the cost of inflating bundles (`spread_ratio >> 1`) or if it does nothing for `crowded_one_side`.
