# Trajectory Cosmology: Condensation of Developmental Phenotype Trajectories
## Mathematical Specification and Implementation Plan

---

## 1. Problem and Context

### 1.1 Biological Setting

We have a set of perturbation conditions (PBX crispants + controls):

- `inj_ctrl` — injection control
- `wik_ab` — wildtype
- `pbx4_crispant`
- `pbx1b_crispant`
- `pbx1b_pbx4_crispant` (double)

Each condition contains multiple embryos tracked across developmental time (~21–79 hpf).
Each embryo has a feature vector at each observed time bin (embedding latents from the VAE).

We want to infer and visualize the **time-varying phenotypic relationship structure** among conditions — not snapshots, but an evolving relational geometry that reveals divergence timing, reconvergence, and within-condition heterogeneity.

### 1.2 Input: Multiclass Probability Vectors

The primary input is a **\( G \)-dimensional softmax probability vector** per embryo-timebin,
where \( G = 5 \) is the number of conditions, from a single multinomial (one-vs-all or
softmax) classifier trained across all conditions simultaneously.

\[
\mathbf{v}^{(i,t)} = \bigl(p_1^{(i,t)}, \dots, p_G^{(i,t)}\bigr) \in \Delta^{G-1}
\]

where \( \sum_g p_g = 1 \) and \( p_g^{(i,t)} \) is the probability that embryo \( i \)
at time \( t \) belongs to condition \( g \).

This is the feature space in which trajectories live. It is more compact and geometrically
cleaner than the 10D pairwise space, while being implicitly equivalent: the pairwise
log-odds ratio between any two conditions \( i, j \) is recoverable as
\( \log(p_i / p_j) \), so no pairwise information is lost.

**Alternative:** the 10D pairwise signed-margin vectors from `phenotypic_positioning_phase2`
(`raw_position_vectors.csv`) encode the same relational structure and could be substituted
if multiclass model fits are unavailable or noisier at sparse time bins.

### 1.3 Relation to the Dynamic Graph Idea

The `trajectory_relationship_graph_idea.md` posed the core challenge:

> How do I turn a time series of pairwise condition-condition relationships into a global, temporally coherent structure?

This document answers it with a concrete formalism. The key insight is:

- **Condition-level graph**: the \( N \times N \) similarity matrix derived from the
  multiclass vectors at each time → edge weights between condition nodes
- **Embryo-level trajectories**: each embryo is a curve in probability simplex space;
  condensation reveals which embryos co-travel, independent of assigned genotype labels
- The two levels are complementary, not competing

---

## 2. Data Contracts

### 2.1 Primary Input

A single table with one row per embryo-timebin:

```
embryo_id | time_bin_center | genotype | p_inj_ctrl | p_wik_ab | p_pbx4_crispant | p_pbx1b_crispant | p_pbx1b_pbx4_crispant
```

- Shape: \( \sim 2500 \) rows × \( (3 + G) \) columns, \( G = 5 \)
- Each \( p_g \in [0,1] \), rows sum to 1
- Missing embryo-timebin observations encoded by absence of row (not NaN fill)

### 2.2 AUROC Tensor (condition-level graph input)

`pairwise_auroc_bins_pbx_controls_embedding_all_pairs.csv`:

```
feature_set | comparison_id | time_bin_center | auroc_obs | pair_id | ...
```

This gives \( D_{ij}(t) \) — the condition-level pairwise distinguishability tensor,
used for the Level 1 condition graph.

---

## 3. Two-Level Architecture

### Level 1: Condition Graph (coarse)

At each time \( t \), build an \( N \times N \) graph:

\[
G_t = (V, E_t), \quad V = \{\text{inj\_ctrl, wik\_ab, pbx4, pbx1b, pbx1b\_pbx4}\}
\]

Edge weight between conditions \( i, j \) at time \( t \):

\[
w_{ij}(t) = 1 - \text{AUROC}_{ij}(t)
\]

High weight = conditions are similar (not distinguishable). Low weight = diverged.

This directly answers the open question in `trajectory_relationship_graph_idea.md` (§A):
edge weights are the direct one-vs-one similarity transform.

**Temporal coherence of the graph layout** is enforced by warm-starting the spring
layout positions from the previous timestep.

### Level 2: Embryo Trajectory Condensation (fine)

Each embryo is a curve in \( \Delta^{G-1} \subset \mathbb{R}^G \) probability simplex
space. We embed and then *condense* these curves to reveal which embryos co-travel,
independent of their assigned genotype label.

This is the cosmological condensation formalism defined below.

---

## 4. Cosmological Condensation: Mathematical Specification

### 4.1 Setup

Let:
- \( N_e \): number of embryos
- \( T \): number of time bins
- \( \mathbf{x}_0^{(i,t)} \in \mathbb{R}^2 \): initial positions from AlignedUMAP applied
  to the multiclass vectors \( \mathbf{v}^{(i,t)} \)
- \( m^{(i,t)} \in \{0,1\} \): observation mask (embryo \( i \) present at time \( t \))

We optimize over 2D positions \( \mathbf{x}^{(i,t)} \), with time indices fixed.

### 4.2 Energy Functional

\[
E = E_{\text{attract}} + E_{\text{repel}} + E_{\text{elastic}} + E_{\text{fidelity}}
\]

---

### 4.3 Attraction (persistence-gated)

Embryos attract only if they have been **consistently co-located** over a backward window:

\[
E_{\text{attract}} = -\sum_{t} \sum_{i \neq j}
K_s\!\left(\mathbf{x}^{(i,t)}, \mathbf{x}^{(j,t)}\right) \cdot C_{ij}(t)
\]

**Spatial kernel:**
\[
K_s(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right)
\]

**Temporal coherence (causal backward window):**
\[
C_{ij}(t) =
\frac{
\displaystyle\sum_{\tau=t-\delta}^{t}
K_s\!\left(\mathbf{x}^{(i,\tau)}, \mathbf{x}^{(j,\tau)}\right) \cdot m^{(i,\tau)} m^{(j,\tau)}
}{
\displaystyle\sum_{\tau=t-\delta}^{t} m^{(i,\tau)} m^{(j,\tau)} + \epsilon_{\text{mask}}
}
\]

- \( \delta \): window size (default: 3 time bins)
- \( \epsilon_{\text{mask}} \): small constant to avoid divide-by-zero
- Causal: only past and present are used; future proximity does not influence current attraction

**Biological interpretation:** two embryos form a persistent bundle only if they have traveled
together, not just been nearby at a single snapshot.

---

### 4.4 Repulsion (soft-core)

Prevents collapse; maintains separation within genotype clouds:

\[
E_{\text{repel}} =
\sum_{t} \sum_{i \neq j}
\frac{\epsilon_r}{\|\mathbf{x}^{(i,t)} - \mathbf{x}^{(j,t)}\|^2 + \eta}
\]

- \( \epsilon_r \): repulsion strength
- \( \eta > 0 \): numerical stabilizer

Can optionally be suppressed *between embryos of the same genotype* to allow
tighter intra-genotype bundles (a hyperparameter).

---

### 4.5 Elasticity

Each embryo is a physically continuous trajectory. Two penalties:

**Stretch (step size):**
\[
E_{\text{stretch}} =
\lambda_1 \sum_i \sum_{t}
m^{(i,t)} m^{(i,t+1)} \|\mathbf{x}^{(i,t+1)} - \mathbf{x}^{(i,t)}\|^2
\]

**Bending (curvature):**
\[
E_{\text{bend}} =
\lambda_2 \sum_i \sum_{t}
m^{(i,t-1)} m^{(i,t)} m^{(i,t+1)}
\|\mathbf{x}^{(i,t+1)} - 2\mathbf{x}^{(i,t)} + \mathbf{x}^{(i,t-1)}\|^2
\]

- \( \lambda_1 \): step smoothness (prevents teleportation)
- \( \lambda_2 \): curvature smoothness (prevents abrupt turns)
- Observation masks are included to handle embryos entering/leaving the dataset

**Biological tuning:** high \( \lambda_2 \) → gradual phenotype transitions enforced;
low \( \lambda_2 \) → allows rapid divergence events (relevant for the ~55 hpf transition
seen in PBX data).

---

### 4.6 Fidelity (annealed anchor)

Anchors positions to AlignedUMAP initialization during early optimization:

\[
E_{\text{fidelity}} =
\mu(n) \sum_i \sum_t
m^{(i,t)} \|\mathbf{x}^{(i,t)} - \mathbf{x}_0^{(i,t)}\|^2
\]

\[
\mu(n) = \mu_0 \cdot \gamma^n, \quad \gamma \in (0.95, 0.99)
\]

- \( n \): iteration number
- Starts high to stabilize; decays to allow structure formation
- Final structure is data-driven, not anchored to UMAP geometry

---

### 4.7 Dynamical Interpretation

Because \( C_{ij}(t) \) depends on current positions, this is **not a static optimization**.
It is a self-consistent dynamical system:

- trajectories that co-move → increased coherence → stronger attraction → tighter bundle
- this is filament formation, analogous to large-scale structure in cosmology

---

## 5. Optimization

### 5.1 Damped gradient dynamics

\[
\mathbf{v}^{(n+1)} \leftarrow \alpha \mathbf{v}^{(n)} - \eta_{\text{lr}} \nabla_{\mathbf{x}} E^{(n)}
\]
\[
\mathbf{x}^{(n+1)} \leftarrow \mathbf{x}^{(n)} + \mathbf{v}^{(n+1)}
\]

- \( \alpha \in (0,1) \): momentum damping
- \( \eta_{\text{lr}} \): learning rate
- All positions updated simultaneously (not sequentially)

### 5.2 Gradient computation

The gradients are tractable analytically:

- \( \nabla E_{\text{attract}} \): involves \( \nabla K_s \) (Gaussian: linear in displacement)
- \( \nabla E_{\text{repel}} \): inverse-square gradient
- \( \nabla E_{\text{elastic}} \): second differences (banded tridiagonal structure per embryo)
- \( \nabla E_{\text{fidelity}} \): linear residual

For \( N_e \sim 500 \) embryos and \( T \sim 30 \) bins, the pairwise terms are \( O(N_e^2 T) \).
This is feasible with vectorized NumPy. If \( N_e > 1000 \), consider approximate repulsion
(tree-based or random sampling).

### 5.3 Hyperparameter defaults (starting point)

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Spatial kernel bandwidth | \( \sigma \) | 0.5 (in UMAP units) | Tune to typical inter-embryo distance |
| Temporal window | \( \delta \) | 3 bins (~6 hpf) | Matches AlignedUMAP alignment window |
| Repulsion strength | \( \epsilon_r \) | 0.01 | Scale with \( \sigma^2 \) |
| Repulsion softener | \( \eta \) | 1e-4 | Numerical stability |
| Stretch weight | \( \lambda_1 \) | 0.1 | Start low; increase if trajectories jump |
| Bending weight | \( \lambda_2 \) | 0.05 | Low → allows rapid divergence |
| Initial fidelity | \( \mu_0 \) | 1.0 | Released over training |
| Fidelity decay | \( \gamma \) | 0.97 | Anneals over ~100 iterations |
| Momentum damping | \( \alpha \) | 0.9 | Standard heavy ball |
| Learning rate | \( \eta_{\text{lr}} \) | 0.01 | Tune with loss curve |
| Max iterations | — | 500 | Or until convergence |

---

## 6. What Emerges

### Early timepoints (21–40 hpf)
All conditions share similar phenotypes. Trajectory bundles condense toward a shared trunk.

### Mid development (40–60 hpf)
PBX crispants begin to diverge. `pbx1b_pbx4` separates earliest/furthest;
`pbx4` alone occupies an intermediate region.

### Late timepoints (60–79 hpf)
Clear branching structure: control bundle (`inj_ctrl`, `wik_ab`) vs. PBX bundles.
Within-genotype heterogeneity visible as bundle thickness.

### Wildtype-like pbx4 subclass
Embryos in `pbx4_crispant` that previously identified as wildtype-like should
remain in the control bundle throughout, appearing as a visible sub-filament.

---

## 7. Outputs

### 7.1 Per-iteration state
- `condensed_positions.npz` — shape \( (N_e, T, 2) \), final positions
- `loss_curve.csv` — \( E_{\text{attract}}, E_{\text{repel}}, E_{\text{elastic}}, E_{\text{fidelity}} \) per iteration

### 7.2 Trajectory figures
- `condensed_trajectories_by_genotype.png/.html` — colored by genotype, lines per embryo
- `condensed_trajectories_by_time.png/.html` — colored by time bin (Plotly)
- `condensed_trajectories_pbx4_split.png` — wildtype-like subclass highlighted

### 7.3 Divergence summary
- `bundle_divergence_time.csv` — estimated divergence time per genotype (when bundle center
  moves > threshold from control bundle center)
- `bundle_width_over_time.csv` — within-genotype spread per time bin

### 7.4 Condition graph overlay
- `condition_centroids_over_time.csv` — mean condensed position per condition per time bin
- Used as node positions in the Level 1 condition graph (replaces spring layout)

---

## 8. Connection to Open Questions in the Graph Idea Document

| Open question | Answer here |
|---------------|------------|
| Best edge definition? | Direct AUROC → similarity transform \( 1 - \text{AUROC} \) at Level 1; at Level 2, edge weight is \( C_{ij}(t) \) (temporal persistence) |
| Best global representation? | Level 1: \( N \times N \) graph with warm-started layout; Level 2: condensed 2D trajectories |
| How to stitch dynamic structure through time? | AlignedUMAP initialization + elasticity term enforces temporal coherence; bending penalty for smoothness |
| Should branches be defined explicitly? | No — branches emerge from filament formation. Label them post hoc from bundle centroids |
| Reconvergence allowed? | Yes — no irreversibility constraint; \( C_{ij}(t) \) can re-activate if embryos co-move again |

---

## 9. Module Structure

The implementation lives in a self-contained folder with no imports from `src/`:

```
results/mcolon/20260329_pbx_crispant_analysis_cont/
└── trajectory_cosmology/
    ├── schema.py            # ingest + canonicalize inputs into (N_e, T, G) tensors
    ├── init_embedding.py    # AlignedUMAP or other initializations → x0
    ├── condition_graph.py   # Level 1: AUROC → G×G graph + warm-started layout
    ├── diagnostics.py       # bundle divergence, width, coherence persistence
    ├── viz.py               # all figures
    └── condensation/
        ├── __init__.py
        ├── state.py         # CondensationConfig, CondensationState, CondensationResult
        ├── coherence.py     # C_ij(t): spatial kernel, causal window, masking
        ├── forces.py        # attraction, repulsion, stretch, bend, fidelity + gradients
        ├── dynamics.py      # update loop, annealing schedule, stopping criteria
        └── api.py           # run_condensation(x0, mask, config) — public entry point
```

### Responsibility split

| File | Owns |
|------|------|
| `schema.py` | Accepts multiclass CSV or pairwise margin CSV; validates; returns canonical `(features, mask, embryo_ids, time_values, labels)` |
| `init_embedding.py` | Turns canonical features into initial 2D positions; condensation never touches UMAP |
| `condensation/state.py` | Dataclasses for config, running state, and result; keeps argument lists clean |
| `condensation/coherence.py` | Spatial kernel \( K_s \), temporal coherence \( C_{ij}(t) \), masking, causal window |
| `condensation/forces.py` | All energy terms and their gradients; takes positions + coherence as inputs |
| `condensation/dynamics.py` | Damped update loop, fidelity annealing, stopping criteria |
| `condensation/api.py` | `run_condensation(x0, mask, config)` — the only function external callers need |
| `diagnostics.py` | Computes quantities from `CondensationResult`; no plotting |
| `viz.py` | Renders all figures from result + diagnostic objects |

### Canonical calling pattern

```python
from trajectory_cosmology import schema, init_embedding
from trajectory_cosmology.condensation import api as condensation

data = schema.from_multiclass_csv(path)
x0 = init_embedding.aligned_umap_init(data.features, data.mask)
result = condensation.run_condensation(x0=x0, mask=data.mask, config=config)
```

Or from pairwise margins:

```python
data = schema.from_pairwise_margin_csv(path)
x0 = init_embedding.aligned_umap_init(data.features, data.mask)
result = condensation.run_condensation(x0=x0, mask=data.mask, config=config)
```

Condensation is purely positional — it never sees genotype labels or raw features.

### Implementation notes

- `condensation/` uses NumPy only; no new dependencies
- \( C_{ij}(t) \) is recomputed from current positions at each iteration, then frozen
  during the gradient step (alternating update, not full differentiation through coherence)
- Pairwise terms are \( O(N_e^2 T) \); feasible at current scale (~500 embryos × 30 bins)
- Runtime estimate: 30s–5min for 500 iterations on CPU

## 10. Implementation Plan

### Phase A: Multiclass Vector Assembly

Train a multinomial classifier on the 5-class problem at each time bin.

**Input:** embedding latents (`z_mu_b*` columns), all 5 genotypes pooled

**Output:** `multiclass_probability_vectors.csv`

```
embryo_id | time_bin_center | genotype | p_inj_ctrl | p_wik_ab | p_pbx4_crispant | p_pbx1b_crispant | p_pbx1b_pbx4_crispant
```

New script: `rerun_20260329_no_yolk/09_multiclass_vectors.py`

### Phase B: Schema + AlignedUMAP Init

Wire `schema.from_multiclass_csv()` and `init_embedding.aligned_umap_init()`.
Output: `multiclass_aligned_umap_2d_coordinates.csv` — the \( \mathbf{x}_0 \).

### Phase C: Condition Graph (Level 1)

**Input:** `pairwise_auroc_bins_pbx_controls_embedding_all_pairs.csv`

**Output:** per-timepoint spring layout graph figures

Buildable from existing AUROC data.

### Phase D: Condensation (Level 2)

**Input:** \( \mathbf{x}_0 \) + mask from Phase B

**Output:** `condensed_positions.npz`, loss curve, figures

New script: `rerun_20260329_no_yolk/10_trajectory_condensation.py`

---

## 10. Known Caveats

### Cosmological analogy is heuristic
The energy formulation is motivated by structure formation, but the analogy is
pedagogical, not mechanistic. The terms are chosen for their biological effect, not
to simulate actual cosmological physics.

### Condensation can overcompress
If \( \sigma \) is too large, all trajectories collapse into one filament regardless of
genotype. Use the fidelity term and a tuning sweep to find \( \sigma \) where bundle
separation is visible but not forced.

### Local minima
The objective is non-convex. Different random initializations (or varying the UMAP seed)
should give qualitatively similar large-scale structure if the signal is real.
Run multiple seeds and report convergence consistency.

### Branch labels are post hoc
This method does not assign branch labels during optimization. Bundle assignments
(e.g., "pbx4 splits at 55 hpf") must be computed afterward from centroid distances.

### Reconvergence is structurally possible but biologically rare in this dataset
The data suggest monotonic divergence for PBX crispants. If a condition does reconverge,
verify it is not an artifact of small sample size at late time bins.

---

## 11. Summary

We define a two-level framework:

**Level 1 (condition graph):** time-varying \( N \times N \) AUROC similarity graph
with temporally coherent spring layout.

**Level 2 (embryo condensation):** each embryo is a 2D curve initialized from
AlignedUMAP applied to the 5D multiclass softmax vectors, then refined under an energy
that enforces:
- attraction between persistently co-moving trajectories
- repulsion to prevent collapse
- elasticity for smooth individual curves
- annealed fidelity to the initialization

The multiclass vectors are geometrically cleaner than the 10D pairwise space and
implicitly encode all pairwise relationships via the log-odds ratios \( \log(p_i/p_j) \).

The result is a **branching developmental structure that emerges from data**,
not from clustering assignments or manual supervision.

The primary biological payoff is:
- divergence timing per condition
- continuous penetrance (bundle membership vs. time)
- within-genotype heterogeneity (bundle width)
- identification of wildtype-like subclasses as persistent cross-bundle embryos
