# Cosmological Condensation of Developmental Trajectories
## A Biologically and Cosmologically Motivated Mathematical Specification

---

## 1. Problem Framing

We observe a population of embryos across developmental time.

Each embryo:
- has a genotype \( g_i \)
- is observed at discrete timepoints \( t = 1, \dots, T \)
- is represented by a feature vector derived from classification probabilities

\[
\mathbf{v}^{(i,t)} \in \mathbb{R}^K
\]

where \( K = \binom{G}{2} \) is the number of pairwise classifiers across \( G \) genotypes.

We embed these vectors into 2D:

\[
\mathbf{x}_0^{(i,t)} \in \mathbb{R}^2
\]

These are *initial positions*, not ground truth geometry.

---

## 2. Biological Principle

Development is a **set of trajectories**, not snapshots.

Each embryo is a curve:

\[
\mathcal{T}_i = \{\mathbf{x}^{(i,1)}, \mathbf{x}^{(i,2)}, \dots, \mathbf{x}^{(i,T)}\}
\]

We assume:

- Early development: trajectories are similar (shared trunk)
- Later development: trajectories diverge (branching)
- Divergence timing and magnitude encode biological information

---

## 3. Cosmological Analogy

We borrow from large-scale structure formation:

| Cosmology | Development |
|----------|------------|
| Matter particles | Embryos |
| Space | Embedding space |
| Time evolution | Development |
| Gravity | Attraction between similar trajectories |
| Filaments | Bundles of embryos with similar trajectories |

Key idea:

> Structure emerges from **persistent proximity**, not instantaneous similarity.

---

## 4. Core Constraint

**Time is fixed.**

We optimize only spatial positions:

\[
\mathbf{x}^{(i,t)} \in \mathbb{R}^2
\]

Time indices \( t \) are immutable.

---

## 5. Energy-Based Formulation

We define a dynamical system over all positions:

\[
E = E_{\text{attract}} + E_{\text{repel}} + E_{\text{elastic}} + E_{\text{fidelity}}
\]

Each term corresponds to a physical and biological principle.

---

## 6. Attraction Term (Key Concept)

### Biological meaning

Embryos should attract each other **only if they behave similarly over time**.

Not:
- "they are close now"

But:
- "they have been close over a window of time"

---

### Mathematical definition

\[
E_{\text{attract}} = -\sum_{t=1}^{T} \sum_{i \neq j}
K_s(\mathbf{x}^{(i,t)}, \mathbf{x}^{(j,t)}) \cdot C_{ij}(t)
\]

Where:

- \( K_s \): spatial kernel (e.g., Gaussian)

\[
K_s(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right)
\]

- \( C_{ij}(t) \): temporal coherence

---

## 7. Temporal Coherence (Teaches the Core Idea)

### Intuition

Two embryos should only attract if they have been consistently close *in the past*.

This prevents:
- noise-driven attraction
- transient crossings from forming false structure

---

### Definition

\[
C_{ij}(t) =
\frac{
\sum_{\tau = t-\delta}^{t}
K_s(\mathbf{x}^{(i,\tau)}, \mathbf{x}^{(j,\tau)})
\cdot m^{(i,\tau)} m^{(j,\tau)}
}{
\sum_{\tau = t-\delta}^{t}
m^{(i,\tau)} m^{(j,\tau)}
}
\]

Where:

- \( \delta \): temporal window
- \( m^{(i,\tau)} \in \{0,1\} \): observation mask

---

### Step-by-step interpretation

For each pair \( (i, j) \) at time \( t \):

1. Look backward in time over window \( [t-\delta, t] \)
2. Measure how close they were at each timepoint
3. Average those similarities
4. Normalize by how often both were observed

---

### What this does

- High \( C_{ij}(t) \): trajectories have moved together → strong attraction
- Low \( C_{ij}(t) \): trajectories were not consistently close → weak attraction

---

### Important property

This is **causal**:
- future proximity does not influence current attraction

This preserves developmental directionality.

---

## 8. Repulsion Term

### Motivation

Without repulsion:
- all points collapse into a single cluster

We want:
- **filaments**, not blobs

---

### Definition (soft-core repulsion)

\[
E_{\text{repel}} =
\sum_{t=1}^{T} \sum_{i \neq j}
\frac{\epsilon}{\|\mathbf{x}^{(i,t)} - \mathbf{x}^{(j,t)}\|^2 + \eta}
\]

Where:
- \( \epsilon \): repulsion strength
- \( \eta > 0 \): numerical stabilizer

---

### Effect

- prevents collapse
- preserves separation within bundles
- allows nearby but distinct trajectories

---

## 9. Elasticity Term

Each embryo is a **continuous trajectory**.

We enforce smoothness using two penalties.

---

### 9.1 Stretch penalty

\[
E_{\text{stretch}} =
\lambda_1 \sum_i \sum_{t=1}^{T-1}
\|\mathbf{x}^{(i,t+1)} - \mathbf{x}^{(i,t)}\|^2
\]

Prevents large jumps between timepoints.

---

### 9.2 Bending penalty

\[
E_{\text{bend}} =
\lambda_2 \sum_i \sum_{t=2}^{T-1}
\|\mathbf{x}^{(i,t+1)} - 2\mathbf{x}^{(i,t)} + \mathbf{x}^{(i,t-1)}\|^2
\]

Penalizes sharp directional changes.

---

### Interpretation

- \( \lambda_1 \): controls step size smoothness
- \( \lambda_2 \): controls curvature smoothness

Biological meaning:
- low \( \lambda_2 \): allows abrupt phenotype shifts
- high \( \lambda_2 \): enforces gradual transitions

---

## 10. Fidelity Term

Anchors positions to initialization:

\[
E_{\text{fidelity}} =
\mu(n) \sum_i \sum_t
\|\mathbf{x}^{(i,t)} - \mathbf{x}_0^{(i,t)}\|^2
\]

With annealing:

\[
\mu(n) = \mu_0 \cdot \gamma^n
\]

---

### Purpose

- stabilizes early optimization
- prevents arbitrary drift
- gradually releases control to allow structure formation

---

## 11. Important Dynamical Property

Because \( C_{ij}(t) \) depends on current positions:

> This is NOT a static energy minimization.

It is a **self-consistent dynamical system**.

---

### Consequence

- trajectories that move together → become more coherent
- increased coherence → stronger attraction
- leads to **filament formation**

---

## 12. Optimization

Use damped dynamics:

\[
\mathbf{v} \leftarrow \alpha \mathbf{v} - \nabla E
\]

\[
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{v}
\]

Where:
- \( \alpha \in (0,1) \): damping

---

## 13. What Emerges

From these rules:

### Early timepoints
- all trajectories condense → shared trunk

### Later timepoints
- trajectories diverge → branches

### Final structure

A **tree-like geometry** representing:

- divergence timing
- divergence magnitude
- trajectory similarity

---

## 14. Biological Interpretation

This method reveals:

### Divergence timing
When embryos stop co-moving

### Continuous penetrance
How long embryos remain WT-like

### Heterogeneity
Variation within genotypes

### Misclassification as structure
Embryos align with different trajectory bundles

---

## 15. Summary

We define a system where:

- trajectories interact through **time-persistent similarity**
- spatial structure evolves under:
  - attraction
  - repulsion
  - elasticity
  - weak anchoring

This produces:

> A branching developmental structure that emerges from data  
> rather than being imposed by clustering or supervision

---

## 16. Implementation Sketch

At each iteration:

1. Compute pairwise distances within each time slice
2. Compute spatial kernel \( K_s \)
3. Compute temporal coherence \( C_{ij}(t) \)
4. Compute gradients of:
   - attraction
   - repulsion
   - elasticity
   - fidelity
5. Update positions using damped dynamics
6. Repeat until convergence

---

## Final Intuition

We are not clustering embryos.

We are letting their trajectories **condense under rules of persistence**,  
until the hidden branching structure of development becomes visible.
