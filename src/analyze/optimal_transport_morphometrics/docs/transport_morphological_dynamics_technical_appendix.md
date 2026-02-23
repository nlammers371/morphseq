# Transport-Based Morphological Dynamics: Technical Appendix

This appendix provides the mathematical foundations for the transport-based morphological dynamics framework. It is intended as a reference for implementation and for readers seeking formal rigor. Section numbers correspond to the main document.

---

## 1. Notation and Preliminaries

### 1.1 Masks and Densities

For embryo $e$, time $t$, and morphological structure $k$:

- **Binary mask:** $M^{(k)}_{e,t}: \Omega \to \{0,1\}$, where $\Omega \subset \mathbb{R}^2$ is the image domain.

- **Smoothed density:** Apply Gaussian smoothing to handle discontinuous structures (e.g., scattered melanocytes):
$$\tilde{\rho}^{(k)}_{e,t}(x) = (K_\sigma * M^{(k)}_{e,t})(x)$$
where $K_\sigma$ is a Gaussian kernel with bandwidth $\sigma$.

- **Probability map (for balanced OT):**
$$\rho^{(k)}_{e,t}(x) = \frac{\tilde{\rho}^{(k)}_{e,t}(x)}{\int_\Omega \tilde{\rho}^{(k)}_{e,t}(u)\,du}$$

- **Unnormalized density (for unbalanced OT):** Use $\tilde{\rho}^{(k)}_{e,t}$ directly to allow mass creation/destruction.

### 1.2 Canonical Coordinate Grid

$\Omega \subset \mathbb{R}^2$ denotes the canonical embryo coordinate grid after alignment. All densities are defined on this common grid to enable comparison across embryos and timepoints.

---

## 2. Coordinate Canonicalization

### 2.1 Yolk-Based Transform

From the yolk mask, estimate:
- Centroid $c_{e,t} \in \mathbb{R}^2$
- Rotation $R_{e,t} \in SO(2)$ (aligning anterior-posterior axis)
- Scale $s_{e,t} > 0$ (e.g., yolk diameter or embryo length)

**Coordinate transform:**
$$x' = \frac{R_{e,t}(x - c_{e,t})}{s_{e,t}}$$

**Pushforward density:** The canonical density is:
$$\rho'_{e,t}(x') = (g_{e,t})_\# \rho_{e,t}$$
where $g_{e,t}(x) = x'$ and $(\cdot)_\#$ denotes the pushforward operator.

### 2.2 Structure-Centered Transform

For analysis robust to gross morphological deformation, replace global alignment with structure-specific alignment:

- Replace $c_{e,t}$ with structure centroid $c^{(k)}_{e,t}$
- Optionally use a structure-specific principal axis for rotation

This yields $\rho'^{(k)}_{e,t}$ invariant to global embryo deformation—useful for severely affected embryos where body-wide coordinates are unreliable.

---

## 3. Optimal Transport Formulations

### 3.1 Discrete Setup

Let source and target distributions be supported on pixel grids $\{x_i\}_{i=1}^n$ and $\{y_j\}_{j=1}^m$ with masses $\rho_t = (\rho_t(x_i))_i$ and $\rho_{t+\Delta} = (\rho_{t+\Delta}(y_j))_j$.

**Cost matrix:**
$$C_{ij} = \|x_i - y_j\|^2$$

### 3.2 Balanced Entropic OT

When total mass is conserved (e.g., normalized probability distributions):

$$\pi^* = \arg\min_{\pi \geq 0} \langle C, \pi \rangle + \varepsilon \sum_{ij} \pi_{ij}(\log \pi_{ij} - 1)$$

subject to:
$$\pi \mathbf{1} = \rho_t, \quad \pi^\top \mathbf{1} = \rho_{t+\Delta}$$

where $\varepsilon > 0$ is the entropic regularization parameter.

**Sinkhorn algorithm:** Solved efficiently via iterative scaling:
$$\pi^{(k+1)} = \text{diag}(u^{(k+1)}) \, K \, \text{diag}(v^{(k+1)})$$
where $K_{ij} = \exp(-C_{ij}/\varepsilon)$.

### 3.3 Unbalanced OT (KL Marginals)

When mass can be created or destroyed (e.g., growth, cell death, migration in/out of frame):

$$\pi^* = \arg\min_{\pi \geq 0} \langle C, \pi \rangle + \lambda_1 \text{KL}(\pi \mathbf{1} \| \rho_t) + \lambda_2 \text{KL}(\pi^\top \mathbf{1} \| \rho_{t+\Delta}) + \varepsilon \, \text{Ent}(\pi)$$

where:
- $\text{KL}(p \| q) = \sum_i p_i \log(p_i/q_i) - p_i + q_i$ is the KL divergence
- $\text{Ent}(\pi) = -\sum_{ij} \pi_{ij} \log \pi_{ij}$
- $\lambda_1, \lambda_2 > 0$ control the penalty for mass imbalance

**Mass residual maps:**
$$r_t = \rho_t - \pi^* \mathbf{1} \quad \text{(mass "removed" / source deficit)}$$
$$r_{t+\Delta} = \rho_{t+\Delta} - (\pi^*)^\top \mathbf{1} \quad \text{(mass "created" / target surplus)}$$

These residuals directly quantify growth and death at each pixel.

---

## 4. Transport Vector Field (Barycentric Projection)

The transport plan $\pi^*$ is a joint distribution. To visualize it as a vector field, we compute the barycentric projection.

### 4.1 Barycentric Map

For each source pixel $x_i$, the transported mass is:
$$m_i = (\pi^* \mathbf{1})_i = \sum_j \pi^*_{ij}$$

The **barycentric map** (conditional mean destination) is:
$$T(x_i) = \frac{\sum_j y_j \, \pi^*_{ij}}{m_i} \quad \text{(defined where } m_i > \tau \text{)}$$

where $\tau$ is a small threshold to avoid division by zero.

### 4.2 Vector Field

The displacement vector field is:
$$v(x_i) = T(x_i) - x_i$$

**Interpretation:** $v(x)$ represents the conditional mean displacement $\mathbb{E}[Y - X \mid X = x]$ under the transport plan. Under entropic regularization, this provides a stable, smooth summary of the transport.

### 4.3 Vector Field Statistics

Useful summary statistics of the vector field:
- **Mean flow magnitude:** $\bar{v} = \frac{1}{n} \sum_i \|v(x_i)\|$
- **Directional bias:** $\bar{v}_{AP} = \frac{1}{n} \sum_i v(x_i) \cdot \hat{e}_{AP}$ (anterior-posterior component)
- **Flow variance:** $\text{Var}(v) = \frac{1}{n} \sum_i \|v(x_i) - \bar{v}\|^2$

---

## 5. Pixel-Level Cost Attribution (Heatmaps)

### 5.1 Source Cost Attribution

The transport cost attributable to each source pixel:
$$h_t(x_i) = \sum_j \|x_i - y_j\|^2 \, \pi^*_{ij}$$

This map shows which source regions contribute most to the total transport cost.

### 5.2 Target Cost Attribution

The transport cost attributable to each target pixel:
$$h_{t+\Delta}(y_j) = \sum_i \|x_i - y_j\|^2 \, \pi^*_{ij}$$

### 5.3 Group Difference Hotspot Map

To identify regions that differ between groups (e.g., mutant vs. wild-type):
$$\Delta h_t(x) = \mathbb{E}[h_t(x) \mid \text{mutant}] - \mathbb{E}[h_t(x) \mid \text{WT}]$$

Statistical significance can be assessed via permutation testing over embryos.

---

## 6. Per-Transition Embeddings

### 6.1 Feature Extraction from Transport Plan

For each transition $t \to t+\Delta$ in embryo $e$, extract a feature vector:
$$z_{e,t} = \phi(\pi^*_{e,t}) \in \mathbb{R}^d$$

**Candidate features $\phi$:**

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| Transport cost | $\langle C, \pi^* \rangle$ | Total "work" to move mass |
| Plan entropy | $\text{Ent}(\pi^*)$ | Diffuseness of transport |
| Mean flow magnitude | $\frac{1}{n}\sum_i \|v(x_i)\|$ | Average displacement |
| AP directional bias | $\frac{1}{n}\sum_i v(x_i) \cdot \hat{e}_{AP}$ | Net anterior-posterior movement |
| DV directional bias | $\frac{1}{n}\sum_i v(x_i) \cdot \hat{e}_{DV}$ | Net dorsal-ventral movement |
| Mass created | $\|r_{t+\Delta}\|_1$ | Total mass appearing |
| Mass destroyed | $\|r_t\|_1$ | Total mass disappearing |
| Flow variance | $\text{Var}(v)$ | Heterogeneity of movement |
| Regional breakdown | $\phi_{\text{head}}, \phi_{\text{tail}}, \ldots$ | Features stratified by region |

### 6.2 Embryo Trajectory

The full developmental trajectory for embryo $e$ is the sequence of transition embeddings:
$$Z_e = \{z_{e,t_1}, z_{e,t_2}, \ldots, z_{e,t_K}\}$$

where $t_1 < t_2 < \cdots < t_K$ are the observed timepoints.

---

## 7. Trajectory Comparison and Clustering

### 7.1 Trajectory Distance via DTW

For embryos with potentially misaligned developmental timing, use Dynamic Time Warping:
$$D(e, e') = \text{DTW}(Z_e, Z_{e'})$$

DTW finds the optimal alignment between sequences, allowing comparison of embryos that develop at different rates.

### 7.2 Euclidean Distance (Aligned Sampling)

If all embryos are sampled at identical timepoints:
$$D(e, e') = \|Z_e - Z_{e'}\|_2$$

where $Z_e$ is treated as a concatenated vector.

### 7.3 Embryo-Level Summary

For simpler clustering, average over time:
$$\bar{z}_e = \frac{1}{K} \sum_{k=1}^K z_{e,t_k}$$

Then cluster on $\{\bar{z}_e\}$ using standard methods (k-means, hierarchical, etc.).

---

## 8. Learned Dynamics (Neural ODE / Flow Matching)

For richer representations, we can learn a continuous-time flow model conditioned on transition embeddings.

### 8.1 Conditional Flow

Model the transport as a continuous flow:
$$\frac{dx}{ds} = v_\theta(x, s; z_{e,t}), \quad s \in [0,1]$$

where:
- $s=0$ corresponds to time $t$
- $s=1$ corresponds to time $t+\Delta$
- $z_{e,t} = E_\psi(\rho_t, \rho_{t+\Delta})$ is a learned transition code

### 8.2 OT-Coupled Flow Matching Loss

Train using samples from the OT plan:
1. Sample $(x_0, x_1) \sim \pi^*$
2. Interpolate: $x_s = (1-s)x_0 + s x_1$
3. Target velocity: $\dot{x}_s = x_1 - x_0$

**Loss function:**
$$\mathcal{L}(\theta, \psi) = \mathbb{E}_{s \sim U[0,1], (x_0, x_1) \sim \pi^*} \left[ \|v_\theta(x_s, s; z_{e,t}) - (x_1 - x_0)\|^2 \right]$$

### 8.3 Clustering on Learned Codes

The learned transition codes $z_{e,t}$ (or trajectory sequences $Z_e$) can be clustered to identify distinct developmental modes.

---

## 9. Displacement Interpolation (Morphological Movies)

### 9.1 McCann Interpolation

Given the optimal transport map $T: x \mapsto y$, the displacement interpolation at time $s \in [0,1]$ is:
$$\rho_s = ((1-s)\text{Id} + sT)_\# \rho_t$$

In the discrete case with transport plan $\pi^*$:
$$x_i^{(s)} = (1-s)x_i + s \, T(x_i)$$

### 9.2 Visualization

Rendering $\rho_s$ for $s \in [0,1]$ produces a smooth "morphological movie" showing how mass flows from $\rho_t$ to $\rho_{t+\Delta}$. This visualization highlights:
- Where mass moves (flow paths)
- When trajectories diverge (comparing WT vs. mutant movies)
- Regions of high transport cost (areas of large displacement)

---

## 10. Statistical Testing

### 10.1 Permutation Tests for Group Differences

To test whether transport features differ between groups:

1. Compute test statistic $T_{\text{obs}}$ (e.g., difference in mean transport cost)
2. Permute group labels $B$ times
3. Compute $T_b$ for each permutation
4. p-value: $p = \frac{1}{B}\sum_{b=1}^B \mathbf{1}[T_b \geq T_{\text{obs}}]$

### 10.2 Spatial Permutation for Hotspot Maps

For pixel-level group differences $\Delta h_t(x)$:

1. Compute observed $\Delta h_t(x)$ at each pixel
2. Permute embryo labels, recompute $\Delta h_t^{(b)}(x)$
3. At each pixel, compute p-value or FDR-corrected significance

---

## 11. Implementation Notes

### 11.1 Software

- **POT (Python Optimal Transport):** `pip install pot` — Sinkhorn, unbalanced OT, barycenters
- **OTT-JAX:** GPU-accelerated OT with autodiff support
- **GeomLoss:** PyTorch library for Sinkhorn divergences

### 11.2 Computational Considerations

- **Grid resolution:** Downsample masks if needed; OT scales as $O(n^2)$ for $n$ pixels (with Sinkhorn) or $O(n^2 \log n)$ for exact solvers
- **Entropic regularization:** Larger $\varepsilon$ = faster convergence, smoother plans; smaller $\varepsilon$ = sharper transport but slower
- **Unbalanced parameters:** $\lambda \to \infty$ recovers balanced OT; smaller $\lambda$ allows more mass flexibility

### 11.3 Recommended Defaults (Starting Points)

| Parameter | Suggested Value | Notes |
|-----------|-----------------|-------|
| $\varepsilon$ (entropy) | 0.01–0.1 × mean cost | Tune for smoothness vs. sharpness |
| $\lambda$ (unbalanced) | 1.0–10.0 | Higher = stricter mass conservation |
| $\sigma$ (smoothing) | 2–5 pixels | For discontinuous structures like melanocytes |
| $\tau$ (barycentric threshold) | $10^{-6}$ | Avoid division by zero |

---

## References

1. Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.

2. Gerber, S., et al. (2018). Exploratory Population Analysis with Unbalanced Optimal Transport. *MICCAI*.

3. Gerber, S., et al. (2022/2023). Optimal Transport Features for Morphometric Population Analysis.

4. Liu, Y., et al. (2023). CellStitch: 3D Cellular Anisotropic Image Segmentation via Optimal Transport.

5. Lipman, Y., et al. (2023). Flow Matching for Generative Modeling. *ICLR*.

6. Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *NeurIPS*.
