# Morphogenetic Dynamics Model: Technical Specification

## 1. Overview

This document specifies a stochastic dynamical model for forward prediction of embryo trajectories in a learned latent space. The model learns a shared developmental potential landscape modulated by a low-dimensional set of perturbation modes. Given a partial observed trajectory, the model infers per-embryo mode loadings and rate parameters via closed-form solves, then forward-simulates under the inferred dynamics.

A kernel regression baseline is specified alongside the primary model to benchmark whether the learned dynamics justify their complexity.

---

## 2. Problem Setting

- Observed data: collections of embryo trajectories in a latent space $z \in \mathbb{R}^d$ (e.g., $d = 10$, from a pretrained VAE or similar embedding model).
- Each embryo $e$ belongs to a perturbation class $p(e)$ (e.g., gene knockdown, drug treatment, wild-type).
- Observations may be irregularly spaced in time and vary in length. Some embryos may have as few as a single observed snapshot.
- Goal: given a partial trajectory $z_{t_0:t_n}$ (possibly from an unseen perturbation class), predict the forward distribution $p(z_{t_n + \tau})$.

---

## 3. Dynamical Model

### 3.1 SDE Framework

The latent state evolves under an Itô SDE:

$$dz = f(z; c_e, R_e) \, dt + \sqrt{2D} \, dW$$

where:
- $f$ is the drift (defined below)
- $c_e \in \mathbb{R}^M$ is the embryo-level mode loading vector
- $R_e > 0$ is the embryo-level rate parameter
- $D > 0$ is a global scalar diffusion coefficient (shared across all embryos)
- $W$ is a standard Wiener process

### 3.2 Drift Structure

The drift decomposes as:

$$f(z; c_e, R_e) = R_e \left[ -\beta \, \nabla_z \phi_0(z) + \sum_{m=1}^{M} c_{e,m} \, (-I + S_m) \, \nabla_z \phi_m(z) \right] + v_e \cdot w_e(z)$$

where:

| Symbol | Description |
|--------|-------------|
| $\phi_0(z)$ | Baseline developmental potential (scalar field, shared across all embryos) |
| $\phi_m(z)$ | Mode potentials (scalar fields, $m = 1, \dots, M$) |
| $S_m$ | Antisymmetric matrix per mode ($S_m = -S_m^\top$), encoding rotational dynamics |
| $\beta > 0$ | Global drift scale for baseline potential |
| $c_e \in \mathbb{R}^M$ | Mode loadings per embryo (unconstrained reals) |
| $R_e > 0$ | Rate parameter per embryo |
| $v_e \in \mathbb{R}^d$ | Embryo-specific local drift correction |
| $w_e(z)$ | Spatial decay kernel for local correction (see §3.5) |

### 3.3 Mode Parameterization (Helmholtz)

Each mode contributes a drift $(-I + S_m) \nabla_z \phi_m(z)$. This admits a clean interpretation:

- When $S_m = 0$: mode $m$ is a pure potential (gradient) mode. Flow descends the landscape $\phi_m$.
- When $S_m \neq 0$: the gradient is rotated before application, producing spiraling or circulation dynamics around the level sets of $\phi_m$.

**Critical property**: the fixed points of mode $m$ are determined entirely by $\nabla_z \phi_m = 0$, regardless of $S_m$. The antisymmetric matrix changes the geometry of approach (spiraling vs. direct descent) without altering the topology (locations of attractors, saddles, basins). This means each mode can be visualized as a scalar landscape even when its dynamics are non-conservative.

To toggle between potential-only and Helmholtz modes: simply freeze $S_m = 0$ for the potential-only variant. This should be a runtime flag, not an architectural change.

### 3.4 Rate Parameters

The embryo-level rate decomposes as:

$$R_e = \lambda_e \cdot \exp\left(-\beta_T (T_{\text{ref}} - T_e)\right)$$

where:
- $T_e$ is the experimentally controlled temperature for embryo $e$ (fixed, known)
- $T_{\text{ref}}$ is a reference temperature (fixed constant)
- $\beta_T$ is a global Arrhenius-like temperature coefficient (learnable or fixed from external estimate)
- $\lambda_e > 0$ is an embryo-specific residual rate factor

**Identifiability constraint**: enforce $\frac{1}{N} \sum_e \lambda_e = 1$ across the training set. This can be implemented via reparameterization: $\lambda_e = \tilde{\lambda}_e \, / \, \bar{\tilde{\lambda}}$, where $\tilde{\lambda}_e$ are unconstrained positive parameters (e.g., $\tilde{\lambda}_e = \exp(\ell_e)$ with learnable $\ell_e$).

$R_e$ modulates the **entire** drift (baseline + modes), not just the baseline. This reflects the assumption that temperature scales developmental rate uniformly. A key consequence: $R_e$ factors out of the least-squares solve for $c_e$, so mode loadings are invariant to developmental tempo.

### 3.5 Embryo-Specific Local Drift Correction

The term $v_e \cdot w_e(z)$ handles OOD trajectories that the mode basis cannot adequately represent:

$$w_e(z) = \exp\left( -\frac{\| z - \bar{z}_e \|^2}{2\sigma_w^2} \right)$$

where $\bar{z}_e$ is the centroid of observed points for embryo $e$, and $\sigma_w$ is a fixed kernel width (set to the median pairwise distance within observed trajectories, or similar).

This correction applies a constant drift vector $v_e$ near the observed data and decays to zero far from it, so the embryo reverts to the mode-predicted dynamics for forward prediction. At each observed transition, $w_e(z_t)$ is a precomputed scalar, so the correction enters the closed-form solve as additional columns in the design matrix (see §4.1).

**Regularization**: $v_e$ should be penalized more strongly than $c_e$ to ensure the modes get first priority in explaining dynamics. Use a separate regularization weight $\lambda_v > \lambda_c$.

---

## 4. Inference (Per-Embryo)

All per-embryo quantities ($c_e$, $R_e$, $v_e$) are inferred via closed-form solves — no inner optimization loop, no encoder network.

### 4.1 Closed-Form Solve for Mode Loadings $c_e$ and Local Correction $v_e$

Under Euler-Maruyama discretization, for transition $(z_t, z_{t+1})$ with time step $\Delta t$:

$$z_{t+1} \sim \mathcal{N}\left(z_t + f(z_t; c_e, R_e) \, \Delta t, \;\; 2D \Delta t \, I \right)$$

With $R_e$ known (or estimated separately, see §4.2), define:

- Observed displacement: $\delta_t = z_{t+1} - z_t$
- Baseline prediction: $g_t = -R_e \, \beta \, \nabla_z \phi_0(z_t) \, \Delta t$
- Mode contribution vectors: $h_{t,m} = R_e \, (-I + S_m) \, \nabla_z \phi_m(z_t) \, \Delta t$
- Local correction columns: $h_{t,d+j} = w_e(z_t) \, \Delta t \, \mathbf{e}_j$ for $j = 1, \dots, d$
- Residual: $r_t = \delta_t - g_t$

The combined unknown vector is $\xi_e = [c_e; v_e] \in \mathbb{R}^{M+d}$.

**Heteroscedasticity correction**: if $\Delta t$ varies across transitions, weight each row by $1 / \sqrt{2D \Delta t_i}$.

Stack all transitions into a system $R = H \xi$ where $R \in \mathbb{R}^{Td}$, $H \in \mathbb{R}^{Td \times (M+d)}$. The regularized solution is:

$$\xi_e^* = (H^\top H + \Lambda)^{-1} (H^\top R + \Lambda \, \xi_0)$$

where $\Lambda = \text{diag}(\lambda_c I_M, \, \lambda_v I_d)$ is a block-diagonal regularization matrix with separate strengths for mode loadings and local correction, and $\xi_0 = [c_{0,p(e)}; \, \mathbf{0}]$ is the prior (class-level loadings for $c$, zero for $v$).

This is an $(M+d) \times (M+d)$ linear system (e.g., $15 \times 15$ for $M=5$, $d=10$). Solved via `torch.linalg.solve`. Fully differentiable — gradients flow through the solve into the network parameters.

### 4.2 Closed-Form Solve for Rate $R_e$

Given $c_e^*$ and $v_e^*$, the total predicted drift direction at each transition is known. $R_e$ is a scalar multiplier on the structured component. The optimal $R_e$ is:

$$R_e^* = \frac{\sum_t \delta_t^\top \hat{f}_t \, \Delta t}{\sum_t \| \hat{f}_t \|^2 \, \Delta t^2}$$

where $\hat{f}_t$ is the total structured drift (baseline + modes, before $R_e$ scaling) at $z_t$. This is a projection of observed displacements onto predicted drift directions.

**Alternating estimation**: solve for $c_e$ given $R_e$, then $R_e$ given $c_e$. Each step is closed-form. Two to three alternations suffice. Identical procedure at train and test time (no train/test asymmetry).

### 4.3 Class-Level Priors $c_{0,p}$

The class-level prior $c_{0,p}$ for perturbation class $p$ is a learnable parameter vector. Given fixed basis functions and $\lambda_c$, the optimal $c_{0,p}$ is the solution to a linear system derived from differentiating the total class likelihood — it is the data-weighted average of embryo-level $c_e^*$ values within the class, where embryos with more transitions are weighted less toward the prior.

$c_{0,p}$ can be solved analytically (nested linear solve) or optimized via backprop alongside network weights. The latter is simpler to implement initially.

---

## 5. Network Architecture

### 5.1 Potential Networks

Each of $\phi_0, \phi_1, \dots, \phi_M$ is a small MLP: $\mathbb{R}^d \to \mathbb{R}$.

- Architecture: 2 hidden layers, width 32–64, smooth activations (softplus or ELU).
- Small capacity is an intentional inductive bias toward simple landscape topography.
- Gradients $\nabla_z \phi_m$ are computed via autodiff.

### 5.2 Antisymmetric Matrices

Each $S_m$ is a learnable constant $d \times d$ antisymmetric matrix. Parameterize by learning the $d(d-1)/2$ upper-triangular entries and antisymmetrizing. Initialize at zero (pure potential modes at start of training).

### 5.3 Initialization

- Mode networks $\phi_m$ ($m \geq 1$): initialize weights near zero so all embryos start on the baseline landscape. Modes are recruited as training progresses.
- Baseline network $\phi_0$: standard initialization (e.g., Xavier/He).
- $S_m$: initialize at zero.

---

## 6. Regularization

### 6.1 Mode Loading Regularization

L2 penalty on $c_e$, implemented as the $\lambda_c$ term in the ridge solve (§4.1). Controls shrinkage toward the class-level prior $c_{0,p}$.

### 6.2 Local Correction Regularization

L2 penalty on $v_e$ with $\lambda_v > \lambda_c$. Ensures modes explain dynamics first; local correction is a residual.

### 6.3 Mode Gradient Magnitude

Penalize the average drift magnitude of each mode across training data:

$$\mathcal{R}_1 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \| (-I + S_m) \nabla_z \phi_m(z) \|^2 \right]$$

This constrains modes to be "small" relative to baseline without creating adversarial coupling between $\phi_0$ and the modes.

### 6.4 Mode Potential Power

Penalize the total scalar power of each mode across the data manifold:

$$\mathcal{R}_2 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \phi_m(z)^2 \right]$$

This constrains modes from dominating the baseline landscape in terms of potential value, preserving the interpretability of $\phi_0$ as the primary developmental stage coordinate.

### 6.5 Hessian Smoothness

Penalize rapid spatial variation in each potential:

$$\mathcal{R}_3 = \mathbb{E}_{z \sim \text{data}} \left[ \sum_m \| \nabla_z^2 \phi_m(z) \|_F^2 \right]$$

Computed via double backward passes. Encourages smooth, broad landscape features per mode.

### 6.6 Rate Parameter Centering

Mean-centering constraint on $\lambda_e$ across the training set: $\bar{\lambda} = 1$. Implemented via reparameterization (§3.4).

---

## 7. Training

### 7.1 Staged Training

Training proceeds in stages to enforce the interpretive hierarchy: $\phi_0$ defines baseline development, modes capture deviations.

**Stage 1: Baseline potential only.** Train $\phi_0$ network weights, $\beta$, $D$, and $R_e$ (per embryo). No modes, no $v_e$. This produces the $\phi_0$-only model, which is saved as a checkpoint and serves as a permanent evaluation baseline alongside the kernel regression.

**Stage 2: Introduce modes.** Freeze $\phi_0$. Introduce mode potentials $\phi_m$, antisymmetric matrices $S_m$ (initialized at zero), class-level priors $c_{0,p}$, and local correction $v_e$. Train with the closed-form $[c_e; v_e]$ solve active. $R_e$ continues to update. Regularization terms $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ are active.

**Stage 3 (optional): Fine-tune baseline.** Unfreeze $\phi_0$ with a heavily reduced learning rate (e.g., $10\times$ lower than stage 2). This allows the baseline to adjust slightly to accommodate the modes. Monitor whether $\phi_0$ drifts significantly from its stage 1 form — if it does, the modes aren't expressive enough. Skip this stage initially.

Within each stage, the data sampling and forward pass logic are identical.

### 7.2 Data Sampling

Each training iteration:

1. For each trajectory in the batch, randomly select a contiguous fragment starting at some $t_0$.
2. Randomly sample a prediction horizon $k \in \{1, 2, 3, 4\}$ (in units of the experiment's time resolution).
3. The context is $z_{t_0:t_n}$ (used to infer $c_e$, $v_e$, $R_e$); the target is $z_{t_n + k}$.

This forces the model to produce calibrated predictions across a range of horizons and trajectory lengths rather than overfitting to full-trajectory reconstruction.

**Time resolution**: within any single experiment, all embryos share a uniform time step $\Delta t$. Across experiments, $\Delta t$ may differ. The model should accept $\Delta t$ as an input per experiment. The heteroscedasticity correction (weighting rows by $1/\sqrt{2D\Delta t}$) applies when training across experiments with different resolutions.

**Horizon curriculum**: consider starting with $k=1$ only during early training (within each stage) and introducing longer horizons once basis functions have stabilized. Multi-step backpropagation through stochastic simulation is noisy; premature exposure to large $k$ may destabilize learning.

### 7.3 Loss Function

$$\mathcal{L} = -\sum_e \log p(z_{t_n + k}^{(e)} | z_{t_n}^{(e)}, \xi_e^*, R_e^*) + \alpha_1 \mathcal{R}_1 + \alpha_2 \mathcal{R}_2 + \alpha_3 \mathcal{R}_3$$

For multi-step prediction ($k > 1$), the likelihood is evaluated by forward-simulating $k$ steps under Euler-Maruyama from $z_{t_n}$ and scoring the target $z_{t_n+k}$. For $k = 1$, the transition log-likelihood under Euler-Maruyama is:

$$\log p(z_{t+1} | z_t) = -\frac{d}{2} \log(4\pi D \Delta t) - \frac{\| z_{t+1} - z_t - f(z_t; \xi_e^*, R_e^*) \Delta t \|^2}{4 D \Delta t}$$

### 7.4 Parameters and Their Optimization Method

| Parameter | Method | Notes |
|-----------|--------|-------|
| $\phi_0$ network weights | Backprop | Baseline potential |
| $\phi_m$ network weights | Backprop | Mode potentials |
| $S_m$ entries | Backprop | Antisymmetric matrices; init at 0 |
| $\beta$ | Backprop | Global drift scale |
| $\beta_T$ | Fixed or backprop | Arrhenius coefficient; external estimate available |
| $D$ | Backprop | Global scalar diffusion |
| $c_{0,p}$ | Backprop (or analytic) | Class-level priors |
| $c_e$ | Closed-form (ridge) | Per-embryo, inside forward pass |
| $v_e$ | Closed-form (ridge) | Per-embryo, inside forward pass |
| $R_e$ | Closed-form (projection) | Per-embryo, inside forward pass |
| $\lambda_e$ | Absorbed into $R_e$ | Via reparameterization |

### 7.5 Forward Pass (Single Training Step)

1. Sample a fragment and prediction horizon $k$ per trajectory (§7.1).
2. Compute $\nabla_z \phi_0(z_t)$ and $\nabla_z \phi_m(z_t)$ at all context transitions via autodiff.
3. Assemble design matrix $H$ and residual vector $R$ per embryo (context only).
4. Solve for $\xi_e^* = [c_e^*; v_e^*]$ via ridge regression.
5. Compute $R_e^*$ via scalar projection.
6. Iterate steps 3–5 once or twice (alternating $c/v$ and $R$).
7. Forward-simulate $k$ steps from $z_{t_n}$ under the inferred dynamics.
8. Evaluate log-likelihood of observed $z_{t_n+k}$ under the predicted distribution.
9. Add regularization penalties.
10. Backpropagate through everything (including the linear solve and forward simulation) into network weights.

---

## 8. Interpretive Framework

### 8.1 Developmental Stage

$\phi_0(z)$ defines a scalar developmental stage for every point in latent space. This is exact and global — not a post-hoc pseudotime estimate.

### 8.2 Developmental Isoclines

Level sets $\{ z : \phi_0(z) = s \}$ are surfaces of equal developmental stage. Two embeddings $z_a$ and $z_b$ with $\phi_0(z_a) = \phi_0(z_b)$ are at the same developmental stage; the distance between them on the level set is pure morphological difference.

### 8.3 Distance Decomposition

For any two embeddings $z_a, z_b$:
- **Developmental progression difference**: $\Delta s = \phi_0(z_a) - \phi_0(z_b)$
- **Morphological difference**: displacement along the shared level set (after projecting to matching stage)

This decomposition is intrinsic to the model and requires no ODE integration, no Euclidean projection, and introduces no path-dependence.

### 8.4 Dynamical Phenotype

The loading vector $c_e \in \mathbb{R}^M$ is a compact dynamical phenotype for embryo $e$. It describes which landscape features are active and at what strength. For comparing perturbation classes, the class-level $c_{0,p}$ serves as a summary phenotype.

Each mode's contribution at a given isocline can be decomposed into:
- **Gradient-parallel component**: changes developmental tempo (accelerates or retards progression)
- **Gradient-orthogonal component**: diverts to a different fate

This provides a richer phenotypic vocabulary than raw trajectory distances.

---

## 9. Baseline: Kernel Trajectory Regression

### 9.1 Method

Given a query fragment $z_{t_0:t_n}$, compute distances to all training trajectories over the overlapping time window:

$$d_i = \frac{1}{|T_{\text{overlap}}|} \sum_{t \in T_{\text{overlap}}} \| z_t^{\text{query}} - z_t^{(i)} \|^2$$

Convert to weights via Gaussian kernel:

$$w_i = \exp\left( -d_i / 2\sigma^2 \right)$$

The forward prediction at horizon $\tau$ is the weighted empirical distribution of training trajectory positions at time $t_n + \tau$.

### 9.2 Properties

- Returns a full predictive distribution (weighted point cloud).
- Degrades gracefully to kernel density regression on initial state for single-snapshot queries.
- One tuning parameter: kernel bandwidth $\sigma$ (cross-validate on held-out trajectories).
- No modeling assumptions. Hard to beat in data-rich, within-distribution regimes.
- Fails for novel perturbations with no nearby training trajectories.

---

## 10. Test Set Design

### Tier 1: Novel Perturbation (Must-Have)

Hold out one or more entire perturbation classes believed to be within the span of the mode basis. This is the primary test of whether the model learns generalizable dynamics. The kernel baseline is blind here by construction.

### Tier 2: Within-Class Generalization (Calibration)

Hold out random embryos from known classes, stratified by trajectory length. Compare performance as a function of number of observed time points. The potential model's advantage should grow as context shrinks.

### Tier 3: Out-of-Span Perturbation (Stress Test)

Hold out a perturbation class suspected to be outside the mode basis span. The model should produce poor predictions but with appropriately high uncertainty (large residuals, large $v_e$ corrections). Confident wrong predictions indicate poor calibration.

### Tier 4: Horizon Sweep

For a well-represented class, predict at increasing horizons $\tau$. Compare degradation curves between the model and baseline. The baseline should degrade faster; if it doesn't, the learned dynamics aren't capturing long-range flow structure.

---

## 11. Evaluation Stack

All evaluation is logged to Weights & Biases (W&B). Every model checkpoint — including the kernel baseline and the $\phi_0$-only model from stage 1 — is evaluated on the same test sets to enable direct comparison.

### 11.1 Quantitative Metrics

The following are computed and logged at each evaluation checkpoint:

- **Held-out NLL**: negative log-likelihood on test transitions. Primary metric. Reported separately for each test tier (§10).
- **Per-horizon NLL**: broken out by prediction horizon $k \in \{1, 2, 3, 4\}$. Reveals where the model struggles.
- **Earth mover's distance (EMD)**: between predicted forward distributions (sampled from the SDE) and observed positions at each horizon. Captures distributional accuracy beyond mean prediction.
- **Baseline comparison**: kernel regression scores on the same test sets, always visible as a reference line on all plots.
- **$\phi_0$-only comparison**: scores from the stage 1 checkpoint, always visible as a second reference line. The gap between kernel → $\phi_0$-only measures the value of learned dynamics; the gap between $\phi_0$-only → full model measures the value of modes.
- **Mode utilization diagnostics**:
  - Distribution of $\|v_e\|$ across test embryos (canary: large values mean modes aren't explaining the data)
  - Distribution of residual norms after the $c_e$ solve (are the modes a good basis?)
  - Per-mode average $|c_{e,m}|$ (which modes are being used?)
  - $\|S_m\|_F$ per mode (how non-conservative are the learned dynamics?)

### 11.2 Three Models Always Compared

Every evaluation report shows three models side by side:

1. **Kernel baseline** — no learned dynamics, pure memorization
2. **$\phi_0$-only** — learned baseline potential, no modes (stage 1 checkpoint)
3. **Full model** — baseline potential + modes + local correction

This decomposition directly answers: does learning dynamics help? Do modes help beyond baseline dynamics?

---

## 12. Visualization

### 12.1 Panel 1: Trajectory View

- X-axis: $\phi_0(z)$ (developmental stage)
- Y-axis: first residual PC (PCA on displacements orthogonal to $\nabla\phi_0$, computed per isocline)
- Observed trajectories as solid colored lines (by perturbation class)
- Model predictions as dashed lines with shaded uncertainty envelopes
- Baseline kernel predictions shown as point clouds

This is the primary diagnostic for whether the model captures overall flow structure. Developmental progression reads left-to-right by construction; vertical spread is morphological variation at fixed stage.

### 12.2 Panel 2: Prediction Fan

For a single selected embryo:

- Context fragment as a solid line
- Fan of 50–100 forward-simulated SDE trajectories from the model (in the same $\phi_0$ vs residual-PC space)
- Kernel baseline's weighted point cloud overlaid
- Observed future (if available) marked

This shows whether the model's uncertainty is calibrated and whether it outperforms the baseline for individual cases. Should be generated for representative embryos from each test tier.

### 12.3 Panel 3: Phenotype Space

- Plot $c_e$ vectors for all embryos, projected into their first 2 PCs
- Colored by perturbation class
- Held-out novel perturbation embryos highlighted distinctly

This reveals whether the dynamical phenotype captures meaningful biological structure: do perturbation classes cluster? Are there subpopulations? Are novel perturbations placed sensibly relative to training classes?

### 12.4 Uncertainty Reporting

High-dimensional uncertainty is reported in two complementary ways:

- **Visual (2D projections)**: marginal uncertainty along the $\phi_0$ axis (temporal uncertainty: "when does it reach stage X?") shown separately from uncertainty orthogonal to $\phi_0$ (fate uncertainty: "which fate does it adopt?"). Displayed in the prediction fan panel.
- **Numerical (full 10D)**: log-determinant of predicted covariance at each horizon, broken out by test tier. Tracked over training as a scalar summary of model confidence. Compared across the three model tiers (kernel, $\phi_0$-only, full).

---

## 13. Hyperparameters Summary

| Hyperparameter | Description | Suggested Starting Point |
|----------------|-------------|--------------------------|
| $M$ | Number of modes | 3–5 |
| Hidden width | MLP width per layer | 32–64 |
| Hidden depth | MLP layers per potential | 2 |
| Activation | Smooth nonlinearity | Softplus |
| $\lambda_c$ | L2 penalty on mode loadings | Cross-validate |
| $\lambda_v$ | L2 penalty on local correction | $5\text{–}10 \times \lambda_c$ |
| $\sigma_w$ | Local correction decay width | Median pairwise distance |
| $\alpha_1$ | Mode gradient penalty weight | Tune |
| $\alpha_2$ | Mode potential power penalty weight | Tune |
| $\alpha_3$ | Hessian smoothness penalty weight | Tune |
| $\sigma$ (baseline) | Kernel bandwidth | Cross-validate |

---

## 14. Future Extensions (Tabled)

The following were discussed and deemed valuable but deferred in the interest of starting lean:

- **Underdamped Langevin (momentum)**: augment state with velocity to capture ballistic arcs. Doubles state dimension, adds damping parameter $\gamma$. Breaks closed-form $c$ solve. Pursue if first-order model fails to capture curved trajectories.
- **Diagonal diffusion**: replace scalar $D$ with per-dimension $D_i$. Minimal complexity increase, likely worthwhile if residual analysis shows anisotropic noise.
- **Time-varying potential**: allow landscape to deform over developmental time. Handles "same state, different stage, different future." Partially preserves closed-form structure if time dependence is simple.
- **Optimal transport predictive distance**: $d(z_a, z_b) = W_p(\mu_{z_a}, \mu_{z_b})$ for dynamically meaningful embryo comparison near bifurcations.

---

## 15. Implementation Notes for Claude Code

### 15.1 Framework

Plain PyTorch. No PyTorch Lightning initially — the non-standard forward pass (inner linear solves, alternating estimation, multi-step simulation) benefits from full transparency during debugging. Lightning can be introduced as a later refactor once core mechanics are validated.

### 15.2 Build Sequence

Implement and test in this order. Each step produces a runnable, evaluable artifact.

1. **Data loading and fragment sampling.** Define the data interface: per embryo, provide trajectory tensor, experiment-level $\Delta t$, temperature $T_e$, perturbation class label. Implement the random fragment and horizon sampling (§7.2).
2. **Evaluation and visualization stack.** Build the W&B logging, metric computation (§11), and three-panel visualization (§12) against dummy/random predictions. This infrastructure must exist before any model is trained.
3. **Kernel baseline.** Implement kernel trajectory regression (§9). Evaluate on all test tiers. This is the floor.
4. **$\phi_0$-only model (Stage 1).** Single baseline potential, $\beta$, $D$, $R_e$. No modes. Train and evaluate. Save checkpoint — this becomes the permanent $\phi_0$-only reference.
5. **Add modes with $S_m = 0$.** Freeze $\phi_0$. Introduce mode potentials, closed-form $[c_e; v_e]$ solve, class-level priors $c_{0,p}$. Train and evaluate (Stage 2). Compare to $\phi_0$-only and kernel baselines.
6. **Unfreeze $S_m$.** Allow non-conservative dynamics. Compare to potential-only modes to assess whether rotational dynamics matter.
7. **Add regularization terms incrementally.** Introduce $\mathcal{R}_1$, $\mathcal{R}_2$, $\mathcal{R}_3$ one at a time, monitoring impact on both performance and interpretability.

### 15.3 Key Implementation Constraints

- All per-embryo inference ($c_e$, $v_e$, $R_e$) must be batched and differentiable. Use `torch.linalg.solve` for the ridge systems.
- The toggle between potential-only ($S_m = 0$) and Helmholtz modes should be a single boolean flag that freezes/unfreezes $S_m$.
- The kernel baseline should be a standalone module sharing the same data loading and evaluation pipeline.
- The three-model comparison (kernel, $\phi_0$-only, full) must be trivially reproducible at any point during development.
