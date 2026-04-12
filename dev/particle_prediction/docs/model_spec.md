# MorphSeq Beta Predictive Model: Technical Specification

## 1. Overview

This document specifies the **beta forward-prediction model** for morphogenetic trajectories in latent space.

The model is intentionally lightweight:

- it uses the existing latent representation as the hard part,
- it performs no heavy end-to-end dynamical training in v1,
- it produces **probabilistic, multimodal forecasts** by sampling local empirical transitions from a reference bank,
- and it is designed to be implemented quickly and debugged visually.

The design goal is not to recover a global mechanistic dynamical system. The goal is to build a practical forecasting layer that can answer:

> Given a morphology snapshot or recent morphology fragment, what morphologies tend to come next?

---

## 2. Scope and non-goals

### In scope for v1

- Load latent trajectories and metadata.
- Smooth noisy latent trajectories.
- Reparameterize trajectories by arc length.
- Build a bank of local transition windows from smoothed, resampled references.
- Forecast forward using weighted empirical transition sampling.
- Support either:
  - snapshot-conditioned prediction, or
  - history-conditioned prediction with a tunable recent-history window.
- Return multimodal predictions via particles / samples.
- Provide visual debugging tools for every core step.

### Explicitly out of scope for v1

- A formal hidden developmental-stage model.
- A global ODE or potential-field model driving the forecasts.
- Absolute hpf prediction inside the forecast kernel.
- A learned neural transition model.
- A full interactive app.
- A formal class-likelihood framework.

Those may be added later, but none are prerequisites for the beta predictor.

---

## 3. Problem setting

### Observed data

Each embryo is represented as a trajectory in latent space:

\[
z(t) \in \mathbb{R}^D
\]

with metadata including at minimum:

- `embryo_id`
- `experiment_id`
- `perturbation_class`
- `temperature`
- observation times `time_seconds`
- experiment-level median frame spacing `delta_t`

The latent coordinates may be noisy. Sampling is regular within an experiment but may differ across experiments.

### Forecast task

Given either:

- a single latent point \(z_0\), or
- a recent latent fragment \(z_{-K}, \dots, z_0\),

predict a distribution over future morphologies after one or more fixed morphological-progress steps.

---

## 4. Core design decisions

### 4.1 Forecast in morphology space, not clock time

The predictive core is **not parameterized by absolute time**.

Instead, each trajectory is smoothed and reparameterized by cumulative arc length \(s\), then resampled at fixed arc-length increments \(\Delta s\). One model step therefore means:

> advance by one fixed increment of morphological change.

This avoids entangling forecasting with variable developmental tempo across embryos and experiments.

### 4.2 Keep stage / hpf separate from forecasting

A scalar stage field may be fit later as a post hoc annotation model, but it is not part of the predictive kernel in v1.

### 4.3 History is for matching, not for state evolution

The recent history window is used only to improve reference-transition selection. It is not treated as an explicit hidden state model in v1.

### 4.4 Visualization is mandatory

Every stage of the model must be inspectable:

- smoothing,
- resampling,
- reference matching,
- local transition distributions,
- rollout behavior,
- and failure modes.

---

## 5. Preprocessing

## 5.1 Trajectory smoothing

Raw latent trajectories are smoothed before any increments are extracted.

### Default method: Savitzky–Golay smoothing

For each trajectory and each latent dimension independently, apply a Savitzky–Golay (SG) smoother.

The SG window is specified in **time units**, not frames, so that the smoothing length scale is consistent across experiments with different frame spacing.

Let:

- `window_seconds` be the smoothing time scale,
- `delta_t_exp` be the experiment-level frame spacing in seconds.

Then:

\[
L_{\text{frames}} = \text{odd\_round}\left(\frac{\text{window\_seconds}}{\Delta t_{\text{exp}}}\right)
\]

subject to:

- odd integer length,
- minimum length consistent with the polynomial order,
- clipped to trajectory length when needed.

### SG defaults

Recommended initial defaults:

- polynomial order: 2 or 3
- window specified in seconds, chosen to span roughly 5–9 frames in a typical experiment

### Output of smoothing

At minimum, smoothing should return:

- smoothed latent positions \(\tilde z(t)\)
- residual diagnostics comparing raw and smoothed trajectories

Derivatives are optional in v1 and not required for the transition kernel.

---

## 5.2 Arc-length parameterization and resampling

After smoothing, define cumulative arc length:

\[
s_0 = 0,\quad
s_n = \sum_{i=1}^{n} \left\lVert \tilde z(t_i) - \tilde z(t_{i-1}) \right\rVert
\]

Each trajectory is then resampled onto a fixed arc-length grid:

\[
0,\ \Delta s,\ 2\Delta s,\ \dots
\]

using linear interpolation along the smoothed curve.

### Why arc length

This creates a common morphology-progress coordinate without requiring a separate developmental-stage model.

### Output of resampling

Each resampled trajectory should contain:

- resampled latent states \(z_s\)
- cumulative arc-length coordinates
- mapping back to original time / frame coordinates when needed for inspection
- metadata inherited from the source trajectory

---

## 6. Transition windows

The core modeling object is a **transition window**.

For each resampled point \(z_i\), define:

- current state \(z_i\)
- optional recent history \(H_i\)
- next-step increment:

\[
\Delta z_i = z_{i+1} - z_i
\]

### Canonical history representation

History is stored as the last \(K\) ordered resampled segments:

\[
H_i = [\delta_{i-K+1}, \dots, \delta_i]
\quad\text{where}\quad
\delta_j = z_j - z_{j-1}
\]

This keeps ordinal information while remaining progression-agnostic.

### Optional fast history summary

For faster discovery and weighting, each transition window may also store a compressed history summary based on the previous \(K\) points:

- mean recent position
- mean recent direction
- optional total recent displacement

The compressed summary is not the canonical history object. It is an optional speedup / approximate matching method.

### Metadata stored with each transition window

Each window must retain at least:

- source embryo ID
- source experiment ID
- perturbation class
- source resampled index
- local arc-length coordinate
- optional original time estimate for visualization only

---

## 7. Query modes

The model supports two query types.

### 7.1 Snapshot mode

Input is a single latent point \(z_0\).

Matching uses current position only.

### 7.2 History mode

Input is a recent observed fragment, which is first smoothed and resampled in the same way as the references. Matching uses:

- current position,
- optional class priors,
- and history-aware reranking.

History mode is the preferred mode when enough recent observations are available.

---

## 8. Reference matching

Matching is performed in two stages.

## 8.1 Stage 1: local candidate discovery

Use the current query position \(z_q\) to retrieve a local pool of candidate transition windows.

A simple default is Euclidean nearest neighbors in latent space, optionally gated by a maximum search radius.

This stage is intentionally cheap and broad.

## 8.2 Stage 2: history-aware reranking

History is used only to rerank the local candidate pool.

### Default method: ordered segment-window matching with small offset tolerance

For the query history \(H_q\) and a candidate history \(H_i\), compute a history mismatch over ordered segments.

To avoid relying on absolute segment index, each candidate is allowed a small local offset band:

\[
\ell \in [-r, r]
\]

and the candidate’s history score is defined by the best local alignment:

\[
d_{\text{hist}}(i) =
\min_{\ell \in [-r,r]}
d\big(H_q,\ H_{i,\ell}\big)
\]

A simple default history distance is:

\[
d^2(H_q, H_{i,\ell}) =
\sum_{m=0}^{K-1} \alpha_m
\left\lVert \delta^{(q)}_{-m} - \delta^{(i,\ell)}_{-m} \right\rVert^2
\]

with larger weights on more recent segments.

This method:

- preserves ordinal information inside the window,
- avoids dependence on absolute stage or absolute segment number,
- and is much cheaper than global whole-trajectory alignment.

### Optional fast method: summary-position-and-direction matching

For faster discovery and weighting, an alternative approximate matcher may compare:

- current position,
- mean recent position,
- mean recent direction,

computed over the previous \(K\) points.

This method is acceptable for coarse filtering or speed-sensitive runs, but the ordered segment method is the default reference implementation.

## 8.3 Optional class priors

If desired, candidate weights may be multiplied by a soft class-prior term:

\[
w_{\text{class}}(c_q, c_i)
\]

This must be optional and configurable. Hard class exclusion is not the default.

---

## 9. Transition kernel

The transition kernel is an empirical mixture over local candidate windows.

For a query state \(q\), let the final weight of candidate \(i\) be:

\[
w_i \propto
w_{\text{pos}}(i)\,
w_{\text{hist}}(i)\,
w_{\text{class}}(i)
\]

with normalization over the candidate set.

A simple default positional term is:

\[
w_{\text{pos}}(i) =
\exp\left(
-\frac{\|z_q - z_i\|^2}{2\sigma_z^2}
\right)
\]

and the history term is:

\[
w_{\text{hist}}(i) =
\exp\left(
-\frac{d_{\text{hist}}(i)^2}{2\sigma_h^2}
\right)
\]

### Sampling rule

Sample a candidate increment \(\Delta z_i\) from the weighted candidate set, then propagate:

\[
z_{q,\text{next}} = z_q + \Delta z_i + \epsilon
\]

where \(\epsilon\) is a small jitter term.

---

## 10. Jitter model

The beta model adds small Gaussian jitter to prevent predictions from collapsing onto a purely discrete set of reference increments.

### Default jitter: tangent-aligned anisotropic Gaussian

Let:

\[
u_i = \frac{\Delta z_i}{\|\Delta z_i\|}
\]

Then define:

\[
\Sigma_i =
\sigma_{\parallel}^2 u_i u_i^\top
+
\sigma_{\perp}^2 (I - u_i u_i^\top)
\]

with:

- small \(\sigma_{\parallel}\)
- smaller or equal \(\sigma_{\perp}\)

and sample:

\[
\epsilon \sim \mathcal N(0, \Sigma_i)
\]

### Fallback debug mode

A simple isotropic jitter mode should also be available:

\[
\epsilon \sim \mathcal N(0, \sigma^2 I)
\]

This is mainly for debugging or ablation.

---

## 11. Forecast algorithm

The predictive engine is a particle-based rollout over the local transition kernel.

### 11.1 Initialization

Initialize \(N\) particles at the final observed state \(z_0\).

If history is available, each particle also carries the recent query history window.

### 11.2 Step update

For each particle at each step:

1. retrieve local candidate transition windows using current position
2. rerank using history-aware matching
3. sample one candidate increment
4. add jitter
5. advance the particle
6. update the particle’s recent history window

### 11.3 Outputs

At each forecast horizon, report:

- particle cloud
- predictive mean
- diagonal covariance or full covariance if cheap
- simple support diagnostics:
  - number of local candidates
  - effective sample size / weight entropy
  - local match radius
  - history mismatch of selected candidates

---

## 12. Graceful failure

The model is intended to be mostly interpolative.

When local support is poor, it should fail gracefully by:

- widening the predictive spread,
- exposing support diagnostics,
- and avoiding false precision.

A forecast should be flagged as low-support when, for example:

- too few local candidates are found,
- the required search radius becomes large,
- candidate weights become diffuse,
- or history mismatch remains high across all candidates.

---

## 13. Evaluation philosophy

The beta model is evaluated with **simple geometric forecast accuracy**, not formal likelihoods.

Primary evaluation focuses on:

- one-step next-state error
- multi-step rollout error
- visual inspection of prediction fans and local transition clouds

Detailed evaluation procedures are specified in `evaluation_plan.md`.

---

## 14. Visualization requirements

Visualization is part of the model specification, not an optional add-on.

The implementation must support inspection of:

- raw vs smoothed trajectories
- arc-length resampling
- local candidate retrieval
- history-reranking behavior
- sampled transition kernels
- particle forecast fans
- rollout error versus horizon

Detailed requirements are specified in `visualization_spec.md`.

---

## 15. Deferred next steps

The following are explicitly deferred:

- post hoc stage / hpf field
- learned progression models
- formal class scoring
- interactive prediction app
- learned parametric transition densities

They are compatible with this architecture but not required for the beta build.
