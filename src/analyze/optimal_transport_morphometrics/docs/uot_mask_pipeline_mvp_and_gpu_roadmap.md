# UOT Mask Pipeline: CPU MVP First, GPU Acceleration Later (ott-jax Roadmap)

See also:
- `src/analyze/optimal_transport_morphometrics/docs/mvp_masterplan_uot.md` (implementation plan + naming/policy)
- `src/analyze/optimal_transport_morphometrics/docs/analysis_goals_transport_morphodynamics.md` (conceptual/biological goals)

This note explains the recommended implementation approach for **unbalanced optimal transport (UOT)** on **binary embryo/tissue masks**, prioritizing:
1) a fast, reliable **CPU MVP**
2) clean best practices (multiscale, warm starts, growth maps)
3) a low-friction upgrade path to **GPU acceleration** using **JAX + ott-jax**

---

## What “switch to JAX/ott-jax later” means

### JAX (the runtime)
JAX is a NumPy-like Python library that can:
- **JIT compile** numerical code (via XLA) into fast fused kernels
- run on **CPU / GPU / TPU**
- vectorize computations with **vmap**
- (optionally) compute gradients if needed

Key idea: JAX turns Python loops (like Sinkhorn iterations) into compiled programs that run efficiently on accelerators.

### ott-jax (the OT solver library)
ott-jax (OTT) is an **optimal transport library built on top of JAX** that provides:
- Sinkhorn solvers (including unbalanced variants)
- design patterns for speed: batching, compilation, accelerator-friendly ops
- a natural path to solving **many OT problems** efficiently

---

## Why implement CPU MVP first (before GPU)

Early risk is not speed, it’s correctness and modeling:
- how to define mass (0A/0B/0C)
- how to crop / align / canonicalize masks
- whether creation maps and flow fields match biological expectations
- which outputs are truly needed downstream

A CPU MVP lets you iterate quickly on these choices with minimal tooling friction.
Once behavior is correct, you upgrade the solver backend to GPU with minimal rewrite.

---

## MVP Architecture Principle: Keep the solver backend pluggable

Design the pipeline so everything except the solver is “backend-agnostic”:

- `qc_and_canonicalize_masks(...)`  (NumPy/SciPy)
- `build_multiscale_pyramid(...)`   (NumPy)
- `uot_solver.solve(mu, nu, params)`  ← pluggable backend

Backends:
- `POTBackend.solve(...)` for CPU MVP
- `OTTBackend.solve(...)` for GPU (JAX/ott-jax), batching, JIT

This keeps the biology pipeline stable while swapping only the numerical engine.

---

## CPU MVP Implementation Plan (recommended order)

### Step 1: Pure preprocessing pipeline (no OT yet)
Implement + unit test:
1) **QC**: largest component, remove islands, optional hole fill
2) **Canonicalize**: centroid shift (yolk-based centering later)
3) **Mass definition**: support 0A / 0B / 0C (physical mass, no per-frame normalization)
4) **BBox crop**: crop to union bbox with padding (biggest CPU speed lever)
5) **Pyramid**: create downsampled levels (e.g. 1/8, 1/4, 1/2, 1)

Output: `(mu_t[level], mu_t1[level])` per level + transform metadata.

### Step 2: UOT at low resolution first
Run UOT on a coarse level (e.g. 1/8 or 1/4):
- verify distance values behave sensibly
- compute creation/destruction maps
- compute a coarse barycentric flow field

This step validates the growth logic fast.

### Step 3: Add multiscale refinement
Solve coarse → fine:
- fewer iterations at fine scales
- early stopping tolerance
- warm-start where possible:
  - across pyramid levels
  - across consecutive time steps (t→t+1 uses init from (t-1→t))

Even without explicit access to dual potentials, multiscale + time warm starts usually yields big gains.

### Step 4: Avoid all-pairs OT by policy
To scale beyond a few embryos:
- primary mode: consecutive frames only
- secondary mode: compare to **k archetypes** (small k)
- optional: neighbor graph based on cheap descriptors (IoU/boundary metrics/sliced OT) before running full UOT

This prevents O(B²) explosions.

---

## CPU Best Practices That Matter Immediately

### 1) Always crop to union bounding box + padding
Before any pyramid/OT:
- compute bbox of union of the two masks
- pad by a scale-appropriate margin
This often reduces compute by orders of magnitude.

### 2) Use contiguous arrays, control dtype
- `np.ascontiguousarray(...)`
- start with float32; if instability occurs, use float64 at coarse scales only

### 3) Parallelize across frame pairs
If you run many independent solves:
- multiprocessing/joblib parallelism over pairs is simple and effective

### 4) Cache expensive pieces
- distance transforms (0C)
- pyramids when reused
- bbox metadata

---

## Outputs to implement in MVP (core biological utility)

Given transport plan/coupling `π` and source/target densities `a(x), b(y)`:

### A) Creation / destruction maps (growth signal)
- **Creation at target**:
  - `create(y) = max(0, b(y) - sum_x π(x,y))`
- **Destruction at source**:
  - `destroy(x) = max(0, a(x) - sum_y π(x,y))`

### B) Flow field (barycentric projection)
- `T(x) = (sum_y y * π(x,y)) / (sum_y π(x,y))`
- `v(x) = T(x) - x`
Guardrail: if `sum_y π(x,y)` is tiny, avoid division blowups (mask or eps floor).

### C) Summary scalars per frame pair
- transported mass fraction
- created mass fraction
- destroyed mass fraction
- mean transport distance (on transported mass)

---

## Phase 2: GPU Acceleration With JAX + ott-jax

Once CPU behavior is validated:
- implement `OTTBackend.solve(...)` using JAX arrays
- batch many frame pairs:
  - use `vmap` to vectorize across pairs
  - `jit` compile the solver for speed

Why this works well for your workload:
- many repeated, similarly shaped solves (frame-to-frame) benefit from batching
- JIT removes Python overhead from Sinkhorn iterations
- GPU excels at large dense tensor ops when batched

---

## Recommended MVP Target (v0.1)

1) Input: two binary masks A and B (frame t and t+1)
2) Crop bbox of A∪B with padding
3) Align by translation (centroid)
4) Mass mode: support 0A and 0C (+ 0B as optional)
5) Multiscale pyramid (1/8, 1/4, 1/2; full-res optional)
6) CPU UOT solve per level + refinement
7) Output:
   - distance
   - creation/destruction maps
   - barycentric flow (at finest computed level)
   - summary scalars

---

## Notes on when GPU is worth it
GPU tends to help most when:
- you can batch many solves at once (many frame pairs)
- cropped grids are still large enough to saturate device
- you’ve already removed avoidable work via bbox cropping + multiscale
