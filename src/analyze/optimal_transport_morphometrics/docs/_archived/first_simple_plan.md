Here’s a quick, concrete MVP plan you can follow that matches the modular structure we discussed. It’s CPU-first, fast-to-iterate, and makes GPU swapping later painless.

---

## MVP Implementation Plan (CPU-first)

### Phase 0: Skeleton + configs (half-day)

**Goal:** lock in a clean API so experiments don’t sprawl.

* Create a small config object/dict:

  * `mass_definition: {0A,0B,0C}`
  * `align_mode: {none, centroid, yolk_centroid}`
  * `qc_mode: {minimal}`
  * `multiscale_levels: [1/8, 1/4, 1/2]` (start without full-res)
  * `uot_params: {epsilon_schedule, reg_m}` (names depend on solver)
  * `bbox_padding_px: int`

* Define the core dataclasses (or dicts):

  * `MaskPair(A, B, meta)`
  * `CanonicalizedPair(levels=[...], transforms={...})`
  * `UOTResult(distance, plan, flow, creation, destruction, summaries)`

---

## Implementation Structure (modules + responsibilities)

### 1) `preprocess_masks.py`

**Functions**

1. `qc_mask(mask, qc_mode) -> mask`

   * largest component
   * remove islands
   * optional hole fill

2. `crop_union_bbox(maskA, maskB, pad_px) -> (A_crop, B_crop, bbox_meta)`

   * **do this first** to shrink compute

3. `align_masks(A, B, mode, yolk_mask=None) -> (A_aligned, B_aligned, transform_meta)`

   * MVP: centroid translation
   * Optional: yolk centroid when available

**Outputs**

* cropped + aligned binary masks
* bbox + transform metadata

---

### 2) `densities.py`

**Goal:** implement 0A/0B/0C cleanly and deterministically.

* `make_density(mask, mode, params) -> mu`

  * 0A: `mu = mask.astype(float)`
  * 0B: boundary band (distance-to-boundary ≤ r)
  * 0C: distance transform mapped by chosen transform

**Important**

* Keep “physical mass”: do **not** normalize across frames.
* Track `mass = mu.sum()` for reporting.

---

### 3) `pyramid.py`

**Goal:** multiscale by default.

* `build_pyramid(mu, levels) -> list[mu_level]`

  * downsample with area-preserving aggregation (sum pooling preferred over mean)
  * keep track of pixel size per level (for interpreting ε in pixels)

---

### 4) `uot_backend_pot.py` (first backend)

**Goal:** one function that takes two densities and returns coupling + cost.

* `solve_uot(mu, nu, uot_params) -> (pi, cost, aux)`

  * Use POT unbalanced Sinkhorn
  * Start at coarse level only; add multiscale loop next

**Notes**

* If full coupling `pi` is too big at some level, allow:

  * `return_pi=False` mode (distance only), but for MVP keep coupling at coarse levels at least.

---

### 5) `postprocess.py`

**Goal:** produce your biological readouts from the plan.

* `compute_marginals(pi) -> (mu_hat, nu_hat)`
* `creation_map = clip(nu - nu_hat, 0, inf)`
* `destruction_map = clip(mu - mu_hat, 0, inf)`
* `barycentric_flow(pi) -> T(x), v(x)` (with eps floor on denominators)
* `summaries`:

  * transported mass = `pi.sum()`
  * created mass = `creation_map.sum()`
  * destroyed mass = `destruction_map.sum()`
  * mean transport distance (if cost available)

---

## Phase 1: Make it work at one scale (1–2 days)

**Goal:** get a correct result for (t → t+1) on a few examples.

1. Run the pipeline at **1/8** or **1/4** resolution only
2. Save outputs as images:

* overlay masks
* flow quiver plot
* creation/destruction heatmaps

3. Confirm qualitative sanity:

* identity: A→A gives ~zero flow, minimal creation/destruction
* growth: dilation produces creation near boundary

---

## Phase 2: Add multiscale refinement (1–2 days)

**Goal:** speed + nicer flows.

* Loop levels coarse → fine
* Use a schedule:

  * more iterations at coarse, fewer at fine
  * ε coarse larger, ε fine smaller
* Warm-start:

  * simplest: reuse previous level’s solution statistics to initialize (even if not true dual warm start)
  * also warm-start across time steps (t-1→t init for t→t+1)

For MVP, you can stop at 1/2 resolution and only go full-res when needed.

---

## Phase 3: Scale across embryos (1–2 days)

**Goal:** many transitions without melting your laptop.

* Parallelize across frame pairs (joblib/multiprocessing)
* Cache:

  * QC’d masks
  * bbox crops
  * distance transforms (0C)
  * pyramids per frame

Add a “policy”:

* default: consecutive frames only
* optional: compare to k archetypes

---

## Phase 4: GPU upgrade path (later)

Once you like the biology:

* add `uot_backend_ottjax.py`
* keep all preprocessing identical
* replace only `solve_uot()`
* use batching (`vmap`) across many frame pairs

---

## Minimal file tree

```
uot_masks/
  preprocess_masks.py
  densities.py
  pyramid.py
  uot_backend_pot.py
  postprocess.py
  viz.py
  run_pair.py
  run_timeseries.py
  config.py
```

---

If you want, I can turn this into:

* a tiny `run_pair.py` CLI spec (input paths, output folder, config yaml),
* and a “golden tests” list (identity/shift/dilation) so you can lock correctness before optimizing.
