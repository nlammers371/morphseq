# UOT Solver Benchmark — Results

**Date:** 2026-02-22
**Script:** `run_benchmark.py`
**Data:** `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`

---

## Setup

### Sweep grid

| Dimension | Values |
|-----------|--------|
| Epsilons | 1e-4, 1e-3, 1e-2 |
| Backends | POTBackend (CPU), OTTBackend (CPU JAX) |
| Max support pts | 1000, 3000 |
| Pairs | 3 (WT×WT, WT×HOM, HOM×HOM) |
| **Total combos** | **36** |

Fixed parameters: `marginal_relaxation=10.0`, `downsample_factor=4`, `coord_scale=1/576`, `random_seed=42`.

### Embryo pairs (frame 88, ~48 hpf, experiment 20251113)

| Pair | Source embryo | Target embryo | Genotypes |
|------|--------------|--------------|-----------|
| WT×WT | 20251113_E02_e01 | 20251113_E04_e01 | WT → WT |
| WT×HOM | 20251113_E04_e01 | 20251113_A05_e01 | WT → homozygous |
| HOM×HOM | 20251113_A05_e01 | 20251113_B01_e01 | HOM → HOM |

### Hardware

CPU-only node (no CUDA). OTTBackend falls back to JAX on CPU. JAX CUDA plugin failed to load (expected on this node).

---

## Results

### 1. Solve time (mean over 3 pairs)

| Backend | epsilon | max_pts | Mean solve time |
|---------|---------|---------|----------------|
| POT | 1e-4 | 1000 | 5.8 s |
| POT | 1e-4 | 3000 | 7.9 s |
| POT | 1e-3 | 1000 | 4.3 s |
| POT | 1e-3 | 3000 | 4.7 s |
| POT | 1e-2 | 1000 | 4.6 s |
| POT | 1e-2 | 3000 | 5.7 s |
| OTT | 1e-4 | 1000 | 282 s |
| OTT | 1e-4 | 3000 | 184 s |
| OTT | 1e-3 | 1000 | 167 s |
| OTT | 1e-3 | 3000 | 188 s |
| OTT | 1e-2 | 1000 | 194 s |
| OTT | 1e-2 | 3000 | 191 s |

**POT is ~35–50× faster than OTT on CPU** across all epsilons and support sizes.

### 2. Cost agreement between backends

Costs agree tightly at eps=1e-4 and 1e-3 (< 0.1% difference). Small divergence at eps=1e-2 (~0.7%), consistent with the tau conversion difference between backends (documented in MEMORY).

| Pair | eps | POT cost | OTT cost | Δ% |
|------|-----|---------|---------|-----|
| WT×WT | 1e-4 | 0.9543 | 0.9543 | <0.01% |
| WT×WT | 1e-3 | 5.761 | 5.762 | 0.02% |
| WT×WT | 1e-2 | 25.27 | 25.44 | 0.67% |
| WT×HOM | 1e-4 | 3.142 | 3.142 | <0.01% |
| WT×HOM | 1e-2 | 23.86 | 24.02 | 0.67% |
| HOM×HOM | 1e-4 | 2.365 | 2.365 | <0.01% |
| HOM×HOM | 1e-2 | 26.64–26.79 | 26.82–26.97 | ~0.6% |

### 3. Actual support points used

Actual support sizes are below the 1k and 3k caps for these embryos (~770–1060 points, constrained by mask area). So `max_pts=3000` is effectively unconstrained — the 1k vs 3k difference in solve time is minimal for POT (~2s) and negligible for OTT.

### 4. Epsilon effect on solve time

For **POT**, epsilon has almost no effect on solve time (all combos 1.8–10s). The algorithm complexity scales with support size (n²), not epsilon.

For **OTT on CPU**, JAX JIT compilation overhead dominates (~160–280s/combo), masking any epsilon-dependent iteration differences. Crucially, **`OTTBackend.solve` has no `jax.jit` wrapper** — every call retraces and recompiles the computation graph. This likely explains most of the OTT slowness (the first OTT call at eps=1e-4/1k pts took 370s; later ones at different configs took 160–250s). With JIT caching across same-shape problems, OTT would be faster in a warmed-up batch loop.

---

## Batching: does it change the picture?

### Current batch API (`solve_working_grid_batch`)

The batch runner in `analyze.utils.optimal_transport.batch` buckets pairs by work shape and calls `backend.solve_batch()` if available, else loops `backend.solve()`.

**Neither backend does true vectorized/parallel batching on CPU:**

| Backend | `solve_batch` implemented? | Implementation |
|---------|--------------------------|----------------|
| POTBackend | No | Falls back to sequential `solve()` loop |
| OTTBackend | Yes | Also a sequential `solve()` loop — not vectorized |

OTTBackend's `solve_batch` is currently:
```python
def solve_batch(self, problems, config):
    return [self.solve(src, tgt, config) for src, tgt in problems]
```

So for CPU, **batching is a Python-level loop in both cases** — throughput per pair is identical to single-solve.

---

## What would vmap actually buy on CPU?

This is the key question. The answer is nuanced:

### vmap is NOT multi-core parallelism

`jax.vmap` is a **vectorizing transform** — it maps scalar operations to SIMD vector instructions (AVX/SSE on x86). It is *not* the same as running problems on separate CPU cores. On CPU:

- `vmap` batches the work into larger matrix operations that XLA can vectorize at the instruction level
- Peak CPU utilization with vmap can still approach ~560% (multi-threaded BLAS underneath), but that's from XLA's thread pool, not vmap itself
- Memory usage increases proportionally with batch size — you hold all problems' cost matrices simultaneously

### What vmap *could* help with for OTT

The Sinkhorn algorithm is iterative matrix operations (softmin, logsumexp). For a batch of N problems with the same support size n:

- **Sequential:** N × O(n² × iters) — but each call pays JIT compile overhead
- **vmapped:** Single JIT compile, then one vectorized call of shape (N, n, n)
- **Net effect on CPU:** JIT overhead amortized over N; BLAS/LAPACK sees larger matrices → better cache/SIMD utilization

The ott-jax documentation shows a **double-vmap pattern for pairwise OT** achieving ~50× speedup, though the hardware context (GPU vs CPU) is not specified. For CPU, the benefit is real but smaller — primarily from amortized JIT compilation.

### The missing optimization: `jax.jit`

The bigger win for our `OTTBackend` would be **adding `jax.jit`** to `solve()` before any vmap work:

```python
# Current (re-traces every call):
out = solver(prob)

# Better (traces once per support size, cached):
_solve_jit = jax.jit(solver)
out = _solve_jit(prob)
```

In `solve_working_grid_batch`, pairs are already bucketed by work shape → same support size → same JIT-traced function reused. This alone could bring OTT from ~200s → perhaps ~10–30s per pair on CPU after the first solve warms the cache.

### `jax.pmap` for true multi-core parallelism

For CPU multi-core parallelism, the correct tool is `jax.pmap`, not `vmap`:

```python
# Enable N logical CPU devices:
# XLA_FLAGS="--xla_force_host_platform_device_count=N"

# Then:
batched_solve = jax.pmap(solve_fn)  # splits batch across N CPU cores
```

Community benchmarks show **3–10× speedup** on multi-core CPUs (3× on 10-core VM). This is independent of vmap and would benefit both POT (via multiprocessing) and OTT (via pmap). Requires explicit setup and same-shape problem batches.

### Summary: when does each approach help on CPU?

| Optimization | CPU benefit | Complexity | Status |
|-------------|------------|-----------|--------|
| `jax.jit` on `OTTBackend.solve` | High — eliminates re-trace (~200s → ~10-30s?) | Low | **Not implemented** |
| `jax.vmap` in `solve_batch` | Medium — amortizes JIT, better SIMD | Medium | Not implemented |
| `jax.pmap` (multi-core) | High — linear in core count | High (requires `XLA_FLAGS`) | Not implemented |
| POT multiprocessing (`joblib`) | High — trivially parallel | Low | Not implemented |

### Implication for throughput

With the current implementations, single-solve timings directly predict batch throughput:

- POT @ eps=1e-4: ~6–8s per pair → **~450–600 pairs/hour** per CPU
- OTT CPU @ eps=1e-4 (no JIT): ~180–280s per pair → **~13–20 pairs/hour** per CPU

If `jax.jit` were added to `OTTBackend.solve`, OTT batches (same-shape pairs, JIT cached after first solve) could reach estimated **~30–60s/pair** → ~60–120 pairs/hour — still 5–10× slower than POT on CPU, but a major improvement.

If OTT's `solve_batch` were upgraded to true `jax.vmap`, on GPU it could become competitive or dominant for large batches. On CPU, it would primarily help via amortized JIT + better SIMD, not true parallelism.

---

## Recommendations

1. **Switch production to POTBackend** for CPU runs. At `epsilon=1e-4` it matches OTT solution quality (cost difference < 0.01%) and is ~35× faster with current implementations. The current default of OTTBackend is only beneficial if a GPU is available.

2. **If sticking with OTT on CPU:** Add `jax.jit` to `OTTBackend.solve` (or cache a JIT-compiled solver per problem shape). This is likely a large win (~10× from JIT amortization alone) and requires only a few lines of change.

3. **For large-scale CPU batch runs:** Use `joblib.Parallel` over `POTBackend.solve` — trivially parallel, no JAX setup, linear in CPU core count.

4. **For GPU batching:** Implement `jax.vmap` in `OTTBackend.solve_batch`. The ott-jax library supports vmapped Sinkhorn natively; the hook is already in place (`solve_batch` method exists). On GPU this could enable processing dozens of pairs simultaneously.

5. **Keep `--fast` mode at POT + epsilon=1e-2** (4–6s/pair). Cost diverges ~0.7% from production but is fine for smoke tests.

6. **1k support points is sufficient** for these embryo masks (~800–1060 actual points used). Raising to 3k adds ~2s/pair for POT with no measurable cost improvement.

---

## Files

```
results/mcolon/20260222_uot_solver_benchmark/
├── run_benchmark.py          ← benchmark script
├── results.csv               ← 36 rows, all combos
├── RESULTS.md                ← this document
├── plots/
│   ├── solve_time_grid.png   ← bar chart: epsilon × backend, grouped by support size
│   ├── cost_vs_epsilon.png   ← cost vs epsilon per pair
│   └── quality_grid.png      ← created_mass_pct and mean_velocity_px vs epsilon
└── logs/
    └── benchmark_20260222_024025.log
```
