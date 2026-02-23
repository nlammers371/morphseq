# Resampling Framework

Unified engine for bootstrap, subsample, and permutation resampling with
deterministic `SeedSequence`-based RNG, unit-aware perturbation, streaming
reducers, and post-hoc aggregation.

> **SeedSequence clean break.** This framework uses `numpy.random.SeedSequence`
> for all RNG derivation. Old resampling code and this framework will produce
> **different** (but equally valid) null distributions for the same seed.
> This is intentional — it buys deterministic, fork-safe seeding at the cost
> of bit-exact reproducibility with legacy code.

---

## Import convention

```python
from analyze.utils.resampling import resample

spec    = resample.indices(replacement=False, frac=0.8)
stat    = resample.statistic("silhouette", scorer_fn)
out     = resample.run(data, spec, stat, n_iters=100, seed=42)
summary = resample.aggregate(out)
```

All public symbols live in the `resample` namespace. Internal modules
(`_spec`, `_engine`, etc.) are not part of the public API.

---

## Quick start

### 1. Groups permutation test (two-sample)

Test whether two point clouds have the same distribution using energy distance.

```python
from analyze.utils.resampling import resample

def energy_distance(data, rng):
    from scipy.spatial.distance import cdist
    X1, X2 = data["X1"], data["X2"]
    d12 = cdist(X1, X2).mean()
    d11 = cdist(X1, X1).mean()
    d22 = cdist(X2, X2).mean()
    return 2 * d12 - d11 - d22

spec = resample.permute_groups(a="X1", b="X2")
stat = resample.statistic("energy_distance", energy_distance, is_nonnegative=True)
out  = resample.run(data={"X1": X1, "X2": X2}, spec=spec, stat=stat,
                    n_iters=999, seed=42)
summary = resample.aggregate(out)
print(f"p = {summary.pvalue:.4f}")
```

### 2. Label permutation test (within strata)

Test whether labels are associated with a response, permuting within time bins.

```python
from analyze.utils.resampling import resample
from scipy.stats import pearsonr

def correlation(data, rng):
    return pearsonr(data["early"], data["labels"])[0]

spec = resample.permute_labels()               # no strata needed here
out  = resample.run(
    data={"labels": late_means, "early": early_means},
    spec=spec, stat=correlation, n_iters=999, seed=42,
    alternative="two-sided",
)
summary = resample.aggregate(out)
```

### 3. Bootstrap subsample with CI

Subsample 80% of embryos without replacement, compute silhouette, get a CI.

```python
from analyze.utils.resampling import resample

def silhouette(data, rng):
    idx = data["indices"]
    D_sub = data["D"][np.ix_(idx, idx)]
    labels_sub = data["cluster_labels"][idx]
    return silhouette_score(D_sub, labels_sub, metric="precomputed")

spec = resample.subsample(frac=0.8)
out  = resample.run(
    data={"n": len(D), "D": D, "cluster_labels": cluster_labels},
    spec=spec, stat=silhouette, n_iters=200, seed=42,
)
summary = resample.aggregate(out)
print(f"observed = {summary.observed:.3f}")
print(f"95% CI   = [{summary.ci_low:.3f}, {summary.ci_high:.3f}]")
```

---

## API reference

### Spec factories

| Function | Returns | Description |
|----------|---------|-------------|
| `indices(*, replacement, size=, frac=, unit=, within=)` | `ResampleSpec` | Low-level index draw spec |
| `labels(*, within=, unit=)` | `ResampleSpec` | Label-permutation spec |
| `groups(*, a="X1", b="X2")` | `ResampleSpec` | Pool-and-redistribute spec |

### Semantic aliases

| Alias | Equivalent |
|-------|------------|
| `bootstrap(*, size=, frac=, unit=, within=)` | `indices(replacement=True, ...)` |
| `subsample(*, size=, frac=, unit=, within=)` | `indices(replacement=False, ...)` |
| `permute_labels(*, within=, unit=)` | `labels(...)` |
| `permute_groups(*, a=, b=)` | `groups(...)` |

### Statistic wrapper

```python
resample.statistic(
    name: str,
    fn: Callable[[dict, Generator | None], scalar | ndarray | dict],
    *,
    description: str = None,
    outputs: list[str] = None,
    default_alternative: str = None,   # "greater" | "less" | "two-sided"
    is_nonnegative: bool = None,
) -> Statistic
```

Raw callables are also accepted — `run()` auto-wraps them into a `Statistic`.

### Engine

```python
resample.run(
    data: dict,
    spec: ResampleSpec,
    statistic: Statistic | Callable,
    *,
    n_iters: int,
    seed: int,
    n_jobs: int = 1,
    store: str = "all",            # "all" | "none"
    reducer = None,                # e.g. PermutationReducer
    max_retries_per_iter: int = 0,
    verbose: bool = False,
    alternative: str = None,
) -> ResampleRun
```

**`ResampleRun` fields:** `observed`, `samples`, `reducer_state`, `n_success`,
`n_failed`, `spec`, `statistic`, `seed`, `resolved_alternative`, `diagnostics`.

### Aggregation

```python
resample.aggregate(
    out: ResampleRun,
    *,
    alpha: float = 0.05,
    ci_method: str = "percentile",
    alternative: str = None,
) -> BootstrapSummary | PermutationSummary
```

Dispatches on `out.spec.kind`:
- **`indices`** → `BootstrapSummary` (mean, se, ci_low, ci_high, ci_method, ci_is_exact)
- **`labels` / `groups`** → `PermutationSummary` (observed, pvalue, null_distribution,
  null_mean, null_std, alternative, n_permutations, statistic_name)

`PermutationSummary.to_permutation_result()` converts to the legacy
`PermutationResult` namedtuple for backward compatibility.

### Preflight

```python
resample.preflight(data, spec, statistic, *, n_iters=1, store="all", reducer=None)
```

Called automatically by `run()`. Validates data keys, population size,
unit/within consistency, memory estimates, and reducer compatibility.

### Streaming reducer

```python
PermutationReducer(*, alternative="two-sided", reservoir_size=200)
```

Exact streaming p-value without storing the full null distribution.
Uses Welford's algorithm for running mean/variance and Algorithm R for
reservoir sampling. Scalar outputs only.

---

## Data bundle convention

The `data` dict carries all inputs. Expected keys depend on `spec.kind`:

### `kind="indices"` (bootstrap / subsample)

| Key | Required | Description |
|-----|----------|-------------|
| `"n"` | Yes (unless `"labels"` present) | Population size |
| `"labels"` | No | Also used for N if `"n"` absent |
| `"indices"` | Injected | Row indices drawn by engine — scorer reads this |
| `"{unit_key}"` | If `unit=` set | Unit membership column |
| `"{within_key}"` | If `within=` set | Stratum membership column |

### `kind="labels"` (label permutation)

| Key | Required | Description |
|-----|----------|-------------|
| `"labels"` | Yes | Array to be shuffled |
| `"{within_key}"` | If `within=` set | Shuffle within strata |
| `"{unit_key}"` | If `unit=` set | One label per unit, broadcast back |

Other keys (features, covariates) pass through untouched.

### `kind="groups"` (pool-and-redistribute)

| Key | Required | Description |
|-----|----------|-------------|
| `"{a_key}"` | Yes (default `"X1"`) | First group array |
| `"{b_key}"` | Yes (default `"X2"`) | Second group array |

The engine concatenates, permutes, and re-splits along axis 0.

---

## Alternative resolution chain

The p-value tail is resolved **once** at the start of `run()`, in this priority:

1. **Caller** — `run(..., alternative="greater")` wins unconditionally
2. **Statistic default** — `statistic(default_alternative="greater")`
3. **Nonneg inference** — if `is_nonnegative=True` → `"greater"`
4. **Fallback** — `"two-sided"`

A warning is emitted if the resolved alternative is `"two-sided"` but the
statistic is marked `is_nonnegative=True` (symmetric-null assumption unlikely).

---

## Default alternative mapping

Known statistics and their natural defaults:

| Statistic | `is_nonnegative` | `default_alternative` | Rationale |
|-----------|-------------------|----------------------|-----------|
| Energy distance | `True` | `"greater"` | Distance ≥ 0; larger = more different |
| MMD | `True` | `"greater"` | Kernel distance ≥ 0 |
| Silhouette | `False` | — | Can be negative; bootstrap CI, not p-value |
| AUROC | `False` | `"greater"` | 0.5 = chance; higher = better separation |
| Pearson *r* | `False` | `"two-sided"` | Can be positive or negative |

---

## P-value formula

Permutation p-values use +1 smoothing:

```
p = (1 + k) / (B + 1)
```

where `k` = number of null values at least as extreme as observed, and
`B` = number of successful permutations. This guarantees `p ∈ (0, 1]`
and is the recommended estimator (Phipson & Smyth, 2010).

**Migration note:** legacy code uses `k / B` (no smoothing). P-values will
shift slightly — by at most `1/B` — toward the conservative direction.

---

## Migration map

Existing resampling loops and their framework equivalents:

| PR | Source function | File | Spec | Status |
|----|----------------|------|------|--------|
| 2 | `permutation_test_distribution()` | `difference_detection/distribution_test.py` | `permute_groups(a="X1", b="X2")` | TODO |
| 3 | `test_anticorrelation()` | `trajectory_analysis/utilities/correlation.py` | `permute_labels()` | TODO |
| 4 | `_permutation_test_ovr()` | `difference_detection/classification_test_multiclass.py` | `permute_labels(within="time_strata")` | TODO |
| 5 | `run_bootstrap_hierarchical()` / `run_bootstrap_kmedoids()` | `trajectory_analysis/clustering/bootstrap_clustering.py` | `subsample(frac=frac)` | TODO |
| 6 | `run_bootstrap_projection()` | `trajectory_analysis/clustering/bootstrap_clustering.py` | `subsample(frac=frac)` | TODO |
| 7 | `_fit_single_spline()` | `spline_fitting/bootstrap.py` | `bootstrap(size=bootstrap_size)` | TODO |
| 8 | Legacy duplicates | `functions/spline_fitting_v2.py`, `functions/improved_build_splines.py` | Deprecation warnings | TODO |

---

## Module structure

```
src/analyze/utils/resampling/
├── __init__.py          # Public API (factories, aliases, re-exports)
├── _spec.py             # ResampleSpec, IndicesParams, LabelsParams, GroupsParams
├── _statistic.py        # Statistic dataclass
├── _engine.py           # run(), ResampleRun, SeedSequence derivation
├── _perturbation.py     # _perturb() dispatch (indices/labels/groups)
├── _aggregator.py       # aggregate(), BootstrapSummary, PermutationSummary
├── _reducer.py          # PermutationReducer (streaming p-value)
├── _preflight.py        # preflight() validation
├── README.md            # This file
└── tests/
    ├── __init__.py
    ├── test_perturbation.py
    ├── test_preflight.py
    ├── test_aggregator.py
    ├── test_engine.py
    └── test_integration.py
```
