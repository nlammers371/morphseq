# Classification API — Agreed Design Spec

Status: approved front-end spec for the next implementation phase.

This document is the canonical reference for the new `run_classification()` API and its comparison
resolution contract. It supersedes the previous conversation dump.

---

## Guiding principles

1. **DataFrame-first for the main dataset.** The primary data input is always a plain
   `pd.DataFrame` (no wrapper object required). Comparison specs may be passed in convenient
   Python forms and are normalized immediately.
2. **Multi-feature is the native unit of thought.** `features={}` dict is always named; the
   results table always has a `feature_set` column.
3. **One entry point.** `run_classification()` covers all modes. No `run_ovr()`, `run_pairwise()` siblings.
4. **Layered signature.** Data contract → comparison spec → features → model/binning → output.
5. **Fail fast.** All validation happens before any computation.
6. **Symmetric pooling.** Pooled positives and pooled negatives work identically at the
   binary-label level. `tuple` = pooled, `list` = enumerate, `str` = single.

---

## Public entry point

```python
def run_classification(
    # ── Layer 1: data contract (required, keyword-only after df) ──────────
    df: pd.DataFrame,
    *,
    class_col: str,                  # column holding class labels
    id_col: str,                     # embryo/unit identity column
    time_col: str,                   # continuous time column for binning

    # ── Layer 2: comparison spec ──────────────────────────────────────────
    # Simple path — explicit positive / negative
    positive: UserComparisonSpec | None = None,
    negative: UserComparisonSpec | None = None,

    # Scheme / advanced path — overrides simple path
    # str scheme accepts positive= as a class-scope LIST filter only
    comparisons: ComparisonScheme = None,
    # Accepts: "all_vs_rest", "all_pairs", pd.DataFrame, list[dict], or None
    # list[dict] rows are normalized to DataFrame immediately at ingestion

    # ── Layer 3: features (always named, always a dict) ───────────────────
    features: dict[str, str | list[str]],   # {"name": "prefix_or_col_list"}

    # ── Layer 4: model / binning ──────────────────────────────────────────
    bin_width: float = 4.0,
    n_permutations: int = 100,
    n_splits: int = 5,
    min_samples_per_group: int = 3,  # group-level minimum (per side of comparison)
    min_samples_per_member: int = 2, # per-constituent minimum in pooled groups
    n_jobs: int = 1,                 # execution strategy is implementation-defined
    random_state: int = 42,          # root seed; child seeds derived for CV, estimator, permutations

    # ── Layer 5: output / control ─────────────────────────────────────────
    verbose: bool = True,
    save_predictions: bool = False,             # tidy binary predictions.parquet
    save_multiclass_predictions: bool = False,  # wide multiclass_predictions.parquet (all-vs-rest only)
    save_null_arrays: bool = False,             # raw per-permutation arrays; summary stats always in scores
    # save_multiclass_predictions defaults to False because the wide table can be
    # large and is only needed by the misclassification pipeline. Users who run
    # run_misclassification_pipeline() should set this to True explicitly.
) -> ClassificationAnalysis:
```

### `random_state` contract

`random_state` is a root seed. The implementation derives deterministic child
seeds from it — one per stochastic component — rather than sharing a single
mutable RNG object. This keeps reproducibility clean even under parallel execution.

Child seeds must be threaded into:
- CV splitter (`StratifiedKFold(random_state=...)`)
- Estimator (`LogisticRegression(random_state=...)`)
- Permutation RNG (per-permutation seed derived from root)

Derived seeds use a deterministic scheme (e.g. `SeedSequence.spawn()` or
`root + offset`) so that results are identical regardless of `n_jobs`.

### `features` type narrowing

The public API accepts `dict[str, str | list[str]]` for convenience:

```python
features={"embedding": "z_mu_b", "shape": ["length", "width"]}
```

The first resolver step (`_resolve_feature_columns`) collapses this to:

```python
dict[str, list[str]]
```

so the rest of the pipeline never sees the `str` variant:
- `"z_mu_b"` → prefix-matched to `["z_mu_b_0", "z_mu_b_1", ...]`
- `["length", "width"]` → passed through as-is

After this point, every feature set value is always a `list[str]`.

### How `positive`, `negative`, and `comparisons` interact

These three parameters work together to define what gets compared.
Their roles are different:

- **`positive`** — Selects *which classes* to focus on. In scheme modes
  (`"all_vs_rest"`, `"all_pairs"`), it acts as a **scope filter** — it limits
  which classes participate but doesn't define the full comparison. In explicit
  mode, it defines the left-hand group(s).

- **`negative`** — Defines the right-hand group(s). Much more restricted:
  it's only valid in explicit mode. Setting `negative` forces explicit mode.
  Cannot be combined with any scheme.

- **`comparisons`** — Chooses the *strategy* for generating pairs. A string
  scheme (`"all_vs_rest"`, `"all_pairs"`) auto-generates pairs from the data.
  A `pd.DataFrame` or `list[dict]` gives full manual control (`list[dict]` is
  normalized to DataFrame immediately at ingestion). When omitted, the mode is
  inferred from whether `negative` is set.

#### Mode resolution table

| `positive` | `negative` | `comparisons` | Mode | What happens |
|---|---|---|---|---|
| omitted | omitted | omitted | all-vs-rest | every class vs all others |
| set (scope) | omitted | omitted | all-vs-rest (scoped) | each listed class vs all others |
| set (scope) | omitted | `"all_vs_rest"` | all-vs-rest (scoped) | same as above, explicit scheme |
| set (scope) | omitted | `"all_pairs"` | all-pairs (scoped) | every unordered pair within scope |
| set | set | omitted | explicit | Cartesian product of positive × negative groups |
| omitted | omitted | `pd.DataFrame` or `list[dict]` | manual design (`explicit_design`) | pairs come from rows |

#### Hard mutual-exclusion errors

These combinations are always rejected at call time by `run_classification()`:

| Combination | Why it's an error |
|---|---|
| `comparisons=DataFrame/list[dict]` + any `positive` or `negative` | Design rows define their own pairs — extra constraints are ambiguous |
| `comparisons=str_scheme` + `negative` | Schemes generate their own negatives — user-provided negative conflicts |
| `comparisons=str_scheme` + `positive` as scalar | Scalar positive with a scheme is ambiguous (is it a scope or a single comparison?). Use a 1-element list to be explicit. |
| `negative` set but `positive` omitted | Negative without positive is almost always a user mistake |

### Three standard usage patterns

```python
# 1. Discovery — all classes, all-vs-rest, one or more feature sets
results = run_classification(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    features={"embedding": "z_mu_b"},
)

# 2. Targeted — explicit positive/negative, multiple named feature sets
results = run_classification(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    positive=["homo", "het"],
    negative="wildtype",
    features={
        "embedding": "z_mu_b",
        "shape": ["total_length_um", "baseline_deviation_normalized"],
    },
    bin_width=4.0,
    n_permutations=100,
)

# 3. Advanced — manual design (list[dict] is the ergonomic form)
results = run_classification(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons=[
        {"positive": "homo", "negative": "wildtype"},
        {"positive": "homo", "negative": "het"},
        {"positive": "het",  "negative": "wildtype"},
    ],
    features={"embedding": "z_mu_b"},
)

# Equivalent — pd.DataFrame also accepted
design = pd.DataFrame({
    "positive": ["homo", "homo", "het"],
    "negative": ["wildtype", "het", "wildtype"],
})
results = run_classification(
    df, class_col="genotype", id_col="embryo_id", time_col="predicted_stage_hpf",
    comparisons=design,
    features={"embedding": "z_mu_b"},
)
```

---

## Comparison vocabulary

| Term | Meaning | How to invoke |
|---|---|---|
| one-vs-one | One binary comparison between exactly two classes | `positive="homo", negative="wildtype"` |
| one-vs-rest | One class against all others pooled | `positive="homo"` (comparisons omitted) |
| all-vs-rest | Every class gets its own one-vs-rest | all omitted, or `comparisons="all_vs_rest"` |
| all-pairs / pairwise | All unordered one-vs-one pairs | `comparisons="all_pairs"` |
| pooled side | Multiple classes merged into one binary side | `tuple`: `("wildtype","het")` |
| enumerated | Multiple comparisons, one per element | `list`: `["wildtype","het"]` |

**Do not use** "all-vs-all" — it is ambiguous (ordered pairs? multiclass model? full confusion?).

---

## Type definitions

```python
ClassLabel      = str
PooledGroup     = tuple[str, ...]           # ≥ 2 elements, sorted+deduped at ingest
ComparisonGroup = ClassLabel | PooledGroup  # one group of labels on one side of a comparison

# What the user may pass to positive= or negative=
UserComparisonSpec = ClassLabel | PooledGroup | list[ClassLabel | PooledGroup]

# What comparisons= accepts
class ComparisonRow(TypedDict):
    positive: ComparisonGroup
    negative: ComparisonGroup

ComparisonScheme = (
    Literal["all_vs_rest", "all_pairs"]
    | pd.DataFrame
    | list[ComparisonRow]                # row-oriented: [{"positive": ..., "negative": ...}]
    | None
)
```

For manual design rows, only `{"positive": ..., "negative": ...}` keys are valid.

**Why `ComparisonGroup`?** Each comparison has two sides (positive vs negative).
A `ComparisonGroup` is one of those sides — either a single class label like `"wt"`,
or a pooled tuple like `("wt", "ctrl")`. The name reads as "one group of labels that
can be compared against another group."

### Encoding convention

| User value | Meaning |
|---|---|
| `"homo"` | single class |
| `("wildtype", "het")` | pooled: wildtype + het merged into one binary side |
| `["homo", "het"]` | enumerate: two separate comparisons |
| `[("wildtype","het"), "crispant"]` | two comparisons: once vs pooled {wt+het}, once vs crispant |

**Symmetry:** `tuple`-as-pool works identically for `positive` and `negative`. There is no
mechanical difference between a pooled positive and a pooled negative — both collapse to a
binary `_y` vector. The biological interpretation differs, but the code does not.

---

## Output type: `ResolvedComparison`

The backend receives a list of these. Everything downstream is unaware of pooling.

```python
@dataclass(frozen=True)
class ResolvedComparison:
    positive_members: tuple[str, ...]    # ≥ 1 label; sorted; maps to _y = 1
    negative_members: tuple[str, ...]    # ≥ 1 label; sorted; maps to _y = 0
    positive_label: str                  # human-readable: "homo" or "homo+crispant"
    negative_label: str                  # human-readable: "wildtype" or "wildtype+het"
    comparison_id: str                   # filesystem-safe: "homo__vs__wildtype_het"

    @property
    def is_pooled_positive(self) -> bool:
        return len(self.positive_members) > 1

    @property
    def is_pooled_negative(self) -> bool:
        return len(self.negative_members) > 1

    @property
    def all_members(self) -> frozenset[str]:
        return frozenset(self.positive_members) | frozenset(self.negative_members)
```

**Invariants:**
- `positive_members` and `negative_members` are disjoint sorted tuples (enforced before construction)
- `comparison_id` contains only `[A-Za-z0-9._-]` (spaces, `+`, `/` etc. → `_`)
- `positive_label` / `negative_label` are human-readable and unsanitized

---

## Comparison resolution pipeline

```
user input (positive, negative, comparisons, available_labels)
    ↓
    ↓  Step 1  — validate input types
    ↓
    ↓  Step 2  — resolve and expand comparisons
    ↓            determine mode
    ↓            normalize user inputs to groups
    ↓            canonicalize pooled groups (sort + dedupe)
    ↓            expand to complete raw_pairs
    ↓
    ↓            After this step, raw_pairs is always:
    ↓              list[(ComparisonGroup, ComparisonGroup)]
    ↓            No None placeholders. No deferred work.
    ↓
    ↓  Step 3  — validate expanded pairs
    ↓            overlap check (same label can't be on both sides)
    ↓            label existence check (catch typos early)
    ↓
    ↓  Step 4  — deduplicate (remove identical pairs)
    ↓
    ↓  Step 5  — convert to ResolvedComparison objects
    ↓            collision detection (error if duplicate comparison_ids)
    ↓
resolve_comparisons() returns here: list[ResolvedComparison]

    ↓  Step 6  — check_min_samples() — called by run_classification(),
    ↓            NOT inside resolve_comparisons(). Needs label_counts
    ↓            (unique id_col units per class), which requires DataFrame.
    ↓
backend receives: list[ResolvedComparison]
```

### Public entry point (pure function)

```python
def resolve_comparisons(
    positive: UserComparisonSpec | None,
    negative: UserComparisonSpec | None,
    comparisons: ComparisonScheme,
    available_labels: set[str],    # = set(df[class_col].unique()), computed by caller
    class_col: str,                # used only in error messages
) -> list[ResolvedComparison]:
    """
    Pure function. No DataFrame access. Steps 1–5 only.
    available_labels must be computed by the caller before calling this.
    Step 6 (min-sample check) is handled separately by check_min_samples().
    """
```

### Step 1 — Validate input types

Check that `positive` and `negative` are valid types before doing anything else.
Each value must be one of:
- a string (single class label)
- a tuple of strings (pooled group, ≥ 2 elements)
- a list of the above (multiple groups to enumerate)
- `None` (not provided)

```python
def _validate_group_input(val, param_name: str) -> None:
    """Validate that val is a legal UserComparisonSpec value."""

    # None means "not provided" — that's fine
    if val is None:
        return

    # A plain string is a single class label — always valid
    if isinstance(val, str):
        return

    # A tuple means "pool these classes into one group"
    if isinstance(val, tuple):
        if not all(isinstance(v, str) for v in val):
            raise TypeError(
                f"{param_name}: every element in a pooled tuple must be a string. "
                f"Got {val!r}"
            )
        if len(val) < 2:
            raise ValueError(
                f"{param_name}: a pooled tuple must have at least 2 elements. "
                f"For a single class, use a plain string instead of a 1-tuple."
            )
        return

    # A list means "enumerate these groups" — validate each element
    if isinstance(val, list):
        for i, item in enumerate(val):
            _validate_group_input(item, f"{param_name}[{i}]")
        return

    # Anything else is an error
    raise TypeError(
        f"{param_name} must be a string, tuple of strings, or list of those. "
        f"Got {type(val).__name__}"
    )

_validate_group_input(positive, "positive")
_validate_group_input(negative, "negative")
```

### Step 2 — Resolve and expand comparisons

This is the one step where mode matters. It determines what the user
intended, normalizes their inputs, and expands everything into complete
`(positive_group, negative_group)` pairs.

After this step finishes, the rest of the pipeline doesn't need to know
which mode was used. Every pair is complete — no `None` placeholders,
no deferred work.

There are four possible modes:

| Mode | When | What happens |
|---|---|---|
| `"explicit"` | both `positive` and `negative` given | Cartesian product of positive × negative groups |
| `"rest"` | no `negative` given | Each positive group vs all remaining classes |
| `"all_pairs"` | `comparisons="all_pairs"` | Every unordered pair of classes |
| `"explicit_design"` | `comparisons` is a DataFrame or list[dict] | Pairs come directly from the rows |

#### Helpers

```python
def _as_group_list(val: UserComparisonSpec) -> list[ComparisonGroup]:
    """Wrap a single group in a list; pass through a list unchanged."""
    if isinstance(val, (str, tuple)):
        return [val]
    return list(val)


def _canonicalize_group(group: ComparisonGroup) -> ComparisonGroup:
    """Sort + dedupe a pooled tuple. Pass strings through unchanged."""
    if isinstance(group, str):
        return group

    result = tuple(sorted(set(group)))
    if len(result) < 2:
        raise ValueError(
            f"Pooled tuple collapsed to fewer than 2 unique elements "
            f"after deduplication: {group!r}"
        )
    return result


def _group_members(group: ComparisonGroup) -> set[str]:
    """Return the set of class labels in a group."""
    if isinstance(group, str):
        return {group}
    return set(group)
```

#### Mode detection and expansion

```python
from itertools import product, combinations

# ── Mutual-exclusion checks happen earlier in run_classification() ──


# ── Normalize list[dict] to DataFrame immediately ────────────────────
if isinstance(comparisons, list):
    comparisons = pd.DataFrame(comparisons)

if isinstance(comparisons, pd.DataFrame):
    # ── explicit_design ───────────────────────────────────────────────
    # Pairs come directly from the user's DataFrame or list[dict].
    # Cells can be strings or pooled tuples.
    _validate_design_table(comparisons)
    raw_pairs = [
        (_canonicalize_group(row["positive"]), _canonicalize_group(row["negative"]))
        for _, row in comparisons.iterrows()
    ]


elif comparisons == "all_pairs":
    # ── all_pairs ─────────────────────────────────────────────────────
    # Every unordered pair of single labels.
    # positive= is accepted as a scope filter (which classes to pair).

    class_scope: list[str] = (
        list(positive) if positive is not None
        else sorted(available_labels)
    )

    # Pooled tuples don't make sense for all_pairs — only single labels
    for i, s in enumerate(class_scope):
        if isinstance(s, tuple):
            raise ValueError(
                f"comparisons='all_pairs': positive[{i}] is a tuple. "
                f"Scope entries must be single class labels (strings)."
            )

    # Direction is alphabetical: earlier label is always "positive"
    # Dedupe in case user passed duplicates in the scope list
    labels = sorted(set(class_scope))
    raw_pairs = [(a, b) for a, b in combinations(labels, 2)]


elif comparisons in ("all_vs_rest", None) and negative is None:
    # ── rest ──────────────────────────────────────────────────────────
    # Each positive group vs everything else.

    # 1. Normalize user inputs to groups
    positive_groups = (
        _as_group_list(positive) if positive is not None
        else [label for label in sorted(available_labels)]
    )

    # 2. Canonicalize pooled groups
    positive_groups = [_canonicalize_group(g) for g in positive_groups]

    # 3. Expand: compute the "rest" complement immediately
    raw_pairs = []
    for pg in positive_groups:
        rest_labels = sorted(available_labels - _group_members(pg))

        if len(rest_labels) == 0:
            raise ValueError(
                f"No remaining classes to form 'rest' for "
                f"positive={pg!r}. "
                f"All available labels are already in the positive group."
            )

        raw_pairs.append((pg, tuple(rest_labels)))


else:
    # ── explicit ──────────────────────────────────────────────────────
    # comparisons is None, negative is provided.
    # Every positive group × every negative group (Cartesian product).
    # Example: positives=[A, B], negatives=[C, D]
    #        → (A,C), (A,D), (B,C), (B,D)
    #
    # Note: positive=None + negative=set is already rejected as a hard
    # error by run_classification(), so positive is always set here.

    # 1. Normalize user inputs to groups
    positive_groups = _as_group_list(positive)
    negative_groups = _as_group_list(negative)

    # 2. Canonicalize pooled groups
    positive_groups = [_canonicalize_group(g) for g in positive_groups]
    negative_groups = [_canonicalize_group(g) for g in negative_groups]

    # 3. Expand: Cartesian product
    raw_pairs = list(product(positive_groups, negative_groups))
```

**Post-condition:** `raw_pairs` is always `list[(ComparisonGroup, ComparisonGroup)]`.
Both elements are present. No `None`s. No deferred work.

### Step 3 — Validate expanded pairs

Now that every pair is complete, validate them uniformly.
No mode-specific branching needed here.

#### Overlap check

The same class label cannot appear on both sides of a comparison.
For example, `positive=("wt", "het")` vs `negative=("het", "mut")` is
invalid because `"het"` is on both sides. Always enforced.

```python
for pos_group, neg_group in raw_pairs:
    overlap = _group_members(pos_group) & _group_members(neg_group)
    if overlap:
        raise ValueError(
            f"Comparison ({pos_group!r} vs {neg_group!r}) has overlapping "
            f"class labels: {sorted(overlap)}. "
            f"The same label cannot appear on both sides."
        )
```

#### Label existence check

Make sure every class label referenced in the pairs actually exists
in the data. This catches typos before any computation starts.

Pure function — no DataFrame access. The caller passes
`available_labels = set(df[class_col].unique())`.

```python
def _check_labels_exist(
    raw_pairs: list[tuple],
    available_labels: set[str],
    class_col: str,
) -> None:
    """Error if any label in raw_pairs is not in available_labels."""

    # Collect every label mentioned across all pairs
    referenced = set()
    for pos_group, neg_group in raw_pairs:
        referenced |= _group_members(pos_group)
        referenced |= _group_members(neg_group)

    # Check for unknown labels
    unknown = referenced - available_labels
    if unknown:
        raise ValueError(
            f"Class labels not found in column {class_col!r}: {sorted(unknown)}. "
            f"Available labels: {sorted(available_labels)}"
        )

_check_labels_exist(raw_pairs, available_labels, class_col)
```

### Step 4 — Deduplicate

Remove identical pairs. Order-preserving.

```python
seen: set[tuple] = set()
deduped: list[tuple] = []
for pair in raw_pairs:
    # Both elements are str or sorted tuple → hashable
    if pair not in seen:
        seen.add(pair)
        deduped.append(pair)
raw_pairs = deduped
```

### Step 5 — Convert to `ResolvedComparison`

Convert each `(positive_group, negative_group)` pair into a
`ResolvedComparison` dataclass. This is the final internal
representation that the backend receives.

```python
def _sanitize_id(label: str) -> str:
    """Replace non-filesystem-safe characters with underscores."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", label)


def _to_resolved(
    pos_group: ComparisonGroup,
    neg_group: ComparisonGroup,
) -> ResolvedComparison:
    """Build a ResolvedComparison from one (positive, negative) pair."""

    # Normalize to tuples so members are always iterable
    pos_members = (pos_group,) if isinstance(pos_group, str) else pos_group
    neg_members = (neg_group,) if isinstance(neg_group, str) else neg_group

    # Human-readable labels: "crispant+homo", "het+wildtype"
    positive_label = "+".join(pos_members)
    negative_label = "+".join(neg_members)

    # Filesystem-safe ID: "crispant_homo__vs__het_wildtype"
    comparison_id = (
        _sanitize_id(positive_label) + "__vs__" + _sanitize_id(negative_label)
    )

    return ResolvedComparison(
        positive_members=pos_members,
        negative_members=neg_members,
        positive_label=positive_label,
        negative_label=negative_label,
        comparison_id=comparison_id,
    )


resolved: list[ResolvedComparison] = [
    _to_resolved(pos, neg) for pos, neg in raw_pairs
]

# ── Collision detection ─────────────────────────────────────────────────
# comparison_ids must be unique. Collisions happen when class labels
# differ only in characters that get sanitized (spaces, slashes, etc.).
from collections import Counter
counts = Counter(rc.comparison_id for rc in resolved)
dupes = sorted(cid for cid, n in counts.items() if n > 1)
if dupes:
    raise ValueError(
        f"comparison_id collision: {dupes}. "
        f"Distinct comparisons produced the same filesystem-safe ID. "
        f"This typically happens when class labels differ only in "
        f"characters that get sanitized (spaces, slashes, etc.)."
    )
```

### Step 6 — Min-sample check (standalone function)

**Called by `run_classification()`, NOT inside `resolve_comparisons()`.**
Requires `label_counts` which needs DataFrame access (counting unique `id_col` units).

`label_counts` must count **unique `id_col` units**, not rows.
Row counts are inflated by time points and must not be used here.

#### Two thresholds, not one

Pooled groups create a risk of **hidden minorities**: a group passes the
total-count check but one constituent contributes almost nothing. For example,
`("rare_mutant", "common_het")` with counts `{rare_mutant: 1, common_het: 50}`
passes a group minimum of 3 but `rare_mutant` is statistically ornamental.

To catch this, validation enforces two levels:

| Check | What it does | Default | Severity |
|---|---|---|---|
| **Group minimum** | total unique units across all members of a group | `min_samples_per_group=3` | hard error |
| **Per-member minimum** | each individual member in a pooled group | `min_samples_per_member=2` | hard error |

For non-pooled groups (single class label), both checks collapse to the same thing.

```python
def check_min_samples(
    resolved: list[ResolvedComparison],
    label_counts: dict[str, int],    # {class_label: n_unique_units}
    min_samples_per_group: int,      # group-level minimum
    min_samples_per_member: int = 2, # per-constituent minimum in pooled groups
) -> None:
    """
    Validate that every group in every comparison has enough data.

    Two checks:
    1. Group total: sum of counts across all members ≥ min_samples_per_group
    2. Per-member: each individual member ≥ min_samples_per_member
       (only meaningful for pooled groups with ≥ 2 members)
    """
    for rc in resolved:
        for members, group_label, side_name in [
            (rc.positive_members, rc.positive_label, "positive"),
            (rc.negative_members, rc.negative_label, "negative"),
        ]:
            # 1. Group-level check
            group_total = sum(label_counts.get(m, 0) for m in members)
            if group_total < min_samples_per_group:
                raise ValueError(
                    f"Comparison {rc.comparison_id!r}: {side_name} "
                    f"group '{group_label}' has only {group_total} units "
                    f"(min_samples_per_group={min_samples_per_group})."
                )

            # 2. Per-member check (catches hidden minorities in pooled groups)
            if len(members) >= 2:
                for m in members:
                    n = label_counts.get(m, 0)
                    if n < min_samples_per_member:
                        raise ValueError(
                            f"Comparison {rc.comparison_id!r}: member '{m}' "
                            f"in {side_name} pool '{group_label}' has only "
                            f"{n} units (min_samples_per_member="
                            f"{min_samples_per_member}). Each constituent "
                            f"in a pooled group must contribute meaningfully."
                        )
```

---

## Design table validation

Design tables accept both string class labels and pooled tuples.
Each cell in the `positive` and `negative` columns must be either
a `str` (single class) or a `tuple[str, ...]` (pooled group).

```python
def _validate_design_table(df: pd.DataFrame) -> None:
    required = {"positive", "negative"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"comparisons DataFrame missing columns: {sorted(missing)}")

    for col in required:
        if df[col].isnull().any():
            raise ValueError(f"comparisons DataFrame column {col!r} contains nulls.")

        for i, val in enumerate(df[col]):
            if isinstance(val, str):
                continue
            if isinstance(val, tuple):
                if not all(isinstance(v, str) for v in val):
                    raise TypeError(
                        f"comparisons[{col!r}][{i}]: tuple elements must all be strings."
                    )
                if len(val) < 2:
                    raise ValueError(
                        f"comparisons[{col!r}][{i}]: pooled tuple must have ≥ 2 elements."
                    )
                continue
            raise TypeError(
                f"comparisons[{col!r}][{i}]: must be a string or tuple of strings. "
                f"Got {type(val).__name__}"
            )

    # Deduplicate check: canonicalize before comparing so
    # ("a","b") and ("b","a") are treated as the same pair
    canon = df.copy()
    for col in required:
        canon[col] = canon[col].map(
            lambda v: tuple(sorted(v)) if isinstance(v, tuple) else v
        )
    if canon.duplicated(subset=["positive", "negative"]).any():
        raise ValueError("comparisons DataFrame contains duplicate (positive, negative) rows.")
```

**Serialization note:** Design tables with pooled tuples are a Python-native
convenience. If serialized externally (e.g. to CSV or JSON), tuple preservation
must be handled carefully — a round-tripped CSV will stringify tuples. For
persistent manual comparison specs, prefer `list[dict]` input (which survives
JSON serialization) or rebuild from code.

---

## The one backend function that knows about pooling

Everything downstream of this function (CV, AUROC, permutation) sees only `_y` and is
unaware that pooling occurred.

```python
def _build_binary_labels(
    df: pd.DataFrame,
    class_col: str,
    comparison: ResolvedComparison,
) -> pd.DataFrame:
    """
    Filter df to rows belonging to either side of the comparison and assign
    binary labels: _y = 1 for positive_members, _y = 0 for negative_members.
    Rows from classes not in either side are dropped.

    The inference target is separability of the pooled groups, not of each
    constituent class individually. Permutation tests shuffle this binary _y
    vector within time strata — pooled positives and pooled negatives are
    handled identically.
    """
    members = set(comparison.positive_members) | set(comparison.negative_members)
    subset = df[df[class_col].isin(members)].copy()
    subset["_y"] = subset[class_col].isin(comparison.positive_members).astype(int)
    return subset
```

**Note:** `_y` is derived from the subset's class labels directly (not from outer mask
indexing) to avoid index-alignment footguns.

---

## Permutation / null testing

Because the null permutation shuffles `_y` within time-bin strata:

- Pooling only changes which rows get `_y=1` vs `_y=0`.
- Once `_y` exists, permutation is identical regardless of how many members are
  on each side: shuffle `_y` within bin, recompute AUROC, collect null distribution.
- The null corresponds to the binary task "positive_members vs negative_members."

**The hypothesis being tested is separability of the pooled groups, not of each
constituent class individually.**

---

## Worked example (end-to-end trace)

**Inputs:**
```python
available_labels = {"wildtype", "het", "homo", "crispant"}
positive = ("homo", "crispant")     # pooled positive
negative = ("wildtype", "het")      # pooled negative
comparisons = None
```

**Step 1:** Both values are valid tuples with ≥ 2 strings. ✓

**Step 2:** Resolve and expand.
- `comparisons=None`, `negative` is not None → `mode = "explicit"`.
- Normalize: `positive_groups = [("homo", "crispant")]`, `negative_groups = [("wildtype", "het")]`
- Canonicalize: `("homo", "crispant")` → `("crispant", "homo")`, `("wildtype", "het")` → `("het", "wildtype")`
- Expand (Cartesian product): `raw_pairs = [(("crispant","homo"), ("het","wildtype"))]`

**Step 3:** Validate.
- Overlap: `{"crispant","homo"} ∩ {"het","wildtype"} = ∅` ✓
- Label existence: all four labels in `available_labels`. ✓

**Step 4:** One pair, no duplicates.

**Step 5:**
```python
ResolvedComparison(
    positive_members = ("crispant", "homo"),
    negative_members = ("het", "wildtype"),
    positive_label   = "crispant+homo",
    negative_label   = "het+wildtype",
    comparison_id    = "crispant_homo__vs__het_wildtype",
)
```

**Step 6:** Check union embryo counts for each group against `min_samples`.

**Backend:**
```python
subset = df[df["genotype"].isin({"crispant","homo","het","wildtype"})].copy()
subset["_y"] = subset["genotype"].isin({"crispant","homo"}).astype(int)
# downstream CV / AUROC / permutation operates on _y only
```
# Backend & Object Spec — Locked

---

## Internal factory line

```
run_classification(df, ...)              ← orchestrator in run_classification.py
    │
    ├─ 1. _resolve_feature_columns()   → dict[str, list[str]]  (feature_set → col list)
    ├─ 2. resolve_comparisons()        → list[ResolvedComparison]  (Steps 1–5)
    ├─ 2b. check_min_samples()         → validate sample counts  (Step 6, data-dependent)
    ├─ 3. _build_binary_labels()       → filtered df with _y column  (per comparison)
    ├─ 4. _bin_and_aggregate()         → df binned by (id_col, time_bin)
    ├─ 5. _run_classification_loop()   → raw per-bin result dicts
    │       ├─ cross_val_predict()     → probabilities
    │       ├─ roc_auc_score()         → auroc_obs
    │       └─ _permutation_test_binary() → null_aurocs → pval, null_mean, null_std
    │                                      (n_jobs execution strategy is implementation-defined)
    ├─ 6. _collect_scores()              → scores DataFrame  ← THE canonical table
    ├─ 7a. _collect_binary_predictions() → tidy binary predictions  (save_predictions=True)
    ├─ 7b. _collect_multiclass_predictions()
    │                                    → wide multiclass predictions  (all-vs-rest only,
    │                                       save_multiclass_predictions=True)
    └─ 8. _collect_confusion()           → confusion DataFrame  (always emitted)
```

Each step runs once per `(feature_set, comparison)` pair. Results are concatenated into
the canonical tables after all pairs complete.

---

## Step 5 — inner loop output (raw per-bin dict)

Produce these keys inside the loop. No renaming at boundaries.

```python
{
    # time
    "time_bin":        int(t),
    "time_bin_center": float(t) + bin_width / 2.0,
    "bin_width":       float(bin_width),
    # results — canonical names, set here, never renamed later
    "auroc_obs":       float,        # was auroc_observed — renamed HERE not at boundary
    "pval":            float,
    "n_positive":      int,
    "n_negative":      int,
    # null summary
    "auroc_null_mean": float,
    "auroc_null_std":  float,
    "n_permutations":  int,
    # raw null array — collected separately, NOT a dict key
    "_null_array":     np.ndarray,   # shape (P,), removed before scores assembly
}
# DROPPED from inner loop: positive_class, negative_class, negative_mode, groupby
```

---

## Step 6 — `_collect_scores(bin_results, comparison, feature_set) -> list[dict]`

The only place that assembles identity keys + results into a canonical scores row.

```python
def _collect_scores(
    bin_results: list[dict],
    comparison: ResolvedComparison,
    feature_set: str,
) -> list[dict]:
    rows = []
    for r in bin_results:
        rows.append({
            # identity (always present)
            "feature_set":     feature_set,
            "comparison_id":   comparison.comparison_id,
            "positive_label":  comparison.positive_label,
            "negative_label":  comparison.negative_label,
            # time
            "time_bin_center": r["time_bin_center"],
            "time_bin":        r["time_bin"],
            "bin_width":       r["bin_width"],
            # results
            "auroc_obs":       r["auroc_obs"],
            "pval":            r["pval"],
            "n_pos":           r["n_positive"],
            "n_neg":           r["n_negative"],
            # null summary
            "auroc_null_mean": r.get("auroc_null_mean"),
            "auroc_null_std":  r.get("auroc_null_std"),
            "n_permutations":  r.get("n_permutations"),
        })
    return rows
```

---

## Step 7 — `_collect_binary_predictions()` — binary per comparison

Works for every mode (all-vs-rest, pairwise, all-pairs). One row per
(id_col, time_bin_center, comparison_id, feature_set).

**`p_pos` is the raw model probability for all modes.** For a
`homo_vs_wildtype` comparison, `p_pos = 0.87` means the model thinks
there's an 87% chance this embryo-at-this-timepoint belongs to the
positive group. You can plot it per embryo over time, check calibration,
or inspect which embryos the model is uncertain about — all from this
one table, regardless of whether the run was all-vs-rest or pairwise.

```python
{
    "feature_set":     str,
    "comparison_id":   str,
    id_col:            str,    # uses the actual id_col name (e.g. "embryo_id")
    "time_bin_center": float,
    "y_true":          int,    # 1 = positive side, 0 = negative side
    "p_pos":           float,  # raw model probability of positive class
    "y_pred":          int,    # hard call at 0.5 threshold
    "is_correct":      bool,
}
```

### Relationship to the misclassification pipeline

**These are two different prediction tables for two different purposes.**

`run_misclassification_pipeline()` consumes a `ClassificationAnalysis` object
and reads its `multiclass_predictions` layer.

The tidy binary table above is the new standard for per-comparison
diagnostics (AUROC breakdowns, per-embryo accuracy in binary tasks).

The misclassification pipeline (`run_misclassification_pipeline`) requires
a **separate wide multiclass format** — `multiclass_predictions.parquet` —
with one row per `(embryo_id, time_bin)` and wide `pred_proba_{class}` columns
for every class. This format is needed because:

- **Trajectory analysis** (`trajectory.py`) builds feature matrices from
  full per-class probability vectors (soft, delta, residual stages).
  `p_pos` alone is insufficient — it doesn't say *which* classes the model
  confused the embryo with.
- **Top-confused-as analysis** (`null.py`) permutes `pred_class` to test
  whether an embryo's confusion pattern is non-random. This needs the
  original multiclass identity, not a binary collapse.
- **`io.py` validation** requires `pred_proba_*` columns that sum to ~1.0
  per row — a hard contract.

The binary predictions table does **not** replace the wide multiclass one.
They coexist:

| Table | Shape | Saved by | Consumed by |
|---|---|---|---|
| `predictions.parquet` (tidy binary) | (id, time_bin, comparison, feature_set) → y_true, p_pos | `run_classification()` | scores plots, per-comparison diagnostics |
| `multiclass_predictions.parquet` (wide multiclass) | (id, time_bin) → true_class, pred_class, pred_proba_* | `run_classification()` (all-vs-rest mode) | misclassification pipeline |

The wide table is only meaningful for all-vs-rest runs where the model
sees all classes simultaneously. For pairwise or explicit comparisons,
the binary table is the right artifact — and `p_pos` is the raw probability
you'd want to inspect.

### Stacking and predictions

`stack()` merges scores and `uns["comparisons"]` only. **Prediction,
confusion, and raw-null layers are not merged by design.**

**Layers are diagnostic artifacts, not canonical merge targets.**
Scores are cross-run comparable summaries. Predictions are run-local
diagnostics. Raw prediction probabilities are not guaranteed to be
directly comparable across runs, because runs may differ in class
composition, comparison structure, feature sets, and training
distributions. The same principle applies to confusion profiles and
null arrays.

```python
# Compare AUROC across runs → use the stacked object
combined = results_ovr.stack(results_pw)
combined.scores  # unified scores table

# Inspect raw probabilities for a specific run → use the original
results_pw.layers["predictions"]              # binary p_pos per embryo
results_ovr.layers["multiclass_predictions"]  # wide pred_proba_* for misclassification
```

**Bridge for v2:** A future utility could reconstruct the wide format from
a complete set of all-vs-rest binary predictions, but this is not required
for the initial implementation.

---

## Step 8 — `_collect_confusion()`

Wraps `extract_temporal_confusion_profile`. Add `feature_set`, `comparison_id`, and
use `time_bin_center` as the canonical time key.

```python
{
    "feature_set":     str,
    "comparison_id":   str,    # "all_vs_rest" for multiclass runs
    "time_bin_center": float,
    "true_class":      str,
    "predicted_class": str,
    "proportion":      float,
    "count":           int,
    "is_correct":      bool,
}
```

Emitted for all comparison modes, including pairwise. Even for binary (2×2)
comparisons, confusion captures error asymmetry at the chosen threshold —
a model that collapses toward wildtype is very different from one that
overcalls mutant, even if AUROC looks similar. Since predictions are
already collected, computing confusion is cheap.

**For pooled binary comparisons,** `true_class` and `predicted_class` refer
to the comparison *side labels* (`positive_label` / `negative_label`), not
the underlying biological class identities. For example, in a comparison
with `positive=("homo", "crispant")` and `negative=("wildtype", "het")`,
the confusion rows use `"crispant+homo"` and `"het+wildtype"` — not the
four individual genotypes. This keeps confusion aligned with the binary
prediction task, not the original class taxonomy.

---

## Canonical `scores` table — column contract

```
REQUIRED (always present, never null):
  feature_set      str     name from features={} dict
  comparison_id    str     "homo__vs__wildtype_het"  (filesystem-safe)
  positive_label   str     "homo" or "homo+crispant"
  negative_label   str     "wildtype" or "wildtype+het"
  time_bin_center  float   canonical x-axis for all plots
  auroc_obs        float   observed AUROC

STANDARD (present unless n_permutations=0):
  pval             float
  auroc_null_mean  float
  auroc_null_std   float
  n_permutations   int
  n_pos            int
  n_neg            int

OPTIONAL:
  time_bin         int     bin start (join key, not for plotting)
  bin_width        float
```

**Unique key:** `(feature_set, comparison_id, time_bin_center)` — enforced by
`_validate_scores()`.

**No JSON blobs, no list-in-cell, no alias columns.**

Comparison membership lives in `uns["comparisons"]` keyed by `comparison_id`:

```json
{
  "comparisons": {
    "homo__vs__wildtype_het": {
      "positive_members": ["homo"],
      "negative_members": ["wildtype", "het"],
      "positive_label": "homo",
      "negative_label": "wildtype+het"
    }
  }
}
```

---

## `ClassificationAnalysis` object

```python
@dataclass
class ClassificationAnalysis:
    scores: pd.DataFrame      # required — always eager
    uns:    dict              # required — always eager; treat as read-only
    layers: _LazyLayers       # optional artifacts — lazy from disk

    def __post_init__(self):
        _validate_scores(self.scores)

    # Properties
    @property
    def feature_sets(self) -> list[str]:
        return sorted(self.scores["feature_set"].unique())

    @property
    def comparison_ids(self) -> list[str]:
        return sorted(self.scores["comparison_id"].unique())

    # Subsetting — forks layer cache so subset and parent don't share state
    def subset(self, feature_set=None, comparison_id=None,
               positive_label=None, time_range=None) -> "ClassificationAnalysis":
        s = self.scores
        if feature_set   is not None: s = s[s["feature_set"].isin(_listify(feature_set))]
        if comparison_id is not None: s = s[s["comparison_id"].isin(_listify(comparison_id))]
        if positive_label is not None: s = s[s["positive_label"].isin(_listify(positive_label))]
        if time_range    is not None: s = s[s["time_bin_center"].between(*time_range)]
        return ClassificationAnalysis(scores=s.copy(), uns=self.uns,
                                      layers=self.layers._fork())

    # ── Stacking ─────────────────────────────────────────────────────────
    #
    # stack() merges scores and uns["comparisons"] only.
    # Prediction, confusion, and raw-null layers are not merged by design.
    #
    # Layers are diagnostic artifacts, not canonical merge targets.
    # Predictions are the clearest example — raw prediction probabilities
    # are not guaranteed to be directly comparable across runs, because
    # runs may differ in class composition, comparison structure, feature
    # sets, and training distributions — but the same principle applies
    # to confusion profiles and null arrays.
    #
    # Use stacked objects for summary-level comparison across runs.
    # Use original ClassificationAnalysis objects to inspect raw
    # predictions or other run-local diagnostics:
    #
    #   results_ovr.layers["predictions"]              # binary p_pos table
    #   results_ovr.layers["multiclass_predictions"]   # wide pred_proba_* table

    def stack(self, other, on_conflict="error") -> "ClassificationAnalysis":
        """Merge scores and uns['comparisons']. Layers are not merged."""
        new_keys = set(zip(other.scores["feature_set"], other.scores["comparison_id"]))
        existing = set(zip(self.scores["feature_set"], self.scores["comparison_id"]))
        overlap  = new_keys & existing
        if overlap and on_conflict == "error":
            raise ValueError(f"Overlapping (feature_set, comparison_id) pairs: {overlap}")
        scores = pd.concat([self.scores, other.scores], ignore_index=True)
        if on_conflict == "overwrite" and overlap:
            scores = scores.drop_duplicates(
                subset=["feature_set", "comparison_id", "time_bin_center"], keep="last")
        # Merge uns["comparisons"] by comparison_id; error on key collision
        merged_uns = {k: v for k, v in self.uns.items() if k != "comparisons"}
        merged_uns.update({k: v for k, v in other.uns.items() if k != "comparisons"})
        self_comps = self.uns.get("comparisons", {})
        other_comps = other.uns.get("comparisons", {})
        collision = set(self_comps) & set(other_comps)
        if collision and on_conflict == "error":
            raise ValueError(f"uns['comparisons'] key collision: {sorted(collision)}")
        if on_conflict == "overwrite":
            merged_uns["comparisons"] = {**self_comps, **other_comps}
        else:
            merged_uns["comparisons"] = {**self_comps, **other_comps}
        return ClassificationAnalysis(scores=scores, uns=merged_uns,
                                      layers=_LazyLayers(None))

    # Plotting — thin sugar, all inference happens inside plot_aurocs_over_time
    def plot_aurocs(self, *, curve_col=None, facet_col=None, **kwargs):
        from .viz.auroc_over_time import plot_aurocs_over_time
        return plot_aurocs_over_time(self.scores, curve_col=curve_col,
                                     facet_col=facet_col, **kwargs)

    def plot_confusion(self, **kwargs):
        conf = self.layers.get("confusion")
        if conf is None:
            raise KeyError(
                "No confusion layer available. "
                "Re-run run_classification() — confusion is saved automatically."
            )
        from .viz.confusion import plot_confusion
        return plot_confusion(self.scores, conf, **kwargs)

    # Persistence
    def save(self, path, overwrite=False) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        _write_parquet(self.scores, path / "scores.parquet", overwrite)
        _write_json(self.uns, path / "metadata.json", overwrite)
        self.layers._save_to_dir(path, overwrite)
        return path

    @classmethod
    def load(cls, path) -> "ClassificationAnalysis":
        path = Path(path)
        scores = pd.read_parquet(path / "scores.parquet")
        with open(path / "metadata.json") as f:
            uns = json.load(f)
        return cls(scores=scores, uns=uns, layers=_LazyLayers(path))

    @classmethod
    def from_legacy(cls, path) -> "ClassificationAnalysis":
        from .legacy import load_legacy_ovr_results
        return load_legacy_ovr_results(path)
```

---

## `_LazyLayers` — complete implementation

```python
class _LazyLayers:
    """
    Lazy-loading dict-like interface for optional artifacts.

    Layers
    ------
    "predictions"              pd.DataFrame         predictions.parquet (tidy binary)
    "multiclass_predictions"   pd.DataFrame         multiclass_predictions.parquet (wide)
    "confusion"                pd.DataFrame         confusion.parquet
    "null_full"                NullDistributions     null_distributions.npz
    """

    _REGISTRY: dict[str, tuple[str, str]] = {
        "predictions":             ("predictions.parquet", "parquet"),
        "multiclass_predictions":  ("multiclass_predictions.parquet", "parquet"),
        "confusion":   ("confusion.parquet",   "parquet"),
        "null_full":   ("null_distributions.npz", "nulls"),
    }

    def __init__(self, base_dir: Path | None) -> None:
        self._dir   = base_dir
        self._cache: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown layer '{key}'. Known: {sorted(self._REGISTRY)}")
        if key in self._cache:
            return self._cache[key]
        if self._dir is None:
            raise KeyError(f"Layer '{key}' not in cache and no backing directory.")
        fname, kind = self._REGISTRY[key]
        path = self._dir / fname
        if not path.exists():
            raise KeyError(f"Layer '{key}' not found at {path}")
        data = self._load(kind, path)
        self._cache[key] = data
        return data

    def get(self, key: str, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        """Pure existence check — never triggers a disk load."""
        if key in self._cache:
            return True
        if self._dir is None:
            return False
        fname, _ = self._REGISTRY.get(key, (None, None))
        return fname is not None and (self._dir / fname).exists()

    def available(self) -> list[str]:
        """Keys that exist on disk, sorted."""
        if self._dir is None:
            return []
        return sorted(
            k for k, (fname, _) in self._REGISTRY.items()
            if (self._dir / fname).exists()
        )

    def cached(self) -> list[str]:
        """Keys currently in memory."""
        return sorted(self._cache)

    def store(self, key: str, data: Any) -> None:
        """Cache an in-memory artifact (written to disk on analysis.save())."""
        if key not in self._REGISTRY:
            raise KeyError(f"Unknown layer '{key}'. Known: {sorted(self._REGISTRY)}")
        self._cache[key] = data

    def _fork(self) -> "_LazyLayers":
        """New _LazyLayers with same backing dir but empty cache. Used by subset()."""
        return _LazyLayers(self._dir)

    # ── load/save internals ───────────────────────────────────────────────

    @staticmethod
    def _load(kind: str, path: Path) -> Any:
        if kind == "parquet":
            return pd.read_parquet(path)
        if kind == "nulls":
            return NullDistributions.load(path)
        raise ValueError(f"Unknown kind: {kind}")

    def _save_to_dir(self, path: Path, overwrite: bool) -> None:
        for key, data in self._cache.items():
            fname, kind = self._REGISTRY[key]
            fpath = path / fname
            if fpath.exists() and not overwrite:
                raise FileExistsError(f"{fpath} exists. Pass overwrite=True.")
            if kind == "parquet":
                data.to_parquet(fpath, index=False)
            elif kind == "nulls":
                data.save(fpath)
            else:
                raise TypeError(f"Don't know how to save layer '{key}'")
        self._dir = path
```

---

## `NullDistributions` — array-indexed null handle

Avoids delimiter hell by storing a parallel index rather than encoding keys into strings.

```python
@dataclass
class NullDistributions:
    """
    Handle for raw per-permutation AUROC null distributions.

    Storage layout (null_distributions.npz):
      null_auc         float32  shape (N, P)   N = cells, P = permutations
      feature_set      U64 str  shape (N,)
      comparison_id    U64 str  shape (N,)
      time_bin_center  float64  shape (N,)

    Access:
      nd.get("embedding", "homo__vs__wildtype", 26.0)  → np.ndarray shape (P,)
      nd.index_df                                       → DataFrame with N rows
    """
    null_auc:        np.ndarray     # (N, P)  float32
    feature_set:     np.ndarray     # (N,)    str
    comparison_id:   np.ndarray     # (N,)    str
    time_bin_center: np.ndarray     # (N,)    float64
    _index: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        N = len(self.feature_set)
        assert self.null_auc.shape[0] == N
        assert len(self.comparison_id) == N
        assert len(self.time_bin_center) == N

    @property
    def index_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature_set":     self.feature_set,
            "comparison_id":   self.comparison_id,
            "time_bin_center": self.time_bin_center,
        })

    def _build_index(self) -> dict:
        return {
            (str(fs), str(cid), float(tbc)): i
            for i, (fs, cid, tbc) in enumerate(
                zip(self.feature_set, self.comparison_id, self.time_bin_center)
            )
        }

    def get(self, feature_set: str, comparison_id: str,
            time_bin_center: float) -> np.ndarray:
        if self._index is None:
            object.__setattr__(self, "_index", self._build_index())
        key = (feature_set, comparison_id, float(time_bin_center))
        if key not in self._index:
            raise KeyError(f"No null distribution for {key}")
        return self.null_auc[self._index[key]]

    @classmethod
    def load(cls, path: Path) -> "NullDistributions":
        npz = np.load(path, allow_pickle=False)
        return cls(
            null_auc        = npz["null_auc"],
            feature_set     = npz["feature_set"],
            comparison_id   = npz["comparison_id"],
            time_bin_center = npz["time_bin_center"],
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            null_auc        = self.null_auc.astype(np.float32),
            feature_set     = self.feature_set,
            comparison_id   = self.comparison_id,
            time_bin_center = self.time_bin_center,
        )
```

**Building a `NullDistributions` during classification:**

```python
# Accumulate per-bin nulls during the loop
null_rows: list[dict] = []
for (fs, cid, tbc), null_array in per_bin_nulls:
    null_rows.append({
        "feature_set":     fs,
        "comparison_id":   cid,
        "time_bin_center": tbc,
        "nulls":           null_array,
    })

# At end of run
nd = NullDistributions(
    null_auc        = np.array([r["nulls"] for r in null_rows], dtype=np.float32),
    feature_set     = np.array([r["feature_set"] for r in null_rows]),
    comparison_id   = np.array([r["comparison_id"] for r in null_rows]),
    time_bin_center = np.array([r["time_bin_center"] for r in null_rows]),
)
analysis.layers.store("null_full", nd)
```

---

## `_validate_scores` — minimum only

```python
_SCORES_REQUIRED = frozenset({
    "feature_set", "comparison_id", "positive_label", "negative_label",
    "time_bin_center", "auroc_obs",
})

def _validate_scores(df: pd.DataFrame) -> None:
    missing = _SCORES_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    dupes = df.duplicated(subset=["feature_set", "comparison_id", "time_bin_center"])
    if dupes.any():
        raise ValueError(
            f"scores has {dupes.sum()} duplicate "
            f"(feature_set, comparison_id, time_bin_center) rows."
        )
```

---

## On-disk layout

```
my_run/
  scores.parquet                          ← always
  metadata.json                           ← always (uns dict)
  predictions.parquet                     ← optional (save_predictions=True) — tidy binary
  multiclass_predictions.parquet    ← optional (all-vs-rest mode) — wide multiclass
  confusion.parquet                       ← always
  null_distributions.npz                  ← optional (save_null_arrays=True)
```

---

## Artifact tiers

| Artifact | Storage | Default | When |
|---|---|---|---|
| Null stats (mean/std/n) | columns in `scores` | always | free, always useful |
| Raw null arrays | `null_distributions.npz` via `NullDistributions` | off (`save_null_arrays=True`) | diagnostic only |
| Confusion profile | `confusion.parquet` | always (all modes) | cheap; captures error asymmetry |
| Predictions (binary) | `predictions.parquet` | off (`save_predictions=True`) | per-comparison diagnostics |
| Predictions (multiclass) | `multiclass_predictions.parquet` | off (`save_multiclass_predictions=True`; all-vs-rest only) | required by misclassification pipeline |

---

## Misclassification pipeline — fail-loud contract

If `run_misclassification_pipeline()` is called and the `multiclass_predictions` layer
is missing, raise immediately with a direct remediation message:

```python
def run_misclassification_pipeline(analysis: ClassificationAnalysis, ...):
    if "multiclass_predictions" not in analysis.layers:
        raise ValueError(
            "Misclassification pipeline requires the multiclass_predictions layer. "
            "Re-run run_classification() with save_multiclass_predictions=True."
        )
    ...
```

This makes the dependency between `save_multiclass_predictions=False` (the default)
and `run_misclassification_pipeline()` explicit at runtime, rather than producing a
cryptic KeyError downstream.

---

## Refactoring contract — renames and boundaries

### Rename table (old → new)

These renames are locked. Apply during implementation; do not preserve old names in new code.

#### Renamed

| Old name | New name | Scope |
|---|---|---|
| `_permutation_test_ovr()` | `_permutation_test_binary()` | engine/loop.py |
| `auroc_observed` | `auroc_obs` | everywhere (inner loop, scores, plotters) |
| `_validate_group()` | `_validate_group_input()` | engine/comparison_resolution.py |
| `_to_groups()` | `_as_group_list()` | engine/comparison_resolution.py |
| `_members()` | `_group_members()` | engine/comparison_resolution.py |
| `_collect_predictions()` | `_collect_binary_predictions()` | engine/loop.py |
| `embryo_predictions_augmented.parquet` | `multiclass_predictions.parquet` | on-disk, `_LazyLayers._REGISTRY` |
| `plot_confusion_profile` | `plot_confusion` | viz/confusion.py |
| `design_table` (mode name) | `explicit_design` | mode resolution, comments |

#### Deleted

| Old name | Replacement / note | Scope |
|---|---|---|
| `positive_class` / `negative_class` | dropped — use `positive_label` / `negative_label` from `ResolvedComparison` | scores assembly |
| `_auroc_col()` / `_time_col()` | deleted — columns are always `auroc_obs` and `time_bin_center` | plotters |

#### Retained only as legacy shim

| Old name | Shim behavior | Scope |
|---|---|---|
| `ComparisonSpec` | legacy shim only — no new internal use | results.py (shim) |
| `run_multiclass_classification_test` | legacy shim only — routes to `run_classification` | classification_test.py (shim) |

### Normalization boundary

All user-provided comparison specs — `pd.DataFrame`, `list[dict]`, `positive`/`negative` —
are normalized into `list[tuple[ComparisonGroup, ComparisonGroup]]` (raw_pairs) inside
`resolve_comparisons()`, Steps 1–4. After Step 5, the only internal representation is
`list[ResolvedComparison]`. No downstream code ever sees raw user input forms.

`list[dict]` is normalized to `pd.DataFrame` immediately at the top of `resolve_comparisons()`.
This keeps the DataFrame validation path as the single code path for manual designs.

### Canonical assembly boundaries

- **`_collect_scores()`** — the only place that assembles identity keys + result keys into a scores row
- **`_build_binary_labels()`** — the only place that knows about pooling and constructs `_y`
- **`_collect_binary_predictions()`** — the only place that assembles tidy binary prediction rows
- **`_collect_multiclass_predictions()`** — the only place that assembles wide multiclass rows

No other code should build these row schemas.

### Persistence defaults (locked)

| Artifact | Default | Rationale |
|---|---|---|
| `scores.parquet` | always | core contract |
| `metadata.json` | always | provenance + comparison membership |
| `confusion.parquet` | always | aggregated at the same time-bin granularity as scores; compact enough to save by default while capturing error asymmetry even in binary tasks |
| `predictions.parquet` | off (`save_predictions=True`) | can be large; diagnostic |
| `multiclass_predictions.parquet` | off (`save_multiclass_predictions=True`) | can be large; only needed by misclassification pipeline |
| `null_distributions.npz` | off (`save_null_arrays=True`) | summary stats always in scores; raw arrays are diagnostic |

### Test seams (boundary tests for refactoring)

These tests validate the seams between modules. They let internals change freely.

| Test | What it validates |
|---|---|
| `test_resolve_comparisons_*` | all modes produce correct `list[ResolvedComparison]`; mutual-exclusion errors fire; label existence checks work |
| `test_check_min_samples_*` | group-level and per-member thresholds; pooled hidden-minority detection |
| `test_build_binary_labels_*` | pooled and unpooled `_y` construction; row filtering; no index-alignment bugs |
| `test_collect_scores_schema` | output schema matches canonical column contract; no extra columns |
| `test_save_load_roundtrip` | `ClassificationAnalysis.save()` → `.load()` produces identical `scores`, `uns`, and available layers |
| `test_lazy_layers_missing` | `_LazyLayers.__getitem__` raises `KeyError` with clear message when layer absent |
| `test_misclassification_missing_layer` | `run_misclassification_pipeline()` raises `ValueError` when `multiclass_predictions` layer missing |
| `test_null_distributions_roundtrip` | `NullDistributions.save()` → `.load()` preserves arrays and index |

---

## `uns` structure

```python
uns = {
    # provenance
    "schema_version": "classification_v1",
    "created_at":     "2026-03-23T...",
    "git_commit":     "abc123",

    # run config
    "class_col":  "genotype",
    "id_col":     "embryo_id",
    "time_col":   "predicted_stage_hpf",
    "bin_width":  4.0,
    "n_permutations": 300,
    "feature_sets": {
        "embedding": {
            "spec":    "z_mu_b",                      # original user input
            "columns": ["z_mu_b_0", "z_mu_b_1", ...], # resolved columns used
        },
        "shape": {
            "spec":    ["total_length_um", "baseline_deviation_normalized"],
            "columns": ["total_length_um", "baseline_deviation_normalized"],
        },
    },

    # comparison membership (replaces negative_members JSON blob in table)
    "comparisons": {
        "homo__vs__wildtype_het": {
            "positive_members": ["homo"],
            "negative_members": ["wildtype", "het"],
            "positive_label":   "homo",
            "negative_label":   "wildtype+het",
        },
    },
}
```

---

## Helper

```python
def _listify(val: str | list[str]) -> list[str]:
    return [val] if isinstance(val, str) else list(val)
```

---

# Plotting Spec — Locked

---

## Design principles

- Plotters take DataFrames, not result objects.
- `plot_aurocs_over_time(scores, ...)` is the one canonical plotter.
- `results.plot_aurocs(...)` is sugar that passes `results.scores` and nothing else.
- All defaults are inferred from the data.
- Overlays are disabled with a warning if their required columns are absent.
- No alias helpers (`_auroc_col`, `_time_col`) — columns are always `auroc_obs` and
  `time_bin_center`.

---

## Return types

```python
from matplotlib.figure import Figure as MplFigure
from plotly.graph_objects import Figure as PlotlyFigure
```

Use `@overload` so type checkers get precise return types:

```python
@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["plotly"] = "plotly", **kw
) -> PlotlyFigure: ...

@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["matplotlib"], **kw
) -> MplFigure: ...

@overload
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: Literal["both"], **kw
) -> tuple[PlotlyFigure, MplFigure]: ...
```

Implementation uses the union internally:

```python
def plot_aurocs_over_time(
    scores: pd.DataFrame, *, backend: str = "plotly", **kw
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    ...
```

Always return `Figure`, not `(fig, ax)`. Caller can access axes via `fig.axes` if needed.
Returning `Figure` is correct for multi-facet grids where there are many axes.

---

## `infer_curve_col(scores)` — smart default, lives in plotter module

```python
def infer_curve_col(scores: pd.DataFrame) -> str:
    """
    Infer the best default curve_col for plot_aurocs_over_time.

    Rules
    -----
    Requires columns: positive_label, negative_label, comparison_id.

    If each positive_label maps to exactly one negative_label:
        → "positive_label"   (unambiguous, clean labels)
    If any positive_label appears with multiple negative_labels:
        → "comparison_id"    (avoids silently merging distinct comparisons)
        → emits UserWarning naming the ambiguous labels

    Notes
    -----
    - Does not handle the mirror case (one negative, multiple positives) because
      that is unambiguous for curve_col purposes. If you want curves per negative,
      pass curve_col="negative_label" explicitly.
    - inference only runs when curve_col=None is passed to the plotter.
    """
    _require_columns(scores, {"positive_label", "negative_label", "comparison_id"})
    pairs = scores[["positive_label", "negative_label"]].drop_duplicates()
    neg_per_pos = pairs.groupby("positive_label")["negative_label"].nunique()
    if (neg_per_pos > 1).any():
        ambiguous = sorted(neg_per_pos[neg_per_pos > 1].index.tolist())
        warnings.warn(
            f"positive_label is ambiguous for {ambiguous} "
            f"(each appears with multiple negative_labels). "
            f"Defaulting to curve_col='comparison_id'. "
            f"Pass curve_col='positive_label' explicitly to override, or "
            f"use facet_col='negative_label' to separate comparisons.",
            UserWarning, stacklevel=3,
        )
        return "comparison_id"
    return "positive_label"
```

---

## `_require_columns` — plotter-internal

```python
def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"scores missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )
```

---

## `plot_aurocs_over_time` — full locked signature

```python
def plot_aurocs_over_time(
    scores: pd.DataFrame,
    *,
    # ── what to plot ──────────────────────────────────────────────────────
    curve_col: str | None = None,
    # None → infer_curve_col(scores); requires positive_label + negative_label + comparison_id
    # explicit str → used directly; only that column is required to exist

    facet_row: str | None = None,
    facet_col: str | None = None,
    # None → "feature_set" if scores["feature_set"].nunique() > 1, else None

    # ── overlays — disabled with UserWarning if columns absent ────────────
    show_null_band: bool = False,
    # requires: auroc_null_mean, auroc_null_std

    show_significance: bool = True,
    # requires: pval

    sig_threshold: float = 0.01,
    show_chance_line: bool = True,

    # ── styling ───────────────────────────────────────────────────────────
    color_lookup: dict[str, str] | None = None,
    ylim: tuple[float, float] = (0.3, 1.05),
    xlim: tuple[float, float] | None = None,
    title: str = "AUROC over time",
    x_label: str = "Hours Post Fertilization (hpf)",
    y_label: str = "AUROC",
    style: dict | None = None,

    # ── output ────────────────────────────────────────────────────────────
    backend: Literal["plotly", "matplotlib", "both"] = "plotly",
    output_path: str | Path | None = None,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
```

### Validation block (runs first, before any computation)

```python
    # 1. Always-required columns
    _require_columns(scores, {"time_bin_center", "auroc_obs"})

    # 2. Infer curve_col (requires pos/neg/comparison_id if curve_col is None)
    if curve_col is None:
        curve_col = infer_curve_col(scores)   # emits warning if ambiguous
    else:
        _require_columns(scores, {curve_col})

    # 3. Infer facet_col
    if facet_col is None:
        facet_col = "feature_set" if scores["feature_set"].nunique() > 1 else None

    # 4. Conditional overlays — warn once per call site, then disable
    if show_null_band:
        missing = {"auroc_null_mean", "auroc_null_std"} - set(scores.columns)
        if missing:
            warnings.warn(
                f"show_null_band=True requires {sorted(missing)}. "
                f"Disabling null band.",
                UserWarning, stacklevel=2,
            )
            show_null_band = False

    if show_significance:
        if "pval" not in scores.columns:
            warnings.warn(
                "show_significance=True requires 'pval'. "
                "Disabling significance markers.",
                UserWarning, stacklevel=2,
            )
            show_significance = False
```

Warning spam in loops: use `warnings.warn(..., stacklevel=2)` — the default Python
warning filter deduplicates by (message, category, module, lineno), so identical calls
in a loop produce one warning. No extra machinery needed.

---

## Object method — no logic, just sugar

```python
def plot_aurocs(
    self,
    *,
    curve_col: str | None = None,
    facet_col: str | None = None,
    **kwargs,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    from .viz.auroc_over_time import plot_aurocs_over_time
    return plot_aurocs_over_time(
        self.scores,
        curve_col=curve_col,
        facet_col=facet_col,
        **kwargs,
    )
```

No logic. All inference happens in `plot_aurocs_over_time`.

---

## Confusion plotter

```python
def plot_confusion(
    scores: pd.DataFrame,       # for time axis reference
    confusion: pd.DataFrame,    # from layers["confusion"]
    *,
    feature_set: str | None = None,
    time_range: tuple[float, float] | None = None,
    backend: Literal["plotly", "matplotlib", "both"] = "plotly",
    output_path: str | Path | None = None,
) -> PlotlyFigure | MplFigure | tuple[PlotlyFigure, MplFigure]:
    """
    Required confusion columns: feature_set, comparison_id, time_bin_center,
    true_class, predicted_class, proportion.
    """

# Object convenience
def plot_confusion(self, **kwargs):
    conf = self.layers.get("confusion")
    if conf is None:
        raise KeyError(
            "No confusion layer available. "
            "Re-run run_classification() — confusion is saved automatically."
        )
    from .viz.confusion import plot_confusion
    return plot_confusion(self.scores, conf, **kwargs)
```

---

## What gets deleted from `classification/viz/`

| File / symbol | Fate |
|---|---|
| `plot_feature_comparison_grid` | deleted — replaced by `facet_col="feature_set"` |
| `plot_multiclass_ovr_aurocs` | deleted — was a wrapper around `MulticlassOVRResults` |
| `plot_multiple_aurocs` | deleted — dict-of-DataFrames pattern is gone |
| `plot_auroc_with_null` | **kept** — useful low-level primitive for custom figures |
| `_auroc_col()` helper | deleted |
| `_time_col()` helper | deleted |
| `classification.py` | becomes a shim with `FutureWarning` on all exports |

`misclassification.py` and `trajectory.py` are unaffected — they take
`embryo_predictions` directly and never touch `scores`.

---

## Compatibility shim — `classification.py`

```python
# classification.py — DEPRECATED, will be removed in a future release

import warnings
from .auroc_over_time import plot_aurocs_over_time

def plot_feature_comparison_grid(*args, **kwargs):
    warnings.warn(
        "plot_feature_comparison_grid is deprecated. "
        "Use plot_aurocs_over_time(scores, facet_col='feature_set') instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time with facet_col='feature_set'.")

def plot_multiclass_ovr_aurocs(*args, **kwargs):
    warnings.warn(
        "plot_multiclass_ovr_aurocs is deprecated. "
        "Use plot_aurocs_over_time(results.scores) instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time(results.scores).")

def plot_multiple_aurocs(*args, **kwargs):
    warnings.warn(
        "plot_multiple_aurocs is deprecated. "
        "Use plot_aurocs_over_time(scores, curve_col=...) instead.",
        FutureWarning, stacklevel=2,
    )
    raise NotImplementedError("Use plot_aurocs_over_time.")
```

Temporary `MulticlassOVRResults` input shim inside `plot_aurocs_over_time`:

```python
# At top of plot_aurocs_over_time, before validation
if isinstance(scores, MulticlassOVRResults):
    warnings.warn(
        "Passing MulticlassOVRResults to plot_aurocs_over_time is deprecated. "
        "Pass result.scores (a DataFrame) instead.",
        FutureWarning, stacklevel=2,
    )
    scores = scores.comparisons.rename(columns={
        "positive":       "positive_label",
        "negative":       "negative_label",
        "auroc_observed": "auroc_obs",
    })
    if "feature_set" not in scores.columns:
        scores = scores.copy()
        scores["feature_set"] = "default"
```

Use `isinstance(scores, MulticlassOVRResults)` not `hasattr` — avoids false matches.

---

## Final viz module layout

```
classification/viz/
  __init__.py           re-exports canonical + legacy shimmed symbols
  auroc_over_time.py    plot_aurocs_over_time + infer_curve_col + _require_columns
  confusion.py          plot_confusion (new)
  classification.py     FutureWarning shims only
  misclassification.py  unchanged (takes embryo_predictions df directly)
  trajectory.py         unchanged
```

---

## User experience summary

```python
# All modes — defaults just work
results.plot_aurocs()
# all-vs-rest + multi-feature → curve=positive_label, facet=feature_set
# pairwise + multi-feature    → warns, curve=comparison_id, facet=feature_set
# single feature              → curve=positive_label or comparison_id, no facet

# Override curve grouping
results.plot_aurocs(curve_col="comparison_id")
results.plot_aurocs(curve_col="negative_label")   # curves per reference

# Ambiguous case: same positive vs multiple negatives
# e.g. homo__vs__wildtype AND homo__vs__het both have positive_label="homo"
# → infer_curve_col warns and defaults to comparison_id
# → pass curve_col="positive_label" explicitly to override if intentional

# Override faceting
results.plot_aurocs(facet_col=None)               # all curves on one panel
results.plot_aurocs(facet_row="negative_label", facet_col="feature_set")

# Overlays
results.plot_aurocs(show_null_band=True, show_significance=True, sig_threshold=0.05)

# Colors
results.plot_aurocs(color_lookup={"homo": "#B2182B", "het": "#F7B267"})

# Backend
results.plot_aurocs(backend="matplotlib", output_path="figures/auroc.png")
results.plot_aurocs(backend="both")   # → (plotly_fig, mpl_fig)

# Standalone — same function, takes DataFrame directly
from analyze.classification.viz import plot_aurocs_over_time
plot_aurocs_over_time(results.scores, facet_col="feature_set")

# Confusion
results.plot_confusion()
results.plot_confusion(feature_set="embedding", backend="matplotlib")
```

---

# Module Organisation — Locked

---

## File layout

```
classification/
  __init__.py                     ← public API surface (see below)
  run_classification.py           ← orchestrator: wires engine pieces together

  # ── engine (internal implementation) ─────────────────────────────────────
  engine/
    __init__.py
    comparison_resolution.py      resolve_comparisons(), ResolvedComparison,
                                  check_min_samples(), all validators
                                  (_validate_group_input, _canonicalize_group,
                                  _as_group_list, _group_members,
                                  _check_labels_exist, etc.)
    loop.py                       _run_classification_loop(), _bin_and_aggregate(),
                                  _build_binary_labels(), _collect_scores(),
                                  _collect_binary_predictions(),
                                  _collect_multiclass_predictions(),
                                  _collect_confusion()
    null.py                       NullDistributions dataclass + save/load
    analysis.py                   ClassificationAnalysis, _LazyLayers,
                                  _validate_scores, _listify

  # ── legacy files (shimmed, not deleted) ──────────────────────────────────
  classification_test.py          FutureWarning shims:
                                    run_classification_test
                                    run_multiclass_classification_test
                                    extract_temporal_confusion_profile
  results.py                      FutureWarning shims:
                                    MulticlassOVRResults, ComparisonSpec
  classification_results.py       FutureWarning shim: ClassificationResults
  permutation_utils.py            unchanged (shared, not classification-specific)

  # ── viz ───────────────────────────────────────────────────────────────────
  viz/
    __init__.py                   exports canonical + shimmed legacy symbols
    auroc_over_time.py            plot_aurocs_over_time, infer_curve_col,
                                  _require_columns  (updated)
    confusion.py                  plot_confusion  (new)
    classification.py             FutureWarning shims only
    misclassification.py          unchanged
    trajectory.py                 unchanged

  # ── misclassification submodule ───────────────────────────────────────────
  misclassification/              unchanged internally
    __init__.py
    pipeline.py
    flagging.py
    io.py
    metrics.py
    null.py
    trajectory.py

  # ── tests ─────────────────────────────────────────────────────────────────
  tests/
    test_run_classification.py    new: run_classification() + ClassificationAnalysis
    test_comparison_resolution.py new: resolve_comparisons()
    test_null_distributions.py    new: NullDistributions save/load roundtrip
    test_classification_test.py   existing (keep until migration complete)
    test_classification_results.py existing (keep until migration complete)
    test_misclassification_*.py   unchanged
```

---

## `__init__.py` — complete proposed surface

```python
"""
analyze.classification
======================

Public API for time-binned AUROC classification with permutation testing.

Primary interface
-----------------
    run_classification(df, ...)       → ClassificationAnalysis
    ClassificationAnalysis.load(path) → ClassificationAnalysis

Legacy (deprecated, will be removed)
-------------------------------------
    run_classification_test           → use run_classification()
    MulticlassOVRResults              → use ClassificationAnalysis
    ClassificationResults             → use ClassificationAnalysis
"""

# ── Primary ───────────────────────────────────────────────────────────────────
from .run_classification import run_classification
from .engine.analysis import ClassificationAnalysis

# ── Submodules ────────────────────────────────────────────────────────────────
from . import viz
from . import misclassification

# ── Misclassification pipeline (unchanged, stays public) ──────────────────────
from .misclassification import run_misclassification_pipeline, run_stage_geometry

# ── Legacy (FutureWarning fires on call, not on import) ───────────────────────
from .classification_test import (
    run_classification_test,
    run_multiclass_classification_test,
    extract_temporal_confusion_profile,
)
from .results import MulticlassOVRResults, ComparisonSpec
from .classification_results import ClassificationResults

__all__ = [
    # Primary
    "run_classification",
    "ClassificationAnalysis",
    # Submodules
    "viz",
    "misclassification",
    # Misclassification pipeline
    "run_misclassification_pipeline",
    "run_stage_geometry",
    # Legacy
    "run_classification_test",
    "run_multiclass_classification_test",
    "extract_temporal_confusion_profile",
    "MulticlassOVRResults",
    "ComparisonSpec",
    "ClassificationResults",
]
```

---

## Public surface — three tiers

| Tier | Symbols | Status |
|---|---|---|
| **Primary** | `run_classification`, `ClassificationAnalysis` | new, canonical |
| **Submodules** | `viz`, `misclassification` | unchanged |
| **Pipeline** | `run_misclassification_pipeline`, `run_stage_geometry` | unchanged, stays public |
| **Legacy** | `run_classification_test`, `MulticlassOVRResults`, `ClassificationResults`, `ComparisonSpec` | importable, `FutureWarning` on call |

---

## FutureWarning shim pattern

Warning fires on **call**, not on import — existing import lines don't break.

```python
# classification_test.py — legacy shim
import warnings

def run_classification_test(df, groupby, groups="all", reference="rest",
                             features="z_mu_b", **kwargs):
    warnings.warn(
        "run_classification_test() is deprecated and will be removed in a future release. "
        "Use run_classification() instead:\n"
        "  from analyze.classification import run_classification\n"
        "  results = run_classification(df, class_col=groupby, id_col=..., time_col=...,\n"
        "                     positive=groups, negative=reference, features={...})",
        FutureWarning, stacklevel=2,
    )
    from .run_classification import run_classification as _run
    # translate old kwargs → new kwargs and delegate
    ...
```

---

## User import surface

```python
# New — canonical
from analyze.classification import run_classification, ClassificationAnalysis

# Old — still works, warns on call
from analyze.classification import run_classification_test, MulticlassOVRResults

# Viz
from analyze.classification.viz import plot_aurocs_over_time   # standalone
results.plot_aurocs()                                           # object sugar

# Misclassification (unchanged)
from analyze.classification import run_misclassification_pipeline, run_stage_geometry
from analyze.classification.viz import plot_confusion

# difference_detection shim (unchanged, adds run_classification + ClassificationAnalysis)
from analyze.difference_detection import run_classification   # FutureWarning on module import
```

---

## `analyze.difference_detection` update

```python
# analyze/difference_detection/__init__.py
import warnings
warnings.warn(
    "analyze.difference_detection is deprecated. "
    "Use analyze.classification instead.",
    FutureWarning, stacklevel=2,
)
from analyze.classification import (
    run_classification,
    ClassificationAnalysis,
    run_classification_test,
    MulticlassOVRResults,
    ClassificationResults,
    viz,
    misclassification,
    run_misclassification_pipeline,
    run_stage_geometry,
)
```
