# QC and Analysis-Ready Pipeline Spec
**Status:** Living document — update as implementation progresses
**Scope:** Chunk A (QC integration) and Chunk B (analysis-ready table). Chunk C (VAE/embeddings) is next after these two.

---

## Context

The pipeline currently ends at `consolidate_features` → `consolidated_snip_features.csv`.
Two stages remain before a dataset is analysis-ready:

1. **QC** — flag bad snips and produce a per-snip `use_snip` gate
2. **Analysis-ready assembly** — join features + QC flags into a single flat table

QC is done at the **snip level**. There is no per-embryo QC at this stage — that is deferred.
All the Python logic for both stages already exists. What is missing is Snakemake wiring and,
where needed, conforming existing modules to the core/entrypoint split.

---

## Status legend

| Status | Meaning |
|---|---|
| **Implemented** | Computation exists and should be wrapped, not redesigned |
| **New** | Computation must be written for this chunk |
| **Stub** | Deliberately returns deterministic placeholder output (e.g. all `False`); wired into the DAG for schema continuity; real implementation deferred |

---

## Core / Entrypoint / IO Split Pattern

Every data pipeline module follows this layout. Enforce it for all new and touched files.

```
src/data_pipeline/<domain>/
  core/
    <logic>.py          ← pure computation, no I/O, no CLI
  entrypoints/
    <verb>_<noun>.py    ← CLI wrapper: parse args → load → call core → validate → write
  io/
    loaders.py          ← reads contract files, returns DataFrames
    writers.py          ← writes CSV and .validated sentinel; calls validators, does not own logic
  validators.py         ← schema and row-level invariant checks; raises on failure
```

**Rules:**
- `core/` functions take DataFrames and plain Python parameters, and return DataFrames or typed result objects. They do not perform file I/O, parse CLI arguments, or read workflow config. No `pd.read_csv`, no `open()`, no `Path` resolution inside `core/`.
- `validators.py` owns schema and contract checks: required columns, unique `snip_id`, non-null boolean flags, null semantics for nullable columns, join row-count assumptions. Functions raise clear exceptions on failure. This logic is testable independently of I/O.
- `io/writers.py` is thin: write CSV → call validator → write `.validated` sentinel. It orchestrates validation but does not embed validation logic.
- `entrypoints/` are the only things the Snakefile calls via `-m`. They own argument parsing, loading, calling core, and handing results to `io/writers.py`.
- `io/loaders.py` and `io/writers.py` are shared within a domain. Don't duplicate load/write logic across entrypoints.
- Snakemake rules pass all config as explicit `params` → CLI flags. No module reads `config.yaml` directly.
- Prefer wrapping existing logic without behavioral rewrites. If an existing module violates the contract (reads files internally, parses config), do the minimal refactor to push I/O to the entrypoint while preserving computation in `core/`.

Assembly line per stage: `core/ computes → validators.py checks → writers.py writes + sentinel`

**Snakemake invocation convention:** All Snakefile Python calls use:
```
"$PYTHON" -m data_pipeline.<domain>.entrypoints.<entrypoint>
```
Never use direct script paths. This enforces package-namespace hygiene and keeps imports consistent.

**`validators.py` structure:** One validator function per output artifact contract, plus shared helper assertions beneath. Example:
```python
# quality_control/validators.py
def assert_unique_snip_id(df): ...
def validate_segmentation_qc_flags(df): ...
def validate_death_detection_flags(df): ...
def validate_qc_flags(df): ...  # consolidated
```
Do not write a single generic validator that grows to cover everything — keep them per-artifact.

Existing reference: `src/data_pipeline/feature_extraction/` — use it as the template (note: it predates `validators.py`; add validators when touching QC and analysis-ready modules).

---

## Sentinel contract

A `.validated` sentinel means:
- the corresponding CSV was validated and written successfully
- schema validation passed (required columns present, correct dtypes)
- row-level invariants for that stage passed (unique `snip_id`, no nulls in flag columns)
- the artifact is safe for downstream consumption

**The CSV is the true output artifact. The `.validated` file is a completion marker.**
Downstream rules must declare both as inputs to ensure validation is part of contract completion.
Do not use `.validated` as the sole freshness artifact — that was the bug we just fixed upstream.

`io/writers.py` is thin: validate the in-memory result against the artifact contract, finalize
the CSV artifact, then write the `.validated` sentinel. The sentinel is only written after
validation passes. Validate before writing the final CSV — do not leave an invalid artifact on
disk if validation fails.

**Dense table rule:** All QC stage outputs are dense per-`snip_id` tables over the same snip
universe as `consolidated_snip_features.csv`. A stage that finds no bad snips still emits one
row per `snip_id` with all flag columns set to `False`. Zero-row outputs are only allowed if
the upstream snip universe is itself empty.

---

## Boolean dtype contract

All **flag columns** in validated QC outputs must be **non-null booleans**.
- Use pandas nullable boolean (`pd.BooleanDtype()`) during intermediate assembly if nulls arise during computation.
- `validators.py` must enforce non-null before the sentinel is written — any null in a flag column is a hard error at validation time.
- `death_inflection_time_int` is **not a flag column** — it is a nullable metadata column and is explicitly exempt from the non-null boolean rule. It must be null/NaN when `dead_flag=False`.

---

## Chunk A: QC Integration

### Goal
Wire the existing `quality_control/` Python into the Snakemake DAG. Produce `qc_flags.csv`
per experiment with one row per `snip_id`.

### Key semantics
- **Unit of analysis:** `snip_id`
- **Row contract:** every QC output CSV has one row per `snip_id`, `snip_id` unique
- **`use_snip` gate:** `use_snip = not(any SNIP_EXCLUSION_FLAGS flag is True)` — derived deterministically in `consolidate_qc`
- **Missing upstream input:** hard error — no defaults, no silent skips
- **Empty flag table:** valid (see empty table rule above)
- **Row count (MVP invariant):** all QC outputs preserve the same `snip_id` universe as `consolidated_snip_features.csv`; row counts must match. This is an MVP constraint — if optional filtering stages are added later, revisit explicitly.

### QC stages — what each one is and what it produces

| Snakemake rule | What it checks | Columns produced | Inputs | Status |
|---|---|---|---|---|
| `compute_segmentation_qc` | Edge contact, discontinuous masks, overlapping masks — geometric integrity of SAM2 output | `snip_id`, `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag` | `segmentation_tracking.csv` | **Implemented** — wrap `segmentation_qc/segmentation_quality_qc.py`. Drop `mask_quality_flag` composite. |
| `compute_viability_qc` | Per-snip viability from mask geometry (area collapse, aspect ratio extremes) — frame-level signal only, not temporal | `snip_id`, `viability_flag` | `mask_geometry_metrics.csv` | **New** — geometry threshold logic, no UNet, no `fraction_alive` |
| `compute_death_detection` | Sustained decline in `fraction_alive` with persistence validation (80% threshold, 4hr buffer, tunable knobs) | `snip_id`, `dead_flag`, `death_inflection_time_int` | `fraction_alive.csv` | **Implemented** — wrap `death_detection.py`. Rename `dead_flag2` → `dead_flag`. |
| `compute_surface_area_qc` | SA vs stage-matched reference curves (k_upper=1.4 × p95, k_lower=0.7 × p5) | `snip_id`, `sa_outlier_flag` | `mask_geometry_metrics.csv`, SA reference CSV | **Implemented** — wrap `surface_area_outlier_detection.py` |
| `compute_auxiliary_mask_qc` | UNet-based imaging artifacts: yolk detection and bubbles. Kept for inspection; being phased out. | `snip_id`, `yolk_flag`, `bubble_flag` | `auxiliary_masks.csv` | **Stub** — returns all `False`. Informational only, does not gate `use_snip`. |
| `compute_focus_qc` | Per-snip focus quality via `entropy_rel` — replaces UNet focus | `snip_id`, `focus_flag` | `frame_contract.csv` + z-stack metrics (TBD) | **Stub** — schema-continuity stub only. Returns `focus_flag=False` for all snips. Must not introduce new upstream dependencies, scan ND2 files, or read motion-artifact result directories. Real impl (`entropy_rel`) deferred. |
| `compute_motion_qc` | Between-frame motion artifacts via `ncc_min` / `bad_pair_frac_ncc` | `snip_id`, `motion_flag` | `frame_contract.csv` + z-stack metrics (TBD) | **Stub** — schema-continuity stub only. Returns `motion_flag=False` for all snips. Must not introduce new upstream dependencies, scan ND2 files, or read motion-artifact result directories. Real thresholds: `ncc_min < 0.85` OR `bad_pair_frac_ncc > 0.10`; deferred. |
| `consolidate_qc` | Merges all flag CSVs; computes `use_snip` | `snip_id`, `use_snip`, + all flag columns | All 7 flag CSVs + `.validated` sentinels | **Implemented** — wrap `consolidation/consolidate_qc.py`; update to `use_snip` and new flag set |

**`death_inflection_time_int` null semantics:** nullable column. Must be `null`/`NaN` when `dead_flag=False`. Populated only when `dead_flag=True`. Downstream consumers must handle nulls.

**`use_snip` logic:** `use_snip = not(edge_flag OR discontinuous_mask_flag OR overlapping_mask_flag OR viability_flag OR dead_flag OR sa_outlier_flag OR focus_flag OR motion_flag)`. `yolk_flag` and `bubble_flag` are informational — they do not gate `use_snip` while the UNet is being phased out.

> **Phaseout plan for `compute_auxiliary_mask_qc`:** Keep wired so `yolk_flag` and `bubble_flag`
> appear in `qc_flags.csv` for inspection. Once `compute_focus_qc` is real and UNet is retired,
> remove `compute_auxiliary_mask_qc` from the DAG and drop `yolk_flag`/`bubble_flag` from the schema.

### Target output contract
```
quality_control/{experiment}/
  segmentation_qc/
    segmentation_qc_flags.csv       ← snip_id, edge_flag, discontinuous_mask_flag, overlapping_mask_flag
    .segmentation_qc_flags.validated
  viability_qc/
    viability_qc_flags.csv          ← snip_id, viability_flag
    .viability_qc_flags.validated
  death_detection/
    death_detection_flags.csv       ← snip_id, dead_flag, death_inflection_time_int (nullable)
    .death_detection_flags.validated
  surface_area_qc/
    surface_area_qc_flags.csv       ← snip_id, sa_outlier_flag
    .surface_area_qc_flags.validated
  auxiliary_mask_qc/
    auxiliary_mask_qc_flags.csv     ← snip_id, yolk_flag, bubble_flag (all False, stub)
    .auxiliary_mask_qc_flags.validated
  focus_qc/
    focus_qc_flags.csv              ← snip_id, focus_flag (stub: all False)
    .focus_qc_flags.validated
  motion_qc/
    motion_qc_flags.csv             ← snip_id, motion_flag (stub: all False)
    .motion_qc_flags.validated
  consolidated/
    qc_flags.csv
    .qc_flags.validated
```

`qc_flags.csv` required columns:
`snip_id`, `use_snip`, `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag`,
`viability_flag`, `dead_flag`, `death_inflection_time_int`, `sa_outlier_flag`,
`focus_flag`, `motion_flag`, `yolk_flag`, `bubble_flag`

`snip_id` unique. All flag columns non-null boolean except `death_inflection_time_int` (nullable).

### Module structure to enforce
```
src/data_pipeline/quality_control/
  core/
    segmentation_quality_qc.py      ← promote from segmentation_qc/; drop mask_quality_flag
    viability_qc.py                 ← new; geometry thresholds on mask_geometry_metrics
    death_detection.py              ← promote existing; rename dead_flag2 → dead_flag
    surface_area_outlier_detection.py ← promote existing
    auxiliary_mask_qc.py            ← promote from auxiliary_mask_qc/; yolk+bubble only, stub
    focus_qc.py                     ← new stub; real impl: entropy_rel
    motion_qc.py                    ← new stub; real impl: ncc_min / bad_pair_frac_ncc
    consolidate_qc.py               ← promote from consolidation/; update flag set + use_snip
  entrypoints/
    compute_segmentation_qc.py
    compute_viability_qc.py
    compute_death_detection.py
    compute_surface_area_qc.py
    compute_auxiliary_mask_qc.py
    compute_focus_qc.py
    compute_motion_qc.py
    consolidate_qc.py
  io/
    loaders.py
    writers.py                      ← thin: write CSV, call validators, write sentinel
  validators.py                     ← validate_qc_stage(df), validate_qc_flags(df), assert_unique_snip_id(df)
```

### DAG order
```
segmentation_tracking  mask_geometry  fraction_alive  auxiliary_masks  (z-stack TBD)
        ↓                  ↓    ↓          ↓               ↓             ↓       ↓
compute_segmentation_qc  compute_viability_qc  compute_surface_area_qc
compute_death_detection  compute_auxiliary_mask_qc  compute_focus_qc  compute_motion_qc
                    ↓ (all 7 flag CSVs + .validated sentinels)
               consolidate_qc
                    ↓
              qc_flags.csv
```

All 7 compute rules are independent and run in parallel. `consolidate_qc` waits for all.

### config layering

Use code defaults plus Snakemake overrides.

- `src/data_pipeline/quality_control/config.py` holds the canonical default QC values.
- `config.yaml` and `snakemake --config` provide override values only.
- Each QC entrypoint merges `defaults + overrides` before calling core logic.
- Snakemake remains the operator-facing tuning surface, but defaults are not duplicated in YAML.

### config.yaml overrides
```yaml
quality_control:
  segmentation_qc:
    edge_margin_pixels: 2
    max_mask_overlap_fraction: 0.3

  viability_qc:
    min_mask_size_px: 100         # TBD — thresholds will be set when rule is implemented

  death_detection:
    persistence_threshold: 0.8   # fraction of post-inflection points that must be dead
    lead_time_hr: 4.0             # buffer before inflection point
    decline_rate_threshold: 0.05

  surface_area_qc:
    k_upper: 1.4                  # flag if SA > k_upper × p95 reference
    k_lower: 0.7                  # flag if SA < k_lower × p5 reference

  auxiliary_mask_qc: {}           # no knobs — stub returning all False

  focus_qc: {}                    # TBD — entropy_rel cutoff set when rule is implemented

  motion_qc:
    ncc_min_threshold: 0.85
    bad_pair_frac_threshold: 0.10
```

Each Snakemake rule reads only its own `config["quality_control"]["<stage>"]` block via `params`.
The entrypoint merges that override block onto the canonical code defaults for the stage.

### Exclusion policy (code, not config)
Which flags gate `use_snip` is an architectural decision, not an operator knob. It lives in:

```python
# src/data_pipeline/schemas/quality_control.py
SNIP_EXCLUSION_FLAGS = [
    "edge_flag",
    "discontinuous_mask_flag",
    "overlapping_mask_flag",
    "viability_flag",
    "dead_flag",
    "sa_outlier_flag",
    "focus_flag",
    "motion_flag",
]

SNIP_INFORMATIONAL_FLAGS = [
    "yolk_flag",
    "bubble_flag",  # auxiliary mask QC, being phased out
]
```

`consolidate_qc` computes `use_snip = not(any(SNIP_EXCLUSION_FLAGS))`.
Adding a flag to the exclusion policy requires a code change (intentional, reviewable).
Tuning its threshold is a config change (routine, operator-facing).

---

## Chunk B: Analysis-Ready Assembly

### Goal
Add a single Snakemake rule after `consolidate_features` + `consolidate_qc`. Produces the
final flat table. Skip embeddings for MVP (`embedding_calculated=False`).

### What already exists
`src/data_pipeline/analysis_ready/assemble_features_qc_embeddings.py` — stub that merges
features + QC and sets `embedding_calculated=False`.

### Join semantics
The assembly join is a **validated 1:1 join on `snip_id`**:
- Both `consolidated_snip_features.csv` and `qc_flags.csv` must have unique `snip_id` before joining
- Duplicate keys on either side are a hard error
- Missing join partners (`snip_id` in one but not the other) are a hard error for MVP
- Row count of `analysis_ready.csv` must equal row count of each input
- Column name collisions between the two inputs are a hard error, except for `snip_id` itself

### Target output contract
```
analysis_ready/{experiment}/
  analysis_ready.csv
  .analysis_ready.validated
  analysis_ready.schema.json   ← generated from canonical schema definition at write time
```

`analysis_ready.csv` required columns (from `src/data_pipeline/schemas/analysis_ready.py`):
All columns from `consolidated_snip_features.csv` + all QC flag columns from `qc_flags.csv`,
joined on `snip_id`. Plus: `embedding_calculated` (always `False` for MVP).

`analysis_ready.schema.json` is a required MVP output. It must be generated by `io/writers.py`
from the canonical schema definition at write time — not hand-authored. Downstream tooling
may depend on it for column discovery.

### Module structure to enforce
```
src/data_pipeline/analysis_ready/
  core/
    assemble.py                  ← join logic only, no I/O
  entrypoints/
    assemble_analysis_ready.py   ← CLI, calls core, hands result to io/writers.py
  io/
    loaders.py
    writers.py                   ← thin: write CSV, call validators, write sentinel
  validators.py                  ← validate_analysis_ready(df), assert_unique_snip_id(df), assert_1to1_join(df, left, right)
```

Promote the existing stub's logic to `core/assemble.py`. Write a proper entrypoint.

### Snakemake rule to add
One rule: `assemble_analysis_ready`. Inputs:
- `consolidated_snip_features.csv` + `.validated`
- `qc_flags.csv` + `.validated`

Outputs:
- `analysis_ready/{experiment}/analysis_ready.csv`
- `analysis_ready/{experiment}/.analysis_ready.validated`

Update the `all` default target to require `analysis_ready.csv` instead of
`feature_extraction_complete`.

---

## Chunk C: VAE / Embeddings (next, not now)

`src/data_pipeline/embeddings/` is currently empty. This is a full implementation task —
not wiring. Defer until A and B are done and smoke-tested.

When implemented, it slots between `assemble_analysis_ready` and a future `embeddings_complete`
target. The `embedding_calculated` column flips from `False` to `True` once embeddings are written.

---

## Verification gates (both chunks)

After A and B are wired and run on the smoke subset (20240418, wells A01/C01):

1. `quality_control/20240418/consolidated/qc_flags.csv` exists and has `use_snip` column.
2. `qc_flags.csv` has unique `snip_id` — no duplicates.
3. `qc_flags.csv` row count matches `consolidated_snip_features.csv` row count exactly (MVP invariant).
4. All flag columns in `qc_flags.csv` are non-null boolean dtype.
5. `focus_flag`, `motion_flag`, `yolk_flag`, `bubble_flag` are uniformly `False` (stubs).
6. `dead_flag=False` rows have `death_inflection_time_int` as null/NaN.
7. `analysis_ready/20240418/analysis_ready.csv` exists with all expected columns.
8. `analysis_ready.csv` has unique `snip_id` — no duplicates.
9. `analysis_ready.csv` row count equals `qc_flags.csv` row count equals `consolidated_snip_features.csv` row count.
10. No row in `analysis_ready.csv` has a missing `snip_id`.
11. `embedding_calculated` is uniformly `False`.
12. Dry-run of `snakemake` (default `all` target) shows no missing jobs.

---

## Agent handoff notes

If splitting A and B across separate agents, give each agent:
- This spec file
- `feature_extraction_module_inventory.md` (core/entrypoint pattern reference)
- `src/data_pipeline/pipeline_orchestrator/Snakefile`
- The relevant `src/data_pipeline/quality_control/` or `src/data_pipeline/analysis_ready/` directory
- The schema files: `src/data_pipeline/schemas/quality_control.py`, `src/data_pipeline/schemas/analysis_ready.py`

**Scope boundaries:**
- Agent A: add QC rules to the Snakefile after `consolidate_features`; create/promote `quality_control/core/`, `quality_control/entrypoints/`, `quality_control/io/`. Do not touch `analysis_ready/` or the `all` target.
- Agent B: add `assemble_analysis_ready` rule; create/promote `analysis_ready/core/`, `analysis_ready/entrypoints/`, `analysis_ready/io/`; update the `all` target. Do not touch `quality_control/`.
- If either agent needs a cross-cutting change (shared utility, schema fix, Snakefile import), note it explicitly rather than silently broadening scope.
