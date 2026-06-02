# Well-ID Through-Line Refactor Plan

**Status:** Draft / pre-implementation. Nothing in this plan has been built yet.
**Owner:** mdcolon
**Related:** `identifier_and_wildcard_contract.md` (the existing grammar doc this plan revises and finishes).

---

## Why this exists

The pipeline already fans out **per well** for segmentation, snips, and auxiliary
masks, then collapses to **experiment grain** for features, QC, and analysis-ready.
Two things make that fan-out fragile and prevent running a single well end-to-end:

1. **`well_id` does not mean one thing.** In some persisted tables it is the local
   label `A01`; in others it is the experiment-qualified `20240418_A01`. Every
   per-well boundary carries defensive code to guess which it is
   (`.split("_")[-1]`, `startswith(f"{exp}_")`, two-candidate `isin(wanted)` sets).
2. **`video_id` is a redundant duplicate** of `(experiment_id, well)` — it is built
   as `f"{experiment_id}_{well_slug}"` and then re-parsed back apart downstream. It
   is, in effect, the globally-unique well identifier that `well_id` should have
   been all along.

Because `well_id` is ambiguous, no stage can be trusted to filter "this well"
the same way, and the experiment-grain merge barrier in the back half cannot be
safely pushed down to well grain. Fixing the identifier model is therefore the
prerequisite for the real goal: **run one well through the entire pipeline.**

### Canonical model (target)

Two names, not three. `well_index` is a **misnomer** — it stores the string label
`A01`, not an integer — and is redundant with the local label. It is renamed to
`well`.

```
experiment_id = 20240418            global experiment identifier
well          = A01                 local well label (was: well_index)
well_id       = 20240418_A01        global well identifier (replaces video_id)
channel       = BF                  local imaging channel token
image_id      = 20240418_A01_BF_t0003
embryo_id     = 20240418_A01_e07
snip_id       = 20240418_A01_e07_t0003
```

Compositional grammar — every ID is `parent_id + local_token`:

```
well_id   = {experiment_id}_{well}
image_id  = {well_id}_{channel}_t{time_int:04d}
embryo_id = {well_id}_e{local_embryo_index:02d}
snip_id   = {embryo_id}_t{time_int:04d}
```

**The sign on the door (put this in every schema comment + README):**
> `well` is the local label (`A01`), unique within an experiment.
> `well_id` is globally unique (`20240418_A01`).

---

## The two kingdoms (do not merge them)

This refactor introduces **two separate helper layers** with different jobs. They
must not collapse into one module.

| Kingdom | Question it answers | Lives in | Knows about |
|---|---|---|---|
| **Identity** | "What is the canonical name of this object?" | `shared/identifiers/` | biology/acquisition naming |
| **Orchestration** | "How do we run one well through a stage without rewriting glue?" | `pipeline_orchestrator/lib/` | workflow / paths / shards |

Rule: the well-runner **imports** identifier constructors; it never mints IDs.
> ID constructors define naming. Well runners define workflow execution.
> Keep the kingdoms separate so the crown doesn't start wearing a lab coat.

---

## Scopes

Five scopes, in strict dependency order. Each is independently reviewable.

```
Scope 1 (identifier grammar)  ──►  Scope 2 (migrate data + call sites to it)
                                          │
                                          ▼
                    Scope 3 (environment + path routing)   ← must precede the well-runner
                                          │
                                          ▼
                          Scope 4 (well-runner layer)
                                          │
                                          ▼
              Scope 5 (per-well back half + run one well E2E)
```

- Scope 1 is **purely additive and zero-risk** (new module, no call sites changed).
- Scope 2 is the **riskiest** (touches persisted data + every per-well boundary).
- **Scope 3 is inserted before the well-runner on purpose.** The well-runner must
  take `output_root` as a parameter; if it is built on `PROJECT_ROOT /
  data_pipeline_output` it bakes in the very repo-bound behavior we are removing,
  then has to be pried loose immediately. Extract path routing first.

The **structural wins** (config layer + 5 mechanical cleanups) are folded into these
scopes — see the "Structural Wins" section below for where each lands.

---

## Structural Wins (review 2026-06-01)

**Diagnosis.** The pipeline is **mid-migration between two conventions and froze
halfway.** A cleaner convention exists in the orphaned, never-`include`d `rules/*.smk`
files (subcommand dispatch, parameterized interpreter/paths, real `validate_*` rules);
the active Snakefile is the cruder half (deep `-m` module calls, hardcoded interpreter,
`echo "ok"` fake validation). Most wins below are *"finish that migration on a clean
foundation."* Win C (the config layer) is the piece neither generation fully built.

### Win C — Environment/config layer (the big one)

*Where your data lives is welded to where your code lives.* Fixed by Scope 3.
See Scope 3 for the full MVP spec; in short: `env.yaml` (machine) vs `config.yaml`
(science), with `output_root` as the one first-class absolute "data output route,"
and the path-contract layout staying in code.

### Wins 1–5 — Mechanical cleanups

| # | Problem (with evidence) | Fix (simple example) | Lands in | Risk |
|---|---|---|---|---|
| **1** | `PYTHONPATH="{PROJECT_ROOT}:..." "{PYTHON}" -m` prefix copy-pasted in **22 rules**; interpreter hardcoded to a home-dir path. | Define prefix once, **built from `env.yaml`**: `shell: "{RUN} <module> --flag {input.x}"`. | **Scope 3** | none — pure cleanup |
| **2** | **Two dispatch styles mixed**: scope/plate/stitch go through `tasks.py` subcommands; features/QC/aux/snips call deep `-m data_pipeline.feature_extraction.entrypoints.compute_mask_geometry`. | Route everything through `tasks.py`: `... tasks compute-mask-geometry ...`. Snakefile knows verbs, not module paths. | **Independent** (anytime; eases Scope 5) | low |
| **3** | `echo "ok" > {output.validated}` asserts **nothing** — written unconditionally (Snakefile lines 108, 206, 261). The `.validated` name lies. | Replace with a real validator that **reads the file and fails** if malformed (the orphaned `.smk` files already have these: `validate_plate_metadata`, `validate_physical_well_mapping`). | **Scope 2** (pairs with `validate_well_id`) | low, high-value |
| **4** | "Which wells run?" answered in **3 places**: `config.yaml`, `targets.py` `DEFAULT_EXPERIMENT_WELLS`, and the `discover_wells` checkpoint. | One `selected_well_ids_for_experiment()` in the well-runner; delete the `targets.py` copy. | **Scope 4** | low |
| **5** | The `discover_wells` checkpoint reads wells from real data, but config **bypasses** it when wells are listed — two code paths; config can name wells that don't exist. | Data (checkpoint) always decides which wells exist; config only **filters** the set. | **Scope 4** | low |

**Mapping summary:**
```
Win C  → Scope 3 (env layer)            Win 1  → Scope 3 (RUN helper from env.yaml)
Win 3  → Scope 2 (real validation)      Win 4  → Scope 4 (one well-selection source)
Win 5  → Scope 4 (checkpoint is truth)  Win 2  → Independent (dispatch facade)
```

**Open archaeology (not yet done):** dig git history for *why* the cleaner `.smk`
generation was abandoned mid-flight — was it intentional, or did someone just stop?
That context informs whether converging on its conventions is safe. The `.smk` files
already use `PYTHON_EXE`/`SRC_ROOT` params (i.e. were *already* moving toward the
Win-C config layer before being dropped).

---

### Scope 1 — Identifier grammar: clean, global, composed

**Goal.** One sacred place that mints IDs. Constructors are `well_id`-first, dumb
(no biology inference), and sanitize once at the root.

**Module layout** (split the flat `shared/identifiers.py` into a package):

```
src/data_pipeline/shared/identifiers/
    __init__.py        # re-exports every public name (compat shim → near-zero import churn)
    constructors.py    # build_well_id, build_image_id, build_embryo_id, build_snip_id
    parsers.py         # split_well_id, parse_embryo_id, normalize_embryo_local_track_id
    validators.py      # validate_well_id (and friends)
    README.md          # the "sign on the door"
```

**Constructors (`well_id`-first, the one join point at `build_well_id`):**

```python
def build_well_id(experiment_id: str, well: str) -> str:
    return f"{sanitize_experiment_id(experiment_id)}_{well}"   # sanitize ONCE here

def build_image_id(well_id: str, channel: str, time_int: int) -> str:
    return f"{well_id}_{channel}_t{time_int:04d}"

def build_embryo_id(well_id: str, local_embryo_index: int) -> str:
    return f"{well_id}_e{local_embryo_index:02d}"

def build_snip_id(embryo_id: str, time_int: int) -> str:
    return f"{embryo_id}_t{time_int:04d}"
```

Notes / decisions baked in:
- **`well_id`-first everywhere downstream.** Drop the old 4-arg
  `build_image_id(experiment_id, well, channel, time_int)` signature; there is
  exactly one place `experiment_id + well` are joined.
- **`sanitize_experiment_id` moves up into `build_well_id`** so `well_id` is born
  clean and every downstream ID inherits it. (Today it lives in `build_image_id`.)
- **`parsers.py` is the only sanctioned decomposition.** `split_well_id(well_id)
  -> (experiment_id, well)` replaces every ad-hoc `.split("_")`. The tracker-native
  `embryo_0 -> 0` regex (`_LOCAL_ID_RE`) moves here too — it is parsing, not minting.
- **`validate_well_id` fails loudly on a bare `A01`.** This is the guard rail that
  stops stale local-`well_id` data from silently flowing once the semantics flip.

**Channel** is treated like `well`: a local token, validated, **normalized
upstream** in metadata ingest (already happens at `apply_series_mapping.py` deriving
`channel_id` from `raw_channel_name`). The constructor stays dumb — it does not map
`Brightfield → BF`.

**Blast radius:** none yet. `__init__.py` re-exports keep all ~15 existing
`from data_pipeline.shared.identifiers import ...` sites working unchanged.

---

### Scope 2 — `well_index → well` rename, retire `video_id` (data-semantics migration)

**Goal.** Make persisted data and all call sites *use* the Scope-1 grammar. This is
where `well_id` flips from local→global in stored tables, and `well_index → well`.

**Schema changes (10 schema files carry these columns):**
- Introduce `well` (local label `A01`) and `well_id` (global `{experiment_id}_{well}`)
  as the two well identifiers in: `frame_contract`, `scope_metadata`,
  `plate_metadata`, `stitched_image_index`, `segmentation`, `snip_processing`,
  `auxiliary_masks`, `stage_predictions`. The old `well_index` column held the same
  `A01` value `well` now holds — so this is effectively `well_index → well`, except
  the integer SAM2 variant is dropped, not preserved (see DELETE note above).
- **Delete `well_index` entirely** (string column + SAM2 integer); no replacement.
- Redefine `well_id` = global `{experiment_id}_{well}` wherever it is stored.
- Delete `video_id` from the 5 segmentation sub-schemas
  (`SEGMENTATION_TRACKING`, `FRAME_DETECTIONS`, `SEED_SELECTION`,
  `TRACK_INSTANCES`, `MASK_RLE`).

**Code changes — replace ambiguity with Scope-1 helpers:**
- `segmentation_and_tracking.py:67-114` — delete the qualified/bare normalization
  (`candidate`, `storage_well_id`, `isin(wanted)`); filter on the unambiguous
  global `well_id`.
- `merge_segmentation_and_tracking_contracts.py:22-29` &
  `merge_snip_manifests.py:22-28` — delete the byte-identical
  `qualified/_slug = name.split("_")[-1]` blocks (absorbed by Scope 3).
- `csv_formatter.py:206` & `render_eval_video.py` — delete the
  `video_id.split("_")[-1]` re-parse sites.

**`well_index` is DELETED, not renamed (decision — verified 2026-06-01).** Earlier
drafts of this plan worried about a "derived integer plate-number" that needed an
honest name. That worry is resolved: **the integer is never used as math.** Trace:
- `normalize_segmentation_tracking.py:17` stores `well_index = int(...)` into the
  contract.
- Every downstream reader is **pure passthrough** —
  `compute_stage_predictions.py:98` reads it, immediately `.astype(str)`, and just
  copies it to an output column (the stage math uses `elapsed_time_s`, not
  `well_index`); `snip_processing/ops.py:219`, `snip_processing.py:125`, and
  `analysis_ready/assemble_features_qc_embeddings.py:42` all just shuttle it forward.
- Joins are on `image_id`/`snip_id`, never on `well_index`.

So `well_index` (both the string column **and** the SAM2 integer) carries no
information that `well` does not. **Remove it entirely.** There is no
`well_plate_number` to invent — nothing does plate-number math. (Confirm during
implementation that nothing *orders by* the integer; from the reads above we are
clear.) The legacy `extract_well_index("A01") -> int` helper that minted it is
retired with it.

**Data migration: regenerate (decision).** Existing on-disk artifacts
(`frame_contract.csv`, `segmentation_tracking.csv`, snip manifests) have the OLD
column headers and OLD `well_id` semantics. These are deterministic pipeline
outputs, so we **regenerate** rather than migrate in place. `validate_well_id`
fails loudly on a bare `A01` so any stale artifact cannot flow silently into new
code.

---

### Scope 3 — Environment + path-routing layer

**Goal.** Separate **where/how the pipeline runs on this machine** (environment) from
**what science it runs** (domain) — and stop welding the data location to the code
location. This must land **before** the well-runner (Scope 4), which depends on a
configurable `output_root`.

**The four layers, each with one job (the mental model this scope establishes):**

| Layer | Job | Lives in |
|---|---|---|
| Identity | how objects are named (`well_id = {exp}_{well}`) | `shared/identifiers/` (Scope 1) |
| Domain config | *what* science to run (experiments, thresholds) | `config.yaml` (version-controlled) |
| Environment config | *where/how* on this machine (python, device, paths) | `env.yaml` (gitignored) |
| **Path contract** | the **fixed layout under `output_root`** | **code** (the well-runner, Scope 4) |

> The **root** is environment. The **layout beneath the root** is code (pipeline
> contract). `output_root` is the env knob; `segmentation_and_tracking/{exp}/per_well/{well_id}/...`
> stays in code.

**The core problem (verified 2026-06-01).** Today
`DATA_ROOT = PROJECT_ROOT / config.get("data_root", "data_pipeline_output")`, and
`PROJECT_ROOT` is derived by walking up 3 dirs from the Snakefile. So **data location
is welded to code location** — every output piles up *inside the git repo*. The
interpreter is also hardcoded to a personal home-dir path
(`PYTHON = "/net/.../mdcolon/software/.../bin/python"`), so no other machine/user can
run the pipeline without editing the Snakefile.

**File placement (decision).** `env.yaml` lives **next to `config.yaml`**, beside the
Snakefile, loaded the same way:
```
src/data_pipeline/pipeline_orchestrator/
    Snakefile
    config.yaml          # domain / science      — committed (already here)
    env.yaml             # environment / paths   — GITIGNORED (new)
    env.example.yaml     # template              — committed (new)
```
Rationale: `config.yaml` is already loaded from `workflow.basedir`; `env.yaml` loads
the same way — one mental model, both files side by side. **Not** in the empty
`config/` subdir (would split the two config files across two locations); **not** at
repo root (too far from the workflow that consumes it).

**Loading (decision — keep the two kingdoms in separate files).** Load `config.yaml`
via Snakemake's `configfile:` (unchanged) and `env.yaml` as a plain dict alongside it.
Do **not** merge env into the Snakemake `config` namespace — that loses the file
separation that makes "gitignore my machine paths" clean.
```python
configfile: str(Path(workflow.basedir) / "config.yaml")   # domain (unchanged)
import yaml
env = yaml.safe_load((Path(workflow.basedir) / "env.yaml").read_text())
DATA_ROOT  = Path(env["paths"]["output_root"])
INPUTS_DIR = Path(env["paths"]["input_root"])
MODELS_DIR = Path(env["paths"]["models_root"])
PYTHON     = env["runtime"]["python"]
```
Add to the repo-root `.gitignore`:
```
# Machine-specific pipeline environment (paths, interpreter)
src/data_pipeline/pipeline_orchestrator/env.yaml
```

**MVP scope (deliberately tight).**
- Three files: `env.yaml` (gitignored, per-machine) + `env.example.yaml` (template,
  committed) + the existing `config.yaml` (domain, committed).
- **`output_root`, `input_root`, `models_root` are ALL first-class absolute paths**
  read from `env.yaml`. Decoupling input from output is a **one-line change**
  (`INPUTS_DIR = Path(env["paths"]["input_root"])` instead of `DATA_ROOT / "inputs"`):
  the 5 rules that reference `RAW_IMAGES_DIR`/`PLATE_METADATA_DIR` keep deriving from
  `INPUTS_DIR` unchanged, and the Python side already receives `--raw-images-dir`
  explicitly (nothing reconstructs input from output). Verified zero blast radius
  beyond that line.
- `python` and `device` move to `env.yaml` (kills the hardcoded interpreter; collapses
  the duplicated `device` that today lives under both `image_building` and
  `segmentation_and_tracking`).
- Flat resource defaults in `env.yaml`: `default_threads`, `default_mem_mb`.

**MVP `env.yaml`:**
```yaml
# env.yaml — gitignored, per-machine
runtime:
  python: "/net/.../envs/segmentation_grounded_sam/bin/python"
  device: "cuda"
paths:
  input_root:  "/net/.../inputs"                 # read-only source data (raw images, plate sheets)
  output_root: "/net/.../data_pipeline_output"   # churning results — the "data output route"
  models_root: "/net/.../models"                 # big static model weights
resources:
  default_threads: 4
  default_mem_mb: 16000
```

**Why all three roots now (not deferred).** Raw images and model weights are large,
static, and typically live on **shared read-only volumes**, while outputs are churning
scratch you write constantly. Forcing them under one root fights real cluster storage
layout — and you hit it the first time you run on someone else's data. The cost to do
it now is **one line each**; the cost to retrofit later is **moving data on disk +
re-validating**. So `input_root` and `models_root` are first-class in MVP, not deferred.

**Decisions baked in (MVP discipline):**
- **`project_root`: keep deriving it** from the Snakefile location. It works with zero
  per-machine setup; only the data/input/model roots need to be explicit. Do not force
  users to hand-set a path the code can compute.
- **Win 1 (command-prefix helper) lands here:** the repeated
  `PYTHONPATH="..." "{PYTHON}" -m` prefix (×22) is defined once and **built from
  `env.yaml`** (`runtime.python` + derived `PYTHONPATH`). See Structural Wins.

**Explicitly deferred (placeholders, not built):**
- `scratch_root`, `gpu_partition` — nothing reads them yet; add the day a rule needs
  them. (Unlike input/models roots, these have **no consumers today**, so deferring is
  correct here.)
- Per-stage resource overrides (`resources.segmentation_and_tracking.threads`) — flat
  defaults only for MVP.
- Feature on/off toggles (`features.compute_curvature: true`) — leave a **commented
  placeholder** in `config.yaml`; do not wire conditional rule inclusion.

---

### Scope 4 — Well-runner orchestration layer

**Goal.** Stop rewriting per-well Snakemake/Python glue. This is the second kingdom
(workflow, not identity). Lives near the workflow code, **not** in `shared/`.

```
src/data_pipeline/pipeline_orchestrator/lib/
    __init__.py
    well_runner.py
    paths.py
```

**What goes in `well_runner.py`** (the absorbed responsibilities, each currently
duplicated or ad-hoc across the codebase):

1. **Well discovery / selection** — one definition of "which wells run for this
   experiment," composing global `well_id`s from config slugs or from the
   `discover_wells` checkpoint. Replaces the **currently-undefined**
   `selected_well_ids_for_experiment` that the orphaned `.smk` files already
   reference, and the `experiment_well_ids` helper in the Snakefile.

   ```python
   def selected_well_ids_for_experiment(experiment_id: str, config: dict) -> list[str]: ...
   ```

2. **Per-well path construction** — one builder for the
   `stage/{experiment_id}/per_well/{well_id}/...` shape, killing the
   `per_well/{well_id}` string repeated across 5+ rule files.

   ```python
   def per_well_stage_dir(root, stage, experiment_id, well_id) -> Path: ...
   def per_well_artifact_path(root, stage, experiment_id, well_id, filename) -> Path: ...
   ```

3. **Shard discovery** — one way to enumerate per-well shard outputs under a stage
   root (replaces the `[p.name for p in per_well_root.iterdir()]` + `qualified`
   filtering duplicated in both `merge_*` modules).

   > **shard** = one well's slice of a stage's output, written to its own dir:
   > `stage/{exp}/per_well/{well_id}/contracts/<artifact>`. The experiment-level
   > table is just every shard concatenated. Segmentation/snips/aux already produce
   > shards today; Scope 5 makes *every* back-half stage produce them too.

   ```python
   def collect_well_shards(root, stage, experiment_id) -> list[WellShard]: ...
   ```

4. **Shard merge** — one generic concat+validate for per-well → experiment tables,
   parameterized by required columns. Replaces the two near-identical
   `merge_segmentation_and_tracking_contracts` / `merge_snip_manifests` bodies
   (the symlink-views logic stays stage-specific; the concat/validate is shared).

   ```python
   def merge_well_shards(input_paths, output_path, required_columns=None) -> None: ...
   ```

5. **`WellRun` value object** — binds `(experiment_id, well, well_id, root)` and
   exposes `stage_dir(stage)`. The single object a stage entrypoint receives so it
   never re-derives the shape.

   ```python
   @dataclass(frozen=True)
   class WellRun:
       experiment_id: str
       well: str          # A01
       well_id: str       # 20240418_A01
       root: Path
       def stage_dir(self, stage: str) -> Path:
           return self.root / stage / self.experiment_id / "per_well" / self.well_id
   ```

6. **`select_well(df, *, well_id)`** — the one filter, keyed on global `well_id`,
   replacing the scattered per-stage filters
   (`segmentation_and_tracking.py`, `auxiliary_masks/inference.py:154`,
   `compute_stage_predictions.py:71`), each of which currently guesses the grain.

**`paths.py`** holds the common non-well pipeline output path constructors
(experiment-level contract/validated/schema paths) so rules and entrypoints agree.

**Hard boundary:** the well-runner imports from `shared/identifiers/`. It never
defines `build_embryo_id` or any ID. Identity flows *into* orchestration, never the
other way.

**Depends on Scope 1/2/3** — the runner can only key on `well_id` once `well_id`
means exactly one thing (1/2), and it must build paths from a configurable
`output_root` (3), never from `PROJECT_ROOT`.

> **Directive for the implementer:** Before writing `well_runner.py`, extract path
> routing out of `PROJECT_ROOT` (Scope 3). The well-runner takes `output_root` as a
> parameter; it never derives paths from the Snakefile's location. Otherwise the
> well-runner encodes the very repo-bound output behavior we are removing.

Wins 4 (one well-selection source) and 5 (checkpoint-is-truth) land here — see
Structural Wins.

---

### Scope 5 — Integrate: push the back half to well grain, run one well E2E

**Goal.** The payoff:
`snakemake .../analysis_ready/{exp}/per_well/20240418_A01/...` runs one well
through **every** stage; the experiment table becomes a thin concat.

**Cohort-QC question: RESOLVED.** Audited 2026-06-01. Every back-half stage is
per-snip or per-embryo:
- `surface_area_qc` compares each snip to an **external** reference curve by
  `predicted_stage_hpf` — no cross-well data.
- `death_detection` iterates **per `embryo_id`** over each embryo's own time-series.
- **Zero `groupby` calls** across all of `quality_control/core` and
  `feature_extraction`.

**Consequence:** there are **no experiment-grain exceptions**. The well-runner does
**not** need an experiment-grain escape hatch. Scope 5 converts *all* back-half
stages to well grain.

**Rule conversions** — adopt the uniform rhythm for every stage:
```
compute_X_per_well   →   merge_X   →   validate_X
```
Stages to convert (currently experiment-grain in the active Snakefile):
- Features: `compute_mask_geometry`, `compute_curvature_metrics`,
  `compute_pose_kinematics`, `compute_fraction_alive`, `compute_stage_predictions`,
  `consolidate_features`.
- QC: `compute_segmentation_qc`, `compute_viability_qc`, `compute_death_detection`,
  `compute_surface_area_qc`, `compute_auxiliary_mask_qc`, `compute_focus_qc`,
  `compute_motion_qc`, `consolidate_qc`.
- `assemble_analysis_ready`.

**Prior art — use as blueprint, then DELETE (decision).** The orphaned,
never-`include`d `rules/*.smk` files (`segmentation_and_tracking.smk`,
`snip_processing.smk`, **`stage_predictions.smk`**) are an earlier per-well design
that was started (~commits `e25d3d93`, `ba9d41b8`), then superseded when the active
rules were inlined into the Snakefile, and left behind. `stage_predictions.smk` is a
working blueprint of the `compute_X_well → merge_X → validate_X` triplet applied to a
feature stage.

Decision: **delete-and-reimplement**, not resurrect. These files predate the clean
identifier grammar (Scopes 1–2) and the well-runner layer (Scope 3); reviving them
means re-inheriting the old ambiguity (and they call the undefined
`selected_well_ids_for_experiment`). Read them for the *shape* (the rhythm is right),
reimplement on the Scope-4 well-runner helpers, then delete the orphaned files.

**Also in Scope 5:**
- Add `wildcard_constraints` pinning `{well_id}` to the global grammar
  (`[0-9A-Za-z-]+_[A-H][0-9]{2}`) and `{experiment}` — currently there are none.
- Move the `merge_*` rules to the **end** of the DAG (thin concat), so they stop
  acting as a mid-pipeline barrier.

**Payoffs:**
- `--well 20240418_A01` runs end-to-end (impossible today — you hit the merge wall).
- Re-running one well recomputes only that well's downstream, not the whole experiment.
- One uniform story: `experiment × well → embryos → snips`, fanned out the whole way,
  merged only to publish.

**Depends on Scope 4** (the well-runner helpers).

---

## Decisions (resolved 2026-06-01)

1. **Scope 2 migration: regenerate.** On-disk artifacts are deterministic pipeline
   outputs; regenerate rather than migrate in place. `validate_well_id` guards
   against stale data flowing silently.
2. **Orphaned `.smk` files: delete-and-reimplement.** Use as a blueprint for the
   per-well rhythm, reimplement on the Scope-4 well-runner helpers, then delete.
   (See Scope 5.)
3. **Integer plate-number: DELETE.** Never used as math — pure passthrough
   everywhere. No `well_plate_number`; `well_index` is removed outright. (See
   Scope 2.)
4. **Environment before well-runner.** Scope 3 (path routing) precedes Scope 4
   (well-runner) so the runner is born taking `output_root` as a parameter, never
   `PROJECT_ROOT`. (See Scope 3 directive.)
5. **Delivery sequencing: Foundation now, integration designed-now / implemented-later.**

   ```
   Foundation refactor (do first):
     Scope 1: ID grammar
     Scope 2: data / call-site migration
     Scope 3: environment + path-routing layer
     Scope 4: well-runner utilities

   Follow-on integration (design now, implement after identifier semantics stable):
     Scope 5: push back half to per-well grain
   ```

   Rationale: Scope 5 is the only scope that **changes the DAG**, so it is where
   behavior can shift or hidden assumptions surface. Keeping it out of the
   foundation pass isolates that turbulence. (Note: the usual back-half risk —
   hidden cohort-level assumptions — was audited to **zero**: no `groupby` across
   QC/feature core; every stage is per-snip, per-embryo, or against an external
   reference curve. Every feature is per-well or derivable per-well. So Scope 5 is
   lower-risk than a typical DAG change, but still correctly sequenced last.)

## Audit facts this plan rests on (verified 2026-06-01)

- `well_id` is ambiguous across persisted tables (local vs experiment-qualified);
  defensive normalization exists at every per-well boundary.
- `video_id = f"{experiment_id}_{well_slug}"` — pure duplicate, re-parsed downstream.
- `well_index` stores the string label `A01`, not an integer; `build_well_id` today
  just strips it, so currently `well_id == well_index == "A01"`.
- A separate derived **integer** plate index exists for SAM2-internal math under the
  same `well_index` name — must not be conflated with the column.
- Only `quality_control.smk` is `include`d; the other 5 `rules/*.smk` are orphaned.
- No back-half stage uses cross-well/cohort statistics (no `groupby`; per-snip /
  per-embryo only).

---

# 👋 START HERE — pick up from this point

> **Hey Claude: read this whole document top-to-bottom to get up and running.**
> This file is the single source of truth for the well-id / per-well / config refactor.
> It must **never move** — its path is the stable anchor:
> `docs/refactors/streamline-snakemake/well_id_throughline_refactor_plan.md`
> (also symlinked into `/net/trapnell/vol1/home/mdcolon/proj/morphseq/docs/refactors/streamline-snakemake/`).
> Two repos are in play: this is **`morphseq-docs`** (the working copy on the docs
> branch); the pipeline code is the same `nlammers371/morphseq` repo.

## Where we are

- **Status:** planning complete, **no code written yet.** Everything above is the
  agreed design. Doc-only sessions so far.
- **Branch:** `mdcolon/20260222_docs_snakemake_remake`.
- **Goal in one line:** make the **global `well_id`** the through-line so the whole
  pipeline can run end-to-end on a single well, while cleaning up identifiers, config,
  and the duplicated per-well glue along the way.

## The agreed scope order (do them in this sequence)

```
FOUNDATION (do first):
  Scope 1  Identifier grammar         shared/identifiers/ package + compat shim (zero-risk, additive)
  Scope 2  Data-semantics migration   well_index→well, retire video_id, REGENERATE artifacts
  Scope 3  Environment + path routing  env.yaml (input/output/models roots first-class), Win 1, Win C
  Scope 4  Well-runner utilities        pipeline_orchestrator/lib/, consumes output_root (Wins 4,5)

INTEGRATION (design done; implement only after Foundation is stable):
  Scope 5  Per-well back half + run-one-well E2E   (Wins 2,3 nearby; delete orphaned .smk)
```

Hard rule (already in Scope 4): **the well-runner takes `output_root` as a parameter,
never derives it from `PROJECT_ROOT`.** That's why Scope 3 precedes Scope 4.

## Decisions already locked (don't relitigate)

- `well` = local label `A01`; `well_id` = global `{exp}_{well}` = `20240418_A01`.
- `video_id` → deleted (it was the global well id all along).
- `well_index` → **deleted outright** (pure passthrough, never math; no `well_plate_number`).
- Migration → **regenerate** artifacts, not migrate in place; `validate_well_id` fails
  loud on bare `A01`.
- Config → two files: `config.yaml` (science, committed) + `env.yaml` (machine,
  GITIGNORED) + `env.example.yaml` (template, committed), **beside the Snakefile**,
  loaded as a plain dict (not merged into Snakemake's `config` namespace).
- `input_root`, `output_root`, `models_root` → **all first-class absolute env paths**
  (decoupling input from output is ~1 line each, verified zero blast radius).
- Deferred (placeholders only): `scratch_root`, `gpu_partition`, per-stage resource
  overrides, feature on/off toggles.
- Orphaned `.smk` → delete-and-reimplement (blueprint only).

## ⏳ OPEN — next thing to think about (the "add a feature easily" exercise)

We started designing the **pattern for adding a new task/feature** and stopped here.
The question that triggered it: *"should the path indices live in config and be
imported by each task?"*

**Finding from the code:** the path **layout** is currently encoded ~3× (Snakefile
rule string, the consolidate rule's input string, AND `feature_extraction/io/paths.py`
which is a real structured registry — `FEATURE_OUTPUT_FILENAMES` +
`feature_output_path(data_root, experiment_id, feature_name)`). They're hand-kept in
sync. `io/paths.py` is *already the right pattern* but only the Python side reads it;
the Snakefile re-types the layout as raw strings.

**The design principle we landed on (three different things, often conflated):**

| Thing | Example | Where it lives | Why |
|---|---|---|---|
| **Root** | `output_root` | `env.yaml` (config) | machine-specific |
| **Layout** | `{stage}/{exp}/per_well/{well_id}/{file}` | **CODE** (a shared `paths.py` registry) | it's a *contract* — must be identical everywhere, never vary per machine |
| **Selection** | "run features: geometry, pose" | `config.yaml` (config) | a science decision |

So the answer to "imported by each task?" = **yes, but from a code registry, NOT from
config.** Putting the layout in config would let two users write to different paths and
nothing could find anyone's outputs — a contract must be fixed, config is for things
that vary.

**Target pattern (one registry → "add a feature = one row"):**
```python
# a pipeline-wide paths registry (code)
STAGES = {
    "mask_geometry": {"family": "computed_features", "file": "mask_geometry_metrics.csv", "grain": "well"},
    "foo":           {"family": "computed_features", "file": "foo_metrics.csv",           "grain": "well"},  # new = 1 line
}
def stage_artifact_path(output_root, stage, experiment_id, well_id=None, *, merged=False):
    spec = STAGES[stage]
    base = output_root / spec["family"] / experiment_id
    if merged or spec["grain"] == "experiment":
        return base / spec["file"]
    return base / "per_well" / well_id / spec["file"]
```
Imported by **both** the Snakefile and the Python entrypoint, so rule-output and
code-output are guaranteed identical. "Add a feature" becomes a recipe: (1) one
registry row, (2) the compute function, (3) a `tasks.py` subcommand, (4) copy a rule
template changing only the stage name.

**DECISION STILL TO MAKE (mdcolon, later):**
1. **Registry scope:** one pipeline-wide `shared/paths.py` registry for *all* stages,
   vs. per-domain registries (like today's `feature_extraction/io/paths.py`) that the
   well-runner composes. (Claude leans pipeline-wide for the "add in one place" goal;
   bigger consolidation.)
2. Where it formally lands (likely Scope 4, alongside `per_well_artifact_path` — same
   function family, both consume `output_root`).

## Suggested first action when resuming

Decide the registry-scope question above, then begin **Scope 1** (the additive,
zero-risk `shared/identifiers/` package + compat `__init__.py` shim) — it changes no
call sites and can land safely on its own.
