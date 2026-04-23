# Agent A Briefing: QC Integration (Chunk A)

**Your job:** Wire the existing `quality_control/` Python into the Snakemake pipeline.
Produce `qc_flags.csv` per experiment with one row per `snip_id`.
Do not touch `analysis_ready/` or the `all` Snakemake target — that is Agent B's scope.

---

## Read these first

1. `docs/refactors/streamline-snakemake/QC_AND_ANALYSIS_READY_SPEC.md` — full contract spec
2. `docs/refactors/streamline-snakemake/feature_extraction_module_inventory.md` — core/entrypoint pattern reference
3. `src/data_pipeline/feature_extraction/` — use as the structural template for your work

---

## What already exists (do not rewrite logic)

| File | What it is |
|---|---|
| `src/data_pipeline/quality_control/segmentation_qc/segmentation_quality_qc.py` | Segmentation flags — wrap into `core/` |
| `src/data_pipeline/quality_control/death_detection.py` | Death persistence detection — wrap into `core/`; rename `dead_flag2` → `dead_flag` |
| `src/data_pipeline/quality_control/surface_area_outlier_detection.py` | SA outlier detection — wrap into `core/` |
| `src/data_pipeline/quality_control/auxiliary_mask_qc/imaging_quality_qc.py` | UNet stub — wrap into `core/`; keep only `yolk_flag` + `bubble_flag` (drop `focus_flag` from this module) |
| `src/data_pipeline/quality_control/consolidation/consolidate_qc.py` | QC consolidation — wrap into `core/`; update to new flag set and `use_snip` |
| `src/data_pipeline/schemas/quality_control.py` | Schema — **update** `REQUIRED_COLUMNS_QC`, rename `use_embryo` → `use_snip`, add `SNIP_EXCLUSION_FLAGS`, add `SNIP_INFORMATIONAL_FLAGS`, drop `mask_quality_flag` |

---

## Target module structure

Create this layout (promote existing logic, add entrypoints, add validators):

```
src/data_pipeline/quality_control/
  core/
    segmentation_quality_qc.py      ← promote; drop mask_quality_flag composite
    viability_qc.py                 ← NEW: geometry thresholds on mask_geometry_metrics
    death_detection.py              ← promote; rename dead_flag2 → dead_flag
    surface_area_outlier_detection.py ← promote
    auxiliary_mask_qc.py            ← promote; yolk_flag + bubble_flag only (no focus_flag)
    focus_qc.py                     ← NEW stub: returns focus_flag=False for all snips
    motion_qc.py                    ← NEW stub: returns motion_flag=False for all snips
    consolidate_qc.py               ← promote; update flag set, use_snip
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
    writers.py                      ← thin: validate → write CSV → write sentinel
  validators.py                     ← per-artifact validators + shared helpers
```

---

## Key contracts

**Output directory:**
```
quality_control/{experiment}/
  segmentation_qc/segmentation_qc_flags.csv       ← snip_id, edge_flag, discontinuous_mask_flag, overlapping_mask_flag
  viability_qc/viability_qc_flags.csv             ← snip_id, viability_flag
  death_detection/death_detection_flags.csv        ← snip_id, dead_flag, death_inflection_time_int
  surface_area_qc/surface_area_qc_flags.csv        ← snip_id, sa_outlier_flag
  auxiliary_mask_qc/auxiliary_mask_qc_flags.csv    ← snip_id, yolk_flag, bubble_flag
  focus_qc/focus_qc_flags.csv                     ← snip_id, focus_flag
  motion_qc/motion_qc_flags.csv                   ← snip_id, motion_flag
  consolidated/qc_flags.csv                        ← all flags + use_snip
```

Each directory also gets a `.validated` sentinel file.

**`qc_flags.csv` columns:**
`snip_id`, `use_snip`, `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag`,
`viability_flag`, `dead_flag`, `death_inflection_time_int`, `sa_outlier_flag`,
`focus_flag`, `motion_flag`, `yolk_flag`, `bubble_flag`

**Dense table rule:** Every QC output has one row per `snip_id` — same universe as
`consolidated_snip_features.csv`. "No bad snips" means all flags are `False`, not zero rows.

**Flag dtype:** All flag columns non-null `bool`. `death_inflection_time_int` is nullable (not a flag column).

**`use_snip` logic:** `not(edge_flag OR discontinuous_mask_flag OR overlapping_mask_flag OR viability_flag OR dead_flag OR sa_outlier_flag OR focus_flag OR motion_flag)`. `yolk_flag` and `bubble_flag` do NOT gate `use_snip`.

---

## Schema changes required

Update `src/data_pipeline/schemas/quality_control.py`:
```python
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
    "bubble_flag",
]
```
Remove `use_embryo`, `mask_quality_flag`, `QC_FAIL_FLAGS` (replaced by `SNIP_EXCLUSION_FLAGS`).

---

## Snakemake rules to add

Add after the `consolidate_features` rule in:
`src/data_pipeline/pipeline_orchestrator/Snakefile`

Seven parallel compute rules + one consolidation rule. Follow the exact same pattern as
`compute_mask_geometry`, `compute_curvature_metrics`, etc. Each rule:
- Declares CSV + `.validated` as **both** inputs and outputs where applicable
- Reads its own config section via `params`: `config["quality_control"]["<stage>"]`
- Calls `"$PYTHON" -m data_pipeline.quality_control.entrypoints.<entrypoint>`

`consolidate_qc` waits for all 7 compute rules (declare all 7 CSVs + sentinels as inputs).

**config.yaml additions** (add under `quality_control:` grouped by stage — see spec for values):
```yaml
quality_control:
  segmentation_qc:
    edge_margin_pixels: 2
    max_mask_overlap_fraction: 0.3
  viability_qc:
    min_mask_size_px: 100
  death_detection:
    persistence_threshold: 0.8
    lead_time_hr: 4.0
    decline_rate_threshold: 0.05
  surface_area_qc:
    k_upper: 1.4
    k_lower: 0.7
  auxiliary_mask_qc: {}
  focus_qc: {}
  motion_qc:
    ncc_min_threshold: 0.85
    bad_pair_frac_threshold: 0.10
```

---

## Stub rules — hard constraints

`compute_focus_qc` and `compute_motion_qc` are schema-continuity stubs only:
- Return all-`False` flag columns over the full `snip_id` universe
- Must NOT scan ND2 files, read motion-artifact result directories, or introduce any new upstream file dependencies
- Accept only what they need for schema continuity (snip manifest or frame contract for snip_id list)

---

## validators.py — structure

```python
# src/data_pipeline/quality_control/validators.py

def assert_unique_snip_id(df): ...          # shared helper
def assert_non_null_flags(df, flag_cols): ... # shared helper

def validate_segmentation_qc_flags(df): ...
def validate_viability_qc_flags(df): ...
def validate_death_detection_flags(df): ...  # includes nullable death_inflection_time_int check
def validate_surface_area_qc_flags(df): ...
def validate_auxiliary_mask_qc_flags(df): ...
def validate_focus_qc_flags(df): ...
def validate_motion_qc_flags(df): ...
def validate_qc_flags(df): ...               # consolidated table
```

All functions raise clear exceptions on failure. `io/writers.py` calls the appropriate
validator before writing the sentinel.

---

## Do not touch

- `src/data_pipeline/analysis_ready/` — Agent B's scope
- The `all` Snakemake target — Agent B updates that
- `src/data_pipeline/feature_extraction/` — already wired, do not modify

## Flag cross-cutting changes

If you need to modify `src/data_pipeline/schemas/quality_control.py` beyond what is listed
above, or touch any shared utility, note it explicitly in a comment or commit message rather
than silently broadening scope.

---

## Done when

1. `snakemake --until consolidate_qc --config experiments='["20240418"]' --dry-run` shows all 8 QC rules in the DAG
2. Running the smoke subset produces `quality_control/20240418/consolidated/qc_flags.csv`
3. `qc_flags.csv` has unique `snip_id`, row count matches `consolidated_snip_features.csv`
4. All flag columns are non-null bool; `focus_flag`, `motion_flag`, `yolk_flag`, `bubble_flag` all `False`
5. `dead_flag=False` rows have `death_inflection_time_int` as null
6. `use_snip` column present and boolean
