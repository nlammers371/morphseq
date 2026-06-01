# Handoff — skill docs restructure

**Last updated:** 2026-04-20
**Status:** classification done. trajectory_condensation schema migration done. morphology_geometry skill docs done.

## Completed (reference implementation)

- `ai/skills/analyze-classification/` → thin pointer (63 lines)
- `src/analyze/classification/README.md` → canonical user-facing reference
- `src/analyze/classification/viz/README.md` → plot cookbook
- `src/analyze/classification/DESIGN.md` → renamed from `feature_what_i_like.md`
- `src/analyze/classification/future_improvement.md` → stamped as historical

See the classification structure as the template for the remaining two.

## Pattern to follow

1. **Inventory every `.md` in the target `src/analyze/<module>/` tree first.** High-effort design specs (multi-hundred-line docs with rationale) must not be lost. Rename them to `DESIGN.md` if they're canonical design rationale; stamp with a "historical snapshot" header if they're out-of-date roadmaps.
2. **One canonical `src/analyze/<module>/README.md`** — user-first:
   - One-paragraph intro + when to use / when NOT to use
   - **Subpackage map** (conceptual branch points) — first-class section, as a table
   - Input contract (df columns, where the data comes from)
   - Quickstart (runnable, with expected output shape)
   - The result object (schema + flag→layer matrix)
   - Visualization (pointer to `viz/README.md`)
   - Troubleshooting (5 common errors → causes → fixes)
   - Full parameter reference
   - Appendix: legacy / migration if relevant
   - Pointers to `DESIGN.md` and any existing algorithm specs
3. **`src/analyze/<module>/viz/README.md`** organized by "I want to see X → use Y", with required layer/flag called out per plot.
4. **Shrink `ai/skills/<skill>/SKILL.md`** to ~25–60 lines: when to use, guardrails, 1 quickstart, pointers to canonical READMEs.
5. **Delete `COMMAND.md` / `VIZ.md`** in the skill dir — content merged.
6. **Cross-check signatures against source** before writing. For each function documented in old skill docs, verify every kwarg is present and named correctly in the current source. Common drift: missing kwargs, renamed/moved modules, legacy-leading examples.

## Completed

### Task #3 — `ai/skills/analyze-trajectory-condensation/` ✅ DONE

- `schema.py`: added `from_classifier_directions(df, vd, *, time_col, ...)` — preferred
  input path using `ValidatedDirections` from `morphology_geometry`. Each feature dim
  is a raw dot product onto a classifier direction; no SVM clipping.
- `from_pairwise_margin_csv`, `from_multiclass_csv`, `subset_pairwise` now emit
  `DeprecationWarning` with explicit "values are clipped" message.
- All three legacy names removed from `__init__.__all__` (still callable via
  `schema.from_pairwise_margin_csv` for historical scripts in
  `results/mcolon/20260329_pbx_crispant_analysis_cont/`).
- `README.md` Quick Start updated to the `load_classifier_directions →
  from_classifier_directions` pattern; legacy section explains the clipping problem.
- `SKILL.md` solver pattern updated; legacy callout added.
- Open question from earlier (contrast coords in `classification/engine/` vs separate
  subpackage): trajectory_condensation is now migrated off contrast coords, so this
  is ready to revisit if desired.

## Completed

### Task #4 — `ai/skills/analyze-morphology-geometry/` ✅ DONE

- Added `src/analyze/morphology_geometry/README.md` as the user-facing overview.
- Added `ai/skills/analyze-morphology-geometry/SKILL.md` as the thin pointer skill.
- Kept the package focused on `validation.py`, `vectors.py`, and `projection.py`.

## Remaining tasks

None from this restructure pass.

## Guardrails that bit me on classification

- **Preserve high-effort docs.** Before deleting any `.md`, grep for whether its content has been fully ported. On classification I nearly dropped `feature_what_i_like.md` (2268 lines of design spec) before recognizing it.
- **Check `difference_detection`-style shim modules.** Cross-module re-exports exist; mention them as deprecated-but-supported.
- **The `save_*` flag → layer matrix is the most error-prone part.** Classification had 12 layer keys; the original skill doc only listed 4. Read `_LazyLayers._REGISTRY` in the analysis module directly.
- **Don't trust skill doc signatures — read the source.** Every skill had at least one stale kwarg.
- **Subpackages that look similar often aren't.** e.g. classification's `directions/` (preferred) and contrast coordinates (legacy-but-supported in `engine/`) are two views of the same trained classifiers but have different APIs and different discoverability. Explain the relationship prominently in the subpackage map.

## Open question not yet resolved

The user raised (and deferred): whether contrast coordinates should physically move out of `classification/engine/` into its own subpackage mirroring `directions/`. Decision was "not now — when `trajectory_condensation` migrates off it." If you find during task #3 that trajectory_condensation is already migrated to `directions/`, flag this as ready to revisit.
