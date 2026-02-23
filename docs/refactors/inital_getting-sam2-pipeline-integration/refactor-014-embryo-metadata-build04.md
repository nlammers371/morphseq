# Refactor-014: EmbryoMetadata Build04 Ingestion

Created: 2025-09-22  
Status: Draft proposal (rev A)  
Owner: TBD (metadata team)  
Depends On: Refactor-013 Build03/Build04 per-experiment outputs

---

## Objective

Let `EmbryoMetadata` instantiate directly from Build04 per-experiment CSVs so biologists can annotate with the full set of QC and staging metrics (death flags, surface area, treatments, temperature, etc.) without touching SAM2 JSON. Preserve existing annotation tooling while cleanly separating ingestion pathways.

---

## Inputs & Discovery

- Callers provide one or more `experiment_id` values.  
- Loader resolves each CSV via `paths.get_build04_output(root, exp)` → `{root}/metadata/build04_output/qc_staged_{exp}.csv`.  
- Embryo IDs in Build04 are globally unique, so concatenating multiple experiments is safe; `experiment_id` remains available for grouping.

---

## Current State Snapshot

- `EmbryoMetadata.__init__` currently *requires* a SAM2 annotations JSON and builds embryo/snips shells via `_extract_embryos_from_sam2` (`segmentation_sandbox/scripts/metadata/embryo_metadta/embryo_metadata.py:37`).
- The generated structure lacks QC metadata—snips only hold empty phenotype/flag arrays.
- Build04 outputs (`metadata/build04_output/qc_staged_{exp}.csv`, see `src/run_morphseq_pipeline/paths.py:131`) already consolidate staging, death logic, treatments, QC flags, and surface area.

Pain points:
- Annotation UX cannot see death timing, stage inference, or perturbation context.
- Legacy SAM2 dependency blocks working with archived experiments when only Build04 remains.
- Multi-experiment annotation sessions need manual JSON stitching today.

---

## Scope & Non-Goals

**In scope**
- New ingestion path that accepts Build04 CSVs and produces the internal embryo/snips dictionary.
- Persist Build04-derived data alongside annotation edits in the existing JSON format.
- Introduce a small utility layer so downstream field derivations (e.g., genotype from pipeline metadata) stay decoupled from the core class.
- Keep SAM2 ingestion available but moved into a separate module (`sam2_ingestion.py`) for archival use.

**Out of scope**
- Changes to Build04 computations or schema.
- Major annotation JSON schema revamps—new fields remain optional.
- Experiment Manager orchestration changes (it already produces the required CSVs).

---

## Proposed Design

1. **Constructor refactor**  
   - Add a `_initialize_from_embryo_dict(embryo_dict, annotations_path)` helper used by all ingestion paths.  
   - Introduce `@classmethod EmbryoMetadata.from_build04(root, experiments, annotations_path=None, loader_opts=None)` that locates CSVs, hydrates the embryo dict, then returns an instance.  
   - Mark the existing `sam2_path` parameter as legacy in docstrings and route it through `sam2_ingestion.load_from_sam2_json` for backward compatibility.

2. **Isolated loaders**  
   - Add `build04_ingestion.py` in the embryo metadata package. Export `load_build04_embryos(root, experiments, *, column_map=None) -> Dict[str, Dict]`.  
   - Retain SAM2 parsing in `sam2_ingestion.py` for archival runs; remove the logic from the main class body.  
   - Both loaders return the same normalized embryo/snips dict interface.

3. **Data model updates**  
   - Each snip stores a new `pipeline_metadata` dict carrying raw Build04 columns (surface area, perimeter, fractional alive, timings, QC flags, etc.).  
   - Maintain `phenotypes` / `flags` arrays for user edits.  
   - Embryo-level metadata (genotype, phenotype, perturbations, temperature) is first populated from non-null values surfaced in `pipeline_metadata`.  

4. **Utility extraction layer**  
   - Provide helper functions (e.g., `pipeline_fields.extract_death_summary(embryo)`, `pipeline_fields.derive_genotype(embryo)`) that operate on `pipeline_metadata` and optionally promote values into the class-level `metrics`/`phenotype`/`genotype` views used elsewhere.  
   - These helpers are imported by notebooks or UI layers; the core class simply stores raw pipeline metadata so future reshaping is isolated to the utilities.  

5. **Persistence & provenance**  
   - When saving, embed `self.data['metadata']['sources'] = {'build04_csvs': [...]}` and a schema version.  
   - Ensure legacy fields (`source_sam2`) remain when the SAM2 loader was used.  
   - Round-trip loading should preserve `pipeline_metadata` exactly as ingested.

6. **Documentation refresh**  
   - Update `biologist_tutorial` to show Build04-backed instantiation and calling the utility helpers to derive death/genotype summaries.  
   - Note the archived SAM2 flow in a short appendix referencing `sam2_ingestion.py`.

---

## Build04 → EmbryoMetadata Mapping

| Build04 column | EmbryoMetadata location | Notes |
| --- | --- | --- |
| `embryo_id` | embryo key & `embryo['embryo_id']` | Primary grouping key |
| `experiment_id` | `embryo['experiment_id']` | Preserved for multi-experiment loads |
| `snip_id` | snip key | Canonical ID |
| `frame_index` | `snip['frame_number']` | Int, zero-based |
| `Time (s)` | `snip['pipeline_metadata']['time_s']` | Raw time stamp |
| `Time Rel (s)` | `snip['pipeline_metadata']['time_rel_s']` | Relative timestamp |
| `predicted_stage_hpf` | `pipeline_metadata` | Float |
| `inferred_stage_hpf` | `pipeline_metadata` | Build04 QC result |
| `surface_area_um`, `perimeter_um`, `centroid_*_um` | `pipeline_metadata['morphometrics']` | Keep grouped for clarity |
| `fraction_alive` | `pipeline_metadata['fraction_alive']` | Drives phenotype utilities |
| `dead_flag`, `dead_flag2` | `pipeline_metadata['flags']['dead']` | Stored as booleans |
| `sam2_qc_flags`, `frame_flag`, `focus_flag`, etc. | `pipeline_metadata['flags']['qc']` | Normalize to list of strings |
| `genotype`, `phenotype`, `chem_perturbation`, `temperature` | `pipeline_metadata['embryo_annotations']` | Derivable via utilities |
| `use_embryo_flag` | `pipeline_metadata['flags']['usable']` | Snip-level indicator |

---

## Implementation Plan (High Level)

1. **Module layout**  
   - Create `build04_ingestion.py` with pandas-based loader and normalization helpers.  
   - Move SAM2-specific parsing into `sam2_ingestion.py` and import it lazily in the legacy constructor path.

2. **EmbryoMetadata refactor**  
   - Introduce `_initialize_from_embryo_dict` and `from_build04`.  
   - Update internal structures to expect `pipeline_metadata` on snips.  
   - Adjust save/load validation to tolerate the richer snip payload.

3. **Utility helpers**  
   - Add `pipeline_fields.py` (or similar) exposing pure functions to derive high-level annotations from `pipeline_metadata`.  
   - Use these utilities in documentation / tutorials, not inside core class methods.

4. **Documentation & tutorial**  
   - Refresh `biologist_tutorial` notebook/text to demonstrate Build04 ingestion, utility-based summaries, and annotation editing.  
   - Document the archived SAM2 workflow pointing to `sam2_ingestion.py` for reference.

---

## Testing & Validation

- Unit test `build04_ingestion` with fixture CSVs covering missing-column and boolean-coercion cases.  
- Unit test round-trip save/load ensuring `pipeline_metadata` contents are unchanged.  
- Integration smoke test: ingest `qc_staged_20250529_36hpf_ctrl_atf6.csv`, verify embryo/snips counts and a few key fields.  
- Notebook/manual check: confirm death-flag utilities reproduce current phenotype logic.

---

## Feasibility Assessment

- **Technical risk:** Low-to-moderate. Build04 schema already captures required fields; work centers on restructuring ingestion.  
- **Performance:** Per-experiment CSVs fit comfortably in memory; grouping remains linear in snip count.  
- **Dependencies:** Pandas already available in Build04 stack; no new external services.  
- **Edge cases:** Must guard against experiments missing optional columns (`fraction_alive`, `chem_perturbation`); loader will log warnings and provide defaults.  
- **Overall feasibility:** High. Estimated effort 2–3 focused dev days plus tutorial edits.

---

## Decisions & Follow-ups

- **SAM2 fallback** – Keep the code path but relocate it to `sam2_ingestion.py`; callers opt in explicitly while documentation prioritises the Build04 loader.  
- **Multi-experiment storage** – Continue emitting a single annotations JSON containing all requested embryos. Future utilities can handle splitting/merging JSON files if needed, but that sits outside this refactor.  
- **Downstream consumers** – Schedule a quick audit of scripts expecting the minimal snip structure and update them (or wrap them) to use the new utility accessors.

