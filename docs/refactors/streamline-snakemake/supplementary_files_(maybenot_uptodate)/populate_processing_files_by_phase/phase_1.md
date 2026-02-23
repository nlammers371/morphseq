# Phase 1 – Input Metadata Alignment

Goal: capture user/microscope metadata, map series numbers to well IDs,
and emit a validated join that downstream phases can consume. All
artifacts live under `input_metadata_alignment/{experiment_id}/…` and
must pass the existing schema checks before progressing to the Phase 1
hand-off (`experiment_metadata/{exp}/…`) described in
`processing_files_pipeline_structure_and_plan.md`.

---

## Required Outputs

- `input_metadata_alignment/{exp}/raw_inputs/plate_layout.csv` – parsed
  plate layout (schema: `REQUIRED_COLUMNS_PLATE_METADATA`).
- `input_metadata_alignment/{exp}/raw_inputs/{microscope}_scope_raw.csv`
  – per-microscope scope metadata (schema:
  `REQUIRED_COLUMNS_SCOPE_METADATA`).
- `input_metadata_alignment/{exp}/series_mapping/series_well_mapping.csv`
  – explicit `series_number → well_index` table with mapping method and
  provenance.
- `input_metadata_alignment/{exp}/series_mapping/mapping_provenance.json`
  – warnings, fallback notes, and source references.
- `input_metadata_alignment/{exp}/aligned_metadata/scope_and_plate.csv`
  – schema-validated join ready for the Snakemake Phase 1 hand-off.

Each file must pass the schemas outlined in
`docs/refactors/streamline-snakemake/data_validation_plan.md`
(`REQUIRED_COLUMNS_*` lists).

---

## Modules to Populate

### `metadata_ingest/plate/plate_processing.py`

- Responsibilities: read Keyence/YX1 plate layout exports, normalize
  columns, coerce types, and emit schema-aligned `plate_metadata.csv`.
- Functions to implement *(see
  [`preprocessing.md`](../populate_process_files/preprocessing.md#preprocessingkeyencemetadatapy)
  and the Week 1 plan)*:
  - `load_plate_layout(raw_path: Path) -> pd.DataFrame`
  - `standardize_plate_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `write_plate_metadata(df: pd.DataFrame, output_csv: Path) -> None`
- Source material: legacy helpers in
  `segmentation_sandbox/scripts/utils/parsing_utils.py`, plus plate
  parsing blocks in `build01A/ build01B`.

### `metadata_ingest/scope/keyence_scope_metadata.py`
### `metadata_ingest/scope/yx1_scope_metadata.py`

- Responsibilities: extract microscope-specific acquisition metadata,
  normalize to shared column names, and write `scope_metadata.csv`.
- Source guidance:
  - [`preprocessing.md`](../populate_process_files/preprocessing.md#preprocessingkeyencemetadatapy)
  - [`preprocessing.md`](../populate_process_files/preprocessing.md#preprocessingyx1metadatapy)
- Must:
  - Output `experiment_id`, `well_id`, `time_int`, `microscope_channel`,
    pixel-to-micron conversions, `frame_interval_s`, etc.
  - De-duplicate/merge multiple exports per experiment.
  - Record provenance columns (`raw_filename`, `acquisition_timestamp`).

### `metadata_ingest/mapping/series_well_mapper.py`

- Responsibilities: generate
  `series_mapping/series_well_mapping.csv` and
  `series_mapping/mapping_provenance.json`.
- Functions to implement:
  - `build_series_well_mapping(plate_df: pd.DataFrame, scope_df: pd.DataFrame) -> pd.DataFrame`
  - `infer_mapping_strategy(plate_df: pd.DataFrame) -> MappingStrategy`
  - `write_mapping_provenance(mapping: pd.DataFrame, provenance_path: Path, strategy: MappingStrategy) -> None`
- Must enforce:
  - Explicit mappings when `series_number_map` exists.
  - Deterministic implicit mapping with validation when missing.
  - Schema for the mapping CSV (new `REQUIRED_COLUMNS_SERIES_MAPPING`).

### `metadata_ingest/mapping/align_scope_plate.py`

- Responsibility: join validated plate + scope metadata using the
  mapping output to write
  `aligned_metadata/scope_and_plate.csv`.
- Functions to script:
  - `align_scope_and_plate(plate_df: pd.DataFrame, scope_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame`
  - `validate_aligned_metadata(df: pd.DataFrame) -> None`
  - `write_aligned_metadata(df: pd.DataFrame, output_csv: Path) -> None`
- Reference: join strategy outlined in
  `processing_files_pipeline_structure_and_plan.md` (Input Metadata
  Alignment section) and the schema doc.

### `data_pipeline/config` foundations

- These configs unblock the metadata modules:
  - `config/paths.py` – canonical directories & filename templates.
  - `config/microscopes.py` – channel normalization, tile geometry.
  - `config/runtime.py` / `resolve_device` helper if GPU-dependent
    stitching helpers are imported while parsing metadata.
- See [`config_module_outline.md`](../populate_process_files/config_module_outline.md)
  for the required exports.

---

## Validation & Handoff

- All CSV writers must call schema validators from `schemas/` before
  writing (`plate_metadata`, `scope_metadata`,
  `scope_and_plate_metadata`, and new `series_mapping` schema).
- Add smoke tests (or TODO placeholders) for:
  - Missing required columns (fail fast with helpful messages).
  - Ambiguous series mappings (warn + fail where needed).
- Ensure Snakemake rule inputs from `snakemake_rules_data_flow.md`
  align:
  - `rule normalize_plate_metadata` + `rule extract_scope_metadata_*`
    write into `raw_inputs/`.
  - `rule map_series_to_wells` produces `series_mapping/`.
  - `rule align_scope_and_plate` writes `aligned_metadata/scope_and_plate.csv`.

With these modules in place, Phase 1 is considered complete: later
phases (built images, image manifest, segmentation, snip extraction,
etc.) can consume the validated metadata files without additional TODOs.
