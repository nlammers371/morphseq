# Analysis Ready Module Population Plan

Goal: provide a single module that assembles experiment-level tables for downstream analytics. The output (`analysis_ready/{experiment_id}/features_qc_embeddings.csv`) merges features, QC flags, and embeddings while tracking whether embeddings are present. Keep this layer lightweight—no heavy analytics, just clean joins and validation.

---

## `analysis_ready/assemble_features_qc_embeddings.py`
**Responsibilities**
- Load consolidated features, QC flags, use-embryo gating, and embedding latents.
- Produce a merged DataFrame with consistent column order and helper flags.
- Write the final CSV to `analysis_ready/{experiment_id}/`.

**Functions to implement**
- `load_inputs(consolidated_features: Path, consolidated_qc: Path, use_flags: Path, embeddings_csv: Path) -> AnalysisInputs`
- `merge_analysis_tables(inputs: AnalysisInputs, params: AnalysisReadyParams) -> pd.DataFrame`
- `compute_embedding_flags(df: pd.DataFrame, embedding_columns: list[str]) -> pd.DataFrame`
- `write_analysis_table(df: pd.DataFrame, output_csv: Path) -> None`
- `validate_analysis_table(df: pd.DataFrame) -> None`

**Source material**
- Recent notebooks assembling “master tables” for model training/visualization.
- Manual pandas merges performed after Build04 runs.

**Cleanup notes**
- Join on `snip_id` for all tables. Validate row counts and uniqueness at each step.
- MVP: single production embedding model (`morphology_vae_2024`). Column names remain `z0..z{dim-1}`; no column prefixes needed.
- Add `embedding_calculated` boolean (true if all requested latent columns are present). Do not drop rows lacking embeddings; allow analysts to filter.
- Keep output sorted by `snip_id` (or `embryo_id`, `time_int`) for human readability.

---

## Schema placeholder (MVP)
- Create `analysis_ready/schema.py` exporting column group tuples (e.g., `FEATURE_COLUMNS`, `QC_COLUMNS`, `EMBEDDING_COLUMNS`).
- Keep definitions minimal (lists/tuples) so downstream notebooks import a single source of truth.

---

## Cross-cutting tasks
- Define `AnalysisReadyParams` in `data_pipeline.config.analysis_ready` (e.g., embedding latent dimension, expected column order).
- Unit test the merge with synthetic CSVs representing common scenarios (full overlap, missing embeddings, multiple models).
- Add CLI example to the documentation (`python -m analysis_ready.assemble_features_qc_embeddings --experiment foo`).
- Ensure logging reports the number of rows included/excluded and missing embeddings to help spot data issues early.
