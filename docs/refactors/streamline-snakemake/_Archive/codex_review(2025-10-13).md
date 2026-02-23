# MorphSeq Snakemake Docs Audit — 2025-10-13

## Priority 1

1. **Scope/plate schema requires embryo IDs before segmentation**  
   - Evidence: `docs/refactors/streamline-snakemake/data_validation_plan.md:76-92` adds `'embryo_id'` (and frame-level fields) to `REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA`.  
   - Impact: Plate/scope alignment happens before any segmentation, so columns like `embryo_id` cannot exist. Enforcing this schema would make the join rule in Phase 1 fail on every run, blocking the rest of the DAG.  
   - Recommendation: Restrict the scope/plate schema to well-level data (experiment, well identifiers, calibration). Move embryo-specific fields into later schemas that operate after segmentation.

2. **Snip ID format oscillates between `_s####` and `_t####` across docs**  
   - Evidence: `docs/refactors/streamline-snakemake/processing_files_pipeline_structure_and_plan.md:354-355` defines snip IDs as `{embryo_id}_s{frame:04d}`, while the same file later (line 733) and the output structure spec (`docs/refactors/streamline-snakemake/data_ouput_strcutre.md:110-118`) require `{embryo_id}_t{frame_index}`.  
   - Impact: Downstream joins (snip manifest, feature tables, embeddings) rely on stable snip IDs. Conflicting formats will break lookups, make validation impossible, and cause duplicated rows when the pipeline mixes both patterns.  
   - Recommendation: Pick one canonical snip ID pattern (likely `_t{frame_index}` to stay aligned with image IDs), update every document/rule/spec to match, and capture the convention in the forthcoming shared entity-ID rules module so future docs/code stay consistent.

3. **Analysis-ready schema attempts to embed every upstream column**  
   - Evidence: `docs/refactors/streamline-snakemake/data_validation_plan.md:228-244` builds `REQUIRED_COLUMNS_ANALYSIS_READY` by concatenating the feature, QC, plate, and scope required lists. Those upstream lists include columns (e.g., `channel`, `microscope_id`, `absolute_start_time`) that never appear in `segmentation_tracking.csv` or downstream merges.  
   - Impact: The final validation step would fail despite correct data because many of the required columns are unavailable after per-snip aggregation. The requirement also forces duplicate column names and makes schema drift hard to manage.  
   - Recommendation: Define an explicit, pared-down analysis-ready schema that lists only the columns guaranteed to survive the merges (IDs, QC flags, aggregated metadata, embeddings). Keep microscope-level metadata optional or move it into a documented enrichment step.

## Priority 2

4. **Segmentation schema mandates `video_id`, but no plan produces it**  
   - Evidence: `docs/refactors/streamline-snakemake/data_validation_plan.md:100-135` includes `'video_id'` in `REQUIRED_COLUMNS_SEGMENTATION_TRACKING`, yet neither the segmentation outputs in `processing_files_pipeline_structure_and_plan.md` nor the Snakemake rules mention creating or propagating that field.  
   - Impact: Validation will fail unless every segmentation row invents a `video_id`. The current plan already captures well-context via `well_id`, which is what the “video” actually represents, so the extra identifier is redundant.  
   - Recommendation: Drop `video_id` from the required column list and document that `well_id` is the canonical video identifier moving forward.

5. **Module paths diverge between documentation sets**  
   - Evidence: The output/module plan uses `metadata/plate_processing.py` and `metadata_mapping/align_metadata.py` (`processing_files_pipeline_structure_and_plan.md:232-244`), while the Snakemake rule draft references `metadata_ingest/plate/plate_processing.py` and `metadata_ingest/mapping/align_scope_plate.py` (`docs/refactors/streamline-snakemake/snakemake_rules_data_flow.md:24-56`).  
   - Impact: Readers cannot tell which package layout is authoritative, increasing the risk of duplicate implementations or broken imports when the Snakefile is wired up.  
   - Recommendation: Consolidate on one directory structure (either `metadata` or `metadata_ingest`) and update both documents to match the final layout.

6. **Snakemake rules cite an undefined schema constant**  
   - Evidence: `docs/refactors/streamline-snakemake/snakemake_rules_data_flow.md:37-45` calls out `REQUIRED_COLUMNS_SERIES_MAPPING`, but `docs/refactors/streamline-snakemake/data_validation_plan.md` never defines that symbol.  
   - Impact: Schema enforcement for the series-to-well map cannot be implemented as written, leaving a gap in metadata validation.  
   - Recommendation: Add the schema definition (and its owning module) to the validation plan or adjust the rule docs to point at the correct constant.

## Priority 3

7. **Cross-references point to missing or differently named docs**  
   - Evidence: `docs/refactors/streamline-snakemake/data_ouput_strcutre.md:5-158` claims alignment with `preliminary_rules.md`, but the only rule doc present is `snakemake_rules_data_flow.md`.  
   - Impact: Following the breadcrumb leads to a dead link, undermining confidence that the documents are in sync.  
   - Recommendation: Update the cross-reference to the actual file name (or restore the referenced document if it should exist).

8. **Refactoring plan placeholder is empty**  
   - Evidence: `docs/refactors/streamline-snakemake/refactoring_plan.md` has zero lines (`wc -l` reports 0).  
   - Impact: Contributors looking for high-level sequencing guidance find a blank file, which may duplicate planning effort elsewhere.  
   - Recommendation: Either populate the file with the agreed plan or remove the stub to avoid implying that guidance exists.
