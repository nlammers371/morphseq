# Segmentation Sandbox Function Inventory

This reference inventories the core segmentation sandbox scripts and explains how their
functions can be containerized into reusable modules for the future `data_pipeline`
package. Each section starts with the file's overarching responsibility followed by
function- or method-level notes describing:

* What the callable currently does.
* How it can be imported "as is" or grouped into slimmer containers.
* Suggested target module groupings (e.g., `segmentation.propagation`,
  `segmentation.mask_io`, `metadata.entity_tracking`).

> **Scope note:** The inventory focuses on the SAM2 propagation and GroundedDINO helper
> stack that will anchor the new segmentation pipeline. Utility stubs with no
> implementation (e.g., empty QC helpers) are omitted.

## `segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py`

**General role.** Orchestrates SAM2 video propagation starting from GroundedDINO seed
annotations. The script bundles model loading, annotation tracking, mask conversion, and
per-video propagation logic. It should be split into loaders, annotation management, mask
format helpers, and propagation runners.

### Top-level functions (containerization targets)

| Function | Current responsibility | Containerization recommendation |
| --- | --- | --- |
| `load_config(path)` | Reads a YAML pipeline config. | Move into a shared `config.io` utility reused across segmentation modules. |
| `load_sam2_model(config_path, checkpoint_path, device)` | Resolves SAM2 paths, manipulates `sys.path`, and loads the torch checkpoint. | Extract into `segmentation.sam2.model_loading` with a thin wrapper that consumes resolved paths from configuration services. |
| `create_snip_id(embryo_id, frame_suffix)` | Builds standardized snip IDs. | Relocate to `identifiers.snip` or merge with `parsing_utils.build_snip_id` to avoid duplication. |
| `extract_frame_suffix(image_id)` | Pulls the `_t####` suffix from an image ID. | Fold into the `parsing_utils` family under a "frame tokens" helper. |
| `convert_sam2_mask_to_rle(mask)` | Converts a binary mask to RLE using `mask_utils`. | Export via `segmentation.mask_io.encode_rle` so downstream consumers can import a single mask formatter. |
| `convert_sam2_mask_to_polygon(mask)` | Converts a mask to polygon coordinates. | Same `segmentation.mask_io` module as above. |
| `extract_bbox_from_mask(mask)` | Computes bounding boxes from binary masks. | Bundle with other mask geometry helpers (`mask_to_bbox`). |
| `run_sam2_propagation(...)` | Runs forward SAM2 propagation from a seed frame. | Become part of `segmentation.sam2.propagation.forward` with explicit dependencies injected (model, frame loader). |
| `run_bidirectional_propagation(...)` | Runs forward/backward propagation when seed frame is not first. | Lives next to the forward propagation routine, sharing a propagation context object. |
| `process_single_video_from_annotations(...)` | High-level per-video driver that loads images, runs propagation, and updates structures. | Refactor into a `SegmentationJob` service built around dependency-injected loaders, propagators, and serializers. |
| `find_seed_frame_from_video_annotations(video_annotations)` | Selects the best seed frame (prefers earliest). | Convert to a pure function inside `segmentation.sam2.seed_selection`. |
| `assign_embryo_ids(...)` | Maps embryos detected in a frame to embryo IDs, creating new IDs as needed. | Merge with entity-tracking utilities inside `metadata.entity_tracking.embryos`. |

### `GroundedSamAnnotations` class

| Method cluster | Purpose today | Containerization hook |
| --- | --- | --- |
| **Initialization (`__init__`, `_load_components`, `_load_seed_annotations`, `_load_or_initialize_results`, `_generate_gsam_id`)** | Reads GroundedDINO seeds, loads metadata, ensures output JSON skeleton, and stamps a GSAM ID. | Create a dedicated `segmentation.sam2.annotations.Loader` that uses shared metadata IO (`ExperimentMetadata`) and `BaseFileHandler`. |
| **Configuration (`set_seed_annotations_path`, `set_sam2_model_paths`, `_load_sam2_model`)** | Allows late binding of paths and lazy SAM2 model load. | Replace with dataclass-style configuration objects; the loader should accept resolved paths and delegate model loading to `model_loading`. |
| **Introspection (`group_annotations_by_video`, `_convert_sam2_results_to_image_ids_format`, `get_processed_video_ids`, `get_missing_videos`, `_extract_gdino_entities`, `get_entity_comparison`)** | Computes bookkeeping views to compare GroundedDINO vs SAM2 coverage. | Break into `segmentation.sam2.annotations.indexing` and `metadata.entity_tracking.comparison` utilities that operate on plain dicts. |
| **Processing (`process_missing_annotations`, `process_video`)** | Calls propagation helpers and updates JSON in place. | Implement a `SegmentationProcessor` service that consumes the loader, propagation API, and mask encoders, returning an updated manifest for serialization. |
| **Reporting (`get_summary`, `print_summary`, `print_entity_comparison`, `__repr__`)** | Generates CLI-friendly summaries. | Convert to structured summary dataclasses and CLI formatters housed in `segmentation.reporting`. |
| **Persistence (`save`)** | Serializes annotations using `BaseFileHandler` with entity validation. | Keep as a thin adapter around shared file handler, ensuring JSON writing lives in `io.json_writer`. |

## `segmentation_sandbox/scripts/detection_segmentation/grounded_dino_utils.py`

**General role.** Manages GroundedDINO inference, annotation storage, and high-quality
filtering. Duplicates logic from the `scripts/utils` variant but tailored for the sandbox
CLI. Separate concerns: model loading, inference execution, annotation CRUD, and
filtering/export routines.

### Top-level functions

| Function | Responsibility | Containerization recommendation |
| --- | --- | --- |
| `load_config(path)` | Read YAML configuration. | Share the same `config.io` helper suggested above. |
| `load_groundingdino_model(config, device)` | Resolve model paths, add to `sys.path`, load the torch model. | Shift into `detection.groundeddino.model_loading` with configuration-injected paths. |
| `get_model_metadata(model)` | Extracts metadata stored on the loaded model. | Provide via `detection.metadata.model_info` so any orchestrator can log provenance. |
| `calculate_detection_iou(box_a, box_b)` | Computes IoU for two `[x1, y1, x2, y2]` boxes (note duplicate definition). | Deduplicate and store under `detection.metrics.iou`. |
| `run_inference(model, image, prompts, thresholds, device)` | Executes GroundedDINO inference and returns normalized detections. | Promote to `detection.groundeddino.inference.run` with typed request/response objects. |
| `visualize_detections(image, detections, output_path)` | Draws detection boxes and labels. | Host under `visualization.detection.overlays`; ensure dependencies (matplotlib) are optional. |
| `gdino_inference_with_visualization(...)` | Convenience wrapper combining inference + visualization. | Provide as part of a high-level CLI adapter separate from core inference logic. |

### `GroundedDinoAnnotations`

| Method cluster | Purpose | Containerization hook |
| --- | --- | --- |
| **Initialization (`__init__`, `_load_or_initialize`, `_get_initial_data`)** | Bootstraps annotation JSON, attaches entity tracking scaffolding. | Build a shared `AnnotationStore` in `detection.annotations.storage` with `BaseFileHandler`. |
| **Metadata wiring (`set_metadata_path`, `_get_image_to_experiment_map`, `get_all_metadata_image_ids`)** | Ties experiment metadata to annotation workflow. | Move into `metadata.experiments.lookup` utilities to supply image/experiment mappings. |
| **State queries (`get_annotated_image_ids`, `get_missing_annotations`, `_get_filtered_image_ids`, `get_images_for_detection`, `has_high_quality`)** | Determine coverage and outstanding work. | Provide as pure functions acting on annotation dicts, enabling reuse in orchestrators or tests. |
| **Processing (`process_missing_annotations`, `add_annotation`, `generate_high_quality_annotations`, `get_or_generate_high_quality_annotations`, `generate_missing_high_quality_annotations`)** | Runs inference on missing images and curates high-quality subsets. | Convert into pipeline tasks (e.g., `detection.pipeline.annotate_missing`, `detection.pipeline.curate_high_quality`) that call shared inference utilities. |
| **Persistence (`save`, `_mark_unsaved`, `has_unsaved_changes`)** | Track dirty state and write JSON. | Replace with `AnnotationStore.save()` built atop `BaseFileHandler` plus `EntityIDTracker`. |
| **Reporting (`get_summary`, `print_summary`, `export_high_quality_annotations`, `import_high_quality_annotations`)** | Provide summary stats and import/export flows. | Move exporting/importing into `detection.annotations.io` while reporting becomes part of `segmentation.reporting`. |

## `segmentation_sandbox/scripts/utils/sam2_utils.py`

**General role.** Earlier refactor of SAM2 utilities used across notebooks and ad-hoc
scripts. Provides similar capabilities to the detection version but with additional
helpers for grouping annotations, preparing frames, and summarizing processing. Treat it
as the source of reusable primitives while porting orchestration into the new pipeline.

### Module-level functions

| Function | Role today | Containerization idea |
| --- | --- | --- |
| `load_config`, `load_sam2_model` | Same as detection variant. | Deduplicate with the new `segmentation.sam2.model_loading` helpers. |
| `run_sam2_propagation`, `run_bidirectional_propagation` | Propagation kernels used across scripts. | Extract into a self-contained `PropagationRunner` class that works with path + metadata adapters. |
| `process_single_video_from_annotations` | Workflow that reads annotations and updates results. | Form the basis of a `SegmentationJob` callable invoked by Snakemake/Nextflow rules. |
| `extract_frame_suffix`, `create_snip_id` | Identifier helpers. | Merge with `parsing_utils` to keep ID rules centralized. |
| `convert_sam2_mask_to_rle`, `convert_sam2_mask_to_polygon`, `extract_bbox_from_mask` | Mask encoding utilities. | Consolidate inside `segmentation.mask_io` alongside `mask_utils`. |
| `extract_seed_annotations_info`, `group_annotations_by_video`, `get_video_metadata_from_annotations` | Build data structures for propagation. | Convert to `annotations.indexing` pure functions (consume dict, return derived dict). |
| `prepare_video_frames_from_annotations`, `prepare_video_frames_from_image_paths` | File-system frame collectors. | Move to `io.frame_loader` with explicit dependencies on metadata/resolvers. |
| `find_seed_frame_from_video_annotations`, `assign_embryo_ids` | Decision logic for seeds and embryo numbering. | Integrate with `metadata.entity_tracking` utilities for deterministic embryo assignment. |

### `GroundedSamAnnotations` methods

| Method | Description | Containerization |
| --- | --- | --- |
| `_generate_gsam_id` | Builds a persistent annotation ID. | Provide as part of `annotations.identity` utilities. |
| `__init__`, `_load_seed_annotations`, `_load_experiment_metadata`, `_load_or_initialize_results` | Compose configuration, metadata, and results state. | Convert into a `@dataclass` factory or loader service inside `segmentation.sam2.annotations`. |
| `get_gsam_id`, `save`, `has_unsaved_changes`, `__repr__` | Accessors and persistence wrappers. | Keep as thin wrappers over shared IO + state dataclasses. |
| `set_seed_annotations_path`, `_load_sam2_model`, `set_sam2_model_paths` | Manage paths and lazy loading. | Replace with dependency injection (pass prepared objects to constructor). |
| `group_annotations_by_video`, `_group_video_annotations`, `get_processed_video_ids`, `get_missing_videos` | Determine processing coverage. | Offer as free functions taking raw annotation dicts. |
| `process_missing_annotations`, `process_video`, `process_videos` | Iterate through pending videos and invoke propagators. | Compose inside a `SegmentationProcessor` service invoked by workflow engine steps. |
| `get_summary`, `print_summary`, `print_processing_summary` | Produce CLI stats. | Provide summary dataclasses consumed by CLI/reporting layer. |

## `segmentation_sandbox/scripts/utils/grounded_sam_utils.py`

**General role.** A monolithic module that fuses GroundedDINO inference and SAM2
propagation for experiments. Many functions overlap with the two specialized modules
above. Use it to source reusable pieces while de-duplicating against the refactored
implementations.

### Module-level functions

| Function | Current behavior | Containerization path |
| --- | --- | --- |
| `load_config` | YAML reader. | Reuse shared `config.io`. |
| `load_groundingdino_model`, `get_model_metadata` | Model loading + metadata capture. | Share with `detection.groundeddino.model_loading`. |
| `calculate_detection_iou` | IoU computation. | Deduplicate with `detection.metrics.iou`. |
| `visualize_detections` | Draws detections. | Export via `visualization.detection`. |
| `run_inference`, `gdino_inference_with_visualization` | Combined inference + optional visualization. | Provide orchestrator wrappers around `detection.groundeddino.inference`. |

### `GroundedDinoAnnotations` methods

The methods mirror the detection module with a few extras for tracking empty IDs and
logging processing status. Treat them identically when extracting reusable services:

| Method subset | Notable behavior | Containerization |
| --- | --- | --- |
| `_load_empty_ids`, `_append_empty_id` | Maintain placeholders for missing annotations. | Provide optional mixin `MissingAnnotationTracker` for annotation stores. |
| `add_from_inference_results`, `get_annotations_for_image`, `get_all_image_ids`, `print_processing_summary` | Convenience accessors for CLI scripts. | Keep as mixin or helper functions in `detection.annotations.query`. |

## `segmentation_sandbox/scripts/utils/mask_utils.py`

**General role.** Pure mask conversion helpers (RLE, polygons, bounding boxes). Already
close to container-ready pure functions.

| Function | Description | Containerization |
| --- | --- | --- |
| `encode_mask_rle`, `encode_mask_rle_full_info` | Encode binary masks to RLE (with optional area/bbox metadata). | Move directly into `segmentation.mask_io.rle`. |
| `decode_mask_rle`, `mask_to_polygon`, `polygon_to_mask` | Convert between representations. | Keep as pure utilities; ensure dependencies (e.g., `pycocotools`) are optional imports. |
| `mask_to_bbox`, `mask_area` | Geometry utilities. | Combine with embryo measurement tooling in `segmentation.metrics.geometry`. |
| `validate_rle_format`, `get_segmentation_format` | Validation helpers. | Provide as `mask_io.validation` to standardize QC steps. |

## `segmentation_sandbox/scripts/utils/parsing_utils.py`

**General role.** Canonical ID parsing, validation, and path builders. Critical for SAM2
propagation because annotation files rely on these identifiers.

| Function group | Responsibilities | Containerization |
| --- | --- | --- |
| **Parsing (`parse_entity_id`, `get_entity_type`, `extract_*`, `validate_id_format`, `get_parent_id`)** | Decompose experiment/video/image/embryo/snip identifiers. | Promote wholesale into `metadata.identifiers.parse`. |
| **Backward parsers (`_parse_backwards_*`)** | Legacy parsing helpers for robustness. | Keep as internal functions inside the identifier module with unit tests. |
| **Builders (`build_*`, `normalize_*`)** | Construct IDs from components with normalization. | Provide as the canonical constructors under `metadata.identifiers.build`. |
| **Filesystem mapping (`get_image_filename_from_id`, `*_path*`, `disk_filename_to_image_id`, `image_id_to_disk_filename`)** | Map IDs ↔ filenames/directories. | Move into `io.pathing.images` so both segmentation and QC tasks share them. |
| **Validation helpers (`is_valid_*`, `parse_multiple_ids`, `extract_unique_*`, `group_by`, `group_entities_by_hierarchy`, `get_entities_by_type`)** | Bulk validation and grouping of IDs. | Provide as vectorized helpers in `metadata.identifiers.grouping`. |
| **Deprecated accessors (`get_image_path_from_id`, `get_video_directory_path`)** | Legacy filesystem lookups. | Replace callers with metadata/IO services; keep wrappers for backward compatibility until deprecation. |

## `segmentation_sandbox/scripts/utils/base_file_handler.py`

**General role.** Handles atomic JSON reads/writes with timestamped backups.

| Method | Description | Containerization |
| --- | --- | --- |
| `load_json`, `save_json` | Safe JSON IO with backup rotation. | Promote to `io.json.AtomicJsonHandler`; parameterize backup retention via config. |
| `_create_backup`, `_cleanup_old_backups`, `cleanup_backups` | Backup management. | Offer as mixin or compose with `pathlib` utilities in `io.backup`. |
| `log_operation`, `get_timestamp`, `has_unsaved_changes`, `get_file_stats` | Operational metadata and status checks. | Provide as optional instrumentation hooks consumed by orchestrators. |

## `segmentation_sandbox/scripts/utils/entity_id_tracker.py`

**General role.** Validates entity hierarchy consistency between files, records ID counts,
and supports diffing trackers.

| Method | Responsibility | Containerization |
| --- | --- | --- |
| `extract_entities`, `_get_entity_type`, `_add_ids`, `_looks_like_entity_id` | Parse IDs out of nested JSON documents. | Convert into `metadata.entity_tracking.extractors` built on `parsing_utils`. |
| `add_entity_tracker`, `update_entity_tracker` | Inject/refresh `entity_tracker` sections in JSON data. | Provide as composable functions used by annotation writers prior to serialization. |
| `compare_files`, `compare_data`, `compare_trackers` | Diff entity trackers between files/dicts. | Promote to `metadata.entity_tracking.compare` with typed result objects. |
| `get_counts`, `validate_hierarchy` | Summaries and validation. | Use within QC utilities to ensure integrity after propagation. |

## `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`

**General role.** CLI helper that reads SAM2 JSON, validates structure, and exports
flattened CSV rows for downstream builds.

| Method | Description | Containerization |
| --- | --- | --- |
| `SAM2MetadataExporter.__init__` | Capture paths/output options. | Convert to dataclass within `segmentation.sam2.exporters`. |
| `load_and_validate_json`, `_validate_json_structure` | Parse + validate SAM2 manifest. | Share schema validators inside `segmentation.sam2.schema`. |
| `export_to_csv`, `_generate_csv_rows`, `_generate_mask_path` | Transform nested SAM2 data into tabular rows with derived mask paths. | Provide as `exporters.sam2.to_csv` returning DataFrames or iterables. |
| `_validate_csv_schema`, `validate_mask_files` | Ensure exported artifacts match expectations. | Integrate with QC suite under `segmentation.sam2.validation`. |
| `main` | CLI entrypoint. | Replace with thin wrapper around the exporter service for command-line use. |

## `segmentation_sandbox/scripts/utils/simple_mask_exporter.py`

**General role.** Creates labeled PNG masks per embryo, updating an export manifest.

| Method | Description | Containerization |
| --- | --- | --- |
| `SimpleMaskExporter.__init__` | Capture paths, overwrite flags, manifest options. | Convert to dataclass-based job spec consumed by orchestrator rules. |
| `process_missing_masks` | Batch driver that filters experiments/images and triggers export. | Promote to `segmentation.sam2.exporters.labeled_masks.process_missing`. |
| `get_export_status` | Return summary counts for exported masks. | Keep as method returning structured status object for logging. |
| `_get_missing_images`, `_export_images`, `_export_single_image`, `_get_embryo_data` | Core export helpers. | Break into pure helpers inside `exporters.labeled_masks.core` to allow reuse in Snakemake rules. |
| `_get_output_path`, `_load_manifest_entities`, `_update_manifest` | Manifest bookkeeping and path building. | Integrate with `io.manifest` utilities shared by other exporters. |

## Prioritized containerization roadmap

1. **Unify model loading + configuration** (`load_config`, `load_sam2_model`,
   `load_groundingdino_model`, `get_model_metadata`). Extract these first so every
   downstream function can receive injected models without touching `sys.path`.
2. **Centralize identifier + entity tracking logic** by merging `create_snip_id`,
   `assign_embryo_ids`, and all `parsing_utils` helpers into a single
   `metadata.identifiers` package with optional validation mixins.
3. **Split propagation/inference kernels from annotation stores.** Implement
   `PropagationRunner`, `SegmentationProcessor`, and `AnnotationStore` classes that wrap
   the pure helpers documented above.
4. **Consolidate mask conversion + export utilities** so both pipeline runs and
   downstream QC/export scripts import from `segmentation.mask_io` and
   `segmentation.sam2.exporters`.
5. **Wrap reporting + CLI glue** as thin adaptors around the reusable services, letting a
   workflow engine (Snakemake or Nextflow) orchestrate tasks without duplicating logic.

This decomposition keeps the core functionality intact while making each group of
functions importable as compact, testable modules that can power SAM2 propagation across
species and experimental setups.
