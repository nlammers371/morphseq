# Data Pipeline Restructuring Plan

## Goals
- Eliminate hardcoded path and model lookups by centralizing configuration and file naming utilities.
- Flatten legacy `buildXX` scripts and the `segmentation_sandbox` helpers into a unified, documented `src/data_pipeline` package.
- Group functionality by capability (ingest, segmentation, QC, feature engineering, outputs) so future pipeline stages can compose shared utilities instead of copying code.

## Pain Points in the Current Layout
- `build03A_process_images.py` embeds direct filesystem traversal logic to locate segmentation outputs and exported SAM2 assets, which breaks when folder names change or new segmentation versions are added.【F:src/build/build03A_process_images.py†L55-L158】
- Both `build03A_process_images.py` and `build04_perform_embryo_qc.py` contain overlapping logic for handling predicted vs. inferred developmental stages, leading to duplicated maintenance effort.【F:src/build/build03A_process_images.py†L115-L199】【F:src/build/build04_perform_embryo_qc.py†L10-L189】
- SAM2 helpers in `segmentation_sandbox/scripts` maintain their own identifier parsing and filesystem assumptions, rather than exporting re-usable modules to the primary codebase.【F:segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py†L1-L158】

## Target `src/data_pipeline` Package Layout
```
src/
└── data_pipeline/
    ├── __init__.py
    ├── config/
    │   ├── __init__.py
    │   ├── paths.py            # centralizes repo-relative locations & environment overrides
    │   └── naming.py           # canonical snip/video/embryo ID builders & parsing rules
    ├── io/
    │   ├── __init__.py
    │   ├── metadata_loader.py  # metadata CSV read/write, schema validation
    │   └── storage.py          # atomic file moves, checksum utilities
    ├── ingest/
    │   ├── __init__.py
    │   ├── image_stitching.py  # from build01A/B modules
    │   └── experiment_import.py
    ├── segmentation/
    │   ├── __init__.py
    │   ├── sam2_runner.py      # SAM2 orchestration extracted from sandbox pipelines
    │   ├── detection.py        # Grounded DINO / detection helpers
    │   ├── mask_postprocess.py # shared mask cleanup utilities
    │   └── exporters.py        # mask export + snip assembly used by build03
    ├── features/
    │   ├── __init__.py
    │   ├── morphology.py       # feature calculation from snips
    │   └── stage_inference.py  # all predicted/inferred stage models
    ├── qc/
    │   ├── __init__.py
    │   ├── flags.py            # boolean QC checks and thresholds
    │   └── reports.py          # summarization + plotting hooks
    ├── transforms/
    │   ├── __init__.py
    │   └── snip_generation.py  # cropping/rotation/export for snips (from build03)
    └── orchestration/
        ├── __init__.py
        ├── pipeline_context.py # dependency injection, config loading
        └── cli.py              # replacement for scattered build scripts
```

## Segmentation Sandbox Audit & Extraction Strategy

The sandbox already groups related behaviour, but functionality is locked inside monolithic CLI scripts. Breaking those scripts
into importable modules first will make the migration less error-prone. The table below inventories the main sandbox folders and
the specific modules that should be decomposed before moving them into `src/data_pipeline`.

| Sandbox area | Current responsibility | Proposed extraction target |
| --- | --- | --- |
| `scripts/pipelines/01_prepare_videos.py` | Raw video discovery, metadata scaffolding, and CLI glue code. | Move the directory crawling and metadata bootstrapping into `data_pipeline.ingest.experiment_import`, keeping CLI-only parsing in a thin wrapper.【F:segmentation_sandbox/scripts/pipelines/01_prepare_videos.py†L1-L160】 |
| `scripts/pipelines/03_gdino_detection.py` | Grounded DINO orchestration, model loading, filter thresholds, and entity selection helpers. | Split model bootstrap + batching into `data_pipeline.segmentation.detection` and metadata-driven filtering into `data_pipeline.segmentation.mask_postprocess` so detection and QC can share them.【F:segmentation_sandbox/scripts/pipelines/03_gdino_detection.py†L1-L120】 |
| `scripts/pipelines/04_sam2_segmentation.py` & `04_sam2_video_processing.py` | SAM2 parameter parsing, segmentation loop, autosave, and frame window logic. | Extract configuration dataclasses and segmentation runners into `data_pipeline.segmentation.sam2_runner`, while video formatting steps become `data_pipeline.transforms.snip_generation` utilities.【F:segmentation_sandbox/scripts/pipelines/04_sam2_segmentation.py†L1-L120】【F:segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py†L1-L140】 |
| `scripts/pipelines/05_sam2_qc_analysis.py` & `scripts/utils/export_sam2_metadata_to_csv.py` | Aggregates segmentation metrics, predicted stage comparisons, and threshold evaluation. | Consolidate numeric checks into `data_pipeline.qc.flags` and reporting helpers into `data_pipeline.qc.reports`, removing repeated dataframe code from build03/04.【F:segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py†L70-L166】【F:segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py†L1-L120】 |
| `scripts/pipelines/06_export_masks.py` & `scripts/utils/simple_mask_exporter.py` | Iterates segmentation outputs and writes mask files/snips with configurable formats. | Wrap exporter primitives in `data_pipeline.segmentation.exporters` so both sandbox and legacy builds call the same functions.【F:segmentation_sandbox/scripts/pipelines/06_export_masks.py†L1-L200】【F:segmentation_sandbox/scripts/utils/simple_mask_exporter.py†L1-L200】 |
| `scripts/pipelines/07_embryo_metadata_update.py` & `scripts/annotations/embryo_metadata.py` | Maintains embryo-level JSON summaries and merges QC fields into experiment metadata. | Move metadata mutation logic into `data_pipeline.features.morphology` and `data_pipeline.qc.reports`, keeping JSON persistence inside `data_pipeline.io.metadata_loader`.【F:segmentation_sandbox/scripts/pipelines/07_embryo_metadata_update.py†L1-L180】【F:segmentation_sandbox/scripts/annotations/embryo_metadata.py†L1-L200】 |
| `scripts/metadata/experiment_metadata.py` & `scripts/utils/base_file_handler.py` | Shared metadata loader, autosave, backup rotation, and entity tracking. | Promote these classes to `data_pipeline.io.metadata_loader` and `data_pipeline.io.storage`, turning sandbox CLI scripts into thin clients of the shared utilities.【F:segmentation_sandbox/scripts/metadata/experiment_metadata.py†L1-L120】【F:segmentation_sandbox/scripts/utils/base_file_handler.py†L1-L120】 |
| `scripts/utils/parsing_utils.py` & `scripts/utils/entity_id_tracker.py` | Canonical entity ID parsing/formatting used by all sandbox steps. | Become the initial contents of `data_pipeline.config.naming`, ensuring `build03`/`build04` no longer reimplement ID logic.【F:segmentation_sandbox/scripts/utils/parsing_utils.py†L1-L200】 |

**Decomposition approach**

1. **Isolate pure logic** – Each sandbox script should expose functions/classes for its core behaviour (e.g., `run_detection`,
   `export_masks`, `summarize_qc`) that are importable without CLI side-effects.
2. **Add unit coverage before moving** – Repurpose the existing sandbox tests (e.g., `scripts/tests/test_pipeline_quick.py` and
   `test_metadata_init.py`) to assert behaviour against the extracted modules, so we can safely relocate them later.【F:segmentation_sandbox/scripts/tests/test_pipeline_quick.py†L1-L120】【F:segmentation_sandbox/scripts/tests/test_metadata_init.py†L1-L80】
3. **Introduce adapters in `build03`/`build04`** – Instead of directly reading sandbox JSON artifacts, the build scripts should
   call the new `data_pipeline` APIs, gradually replacing duplicated utilities.

### Parsing Utilities Migration Details

`segmentation_sandbox/scripts/utils/parsing_utils.py` currently mixes together three distinct responsibilities: (1) canonical ID
builders (`build_snip_id`, `build_embryo_id`, etc.), (2) backwards parsers that infer experiment/video/embryo/snips from any ID,
and (3) filesystem helpers that project IDs into folder/filename conventions.【F:segmentation_sandbox/scripts/utils/parsing_utils.py†L1-L200】
To make these pieces reusable across the reorganized pipeline:

1. **Create dedicated modules**
   - `data_pipeline/config/naming.py` – host ID constants (regex patterns, padding widths), format/parse functions, validation,
     and parent-child navigation helpers.
   - `data_pipeline/io/paths.py` (or extend `config/paths.py`) – expose path derivation helpers that translate IDs into relative
     directories or filenames, separating filesystem concerns from pure parsing.
   - `data_pipeline/config/entities.py` – optional thin dataclasses (`ExperimentID`, `VideoID`, etc.) that encapsulate parsing
     output and provide typed accessors.

2. **Define a single parsing entry point**
   - Implement `parse_entity_id` in `config/naming.py` using the sandbox logic, but return typed dataclasses or a `NamedTuple`
     so downstream code receives structured data instead of raw dictionaries.
   - Surface convenience helpers (`get_entity_type`, `extract_frame_number`, etc.) that delegate to the shared parser, keeping
     `build03`/`build04` compatible while enabling future deprecations of bespoke parsing snippets.

3. **Backfill tests before moving**
   - Port the sandbox quick tests to target the new module directly, verifying round-trip builds/parses, parent resolution, and
     legacy image ID support. The goal is to match existing behaviour verbatim before additional refactors.

4. **Roll out in two steps**
   - Update sandbox scripts to import from `data_pipeline.config.naming` once the module exists, ensuring both environments share
     the same parsing logic.
   - Replace ad-hoc helpers in `build03A_process_images.py`, `build04_perform_embryo_qc.py`, and `src/functions` with the shared
     API, removing duplicated regex definitions and manual string slicing.

This approach keeps parsing logic authoritative in one location, splits path formatting into the IO layer, and sets the stage for
future schema-driven validation of entity identifiers throughout the data pipeline.

## Migration Plan
### Phase 1 – Package Skeleton & Configuration
1. Create the `src/data_pipeline` package skeleton above, with empty modules and docstrings describing intended usage.
2. Move the `build`-level constants (e.g., repo root calculation, default segmentation directories) into `data_pipeline/config/paths.py` with environment-variable overrides to eliminate hardcoded absolute paths.【F:segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py†L83-L158】
3. Extract `build_snip_id` and related identifier helpers from the sandbox into `data_pipeline/config/naming.py`, and update both sandbox scripts and legacy build code to import from the shared location.【F:src/build/build03A_process_images.py†L37-L39】【F:segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py†L97-L106】

### Phase 2 – Shared Utilities and IO
1. Consolidate generic helpers (`path_leaf`, metadata CSV readers, and `segmentation_sandbox/scripts/utils/base_file_handler.py`) into `data_pipeline/io` and `data_pipeline/utils` modules; update imports in `src/functions` and build scripts accordingly.【F:src/build/build03A_process_images.py†L17】【F:segmentation_sandbox/scripts/utils/base_file_handler.py†L1-L160】
2. Build a metadata schema definition that standardizes required columns (e.g., `predicted_stage_hpf`, `exported_mask_path`) so downstream code can validate inputs before processing.【F:src/build/build03A_process_images.py†L115-L158】【F:src/build/build04_perform_embryo_qc.py†L10-L189】

### Phase 3 – Segmentation Module
1. Port SAM2 orchestration (`segmentation_sandbox/scripts/pipelines/04_sam2_segmentation.py`, `06_export_masks.py`, and `detection_segmentation/sam2_utils.py`) into `data_pipeline/segmentation`, collapsing repeated filesystem handling into reusable functions.
2. Define an interface class (e.g., `SAM2SegmentationPipeline`) that exposes methods for running detection, segmentation, QC, and export without depending on sandbox-specific folder names.
3. Relocate mask export helpers so `build03` can request masks via the new API rather than using globbed paths inside script logic.【F:src/build/build03A_process_images.py†L54-L199】

### Phase 4 – Snip Generation & Feature Engineering
1. Move cropping, rotation, and noise-synthesis logic from `build03A_process_images.py` into `transforms/snip_generation.py` with pure functions and dataclasses for input parameters.【F:src/build/build03A_process_images.py†L115-L199】
2. Centralize stage inference utilities from `build04_perform_embryo_qc.py` into `features/stage_inference.py`, exposing canonical entry points for “original”, “sigmoid”, and “reference” models. Both QC routines and visualization notebooks should call these shared functions.【F:src/build/build04_perform_embryo_qc.py†L10-L189】

### Phase 5 – QC Package
1. Collect QC thresholds, flag generation, and reporting helpers from `build04_perform_embryo_qc.py` and the SAM2 QC pipeline into the `qc` subpackage so segmentation and downstream analytics use the same logic.【F:src/build/build04_perform_embryo_qc.py†L10-L189】【F:segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py†L74-L166】
2. Provide composable QC pipelines (e.g., `SegmentationQC`, `EmbryoMetadataQC`) that operate on typed dataclasses instead of free-form DataFrames.

### Phase 6 – Orchestration & CLI
1. Replace individual `build0X` entry scripts with a thin CLI (`data_pipeline/orchestration/cli.py`) that loads configuration, instantiates pipeline components, and executes stages.
2. Supply Hydra/Lightning integration points for experiments so training/inference code can subscribe to the same pipeline outputs without duplicating path logic.

### Phase 7 – Deprecation & Cleanup
1. After modules are ported, convert `src/build/build0X` files into compatibility shims that call the new package (logging a deprecation warning) before removing them entirely in a subsequent release.
2. Mirror the sandbox folder into the package by moving maintained scripts into `src/data_pipeline` and archiving exploratory notebooks under `segmentation_sandbox/archive`.

## Implementation Considerations
- Schedule the migration in short-lived PRs (config, segmentation, QC, orchestration) to keep reviews tractable.
- Add unit tests covering ID parsing, path resolution, stage inference, and snip export behaviour as modules are migrated.
- Document the new package usage in the project README and developer docs, including instructions for overriding paths and running segmentation pipelines end-to-end.
