# Workflow Orchestrator Assessment: Snakemake vs. Nextflow

## Current Pain Points Driving a Workflow Upgrade

- The legacy `build03A_process_images.py` script hardcodes segmentation model discovery, path globbing, and snip export logic instead of calling shared utilities, so every run reimplements environment discovery and fails when directory layouts shift.【F:src/build/build03A_process_images.py†L4-L199】
- `build04_perform_embryo_qc.py` layers complex stage inference logic and dataset-specific exceptions directly inside a monolithic script, making it difficult to share QC behaviour across experiments or new organisms without copy/paste edits.【F:src/build/build04_perform_embryo_qc.py†L1-L189】
- `ExperimentManager` inside `src/build/pipeline_objects.py` scans the filesystem and orchestrates multi-step runs with handcrafted dependency checks, increasing the maintenance surface area and making it hard to substitute new sequencing or segmentation stages.【F:src/build/pipeline_objects.py†L338-L520】

These observations confirm that the current orchestration relies on bespoke Python entry points, repeated path handling, and implicit state. A dedicated workflow engine should:

1. Provide declarative dependencies so segmentation, feature extraction, and QC re-run only when their inputs change.
2. Allow a thin CLI layer while keeping actual computation in reusable `data_pipeline` modules.
3. Remain approachable for scientists extending the pipeline to new organisms or microscopes.

## Option 1: Adopt Snakemake

**Fit with existing codebase**
- Snakemake rules are Python-first and embed arbitrary Python functions or modules, so the planned `src/data_pipeline` package can be imported directly without shell glue. This keeps the workflow layer shallow and leans on the refactored utilities.
- Native support for templated file patterns and `checkpoint` rules aligns with the repo's heavy use of ID-based path conventions that will live in `data_pipeline.config.naming` and `data_pipeline.config.paths`.

**Developer experience**
- Scientists already familiar with Python can add rules with minimal new syntax, and Snakemake's `--cores`/`--use-conda` flags can replace bespoke multiprocessing blocks when scaling within a workstation cluster.
- Built-in dry runs and `--summary` reports simplify reasoning about which experiments will run, replacing the bespoke `_run_step` bookkeeping in `ExperimentManager`.

**Operational considerations**
- Snakemake runs comfortably on shared HPC schedulers and supports containerized execution if you later standardize environments via `mamba` or `docker` images.
- Configuration can stay in YAML/JSON files close to the codebase, mirroring the existing metadata layout.

## Option 2: Adopt Nextflow

**Fit with existing codebase**
- Nextflow encourages containerized processes and JVM-based DSL scripting; integrating tightly with Python modules requires wrapping each stage in shell scripts or entry points, which reintroduces the separation between orchestration and utilities you are trying to collapse.
- The pipeline currently depends on Python-specific libraries (e.g., `skimage`, `pandas`, `scipy`), so every Nextflow process would need container definitions or Conda environments to expose those dependencies.

**Developer experience**
- DSL2 introduces modular workflow definitions, but team members would need to learn Groovy-based syntax and new debugging workflows. This is a larger leap for scientists accustomed to Python notebooks and scripts.
- While Nextflow's channel abstractions are powerful for streaming genomics data, they add conceptual overhead for batch-style microscopy jobs that already exist as discrete CSV/NPY artifacts.

**Operational considerations**
- Nextflow excels in large, distributed environments (AWS Batch, Google Life Sciences) with container registries. If long-term goals include multi-cloud scalability, it offers first-class support, but you pay the complexity cost upfront.
- Managing reproducible environments without containers is harder; local Conda support exists but is less ergonomic than Snakemake's native integration.

## Decision Matrix

| Criteria | Snakemake | Nextflow |
| --- | --- | --- |
| **Learning curve for current team** | Low – extends Python mindset and can call new `data_pipeline` modules directly. | Medium/High – requires Groovy DSL, channel semantics, and container fluency. |
| **Integration with planned refactor** | Direct – rules can import `data_pipeline` utilities and reuse parsing/naming helpers without wrappers. | Indirect – would need shims or CLI scripts, risking a return to "glue" files. |
| **Workflow transparency** | Built-in DAG visualization and dry runs clarify which steps execute per experiment, replacing bespoke status flags. | Strong, but channel-based execution can obscure simple batch dependencies. |
| **Portability / future scaling** | Works locally, on SLURM, or cloud via Snakemake executor plugins; container support optional. | Excellent cloud/HPC portability when containerized, but more setup required today. |
| **Time to adoption** | Weeks – start by mirroring existing build steps as Snakemake rules while utilities migrate. | Months – team must learn DSL and design containers before parity. |

## Recommendation and Proposed Path

Given the desire to "keep things shallow and simple" while flattening script logic into reusable modules, Snakemake is the better near-term fit. It lets you:

1. Replace `ExperimentManager` with a `Snakefile` that maps experiments (dates) to rules invoking shared Python functions, reducing bespoke orchestration code while keeping configuration in YAML.
2. Express segmentation and QC dependencies declaratively so Snakemake can automatically re-run stages when underlying metadata or masks change, avoiding manual state flags.
3. Preserve flexibility to target new organisms by swapping configuration files and rule parameters rather than editing multiple Python entry points.

Nextflow becomes attractive if you later need container-native, cloud-distributed execution or to integrate with other Nextflow-based genomics workflows. For the current microscope-first, Python-heavy stack, Snakemake offers the shortest path to clearer scripts, better reproducibility, and smoother adoption of the reorganized `data_pipeline` package. Once the Snakemake pipeline is stable, you can reassess whether container-heavy deployments warrant a second-stage migration to Nextflow.
