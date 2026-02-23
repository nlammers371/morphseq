# Pipeline Orchestrator Population Plan

Goal: provide a clean Snakemake-based orchestration layer that replaces `ExperimentManager`. The orchestration package should remain thin—no business logic, only DAG definitions, experiment discovery, and a CLI bridge.

---

## Layout
```
src/pipeline_orchestrator/
├── Snakefile                    # Main DAG definition
├── config/
│   ├── defaults.yaml            # Baseline Snakemake config (experiments, thresholds)
│   └── profiles/…               # Optional profiles (SLURM, local, CUDA)
├── experiment_discovery.py      # Experiment selection/resolution helpers
└── cli.py                       # Thin CLI wrapper (e.g., `python -m pipeline_orchestrator`)
```

---

## `Snakefile`
**Responsibilities**
- Declare rules for preprocessing, segmentation, snip processing, QC, embeddings.
- Import fused config values from `data_pipeline.config.registry`.
- Define top-level targets (`rule all`, `rule per_experiment`, etc.).
- Handle optional steps (e.g., skip embeddings) through Snakemake `config` toggles.

**Implementation notes**
- Use plain Python functions (`onstart`, `onsuccess`) for logging.
- Keep rule bodies thin—call into `data_pipeline` modules rather than inlining logic.
- Support per-experiment wildcards based on `EXPERIMENTS = resolve_experiments(config)`.
- Expose convenience targets (e.g., `make sam2`) via `rule sam2_only`.

---

## `config/defaults.yaml`
**Responsibilities**
- Provide baseline Snakemake config values so users can simply run `snakemake all`.
- Example keys:
  ```yaml
  experiments: discover        # use auto-discovery by default
  device: cuda                 # or `auto`, passed to preprocessing modules
  gdino:
    score_threshold: 0.50
    nms_iou: 0.35
  qc:
    refresh_metadata: false
  ```
- Link to optional `configfile:` include in `Snakefile` for easy loading.

**Implementation notes**
- Keep the file minimal and friendly to overrides.
- Document each key inline with comments.
- Provide sample profile YAMLs for HPC vs. local execution if needed.

---

## `experiment_discovery.py`
**Responsibilities**
- Implement experiment selection priority:
  1. CLI override (`config["experiments"]`).
  2. Curated inventory file (`config["inventory"]`).
  3. Auto-discovery (`raw_image_data/{microscope}/{experiment_id}`).
- Validate existence of directories and optional metadata.
- Expose helper functions for CLI reuse.

**Functions to implement**
- `resolve_experiments(config: dict) -> list[str]`
- `load_inventory(csv_path: Path) -> list[str]`
- `auto_discover(raw_root: Path) -> list[str]`
- `validate_experiments(experiments: Iterable[str], raw_root: Path) -> list[str]`

**Cleanup notes**
- Reuse helpers from `data_pipeline.config.registry` if available; otherwise keep logic centralized here and import into the registry.
- Log missing experiments but allow a strict mode to raise errors when desired.

---

## `cli.py`
**Responsibilities**
- Offer a streamlined CLI (e.g., `python -m pipeline_orchestrator run sam2 --experiments=foo`).
- Parse CLI flags, map them to Snakemake `snakemake`/`snakemake.snakemake` calls.
- Surface common toggles (experiment list, dry-run, cores, profiles).

**Implementation notes**
- Use `argparse` with subcommands: `run`, `status`, `graph`.
- `run` should translate to `snakemake.snakemake(...)` with appropriate config overrides.
- Provide helpful error messages when Snakemake returns `False`.
- Keep CLI pure Python (no shell calls).

---

## Migration checklist
- Remove `ExperimentManager`-specific shell scripts once Snakemake parity is confirmed.
- Update documentation to point users to `pipeline_orchestrator/cli.py`.
- Add tests for `experiment_discovery` (filesystem mocked).
- Supply example commands in the README (`snakemake all`, `python -m pipeline_orchestrator run sam2 --experiments=exp1`).
