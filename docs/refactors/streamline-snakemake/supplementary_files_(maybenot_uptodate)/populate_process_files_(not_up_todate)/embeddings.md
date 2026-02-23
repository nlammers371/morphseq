# Embeddings Module Population Plan

Goal: encapsulate the legacy VAE embedding workflow into three focused modules that can be invoked from Snakemake. The implementation must respect the Python 3.9 subprocess requirement, operate on `use_embryo == True` snips only, and emit deterministic latent CSVs.

---

## `embeddings/inference.py`
**Responsibilities**
- Coordinate latent generation for a given experiment/model pair.
- Filter snip inputs using `use_embryo_flags.csv`.
- Reuse cached embeddings when possible.

**Functions to implement**
- `prepare_embedding_inputs(snips_dir: Path, manifest_csv: Path, use_flags_csv: Path, params: EmbeddingParams) -> EmbeddingJob`
- `run_embedding_job(job: EmbeddingJob, subprocess_runner: Callable[..., None]) -> Path`
- `write_embedding_csv(latents: np.ndarray, snip_ids: list[str], output_csv: Path) -> None`
- `ensure_embeddings(experiment_id: str, model_name: str, params: EmbeddingParams) -> Path`

**Source material**
- `src/analyze/gen_embeddings/*.py` (latent generation scripts)
- Build04 gating logic (`use_embryo` handling)

**Cleanup notes**
- Keep `EmbeddingJob` as a simple dataclass with explicit paths.
- Never read QC files directly—consume the consolidated `use_embryo_flags.csv`.
- Support multiple models by parameterizing model name, checkpoint path, latent dimension.
- Log only high-level progress (counts, cache hits) to avoid noisy outputs.

---

## `embeddings/subprocess_wrapper.py`
**Responsibilities**
- Launch the Python 3.9 environment required by the production VAE.
- Handle argument marshalling, temporary directories, and error propagation.

**Functions to implement**
- `build_subprocess_command(job: EmbeddingJob, runtime: EmbeddingRuntimeConfig) -> list[str]`
- `run_embedding_subprocess(command: list[str], timeout: int | None = None) -> None`
- `parse_embedding_output(latent_path: Path) -> np.ndarray`

**Source material**
- Legacy `load_model_subprocess.py` and CLI stubs in `gen_embeddings`
- Ops notes on the dedicated embedding environment

**Cleanup notes**
- Allow configurable interpreter path / conda env via `config.embeddings`.
- Capture stderr/stdout for debugging; raise `RuntimeError` with the original message.
- Support dry-run/test mode that bypasses subprocess execution (useful for CI).

---

## `embeddings/file_validation.py`
**Responsibilities**
- Confirm that embedding CSVs are well-formed and correspond to the intended snip set.
- Provide hooks to skip regeneration when existing files pass validation.

**Functions to implement**
- `embedding_exists(output_csv: Path, expected_snips: Sequence[str], latent_dim: int) -> bool`
- `validate_embedding_csv(output_csv: Path, expected_snips: Sequence[str], latent_dim: int) -> bool`
- `mark_embedding_complete(output_csv: Path, metadata_path: Path, job: EmbeddingJob) -> None`

**Source material**
- Helper functions scattered across `gen_embeddings` scripts
- Analysis notebooks verifying latent dimensions

**Cleanup notes**
- Validate both column headers (e.g., `z0..z{dim-1}`) and row counts.
- Store metadata (`model_name`, `timestamp`, `latent_dim`, `snip_count`) alongside the CSV for reproducibility.
- Prefer Pandas for quick validation, but keep logic simple enough for unit tests without heavy dependencies.

---

## Cross-cutting tasks
- Define `EmbeddingParams` and `EmbeddingRuntimeConfig` in `data_pipeline.config.embeddings` so Snakemake can pass structured overrides (including Python 3.9 interpreter path or conda env).
- Add unit tests that simulate: (1) cache hit, (2) new job, (3) subprocess failure.
- Provide a CLI smoke test (`python -m embeddings.inference --dry-run`) documented in the README.
- Ensure all logging goes through the shared logger to keep Snakemake output clean.
