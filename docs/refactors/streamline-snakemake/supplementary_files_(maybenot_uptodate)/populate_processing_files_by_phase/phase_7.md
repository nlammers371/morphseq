# Phase 7 – Embeddings

Goal: generate latent vector representations from processed embryo snips
that pass QC filtering. Embeddings run in an isolated Python 3.9
environment to match VAE training dependencies, with strict validation
of inputs and outputs.

---

## Inputs

- `processed_snips/{exp}/processed/` (Phase 4) – fully processed JPEG
  crops.
- `quality_control/{exp}/consolidated_qc_flags.csv` (Phase 6) – QC gate
  with `use_embryo_flag`.
- `models/embeddings/{model_name}/` – VAE checkpoint + config.

---

## Outputs

- `latent_embeddings/{model_name}/{exp}_embedding_manifest.csv` –
  validated manifest of eligible snips (`REQUIRED_COLUMNS_EMBEDDING_MANIFEST`).
- `latent_embeddings/{model_name}/{exp}_latents.csv` – latent vectors
  per snip (`REQUIRED_COLUMNS_LATENTS`: `snip_id`, `embedding_model`,
  `z0` … `z{dim-1}`).

---

## Modules to Populate

### `embeddings/prepare_manifest.py`

- Responsibility: join QC gate with processed snip paths, filter to
  `use_embryo_flag == True`.
- Schema enforcement: `REQUIRED_COLUMNS_EMBEDDING_MANIFEST`.
- Key columns:
  - `snip_id`, `processed_snip_path`, `use_embryo_flag`,
    `file_size_bytes`
- Validation:
  - File existence check (all snip paths must be valid)
  - File size check (no empty/corrupted images)
  - Only rows with `use_embryo_flag == True` included
- Emit one manifest per experiment/model (idempotent if reruns occur).

### `embeddings/inference.py`

- Responsibility: VAE inference inside isolated Python 3.9 environment.
- Invoked via `subprocess_wrapper.py` to match training dependencies.
- Processing:
  - Stream manifest rows into model
  - Produce latent vectors
  - Write CSV aligned with manifest ordering
- Tag each row with `embedding_model` so downstream aggregation can mix
  models safely.
- Key parameters (from Snakemake config):
  - `model_name` (default: `morphology_vae_2024`)
  - `batch_size` (default: 256)
  - `device` (resolved via `config/runtime.py`)

### `embeddings/subprocess_wrapper.py`

- Responsibility: launch VAE inference in Python 3.9 subprocess with
  proper environment isolation.
- Handles:
  - Environment activation
  - Subprocess invocation
  - Error propagation
  - Output capture

### `embeddings/file_validation.py`

- Responsibility: post-inference validation of latent CSV.
- Checks:
  - Column count matches model dimensionality
  - Every manifest row appears exactly once
  - No NaN values in latent entries
  - `snip_id` alignment with manifest
- Validation failures halt pipeline with clear error messages.

---

## Contracts & Validation

- **QC filtering:** Only snips with `use_embryo_flag == True` are
  embedded; no exceptions.
- **Environment isolation:** VAE inference must run in Python 3.9
  subprocess to match training environment.
- **Schema enforcement:**
  - `embedding_manifest.csv` validates against
    `REQUIRED_COLUMNS_EMBEDDING_MANIFEST`
  - `latents.csv` validates against `REQUIRED_COLUMNS_LATENTS`
- **Idempotency:** Manifest generation is idempotent; reruns with
  additional snips only add new rows.
- **Model tagging:** Every latent row includes `embedding_model` column
  for multi-model support.
- **Dimensionality contract:** Latent CSV must have exactly `dim`
  columns (`z0` … `z{dim-1}`) matching model config.

---

## Handoff

- Phase 7 outputs feed into analysis-ready data generation (Phase 8,
  pending) and any downstream phenotypic analysis or clustering
  workflows.
- `latents.csv` provides low-dimensional representations for
  visualization, clustering, and statistical modeling.
