# Refactor-013 Addendum (2025-09-14): Build06 Per-Experiment Plan

Status: Proposal for audit and review  
Depends On: Refactor‑013 per‑experiment Build03/Build04; Embedding services readiness

---

## Purpose

Make Build06 a per‑experiment step, mirroring Build03/Build04. For each experiment, consume the per‑experiment Build04 output, ensure/load per‑experiment latent embeddings, and produce a per‑experiment df03 with embeddings added. Keep a simple optional utility to concatenate per‑experiment results into a combined df03 for legacy consumers.

---

## Summary

- Feasibility: High. Existing embedding orchestration, snip normalization, and merge logic are reusable.
- Outcome: A per‑experiment Build06 step that reads `qc_staged_{exp}.csv` (Build04), ensures/loads per‑exp latents, merges on `snip_id`, and writes a per‑exp df03 plus concise coverage logs.

---

## Interfaces

Library (new)
```python
from pathlib import Path
from typing import Optional

def build06_merge_per_experiment(
    root: Path,
    exp: str,
    model_name: str,
    in_csv: Optional[Path] = None,               # defaults to root/metadata/build04_output/qc_staged_{exp}.csv
    latents_csv: Optional[Path] = None,          # defaults to data_root/analysis/latent_embeddings/legacy/{model}/morph_latents_{exp}.csv
    out_dir: Optional[Path] = None,              # defaults to root/metadata/build06_output/
    generate_missing: bool = False,              # generate latents if missing (env dependent)
    overwrite: bool = False,                     # overwrite per-exp df03
    coverage_warn_threshold: float = 0.90,       # warn if join coverage below this
) -> Path:
    """Per‑experiment df03 merge: df02 (Build04) + latents → df03 for `exp`. Returns output path."""
```

CLI (new)
```
python -m src.run_morphseq_pipeline.steps.run_build06_per_exp \
  --data-root <root> \
  --exp <experiment_name> \
  --model-name <model> \
  [--no-generate-latents] [--overwrite] [--out-dir <path>] [--coverage-warn-threshold 0.90]
```

CLI semantics (MVP)
- Default: generates missing latents automatically for the requested experiment, then merges.
- `--no-generate-latents`: do not generate; fail if latents missing.
- `--overwrite`: overwrite the per‑experiment df03 AND force regeneration of latents for the experiment (even if present).
  - If `--overwrite` and `--no-generate-latents` are both provided, treat as a configuration error (mutually exclusive).

---

## Inputs and Outputs

Inputs (per experiment)
- Build04 per‑exp df02: `root/metadata/build04_output/qc_staged_{exp}.csv`
- Latents per‑exp: `data_root/analysis/latent_embeddings/legacy/{model_name}/morph_latents_{exp}.csv`
  - Optional generation (`--generate-missing-latents`) using the legacy embedding generator.

Outputs (per experiment)
- Per‑exp Build06 df03: `root/metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv`
  - Note: filename string matches request exactly (includes `final_ouput_with_latents`).

Optional combined output (legacy)
- Combined df03: `root/metadata/combined_metadata_files/embryo_metadata_df03.csv` (via a separate combine utility).

---

## Algorithm (Per‑Experiment)

1) Load df02 (Build04)
- Read `in_csv or root/metadata/build04_output/qc_staged_{exp}.csv` into DataFrame.
- Fail loudly if missing or malformed (required columns: `snip_id`, `use_embryo_flag`, `experiment_date`).

2) Filter quality
- Keep `use_embryo_flag == True` rows (consistent with current Build06 services).

3) Normalize IDs
- Normalize `snip_id` formats in df02 (e.g., `_t####` vs `_s####`), using the existing normalization utility.

4) Ensure/Load latents
- Locate latents for this experiment: `morph_latents_{exp}.csv` in the selected `{model_name}` directory.
- Generation behavior:
  - Default: generate if missing.
  - `--overwrite`: force regeneration even if present (then use the new file).
  - `--no-generate-latents`: do not generate; fail if missing (mutually exclusive with `--overwrite`).
- Load latents and normalize their `snip_id`s using the same normalizer.

5) Merge
- Left-join df02 onto latents by `snip_id` to produce df03 (preserve all df02 rows).
- Compute join coverage using the first `z_mu_*` column as presence indicator.

6) Write output
- Ensure `root/metadata/build06_output/` exists; write `df03_final_ouput_with_latents_{exp}.csv`.
- Print a concise summary (see Validation & Logging) and return the path.

---

## Latent Generation (Overview)

- Inputs: `training_data/bf_embryo_snips/{exp}/*` and a legacy model under `data_root/models/legacy/{model_name}/`.
- Outputs: one per‑experiment CSV `morph_latents_{exp}.csv` under `data_root/analysis/latent_embeddings/legacy/{model_name}/`.
- Integrity: service checks coverage of `snip_id`s (snips vs latents) and supports regeneration (optional) per experiment.
- Environment: may require Python 3.9 and ML deps (torch, einops). When unavailable in‑process, route via subprocess/py39 env; log the env path clearly.

---

## Validation & Logging

- Print per‑experiment summary:
  - Input df02 rows, filtered rows (use_embryo_flag==True), latent rows loaded, join coverage (matched/total).
  - If coverage < `coverage_warn_threshold` (default 0.90), print a warning and sample a few missing `snip_id`s.
- Fail loudly on missing required files unless `--generate-missing-latents` is set.
- Always log actual paths used for in/out CSVs and latents.

---

## Risks & Mitigations

- Join coverage gaps (snip_id mismatches)
  - Mitigation: robust normalization on both sides; warn below threshold; optionally write a small debug CSV of unmatched `snip_id`s for audit.

- Environment compatibility for latent generation
  - Mitigation: detect/import failure; fall back to a configured py3.9 subprocess route; log the exact env path and command used.

- Path conventions drift
  - Decision: use `metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv` for per‑experiment outputs (consistent naming with Build04_output directory pattern).

---

## ExperimentManager Integration

Add properties (Experiment class)
```python
@property
def build06_path(self) -> Path:
    return self.data_root / "metadata" / "build06_output" / f"df03_final_ouput_with_latents_{self.date}.csv"
```

Freshness checks
```python
def needs_build06_per_experiment(self) -> bool:
    # If per-exp df03 missing → need
    if not self.build06_path.exists():
        return True
    # Rerun if df02 newer than df03
    if self.build04_path.exists() and self.build04_path.stat().st_mtime > self.build06_path.stat().st_mtime:
        return True
    # Rerun if latents newer than df03
    latents = self.latents_path_for(self.model_name)  # implementation: locate morph_latents_{exp}.csv
    if latents and latents.exists() and latents.stat().st_mtime > self.build06_path.stat().st_mtime:
        return True
    return False
```

Optional global combine
- Provide `combine_build06_experiments(root, experiments, out_csv)` to concatenate per‑exp df03s into a single df03 for legacy consumers.

---

## Reuse From Existing Code

- `src/run_morphseq_pipeline/services/gen_embeddings.py` already provides:
  - `filter_high_quality_embryos`
  - `ensure_latents_for_experiments` (works with a single experiment)
  - `normalize_snip_ids`
  - Merge logic patterns to adapt for per‑exp df02 instead of global df02.

Minimal new code: a thin per‑exp wrapper + CLI glue + optional combine utility; ExperimentManager hooks.

---

## Testing & Acceptance

Acceptance criteria
- Given a valid per‑exp df02 and latents, build06 writes `df03_final_ouput_with_latents_{exp}.csv` with:
  - All df02 rows preserved and embedding columns present (left join by `snip_id`).
  - Join coverage printed; warnings if below threshold.
  - Deterministic outputs given same inputs.

Recommended tests
- Happy path: normal experiment with snips/latents present; coverage ≥ 95%.
- Missing latents: with `--generate-missing-latents`, generator creates per‑exp latents; verify join and output.
- Low coverage: inject mismatched `snip_id` patterns; verify normalization & warning.
- Strict validation: missing df02 or model dir → fail loudly with clear messages.

---

## Timeline / Effort

- Per‑exp wrapper + CLI: ~0.5 day
- ExperimentManager hooks + optional combine utility: ~0.5 day
- Smoke tests/docs: ~0.5 day

---

## Open Questions and Decisions

- Output path naming: DECIDED → `metadata/build06_output/df03_final_ouput_with_latents_{exp}.csv`.
- Low coverage behavior: Warn (default threshold 0.90), do not fail; print a small sample of missing `snip_id`s.
- Debug artifacts: Optional per‑exp CSV of unmatched `snip_id`s (not required for MVP).

---

## Next Steps

1) Implement `build06_merge_per_experiment(...)` and a `run_build06_per_exp.py` CLI wrapper.  
2) Add ExperimentManager freshness checks and property for `build06_path`.  
3) Optional: add `combine_build06_experiments(...)` utility.  
4) Update docs/CLI help; add a minimal synthetic test for coverage logging and failure modes.

---

## Status

Proposal ready for review. No breaking changes to global Build06 flow; per‑experiment step can be introduced incrementally and combined later if needed.
