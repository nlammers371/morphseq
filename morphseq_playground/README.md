# `morphseq_playground/` (data root) layout + symlinks

This directory is the **default `--data-root`** used by the MorphSeq pipeline runners in this repo.

It is intentionally **NOT** fully version-controlled (it can be huge and machine-specific), but this
README **is tracked** so we remember the expected layout and avoid “it worked yesterday” issues.

## What the pipeline expects

At minimum, code that uses `ExperimentManager` expects:

- `raw_image_data/Keyence/<EXP>/` and/or `raw_image_data/YX1/<EXP>/`
- `metadata/` (writable; pipeline writes state + outputs here)
  - `metadata/experiments/` (state JSONs per experiment)
  - `metadata/well_metadata/<EXP>_well_metadata.xlsx` (well layout inputs)
- `models/legacy/<MODEL_NAME>/` (legacy embedding model assets)

## Hybrid symlink convention (recommended)

To keep this repo fast/light while still pointing at the canonical data stores, we use a **hybrid**
approach:

- Keep `metadata/` **local + writable** in this repo’s `morphseq_playground/`
  - This prevents permission issues when the pipeline writes state files and per-run outputs.
- Symlink “big / shared” trees to a canonical location (often under nlammers’ central store or an old
  playground snapshot).

Typical symlinks you’ll see here:

- `raw_image_data` → `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data`
- `models` → `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/models`
- `outside_models` → `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/outside_models`
- `segmentation` → `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/segmentation`
- `built_image_data` → (often indirectly) `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data`

And optionally (if you want legacy artifacts available without copying):

- `analysis` → `<old_playground>/analysis`
- `sam2_pipeline_files` → `<old_playground>/sam2_pipeline_files`
- `training_data` → `<old_playground>/training_data`
- `videos` → `<old_playground>/videos`
- `mask_data` → `<old_playground>/mask_data`

The historical “old playground” we’ve used is:

- `/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/morphseq_playground`

To (re)create the hybrid layout quickly, use:

```bash
bash scripts/setup_morphseq_playground_hybrid.sh
```

## Quick sanity checks

Verify symlinks:

```bash
ls -la morphseq_playground | head
readlink morphseq_playground/raw_image_data
readlink morphseq_playground/models
readlink morphseq_playground/built_image_data
```

Verify an experiment is discoverable:

```bash
ls -la morphseq_playground/raw_image_data/YX1/<EXP> | head
```

Verify well metadata is findable (expected name):

```bash
ls -la morphseq_playground/metadata/well_metadata/<EXP>_well_metadata.xlsx
```

## Avoiding duplicated qsub array runs

If you re-submit the same array job twice, SGE will happily run both (double-processing).

Check:

```bash
qstat -u "$USER"
```

Cancel an older run:

```bash
qdel <JOB_ID>
```

## Notes on Git tracking

- `morphseq_playground/` contents are ignored by default to avoid committing large data.
- This README is explicitly un-ignored in the repo `.gitignore` so it stays tracked.
