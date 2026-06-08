# Handoff — Missing Images and Embryos Investigation

**Created:** 2026-06-07 · mdcolon / Claude  
**Context:** Resume after context clear. This is a focused sub-task of the cilia QC first-pass
analysis. Read `HANDOFF.md` for the broader project state.

---

## What we're doing and why

We audited `sequenced_registry.csv` against the plate Excel files and found 604 sequenced wells
across b9d2/cep290 plates, but only 491 are in the registry. `MISSING_SEQUENCED_AUDIT.md` (same
dir) is the working document — mdcolon has added per-plate "Next step" notes after reviewing
the images. Your job is to investigate the open cases, update that doc with findings, and recover
any embryos that can be recovered.

---

## Key files

| File | Purpose |
|------|---------|
| `MISSING_SEQUENCED_AUDIT.md` | **The working doc** — per-plate table of missing wells with mdcolon's notes. Append findings here. |
| `HANDOFF.md` | Broader project state (scripts, label conventions, architecture) |
| `transfer_results/sequenced_registry.csv` | 491 sequenced embryos currently in the registry |
| `src/build/qc/embryo_flags.py` | QC exclusion logic — `frame_flag` already removed from exclusion |
| `results/mcolon/20260605_sci_cilia_qc_first_pass/patch_frame_flag_exclusion.py` | Script that patched qc_staged CSVs |
| `label_transfer_snapshots.py` | Regenerates sequenced_registry.csv — run after any fix |

**Data paths:**
- Raw images: `morphseq_playground/sam2_pipeline_files/raw_data_organized/<exp>/images/`
- build04 QC CSVs: `morphseq_playground/metadata/build04_output/qc_staged_<exp>.csv`
- build06 embeddings: `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_<exp>.csv`
- Plate Excel metadata: `metadata/plate_metadata/<exp>_well_metadata.xlsx`

**Python env:** `conda run -n segmentation_grounded_sam --no-capture-output python ...`

---

## What's already resolved (do NOT re-investigate)

- **`frame_flag` removed from exclusion logic** (`embryo_flags.py`). qc_staged CSVs patched.
  Wells with frame_flag-only exclusion will be recovered on next build06 rerun automatically.
- **Excel header bug fixed** (`20260414_b9d2_14hpf_plate01`): cols 6–9 were `0`, fixed to `6,7,8,9`.
  E06, E07, G07, H07 now in registry.
- **B01 genotype fixed** (`20260415_b9d2_30to48hpf_plate01_t02`): `b9d2_wt` → `b9d2_unknown`.
- **Confirmed unrecoverable** (clipped off frame, sam2 failed): `30hpf_plate01` F02,
  `30hpf_plate02` A07, `30to48hpf_plate01_t02` B06 + E07, `30hpf_plate01` G07 (cep290),
  `cep290_30to48hpf_plate02_t01` A02, `cep290_30to48hpf_plate01_t02` E12.
- **Confirmed recoverable** (frame_flag false positive, good embryo): `cep290_30to48hpf_plate01_t02`
  D09 + H03, `cep290_30to48hpf_plate02_t02` B02 (keep), `b9d2_30to48hpf_plate01_t02` B06
  (confirmed clipped — already marked unrecoverable above, keep consistent).
- **sa_outlier B01** (`plate02_t02`): mdcolon says "not a problem, sa_outlier too restrictive".
  Needs sa_outlier threshold relaxed + build06 rerun.

---

## Open investigations (your job)

### PRIORITY 1 — Pipeline issue: `20260331_b9d2_18hpf_plate01` G/H rows

**mdcolon note:** "Everything after G01 is missing. Seems like there is a pipeline issue processing
this experiment. This needs to be recovered. Thoroughly figure this out."

**What we know:**
- Raw images exist only up to G01:
  `morphseq_playground/sam2_pipeline_files/raw_data_organized/20260331_b9d2_18hpf_plate01/images/`
  lists A01–G01 only. G02–H12 directories do NOT exist.
- build04 CSV has 70 wells (A–F all 12 cols + G01 only).
- 11 sequenced wells are missing: A06, G03, G05, G08, G12, H01, H04, H06, H08, H10, H12.
- A06 is also missing — it's in row A which otherwise has images, so A06 is a separate question.

**Investigate:**
1. Is A06 present in the Keyence raw ND2 files? Check the raw data source (look for nd2 files or
   the original keyence export that feeds `raw_data_organized`). Determine if A06 was ever exported.
2. For G02–H12: why did the image export stop at G01? Was this a truncated export, a microscope
   acquisition that stopped early, or a pipeline processing cutoff? Look at:
   - The nd2/raw source for this experiment
   - Any pipeline logs for this experiment
   - The Keyence image file directory (upstream of `raw_data_organized`)
3. If the raw images exist upstream but weren't exported into `raw_data_organized`, this is a
   pipeline recovery task — re-run the export step for this plate.

### PRIORITY 2 — A06 in `20260331_b9d2_18hpf_plate01`

Separate from G/H: A06 (`b9d2_homo`, seq_code=2) is missing even though rows A–F are otherwise
complete. Check if the raw image exists at:
`morphseq_playground/sam2_pipeline_files/raw_data_organized/20260331_b9d2_18hpf_plate01/images/20260331_b9d2_18hpf_plate01_A06/`

### PRIORITY 3 — `20260414_b9d2_14hpf_plate01` H06

**mdcolon note:** "H06 clearly embryo here — investigate what happened"

H06 (`b9d2_homo`, seq_code=2) is absent from build04. Raw image dir EXISTS:
`morphseq_playground/sam2_pipeline_files/raw_data_organized/20260414_b9d2_14hpf_plate01/images/20260414_b9d2_14hpf_plate01_H06/`

So the image was captured but didn't make it into build04. Investigate:
- Is there an image in that directory?
- Is the well present in build03 output?
  `morphseq_playground/metadata/build03_output/expr_embryo_metadata_20260414_b9d2_14hpf_plate01.csv`
- If it's in build03 but not build04, something went wrong in the build04 run for this well.

### PRIORITY 4 — `20260324_cep290_30hpf_plate01` A09

**mdcolon note:** "Really odd. We clearly sequenced it, but there's nothing in the image. Want to
check if there was a parsing error in how we prescribed the well name."

A09 (`cep290_homo`, seq_code=2) is absent from build04. BUT the raw image directory exists:
`morphseq_playground/sam2_pipeline_files/raw_data_organized/20260324_cep290_30hpf_plate01/images/20260324_cep290_30hpf_plate01_A09/`
and contains `20260324_cep290_30hpf_plate01_A09_ch00_t0000.jpg`.

This is a true pipeline gap — image exists but didn't get processed into build04. Investigate:
- Is A09 in build03? Check `morphseq_playground/metadata/build03_output/expr_embryo_metadata_20260324_cep290_30hpf_plate01.csv`
- If yes, why wasn't it carried into build04?
- If no, why did build03 skip it even though the raw image exists?

### PRIORITY 5 — cep290 18hpf plates G/H rows absent

For `20260324_cep290_18hpf_plate01` and `20260324_cep290_18hpf_24hpf_plate02`:
- Raw image dirs for G07–G12, H01–H12 do NOT exist in `raw_data_organized`.
- These are likely truncated acquisitions (microscope stopped before finishing the plate).
- **mdcolon note:** "Check the raw file to see if they were actually captured."
- Check the upstream Keyence source for these experiments to confirm images were never acquired.
  If they exist upstream, this is a re-export task.

### PRIORITY 6 — `20260415_cep290_18hpf_plate03` C05 + F04

**mdcolon note:** "No embryo in here at all. Need to double check if it was sequenced."
- Check if raw image dirs exist for C05 and F04 in this plate.
- If no image, check whether the sequenced code in the Excel is a pre-fill artifact (the genotype
  sheet shows `cep290_wt` for both — but verify against any sequencing records).

### PRIORITY 7 — `20260415_b9d2_30to48hpf_plate01_t02` F11

**mdcolon note:** "Clearly there, not sure what happened."
F11 (`ab`, seq_code=1) absent from build04. Check raw image directory and build03.

---

## How to update the working doc

Append findings directly to `MISSING_SEQUENCED_AUDIT.md` under each plate's "Next step" section.
Use this format:

```
### Investigation result (2026-06-XX)
- A06: [what you found — present/absent in raw, why it's missing, action taken]
- G02+: [pipeline truncation confirmed / re-export needed / etc.]
```

Do NOT rewrite the existing mdcolon notes — append below them.

---

## How to re-run the registry after any fix

```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260605_sci_cilia_qc_first_pass/label_transfer_snapshots.py
```

Registry should grow from 491. Current breakdown: b9d2=133, cep290=141, crispant=217.

---

## Broader context / gotchas

- **Excel `genotype` = ground truth.** Don't reconcile vs sequenced codes. Low het counts are real.
- **sci_ plates** (`20260414_sci_b9d2_48hpf_plate01`, `20260415_sci_cep290_48hpf_plate01`) are
  intentionally excluded from the registry — analyzed separately via `make_3d_pca_sci.py`.
- **build06 rerun needed** to actually recover frame_flag-only embryos — the qc_staged CSVs are
  patched but build06 was generated before the patch and doesn't have embeddings for those wells.
- **End goal** (per mdcolon): a portfolio canvas — per-embryo image grid with sequenced genotype /
  predicted genotype / predicted phenotype / QC status shown above each photo. The registry audit
  is a prerequisite so the canvas has complete, accurate coverage.
- **Commit:** last commit is `17b3a3a4` ("Sequenced embryo audit: fixes, notes, and frame_flag
  demotion"). All pipeline fixes and audit docs are committed to main.
