# Missing Sequenced Embryo Audit — b9d2 and cep290 plates

**Last updated:** 2026-06-08 · mdcolon / Claude

For each b9d2 and cep290 plate (sci_ excluded — analyzed separately), sequenced wells that are
absent from `sequenced_registry.csv`. Use this to go through each missing embryo and decide action.

Run the audit script to regenerate current numbers:
```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260605_sci_cilia_qc_first_pass/audit_sequenced_coverage.py
```

---

## Overall breakdown — current state (2026-06-08)

**297 OK / 16 ABSENT / 0 QC_EXCLUDED** out of 313 sequenced wells across 16 plates
(1 plate missing Excel: `cep290_24hpf_plate01`).

All previously QC_EXCLUDED wells are now resolved:
- `frame_flag` removed from exclusion logic + build06 rerun → recovered
- SAM2 failures + clipped embryos confirmed unrecoverable by mdcolon → accepted as lost
- `sa_outlier` B01 (`plate02_t02`) → recovered (threshold relaxed)

Remaining 16 ABSENT wells fall into two buckets:

| Bucket | Wells | Disposition |
|--------|-------|-------------|
| **Truncated acquisition** — Keyence stopped before G/H rows | 10 wells (cep290 18hpf plates) | Confirmed never imaged. Not recoverable. |
| **GDino false negative** — image exists, detector found nothing | 4 wells | Need image review. May be empty well or true false negative. |
| **Truly empty well** — image exists, mdcolon confirmed no embryo | 2 wells | Confirmed empty (cep290_18hpf_plate03 C05, F04). |

**Status codes used below:**
- `ABSENT` — well has genotype in Excel but no image in build04.
- `ABSENT (truncated acq.)` — Keyence acquisition stopped early; well was never imaged.
- `ABSENT (GDino FN)` — image exported but GDino detected 0 embryos; needs review.
- `ABSENT (empty well)` — mdcolon confirmed no embryo visible in the image.
- `OK` — in build04, passes QC.
- `✅` — plate fully covered, all sequenced wells OK.

---

## 20260324_cep290_18hpf_24hpf_plate02
Sequenced in Excel: 26 | **OK: 20 | ABSENT: 6** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| G08 | 1 | None | ABSENT (truncated acq.) | Template well — no genotype. Acquisition stopped before G/H rows. |
| G11 | 1 | cep290_wt | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H01 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H05 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H06 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H08 | 1 | None | ABSENT (truncated acq.) | Template well — no genotype. Acquisition stopped before G/H rows. |

**Note:** D09 was previously QC_EXCLUDED (frame_flag) — now OK after build06 rerun.
Keyence dir has G/H only up to G06 (78 XY dirs). Not recoverable.


## 20260324_cep290_18hpf_plate01
Sequenced in Excel: 26 | **OK: 22 | ABSENT: 4** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| G09 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H02 | 1 | cep290_wt | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H06 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |
| H09 | 2 | cep290_homo | ABSENT (truncated acq.) | Keyence stopped at G06 — never imaged. |

Keyence dir stops at G06 (78 XY dirs), same pattern as `cep290_18hpf_24hpf_plate02`. Not recoverable.

## 20260324_cep290_24hpf_plate02
✅ Sequenced in Excel: 15 | **OK: 15** (2026-06-08)

F01 was previously QC_EXCLUDED (frame_flag) — now OK after build06 rerun.

## 20260324_cep290_30hpf_plate01
Sequenced in Excel: 23 | **OK: 22 | ABSENT: 1** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| A09 | 2 | cep290_homo | ABSENT (GDino FN) | Image exported, GDino found 0 embryos. Needs image review — determine if truly empty or false negative. |

**Note:** G07 was QC_EXCLUDED (frame_flag+sam2_qc_flag) — tail clipped, confirmed unrecoverable by mdcolon. Accepted as lost.

## 20260324_cep290_30hpf_plate02
✅ Sequenced in Excel: 9 | **OK: 9** (2026-06-08)

## 20260331_b9d2_18hpf_plate01
Sequenced in Excel: 32 | **OK: 31 | ABSENT: 1** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| A06 | 2 | b9d2_homo | ABSENT (GDino FN) | Re-acquired well (2 XY dirs → 6 tiles stitched). Image exported, GDino found 0 embryos. Needs image review. |

**Recovery complete for G02–H12:** Pipeline bug fixed (2026-06-08). Root cause was
`stitch_experiment()` crashing with `KeyError: 6` on A06's 6-tile image, stopping the stitch
before G02–H12. Fix: added 6-tile target size + wrapped `mosaic.load_params()` in try/except.
G02–H12 (+ most of G/H) are now OK after force rerun. A06 still absent — GDino false negative
on the 6-tile stitched image (double-width image may confuse detector).


## 20260331_b9d2_18hpf_plate02
✅ Sequenced in Excel: 13 | **OK: 13** (2026-06-08)

## 20260414_b9d2_14hpf_plate01
Sequenced in Excel: 31 | **OK: 30 | ABSENT: 1** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| H06 | 2 | b9d2_homo | ABSENT (GDino FN) | Image exists, GDino found 0 embryos. mdcolon: "clearly an embryo". GDino false negative. |

## 20260414_b9d2_30hpf_plate01
✅ Sequenced in Excel: 32 | **OK: 32** (2026-06-08)

F02 was QC_EXCLUDED (frame_flag+sam2_qc_flag) — tail clipped, confirmed unrecoverable by mdcolon. Accepted as lost (not sequenced well we care about tracking).

## 20260414_b9d2_30hpf_plate02
✅ Sequenced in Excel: 8 | **OK: 8** (2026-06-08)

A07 was QC_EXCLUDED (frame_flag+sam2_qc_flag) — clipped, confirmed unrecoverable by mdcolon. Accepted as lost.

## 20260415_b9d2_30to48hpf_plate01_t02
Sequenced in Excel: 29 | **OK: 28 | ABSENT: 1** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| F11 | 1 | ab | ABSENT (GDino FN) | Image exists, GDino found 0 embryos. mdcolon: "clearly there". GDino false negative. |

B06 and E07 were QC_EXCLUDED (frame_flag / frame_flag+sam2_qc_flag) — both confirmed clipped by mdcolon. Accepted as lost.

## 20260415_b9d2_30to48hpf_plate02_t02
✅ Sequenced in Excel: 10 | **OK: 10** (2026-06-08)

B01 was QC_EXCLUDED (sa_outlier_flag) — mdcolon confirmed good embryo, false positive. Recovered after sa_outlier threshold relaxed + build06 rerun.

## 20260415_cep290_18hpf_plate03
Sequenced in Excel: 11 | **OK: 9 | ABSENT: 2** (2026-06-08)

| Well | seq_code | genotype | status | disposition |
|------|----------|----------|--------|-------------|
| C05 | 1 | cep290_wt | ABSENT (empty well) | Image exported, GDino found 0 embryos. mdcolon confirmed: no embryo visible. Lost before imaging. |
| F04 | 1 | cep290_wt | ABSENT (empty well) | Image exported, GDino found 0 embryos. mdcolon confirmed: no embryo visible. Lost before imaging. |

## 20260415_cep290_30to48hpf_plate02_t01
✅ Sequenced in Excel: 5 | **OK: 5** (2026-06-08)

A02 was QC_EXCLUDED (frame_flag+sam2_qc_flag) — confirmed unrecoverable (clipped). Accepted as lost.

## 20260416_cep290_30to48hpf_plate01_t02
✅ Sequenced in Excel: 38 | **OK: 38** (2026-06-08)

D09 and H03 were QC_EXCLUDED (frame_flag only) — mdcolon confirmed both are good embryos. Recovered after build06 rerun.
E12 was QC_EXCLUDED (frame_flag+sam2_qc_flag) — slightly clipped, mdcolon confirmed unrecoverable. Accepted as lost.
## 20260416_cep290_30to48hpf_plate02_t02
✅ Sequenced in Excel: 5 | **OK: 5** (2026-06-08)

B02 was QC_EXCLUDED (frame_flag only) — mdcolon: "probably should be excluded but we can keep it." Recovered after build06 rerun. 