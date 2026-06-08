# Missing Sequenced Embryo Audit — b9d2 and cep290 plates

**Last updated:** 2026-06-07 · mdcolon

For each b9d2 and cep290 plate (sci_ excluded — analyzed separately), sequenced wells that are
absent from `sequenced_registry.csv`. Use this to go through each missing embryo and decide action.

---

## Overall breakdown (non-sci b9d2 + cep290 plates only)

| Reason | Count | What to do |
|--------|-------|------------|
| **ABSENT — not imaged** | ~20 wells | Nothing — embryo existed and was sequenced but no image data was ever collected (empty well at collection, or lost before imaging). Confirmed by checking genotype sheet. |
| **TEMPLATE_ARTIFACT** | 2 wells | Skip — genotype=None in Excel, well was never actually used. (`20260324_cep290_18hpf_24hpf_plate02` G08, H08) |
| **QC_EXCLUDED — frame_flag only** | ~6 wells | Recoverable after build06 rerun. `frame_flag` has been removed from exclusion logic in `embryo_flags.py`; qc_staged CSVs already patched. Just needs new build06. |
| **QC_EXCLUDED — frame_flag + sam2_qc_flag** | ~5 wells | SAM2 segmentation genuinely failed. Review images — likely unrecoverable but worth a look. |
| **QC_EXCLUDED — sa_outlier_flag** | 1 well | `plate02_t02` B01 — unusual orientation triggers SA outlier. False positive. Needs sa_outlier threshold review. |
| **Excel header bug (fixed)** | 5 wells | `14hpf_plate01` E06, E07, G07, H07 (+ H06 absent). Header cols 6–9 were `0` instead of `6,7,8,9`. Fixed in Excel; registry now has these. |

**Status codes used below:**
- `ABSENT` — well has genotype in Excel but no image in build04. Not a pipeline bug.
- `TEMPLATE_ARTIFACT` — genotype=None in Excel, not a real embryo.
- `QC_EXCLUDED` — in build04 but filtered out; flags column shows why.
- `✅` — plate is complete, all sequenced wells in registry.

---

## 20260324_cep290_18hpf_24hpf_plate02
Sequenced in Excel: 26 | In registry: 19 | Missing: 7

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| D09 | 1 | cep290_wt | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |
| G08 | 1 | None | TEMPLATE_ARTIFACT |  | skip — no genotype, not a real embryo |
| G11 | 1 | cep290_wt | ABSENT |  | not imaged — lost at collection or empty well |
| H01 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H05 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H06 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H08 | 1 | None | TEMPLATE_ARTIFACT |  | skip — no genotype, not a real embryo |

### Next step
Check the raw file to see if they were actually captured. 

### Investigation result (2026-06-07)
- **G11, H01, H05, H06**: Raw Keyence data CONFIRMED ABSENT. Keyence dir
  `raw_image_data/Keyence/20260324_cep290_18hpf_24hpf_plate02/cep290_18hpf_24hpf_plate02/`
  has G/H rows only up to G06. XY dirs stop at G06 (78 total). These wells were never acquired
  on the microscope. Confirmed truly absent.
- **Note on G/H truncation**: Keyence has G01–G06 and no H rows at all. 78 stitched images matches
  exactly. The acquisition stopped at G06 on this plate (likely microscope ran out of time or was
  stopped early). These are not recoverable — they were never imaged.


## 20260324_cep290_18hpf_plate01
Sequenced in Excel: 26 | In registry: 22 | Missing: 4

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| G09 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H02 | 1 | cep290_wt | ABSENT |  | not imaged — lost at collection or empty well |
| H06 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H09 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |

### Next step
Check the raw file to see if they were actually captured.

### Investigation result (2026-06-07)
- **G09, H02, H06, H09**: Raw Keyence data CONFIRMED ABSENT. Keyence dir
  `raw_image_data/Keyence/20260324_cep290_18hpf_plate01/cep290_18hpf_plate01/`
  has G/H rows only up to G06 (78 wells total, matches stitched). No H rows at all.
  The acquisition stopped at G06 — same pattern as `cep290_18hpf_24hpf_plate02`.
  These wells were never imaged. Not recoverable.

## 20260324_cep290_24hpf_plate02
Sequenced in Excel: 15 | In registry: 14 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| F01 | 2 | cep290_homo | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |

## 20260324_cep290_30hpf_plate01
Sequenced in Excel: 23 | In registry: 21 | Missing: 2

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| A09 | 2 | cep290_homo | ABSENT |  | not imaged — lost at collection or empty well |
| G07 | 2 | cep290_homo | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverabl

### Next step
 
This is really odd. That A09 isn't there. We clearly sequenced it, but there's nothing at all in the image. I want to check to see if there was a parsing error here and how we prescribed the well name. 

ALso i confirmed that G07 is uncrecoverable its tail it largely clipped off frame

### Investigation result (2026-06-07)
- **A09**: Raw image EXISTS at
  `raw_data_organized/20260324_cep290_30hpf_plate01/images/20260324_cep290_30hpf_plate01_A09/`
  (file: `20260324_cep290_30hpf_plate01_A09_ch00_t0000.jpg`). BUT A09 is absent from
  build03 and build04. Root cause: **GDino detected 0 embryos** in this image
  (`num_detections=0` in `gdino_detections_20260324_cep290_30hpf_plate01.json`). The image
  was exported but the embryo detector failed to find anything. Likely an empty well or
  the embryo fell out of focus and GDino missed it. Review the JPG to confirm.
  To recover: would require re-running GDino with a lower box_threshold OR manually adding
  the detection. Recommend: view the JPG first — if there's truly nothing there, mark ABSENT.
- **G07**: Confirmed unrecoverable (tail clipped). ✓


### Next Next step
A09 deterimine if we actually sequenced it as there was nothing in the wel...

## 20260324_cep290_30hpf_plate02
✅ All 9 sequenced wells in registry.

## 20260331_b9d2_18hpf_plate01
Sequenced in Excel: 32 | In registry: 21 | Missing: 11

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| A06 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| G03 | 1 | b9d2_wt | ABSENT |  | not imaged — lost at collection or empty well |
| G05 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| G08 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| G12 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H01 | 1 | ab | ABSENT |  | not imaged — lost at collection or empty well |
| H04 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H06 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H08 | 1 | b9d2_wt | ABSENT |  | not imaged — lost at collection or empty well |
| H10 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |
| H12 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |

### Next step
- A06 Check to see why it was skipped / is it present in keyence raw data
- Everything after G01 is missing. Seems like there is a pipeline issue processing this experiment. This needs to be recovered. Thoroughly figure this out.

### Investigation result (2026-06-07)
**Root cause: `stitch_experiment()` had a `KeyError: 6` bug — A06 was re-acquired with 6 tiles,
and the `target` dict in `build01A_compile_keyence_torch.py` only handled 2 and 3 tiles.**

- **G02–H12 (and A06)**: Raw Keyence data CONFIRMED PRESENT. All G/H rows have XY dirs
  in `raw_image_data/Keyence/20260331_b9d2_18hpf_plate01/b9d2_18ss_plate01/` with real
  multi-Z TIFF files (XY73=G01, XY74=G02, ..., XY85=H12, XY97=A06). 97 total XY dirs,
  all with real data. FF projections (`built_image_data/Keyence/FF_images/`) exist for all
  96 wells (the projection step handled mixed tile counts via batch_size=1).
- **A06 is the culprit**: A06 was re-acquired (XY97). Keyence has both XY06 and XY97 both
  labeled `_A06`. `get_image_paths` appends both XY dirs into one `A06_t0000` entry → 6
  tiles. `stitch_experiment()` does `target = {2:.., 3:..}[n_tiles]` → `KeyError: 6` when
  it hits A06. This killed the stitch pass before G02–H12 were processed.
- **A06 duplicate not yet resolved**: How to handle two acquisitions of the same well
  (keep first? keep last? treat as separate time points?) is left for a future decision.
  For now, A06 will stitch with all 6 tiles (fix applied — see below), which is not
  biologically meaningful but won't crash. A06 likely needs manual review.
- **Logs confirm**: `logs/morphseq_experiment_mngr.o20781949.2` shows FF projections completed
  all 96/96, then stitch failed with `KeyError: 6`, and GDino + SAM2 ran on only 72 images.
- **Fix applied**: Added `6: np.array([2280, 630])` (and fallback `return {}` for any
  unknown tile count) to `stitch_experiment()` in `build01A_compile_keyence_torch.py`.
- **Action needed**: Re-run build01 stitch step for this experiment to stitch G02–H12 + A06.
  The FF tile images already exist, so only the stitch step needs to run:
  ```bash
  conda run -n segmentation_grounded_sam --no-capture-output python -m src.run_morphseq_pipeline.cli \
    pipeline e2e --data-root morphseq_playground \
    --experiments 20260331_b9d2_18hpf_plate01 --force
  ```
  Then re-run sam2 for the experiment to get GDino + SAM2 for the 25 new wells.

**cep290_18hpf plates (Priority 5) — NOT a pipeline bug**: Both `cep290_18hpf_plate01`
and `cep290_18hpf_24hpf_plate02` Keyence dirs only go to G06 (no H rows). Those 78 stitched
images match exactly. The microscope acquisition stopped at G06. NOT recoverable.


## 20260331_b9d2_18hpf_plate02
✅ All 13 sequenced wells in registry.

## 20260414_b9d2_14hpf_plate01
Sequenced in Excel: 31 | In registry: 30 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| H06 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |

### Next step
H06 clearly embryo here invesrivate what happened

### Investigation result (2026-06-07)
- **H06**: Raw image EXISTS at
  `raw_data_organized/20260414_b9d2_14hpf_plate01/images/20260414_b9d2_14hpf_plate01_H06/`
  (file: `20260414_b9d2_14hpf_plate01_H06_ch00_t0000.jpg`). BUT H06 is absent from
  build03 and build04. Root cause: **GDino detected 0 embryos** in this image
  (`num_detections=0` in `gdino_detections_20260414_b9d2_14hpf_plate01.json`).
  The image exists, GDino ran on it, but it found nothing. Review the JPG — mdcolon says
  "clearly an embryo" so this is a GDino false negative. To recover: either lower the
  GDino box_threshold and re-run detection for just this well, or manually add the detection
  and re-run SAM2 for H06.

## 20260414_b9d2_30hpf_plate01
Sequenced in Excel: 32 | In registry: 31 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| F02 | 2 | b9d2_homo | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverable |

### Next Steps
F02 Confirmed unsavable clipped off

## 20260414_b9d2_30hpf_plate02
Sequenced in Excel: 8 | In registry: 7 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| A07 | 1 | ab | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverable |

### Next Steps
A07 Confirmed unsavable clipped off

## 20260415_b9d2_30to48hpf_plate01_t02
Sequenced in Excel: 29 | In registry: 26 | Missing: 3

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| B06 | 2 | b9d2_homo | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |
| E07 | 2 | b9d2_homo | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverable |
| F11 | 1 | ab | ABSENT |  | not imaged — lost at collection or empty well |
### Next Steps
- B06 Confirmed unsavable clipped off
- E07 Confirmed clipped off
- F11 clearly there, not sure what happened

### Investigation result (2026-06-07)
- **F11**: Raw image EXISTS at
  `raw_data_organized/20260415_b9d2_30to48hpf_plate01_t02/images/20260415_b9d2_30to48hpf_plate01_t02_F11/`
  (file: `20260415_b9d2_30to48hpf_plate01_t02_F11_ch00_t0000.jpg`). BUT F11 is absent
  from build03 and build04. Root cause: **GDino detected 0 embryos**
  (`num_detections=0` in `gdino_detections_20260415_b9d2_30to48hpf_plate01_t02.json`).
  GDino false negative. mdcolon says "clearly there". Same recovery path as H06 above —
  review JPG, then re-run GDino with lower threshold or manually seed the detection.

## 20260415_b9d2_30to48hpf_plate02_t02
Sequenced in Excel: 10 | In registry: 9 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| B01 | 1 | b9d2_wt | QC_EXCLUDED | sa_outlier_flag | sa_outlier — review image, likely false positive |
### Next Steps
- B01 Not a problem at all. We need to fix the surface area outlier flag. It's a bit too restrictive. 

## 20260415_cep290_18hpf_plate03
Sequenced in Excel: 11 | In registry: 9 | Missing: 2

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| C05 | 1 | cep290_wt | ABSENT |  | not imaged — lost at collection or empty well |
| F04 | 1 | cep290_wt | ABSENT |  | not imaged — lost at collection or empty well |

### Next Steps
- C05 No embryo in here at all in the image. Need to double check if it was sequence
- F04 No embryo in here at all in the image. Need to double check if it was sequence

### Investigation result (2026-06-07)
- **C05 and F04**: Raw images EXIST at
  `raw_data_organized/20260415_cep290_18hpf_plate03/images/20260415_cep290_18hpf_plate03_C05/`
  and `..._F04/`. BUT both are absent from build03 and build04. Root cause: **GDino detected
  0 embryos** for both (`num_detections=0` in
  `gdino_detections_20260415_cep290_18hpf_plate03.json`).
  mdcolon says "no embryo at all in the image" — consistent with GDino finding nothing.
  Since the user confirmed there's nothing in the images, these are true empty wells.
  The sequencing record exists but the embryo was likely lost before imaging.
  Status: **ABSENT** (empty well at imaging time, not a pipeline bug). No recovery needed.

## 20260415_cep290_30to48hpf_plate02_t01
Sequenced in Excel: 5 | In registry: 4 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| A02 | 1 | cep290_wt | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverable |

## 20260416_cep290_30to48hpf_plate01_t02
Sequenced in Excel: 38 | In registry: 35 | Missing: 3

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| D09 | 2 | cep290_homo | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |
| E12 | 2 | cep290_homo | QC_EXCLUDED | frame_flag+sam2_qc_flag | sam2 failure — review image, likely unrecoverable |
| H03 | 1 | cep290_wt | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |

### Next Steps
- D09 confirmed it it good
- E12 unsalvagble but slighly clipped (could get curvature out of it) 
- H03 confirmed it is good
## 20260416_cep290_30to48hpf_plate02_t02
Sequenced in Excel: 5 | In registry: 4 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| B02 | 1 | cep290_wt | QC_EXCLUDED | frame_flag | frame_flag only — recoverable after build06 rerun |

### Next Steps
B02 Probably should be excluded, but we can keep it. 