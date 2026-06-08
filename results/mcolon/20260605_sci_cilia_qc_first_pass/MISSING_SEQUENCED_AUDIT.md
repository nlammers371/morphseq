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
 
This is really odd. That 809 isn't there. We clearly sequenced it, but there's nothing at all in the image. I want to check to see if there was a parsing error here and how we prescribed the well name. 

ALso i confirmed that G07 is uncrecoverable its tail it largely clipped off frame 

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


## 20260331_b9d2_18hpf_plate02
✅ All 13 sequenced wells in registry.

## 20260414_b9d2_14hpf_plate01
Sequenced in Excel: 31 | In registry: 30 | Missing: 1

| Well | seq_code | genotype | status | flags | action |
|------|----------|----------|--------|-------|--------|
| H06 | 2 | b9d2_homo | ABSENT |  | not imaged — lost at collection or empty well |

### Next step
H06 clearly embryo here invesrivate what happened

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