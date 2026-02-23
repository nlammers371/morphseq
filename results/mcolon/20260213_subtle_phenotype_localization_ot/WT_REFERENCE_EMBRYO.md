# WT Reference Embryo (Locked for Pilot)

Date: 2026-02-13

## Locked Selection

- Reference embryo ID: `20251212_H07_e01`
- Frame index: `90`
- Genotype: `cep290_wildtype`
- Matched stage: `47.8071846890625` hpf
- Stage error vs 48 hpf: `0.1928153109374974`

## Selection Contract

- Contract file: `data/cohort_contract_48hpf.json`
- Source manifests:
  - `results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_selected_embryos.csv`
  - `results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_bin_frame_manifest.csv`

## Cohort Sizes (Locked)

- `reference_wt`: 1
- `heldout_wt` (controls): 10
- `mutant`: 20

## Temporal Policy

- Primary mode: single 2 hpf bin at 48 hpf with `tolerance_hpf=1.25`.
- Optional robustness mode: expanded 4 hpf window `[46, 50]`.
- If expanded mode is used, collapse to one row per embryo per feature via median before WT-vs-mutant statistics.
