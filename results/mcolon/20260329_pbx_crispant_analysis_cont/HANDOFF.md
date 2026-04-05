# PBX Continuation Handoff

## Analysis home

Continue PBX refresh and follow-up work from:

- `results/mcolon/20260329_pbx_crispant_analysis_cont`

The copied rerun bundle lives at:

- `results/mcolon/20260329_pbx_crispant_analysis_cont/rerun_20260329_no_yolk`

This bundle was moved out of the older `20260326_pbx_crispant_analysis` tree and patched so new reruns write into the `20260329_pbx_crispant_analysis_cont` results/figures roots.

## QC policy change

The core upstream change is that `no_yolk_flag` was removed from the `use_embryo_flag` exclusion logic.

Files:

- `src/build/qc/embryo_flags.py`
- `src/build/patch_build04_use_embryo_flag.py`
- `tests/test_embryo_flags.py`

Operational follow-up:

- patched affected `build04` QC CSVs in place
- updated `src/run_morphseq_pipeline/run_build03_onwards_force.sh` defaults to the PBX rerun experiment set
- reran / planned rerun of downstream `build06` outputs from the patched `build04`

## Current PBX interpretation state

- The multiclass aligned UMAPs currently look better than the pairwise support-aware embedding for the global phenotypic positioning view.
- The `4 hpf` bin-width runs look preliminarily more stable and may have more statistical power than the finer bins, but that is still provisional and should be checked quantitatively rather than assumed from appearance alone.
- The no-yolk QC looked effectively broken in the PBX data and was excluding a large number of otherwise useful `snip_id`s.
- For the bridge experiment follow-up, the current preference is to proceed with the `20251207_pbx + 20260304 + 20260306` bridge-enabled multiclass run **without** `wik_ab`. Including `wik_ab` from the current experiments only was useful as a sensitivity check, but the no-`wik_ab` bridge run currently appears to give denser, cleaner clusters and is the preferred version for interpretation unless a later comparison changes that.
- Refactored attraction/coherence amplitude sweeps on the bifurcating-trunk benchmark were useful diagnostically, but the separated-amplitude path is **not** the preferred production behavior. The inspectable evidence is recorded in `FORCE_SEPARATION_NOTE.md`; the current preference is to keep the naming cleanup (`temporal_cohere_*`, `attract_*`, `solver_*`) but revert solver behavior to the original multiplicative attraction × coherence interaction.

## Rerun status

Already done:

- copied the no-yolk rerun scripts into the new `20260329` continuation root
- patched the copied rerun driver to use the newer heatmap plotting API
- patched the copied rerun driver so both the `inj_ctrl` and `wik_ab` classification branches are intended to use `500` permutations
- copied the existing rerun outputs from the old `20260326` tree into the new `20260329` tree to avoid wasting recomputation

Known state at handoff:

- the `inj_ctrl` comparison branch was successfully refreshed to `500` permutations
- the `all_crispants_vs_wik_ab` branch still needed a clean completed `500`-permutation rerun in the new tree
- because of that, the classification heatmap figures in the new tree should be treated as not yet final until the `wik_ab` branch is confirmed updated and the plotting scripts are rerun

## Immediate next steps

1. Confirm `n_permutations == 500` in:
   - `results/mcolon/20260329_pbx_crispant_analysis_cont/results/rerun_20260329_no_yolk_removed/bin_width_2.0hpf/classification/20260304_20260306_all_crispants_vs_wik_ab_*_comparisons.csv`
2. Rerun the three classification heatmap scripts in the new tree:
   - `data_viz/01_plot_classification_heatmaps.py`
   - `data_viz/03_plot_all_vs_inj_ctrl_heatmaps.py`
   - `data_viz/04_plot_control_comparison_heatmaps.py`
3. Then continue the rest of the misclassification refresh out of `20260329_pbx_crispant_analysis_cont`, not the older `20260326` root.
