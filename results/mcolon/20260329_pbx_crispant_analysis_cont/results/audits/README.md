# Audits

This family contains the diagnostic runs that answer whether the geometry or representation is behaving the way we expect.

High-level script sources:
- `01_bridge_qc_heatmaps.py`
- `04_compare_bridge_with_without_wik_ab.py`
- `10_wikab_injctrl_batch_audit.py`
- `11_wikab_injctrl_lineage_audit.py`
- `12_wikab_injctrl_within_bin_null_audit.py`
- `13_wikab_injctrl_probe_and_init_audit.py`
- `14_wikab_injctrl_representation_ablation_audit.py`
- `15_wikab_injctrl_focal_probe_zoom.py`

Subfolders:
- `batch/`: experiment-separation checks
- `lineage/`: upstream lineage and time-bin source checks
- `null_checks/`: matched-bin null comparisons
- `probe/`: probe-level zoom-ins and init sanity checks
- `qc/`: upstream QC and rerun checks
- `representation/`: layout and encoding ablations

Typical outputs:
- `*_SUMMARY.md`
- `*.csv` tables for AUROC, support, and per-bin metrics
- `*.png` diagnostic panels
- `*.gif` when a geometry animation is useful
