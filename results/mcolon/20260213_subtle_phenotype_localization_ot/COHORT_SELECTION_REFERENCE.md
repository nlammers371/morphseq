# Cohort Selection: Migration Reference

**Date**: 2026-02-13  
**Status**: Stream D cohort selection is tracked under `results/mcolon/20260213_stream_d_reference_embryo/`

---

## Current Location

```
results/mcolon/20260213_stream_d_reference_embryo/
├── pipeline/
│   └── 01_build_cohort_manifest.py  ← Cohort selection script
├── output/
│   └── cohort_selection/
│       ├── cohort_selected_embryos.csv       ← Reference + heldout WT + mutants
│       ├── cohort_bin_frame_manifest.csv     ← Per-bin frame assignments
│       ├── cohort_qc_table.csv               ← All candidates with QC metrics
│       ├── cohort_qc_scatter.png             ← Selection visualization
│       ├── cohort_selected_bin_heatmap.png   ← Coverage heatmap
│       ├── cohort_transition_manifest.csv    ← Consecutive-frame pairs
│       └── wt_ranked_candidates.csv          ← All WT candidates ranked
└── notes/
    └── README.md                              ← Migration notes
```

---

## Selected Embryos (48 hpf pilot)

### Reference WT (n=1)
- **Role**: OT target/template (defines spatial coordinate system)
- **Selection**: `set_type='reference_wt'`, `set_rank=1` (best coverage, lowest curvature)
- **Not used in statistical tests** (just provides template space)

### Heldout WT (n=10)
- **Role**: WT controls for statistical comparison
- **Selection**: `set_type='heldout_wt'` (next-best after reference)
- **Used in permutation tests** (WT control vs mutant labels shuffled)

### Mutants (n=20)
- **Role**: cep290 homozygous mutants
- **Selection**: `set_type='mutant'` (best coverage, lowest curvature among mutants)
- **Compared to WT controls** in statistical tests

---

## Selection Criteria

1. **Maximize 24-48 hpf bin coverage** (13 bins at 2h intervals)
2. **Minimize curvature** (among tied coverage candidates)
3. **Sufficient frame count** (tie-breaker)

**Algorithm** (from `01_build_cohort_manifest.py`):
```python
# Sort by: (n_bins_covered DESC, curvature_median ASC, n_frames DESC, embryo_id ASC)
# Pick top n embryos per cohort:
#   - reference_wt: 1 (rank 1 at 48 hpf for pilot)
#   - heldout_wt: 10 (all used as WT controls)
#   - mutant: 20
```

---

## How to Use in This Project

### Load pre-selected cohorts:

```python
import pandas as pd
from pathlib import Path

# Load manifests
cohort_df = pd.read_csv(
    "results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_selected_embryos.csv"
)
bin_manifest = pd.read_csv(
    "results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_bin_frame_manifest.csv"
)

# Extract 48 hpf embryos
bin_48 = bin_manifest[bin_manifest["bin_hpf"] == 48.0]

# Filter by role
reference = bin_48[bin_48["set_type"] == "reference_wt"].iloc[0]  # Rank 1
controls = bin_48[bin_48["set_type"] == "heldout_wt"]
mutants = bin_48[bin_48["set_type"] == "mutant"]

print(f"Reference: {reference['embryo_id']}, frame {reference['frame_index']}")
print(f"Controls: n={len(controls)}")
print(f"Mutants: n={len(mutants)}")
```

### Re-run cohort selection (if needed):

```bash
cd results/mcolon/20260213_stream_d_reference_embryo

python pipeline/01_build_cohort_manifest.py \
  --csv ../20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv \
  --output-root output/cohort_selection \
  --start-hpf 24.0 \
  --end-hpf 48.0 \
  --step-hpf 2.0 \
  --match-tolerance-hpf 1.25 \
  --n-ref-wt 1 \
  --n-holdout-wt 10 \
  --n-mutants 20
```

---

## Fallback: Search for Migrated Files

If files are not at expected location, use this search pattern:

```python
from pathlib import Path

def find_cohort_manifest(base_dir: Path) -> Path:
    """Search for cohort manifest in results/mcolon/2026* directories."""
    search_dirs = sorted(base_dir.glob("2026*"), reverse=True)
    for d in search_dirs:
        candidates = list(d.rglob("cohort_selected_embryos.csv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("Cohort manifest not found")

manifest_path = find_cohort_manifest(Path("results/mcolon"))
```

---

## Notes

- **Migration-safe**: Script `00_select_reference_and_load_data.py` includes fallback search
- **Reproducible**: Original selection script preserved in `pipeline/`
- **QC plots available**: Check `cohort_qc_scatter.png` for selection visualization
- **All 13 bins covered**: Each selected embryo has frames at all 24-48 hpf bins (2h intervals)

---

## Related Documents

- [PLAN.md](PLAN.md) - Full specification (Section 1.1 references cohort selection)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Section 0 implementation details
- [README.md](README.md) - Overview and infrastructure notes
- [config.yaml](config.yaml) - Configuration parameters

---

**Last updated**: 2026-02-13
