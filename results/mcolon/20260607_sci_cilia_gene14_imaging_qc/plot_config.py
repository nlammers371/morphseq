"""
plot_config.py — shared styling + constants for the 3x plotting scripts.

Colors and a few constants ONLY (no plotting functions). Keep it lean: factor a shared
helper out of the 3x scripts into here only when a second script actually repeats it.

Palette matches the earlier first-pass figures
(results/mcolon/20260605_sci_cilia_qc_first_pass) so the published plots stay consistent:
b9d2 phenotype green/orange, cep290 phenotype pink/teal, genotype blue/amber/crimson.
"""

from __future__ import annotations

# ── phenotype (homozygous binary, the confidence-plot v1 classes) ────────────────
PHENOTYPE_COLORS = {
    # b9d2
    "CE":  "#1b9e77",   # green
    "HTA": "#d95f02",   # darker orange
    # cep290
    "High_to_Low": "#E76FA2",   # pink
    "Low_to_High": "#2FB7B0",   # teal
    # kept for reference plots that still show the pooled-away class
    "Not Penetrant": "#BBBBBB",
}

# ── genotype / zygosity ──────────────────────────────────────────────────────────
GENOTYPE_COLORS = {
    "wildtype":     "#2166AC",   # blue
    "heterozygous": "#F7B267",   # amber
    "homozygous":   "#B2182B",   # crimson
    "unknown":      "#808080",
    # crispant genotype labels
    "ab_wildtype":          "#999999",
    "foxj1a_crispant":      "#9467bd",
    "ift88_crispant":       "#1f77b4",
    "ift88_ift74_crispant": "#ff7f0e",
    "sspo_crispant":        "#2ca02c",
}

# ── status / source (audit + portfolio) ──────────────────────────────────────────
STATUS_COLORS = {
    "included":  "#1b9e77",
    "excluded":  "#B2182B",
    "missing":   "#777777",
    "reference": "#cccccc",
}

# ── developmental ages a plate is collected at (hpf) ─────────────────────────────
DESIGN_STAGES_HPF = [14, 18, 24, 30, 48]


def snap_to_design_stage(hpf, stages=DESIGN_STAGES_HPF, tol: float = 2.0):
    """Nearest design stage within +/- tol hpf, else None.

    Shared by 3b/3c/3d so the plate x design-stage binning is identical everywhere
    (was duplicated in the old monolith 3_plot script).
    """
    import pandas as pd  # local import keeps this module dependency-light at import time

    if pd.isna(hpf):
        return None
    for s in stages:
        if abs(hpf - s) <= tol:
            return s
    return None


# ── confidence-plot columns = collection x support (PER GENE) ────────────────────
# The collection times are NOT shared across genes (verified against the data):
#   b9d2   collected at 14, 18, 30, 48 hpf  -> HAS 14 hpf, has NO 24 hpf.
#   cep290 collected at 18, 24, 30, 48 hpf  -> has 24 hpf, has NO 14 hpf.
# For both, the 48 hpf collection appears TWICE: plate02 single snapshot vs plate01
# 30->48 timeseries. The timeseries column should look BETTER (more support).
# Each tuple is (collection_time_hpf, data_source, column label).
PHENOTYPE_COLUMNS = {
    "b9d2": [
        (14, "snapshot",   "14 hpf"),
        (18, "snapshot",   "18 hpf"),
        (30, "snapshot",   "30 hpf"),
        (48, "snapshot",   "48 hpf\nsnapshot (plate02)"),
        (48, "timeseries", "48 hpf\ntimeseries (plate01)"),
    ],
    "cep290": [
        (18, "snapshot",   "18 hpf"),
        (24, "snapshot",   "24 hpf"),
        (30, "snapshot",   "30 hpf"),
        (48, "snapshot",   "48 hpf\nsnapshot (plate02)"),
        (48, "timeseries", "48 hpf\ntimeseries (plate01)"),
    ],
}
# crispant -> 4 columns, snapshot only (no timeseries).
CRISPANT_COLUMNS = [
    (18, "snapshot", "18 hpf"),
    (24, "snapshot", "24 hpf"),
    (30, "snapshot", "30 hpf"),
    (48, "snapshot", "48 hpf"),
]


def color_for(label: str, kind: str = "phenotype") -> str:
    """Look up a label's color; gray fallback. kind in {'phenotype','genotype','status'}."""
    table = {"phenotype": PHENOTYPE_COLORS, "genotype": GENOTYPE_COLORS,
             "status": STATUS_COLORS}[kind]
    return table.get(label, "#808080")
