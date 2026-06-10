"""
3a - Sequenced-vs-pipeline coverage audit (data-sanity, not a matplotlib figure).

"The Excel `sequenced` sheet says these wells were sequenced — did they actually make it
through the pipeline?" For each non-sci_ b9d2/cep290 plate, read the `sequenced` sheet from
the plate Excel and the build04 qc_staged CSV, and classify every sequenced well:
  OK          — in build04 and passes QC (usable_embryo=1)
  QC_EXCLUDED — in build04 but flagged out (usable_embryo=0); records which *_flag fired
  ABSENT      — not in build04 at all (never stitched / GDino miss / never imaged)
  NO_BUILD04  — build04 CSV doesn't exist for this experiment

Ported from ../20260605_sci_cilia_qc_first_pass/audit_sequenced_coverage.py; the EXPS list
is now derived from this cohort's tables/experiment_manifest.csv (non-sci b9d2/cep290),
and it WRITES its outputs (the original only printed):
    MISSING_SEQUENCED_AUDIT.md            (the audit narrative)
    tables/sequenced_coverage_audit.csv   (one row per sequenced well)
    plots/audit/sequenced_coverage_heatmap.png   (plate × status counts)

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3a_audit_sequenced_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REPO = PROJECT_ROOT
PLAY = REPO / "morphseq_playground"
PLATE_META = REPO / "metadata/plate_metadata"
BUILD04 = PLAY / "metadata/build04_output"

TABLE_DIR = RUN_DIR / "tables"
AUDIT_PLOT_DIR = RUN_DIR / "plots" / "audit"
AUDIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

WELLS = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]
STATUS_ORDER = ["OK", "QC_EXCLUDED", "ABSENT", "NO_BUILD04"]


def cohort_experiments() -> list[str]:
    """Non-sci_ b9d2/cep290 plates from this cohort's manifest (these carry a `sequenced` sheet)."""
    m = pd.read_csv(TABLE_DIR / "experiment_manifest.csv")
    sel = m[(~m["is_sci_timelapse"]) & (m["gene"].isin(["b9d2", "cep290"]))]
    return sorted(sel["experiment"].astype(str).unique())


def sequenced_grid(exp: str) -> dict[str, int] | None:
    """Parse the 8×12 `sequenced` sheet → {well: code}. Returns None if no Excel/sheet found."""
    for cand in (f"{exp}_well_metadata.xlsx", f"{exp}.xlsx"):
        p = PLATE_META / cand
        if not p.exists():
            continue
        with pd.ExcelFile(p) as xlf:
            if "sequenced" not in xlf.sheet_names:
                return None
            df = xlf.parse("sequenced", header=0)
            block = df.iloc[:8, 1:13].reindex(index=range(8), columns=range(1, 13), fill_value="")
            arr = block.to_numpy(dtype=str).ravel()
        out: dict[str, int] = {}
        for w, v in zip(WELLS, arr):
            s = v.strip()
            try:
                out[w] = int(float(s)) if s not in ("", "nan") else 0
            except ValueError:
                out[w] = 0
        return out
    return None


def audit(exps: list[str]) -> pd.DataFrame:
    rows = []
    for exp in exps:
        grid = sequenced_grid(exp)
        if grid is None:
            print(f"  WARNING: no plate Excel / `sequenced` sheet found for {exp}")
            continue

        seq_wells = {w for w, v in grid.items() if v in (1, 2)}
        if not seq_wells:
            continue

        b04_path = BUILD04 / f"qc_staged_{exp}.csv"
        if not b04_path.exists():
            for w in sorted(seq_wells):
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": "NO_BUILD04", "flags": ""})
            continue

        b04 = pd.read_csv(b04_path)
        b04_wells = set(b04["well"].astype(str).str.strip())

        for w in sorted(seq_wells):
            if w not in b04_wells:
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": "ABSENT", "flags": ""})
            else:
                row = b04[b04["well"] == w].iloc[0]
                usable = bool(row.get("usable_embryo", 1))
                flags = [c for c in b04.columns if c.endswith("_flag") and row.get(c, 0)]
                status = "OK" if usable else "QC_EXCLUDED"
                rows.append({"exp": exp, "well": w, "seq_code": grid[w],
                             "status": status, "flags": "|".join(flags)})
    return pd.DataFrame(rows)


def coverage_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby(["exp", "status"]).size().unstack(fill_value=0)
    for col in STATUS_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[STATUS_ORDER]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_index()


def write_markdown(df: pd.DataFrame, pivot: pd.DataFrame, path: Path) -> None:
    counts = df["status"].value_counts()
    lines = ["# Sequenced-vs-pipeline coverage audit", "",
             "Excel `sequenced` sheet vs build04 QC, for non-sci_ b9d2/cep290 plates. "
             "Generated by `3a_audit_sequenced_coverage.py`.", "",
             "## Totals", ""]
    lines += [f"- **{s}**: {int(counts.get(s, 0))}" for s in STATUS_ORDER]
    lines += [f"- **TOTAL sequenced wells**: {len(df)}", "", "## By experiment", "",
              "| experiment | " + " | ".join(STATUS_ORDER) + " | total |",
              "|" + "---|" * (len(STATUS_ORDER) + 2)]
    for exp, r in pivot.iterrows():
        lines.append("| " + exp + " | "
                     + " | ".join(str(int(r[s])) for s in STATUS_ORDER)
                     + f" | {int(r['total'])} |")

    for label, status in [("ABSENT", "ABSENT"), ("QC_EXCLUDED", "QC_EXCLUDED")]:
        sub = df[df["status"] == status]
        lines += ["", f"## {label} detail", ""]
        if sub.empty:
            lines.append("_(none)_")
        else:
            cols = ["exp", "well", "seq_code"] + (["flags"] if status == "QC_EXCLUDED" else [])
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "---|" * len(cols))
            for _, rr in sub.iterrows():
                lines.append("| " + " | ".join(str(rr[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(RUN_DIR)}")


def plot_heatmap(pivot: pd.DataFrame, path: Path) -> None:
    exps = list(pivot.index)
    M = pivot[STATUS_ORDER].to_numpy()
    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(STATUS_ORDER), 1.2 + 0.34 * len(exps)))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(STATUS_ORDER)))
    ax.set_xticklabels(STATUS_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=7)
    for i in range(len(exps)):
        for j in range(len(STATUS_ORDER)):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > M.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="sequenced wells")
    ax.set_title("Sequenced-well coverage by plate × status", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


print("3a - sequenced-vs-pipeline coverage audit")
exps = cohort_experiments()
print(f"Auditing {len(exps)} non-sci b9d2/cep290 plates")
df = audit(exps)
if df.empty:
    print("No sequenced wells found across the cohort — nothing to audit.")
    sys.exit(0)

print("\n=== SEQUENCED WELL COVERAGE ===")
print(df["status"].value_counts().to_string())
print(f"Total sequenced wells in Excel: {len(df)}")

pivot = coverage_pivot(df)
print("\n=== BY EXPERIMENT ===")
print(pivot.to_string())

csv_path = TABLE_DIR / "sequenced_coverage_audit.csv"
df.to_csv(csv_path, index=False)
print(f"\n  wrote {csv_path.relative_to(RUN_DIR)}")
write_markdown(df, pivot, RUN_DIR / "MISSING_SEQUENCED_AUDIT.md")
plot_heatmap(pivot, AUDIT_PLOT_DIR / "sequenced_coverage_heatmap.png")
print("\nDone.")
