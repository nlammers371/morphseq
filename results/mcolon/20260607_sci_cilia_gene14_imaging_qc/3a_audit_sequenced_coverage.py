"""
3a - Sequenced-vs-pipeline coverage audit with real failure modes.

"The Excel `sequenced` sheet says these wells were sequenced — what actually happened to each one
in the pipeline?" For every non-sci_ b9d2/cep290 plate, read the `sequenced` sheet and the build04
qc_staged CSV, and classify each sequenced well into the failure-mode taxonomy:

  OK                 — in build04 AND use_embryo_flag truthy (passed QC)
  EXCLUDED           — in build04 AND use_embryo_flag falsy; records which REAL QC flags fired
  ABSENT_IMAGED      — not in build04 but a stitched image exists (GDino FN / empty-well candidate)
  ABSENT_NO_IMAGE    — not in build04 and no stitched image (truncated-acq candidate)
  QC_NOT_RUN         — no build04 CSV for this experiment; QC/latent pipeline never run
                       (images may already be stitched — recoverable by rerunning build04/06)

The auto status is then refined by a human-curated disposition sidecar
(tables/well_dispositions.csv: exp,well,disposition,note) carried over from the curated
MISSING_SEQUENCED_AUDIT_curated_20260608.md. Auto-detection fills what it can; the sidecar
supplies the final reason (truncated_acq / gdino_fn / empty_well / clipped_lost / needs_review)
where only a human could know. Re-running never destroys those notes.

Why this rewrite: the previous version tested a NON-EXISTENT `usable_embryo` column, so every
in-build04 well defaulted to OK (0 EXCLUDED — wrong; e.g. plate02_t01 A02 is clipped/excluded),
and ABSENT was a single flat bucket that hid the real failure modes.

Outputs:
    MISSING_SEQUENCED_AUDIT.md            (regenerated narrative)
    tables/sequenced_coverage_audit.csv   (one row per sequenced well)
    tables/embryo_loss_map.csv            (embryo_id -> status/disposition, machine-readable)
    plots/audit/sequenced_coverage_heatmap.png   (plate x status counts)
    plots/audit/status_stacked_bar.png           (per-plate stacked status bars)
    plots/audit/wellgrids/wellgrid_<exp>.png     (per-plate 8x12 well maps)

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3a_audit_sequenced_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
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
STITCHED_FF = PLAY / "built_image_data/stitched_FF_images"

TABLE_DIR = RUN_DIR / "tables"
AUDIT_PLOT_DIR = RUN_DIR / "plots" / "audit"
WELLGRID_DIR = AUDIT_PLOT_DIR / "wellgrids"
AUDIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
WELLGRID_DIR.mkdir(parents=True, exist_ok=True)

DISPOSITIONS_CSV = TABLE_DIR / "well_dispositions.csv"
CURATED_MD = "MISSING_SEQUENCED_AUDIT_curated_20260608.md"

ROWS = "ABCDEFGH"
COLS = list(range(1, 13))
WELLS = [f"{r}{c:02}" for r in ROWS for c in COLS]

# The REAL QC-exclusion flag columns in build04. Informational columns
# (well_qc_flag, control_flag, use_embryo_flag) are deliberately NOT here.
REAL_QC_FLAGS = [
    "frame_flag", "sam2_qc_flag", "no_yolk_flag", "focus_flag",
    "bubble_flag", "dead_flag", "dead_flag2", "sa_outlier_flag",
]

STATUS_ORDER = ["OK", "EXCLUDED", "ABSENT_IMAGED", "ABSENT_NO_IMAGE", "QC_NOT_RUN"]

# Plotting colors per status (well-grid + bars). non-sequenced wells render gray.
STATUS_COLORS = {
    "OK": "#2ca02c",                 # green
    "EXCLUDED": "#ff7f0e",           # orange
    "ABSENT_IMAGED": "#d62728",      # red
    "ABSENT_NO_IMAGE": "#7f1d1d",    # dark red
    "QC_NOT_RUN": "#9467bd",  # purple
    "NOT_SEQUENCED": "#e0e0e0",      # light gray (background)
}


def cohort_experiments() -> list[str]:
    """Non-sci_ b9d2/cep290/crispant plates from this cohort's manifest (these carry a `sequenced` sheet)."""
    m = pd.read_csv(TABLE_DIR / "experiment_manifest.csv")
    sel = m[(~m["is_sci_timelapse"]) & (m["gene"].isin(["b9d2", "cep290", "crispant"]))]
    return sorted(sel["experiment"].astype(str).unique())


def sequenced_grid(exp: str) -> dict[str, int] | None:
    """Parse the 8x12 `sequenced` sheet -> {well: code}. Returns None if no Excel/sheet found."""
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


def stitched_image_exists(exp: str, well: str) -> bool:
    """Was a stitched FF image produced for this well? (image-exists signal for ABSENT split)."""
    d = STITCHED_FF / exp
    if not d.is_dir():
        return False
    return any(d.glob(f"{well}_*stitch*"))


def load_dispositions() -> pd.DataFrame:
    """Human-curated (exp,well) -> disposition/note. Empty frame if the sidecar is missing."""
    if not DISPOSITIONS_CSV.exists():
        print(f"  WARNING: no dispositions sidecar at {DISPOSITIONS_CSV.relative_to(RUN_DIR)}")
        return pd.DataFrame(columns=["exp", "well", "disposition", "note"])
    d = pd.read_csv(DISPOSITIONS_CSV, dtype=str).fillna("")
    return d[["exp", "well", "disposition", "note"]]


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
        b04 = pd.read_csv(b04_path) if b04_path.exists() else None
        b04_wells = (
            set(b04["well"].astype(str).str.strip()) if b04 is not None else set()
        )

        for w in sorted(seq_wells):
            embryo_id, flags = "", ""
            if b04 is None:
                status = "QC_NOT_RUN"
            elif w not in b04_wells:
                status = "ABSENT_IMAGED" if stitched_image_exists(exp, w) else "ABSENT_NO_IMAGE"
            else:
                row = b04[b04["well"].astype(str).str.strip() == w].iloc[0]
                embryo_id = str(row.get("embryo_id", "") or "")
                use_ok = bool(row.get("use_embryo_flag", False))
                fired = [f for f in REAL_QC_FLAGS if bool(row.get(f, False))]
                status = "OK" if use_ok else "EXCLUDED"
                flags = "|".join(fired)

            if not embryo_id:
                embryo_id = f"{exp}_{w}"  # stable key for wells with no build04 embryo

            rows.append({
                "embryo_id": embryo_id, "exp": exp, "well": w,
                "seq_code": grid[w], "status": status, "exclusion_flags": flags,
            })

    df = pd.DataFrame(rows)

    # Join human dispositions. Fill auto-defaults where the sidecar is silent.
    disp = load_dispositions()
    df = df.merge(disp, on=["exp", "well"], how="left")
    df["disposition"] = df["disposition"].fillna("")
    df["note"] = df["note"].fillna("")
    df["disposition_conflict"] = ""

    # Self-heal stale dispositions: a sidecar "absence" reason that now contradicts a live
    # present-in-build04 status (OK/EXCLUDED) means the data was reprocessed since the curated
    # snapshot. Trust the data — drop the stale disposition and record the conflict so the
    # sidecar can be pruned. (e.g. cep290_18hpf G/H wells were "truncated_acq" on 2026-06-08
    # but build04 has since gained those rows and they are OK.)
    ABSENCE_DISPS = {"truncated_acq", "gdino_fn", "empty_well"}
    PRESENT = {"OK", "EXCLUDED"}
    stale = df["status"].isin(PRESENT) & df["disposition"].isin(ABSENCE_DISPS)
    df.loc[stale, "disposition_conflict"] = (
        "stale:" + df.loc[stale, "disposition"] + " (well now " + df.loc[stale, "status"] + ")"
    )
    df.loc[stale, ["disposition", "note"]] = ""

    # Auto-default dispositions for rows the sidecar didn't cover (or were just cleared).
    auto_disp = {
        "OK": "ok",
        "EXCLUDED": "needs_review",
        "ABSENT_IMAGED": "needs_review",
        "ABSENT_NO_IMAGE": "needs_review",
        "QC_NOT_RUN": "qc_not_run",
    }
    blank = df["disposition"] == ""
    df.loc[blank, "disposition"] = df.loc[blank, "status"].map(auto_disp)

    # For QC_NOT_RUN wells, note whether the stitched image already exists (recoverable)
    # so the label isn't misread as "never imaged".
    qnr = (df["status"] == "QC_NOT_RUN") & (df["note"] == "")
    df.loc[qnr, "note"] = df[qnr].apply(
        lambda r: ("image stitched — rerun build04/06 to recover"
                   if stitched_image_exists(r["exp"], r["well"])
                   else "no stitched image — QC pipeline not run"), axis=1)

    # Attach gene from the manifest (for the gene-split loss-reason bar plot).
    man = pd.read_csv(TABLE_DIR / "experiment_manifest.csv")[["experiment", "gene"]]
    df = df.merge(man.rename(columns={"experiment": "exp"}), on="exp", how="left")
    df["gene"] = df["gene"].fillna("unknown")
    return df


def coverage_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby(["exp", "status"]).size().unstack(fill_value=0)
    for col in STATUS_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[STATUS_ORDER]
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_index()


# ---------------------------------------------------------------------------- writers

def write_markdown(df: pd.DataFrame, pivot: pd.DataFrame, path: Path) -> None:
    counts = df["status"].value_counts()
    lines = [
        "# Sequenced-vs-pipeline coverage audit", "",
        "Excel `sequenced` sheet vs build04 QC, for non-sci_ b9d2/cep290 plates. "
        f"Generated by `3a_audit_sequenced_coverage.py`. Human dispositions sourced from "
        f"[`{CURATED_MD}`](./{CURATED_MD}) via `tables/well_dispositions.csv`.", "",
        "Status vocabulary: `OK` (passed QC) · `EXCLUDED` (in build04, use_embryo_flag=0) · "
        "`ABSENT_IMAGED` (stitched image exists, no build04 row) · `ABSENT_NO_IMAGE` (no stitched "
        "image) · `QC_NOT_RUN` (no build04 CSV — QC pipeline never run; images may be stitched).", "",
        "## Totals", "",
    ]
    lines += [f"- **{s}**: {int(counts.get(s, 0))}" for s in STATUS_ORDER]
    lines += [f"- **TOTAL sequenced wells**: {len(df)}", "", "## By experiment", "",
              "| experiment | " + " | ".join(STATUS_ORDER) + " | total |",
              "|" + "---|" * (len(STATUS_ORDER) + 2)]
    for exp, r in pivot.iterrows():
        lines.append("| " + exp + " | "
                     + " | ".join(str(int(r[s])) for s in STATUS_ORDER)
                     + f" | {int(r['total'])} |")

    detail_specs = [
        ("EXCLUDED", "EXCLUDED", ["well", "seq_code", "exclusion_flags", "disposition", "note"]),
        ("ABSENT (imaged + no-image)", ("ABSENT_IMAGED", "ABSENT_NO_IMAGE"),
         ["well", "seq_code", "status", "disposition", "note"]),
        ("QC_NOT_RUN", "QC_NOT_RUN", ["well", "seq_code", "disposition", "note"]),
    ]
    for label, status, cols in detail_specs:
        sel = df["status"].isin(status if isinstance(status, tuple) else (status,))
        sub = df[sel].sort_values(["exp", "well"])
        lines += ["", f"## {label} detail", ""]
        if sub.empty:
            lines.append("_(none)_")
            continue
        head = ["exp"] + cols
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for _, rr in sub.iterrows():
            lines.append("| " + " | ".join(str(rr[c]) for c in head) + " |")

    conflicts = df[df["disposition_conflict"] != ""].sort_values(["exp", "well"])
    lines += ["", "## Stale dispositions (auto-status overrides curated note)", "",
              "Wells where the curated sidecar marked an absence reason but build04 has since "
              "been reprocessed and the well is now present. Data wins; prune these rows from "
              "`tables/well_dispositions.csv`.", ""]
    if conflicts.empty:
        lines.append("_(none)_")
    else:
        lines.append("| exp | well | status (now) | stale note |")
        lines.append("|---|---|---|---|")
        for _, rr in conflicts.iterrows():
            lines.append(f"| {rr['exp']} | {rr['well']} | {rr['status']} | "
                         f"{rr['disposition_conflict']} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path.relative_to(RUN_DIR)}")


# ---------------------------------------------------------------------------- plots

def plot_heatmap(pivot: pd.DataFrame, path: Path) -> None:
    exps = list(pivot.index)
    M = pivot[STATUS_ORDER].to_numpy()
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * len(STATUS_ORDER), 1.2 + 0.34 * len(exps)))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(STATUS_ORDER)))
    ax.set_xticklabels(STATUS_ORDER, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=9)
    vmax = M.max() if M.max() > 0 else 1
    for i in range(len(exps)):
        for j in range(len(STATUS_ORDER)):
            v = int(M[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if v > vmax * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="sequenced wells")
    ax.set_title("Sequenced-well coverage by plate x status", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_stacked_bar(pivot: pd.DataFrame, path: Path) -> None:
    exps = list(pivot.index)
    y = np.arange(len(exps))
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.36 * len(exps)))
    left = np.zeros(len(exps))
    for s in STATUS_ORDER:
        vals = pivot[s].to_numpy()
        ax.barh(y, vals, left=left, color=STATUS_COLORS[s], label=s,
                edgecolor="white", linewidth=1.4, height=0.7)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(exps, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("sequenced wells", fontsize=11)
    ax.set_title("Per-plate sequenced-well status", fontsize=12)
    ax.legend(fontsize=9, ncol=len(STATUS_ORDER), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_loss_reasons_by_gene(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars: count of lost (non-OK) sequenced wells per loss-reason, split by gene."""
    lost = df[df["status"] != "OK"]
    if lost.empty:
        print("  (no lost wells — skipping loss-reasons-by-gene plot)")
        return
    ct = lost.groupby(["disposition", "gene"]).size().unstack(fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]  # busiest reason first
    reasons = list(ct.index)
    genes = list(ct.columns)
    x = np.arange(len(reasons))
    w = 0.8 / max(len(genes), 1)
    gene_colors = {"b9d2": "#1f77b4", "cep290": "#ff7f0e", "crispant": "#2ca02c",
                   "unknown": "#888888"}
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(reasons), 4.2))
    for gi, g in enumerate(genes):
        vals = ct[g].to_numpy()
        bars = ax.bar(x + gi * w - 0.4 + w / 2, vals, w, label=g,
                      color=gene_colors.get(g, "#888888"),
                      edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v, str(int(v)),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("lost sequenced wells", fontsize=11)
    ax.set_title("Loss reasons by gene", fontsize=12)
    ax.legend(title="gene", fontsize=10, title_fontsize=10, frameon=False)
    ax.margins(y=0.12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/audit/{path.name}")


def plot_wellgrids(df: pd.DataFrame, out_dir: Path) -> None:
    """One 8x12 plate map per experiment; sequenced wells colored by status, others gray."""
    for exp, sub in df.groupby("exp"):
        status_by_well = dict(zip(sub["well"], sub["status"]))
        seq_by_well = dict(zip(sub["well"], sub["seq_code"]))
        fig, (ax, ax_side) = plt.subplots(
            1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [2.0, 1.0]})
        for ri, r in enumerate(ROWS):
            for ci, c in enumerate(COLS):
                w = f"{r}{c:02}"
                st = status_by_well.get(w, "NOT_SEQUENCED")
                ax.add_patch(mpatches.Rectangle(
                    (ci, ri), 0.92, 0.92, facecolor=STATUS_COLORS[st],
                    edgecolor="white", linewidth=0.8))
                if w in status_by_well:
                    ax.text(ci + 0.46, ri + 0.46, str(int(seq_by_well[w])),
                            ha="center", va="center", fontsize=8, fontweight="bold",
                            color="white" if st != "NOT_SEQUENCED" else "black")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([c - 0.54 for c in range(1, 13)])
        ax.set_xticklabels(COLS, fontsize=10)
        ax.set_yticks([r + 0.46 for r in range(8)])
        ax.set_yticklabels(list(ROWS), fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(exp, fontsize=11)
        handles = [mpatches.Patch(color=STATUS_COLORS[s], label=s) for s in STATUS_ORDER]
        ax.legend(handles=handles, fontsize=8, ncol=len(STATUS_ORDER), loc="upper center",
                  bbox_to_anchor=(0.5, -0.06), frameon=False)

        # Side panel: "Well: lost — reason" for every non-OK sequenced well.
        lost = sub[sub["status"] != "OK"].sort_values("well")
        ax_side.axis("off")
        ax_side.set_title("Lost wells", fontsize=10, loc="left")
        if lost.empty:
            ax_side.text(0.0, 0.97, "(all sequenced wells OK)", fontsize=9,
                         va="top", ha="left", color="gray")
        else:
            y = 0.97
            for _, rr in lost.iterrows():
                reason = rr["disposition"] if rr["disposition"] not in ("", "needs_review") \
                    else (rr["exclusion_flags"] or rr["status"].lower())
                ax_side.text(0.0, y, rr["well"], fontsize=9, va="top", ha="left",
                             color=STATUS_COLORS[rr["status"]], fontweight="bold")
                ax_side.text(0.16, y, f"{rr['status']} — {reason}", fontsize=8,
                             va="top", ha="left", color="black")
                y -= 0.05
        fig.savefig(out_dir / f"wellgrid_{exp}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(df['exp'].unique())} well-grids to plots/audit/wellgrids/")


# ---------------------------------------------------------------------------- main

print("3a - sequenced-vs-pipeline coverage audit (failure-mode taxonomy)")
exps = cohort_experiments()
print(f"Auditing {len(exps)} non-sci b9d2/cep290/crispant plates")
df = audit(exps)
if df.empty:
    print("No sequenced wells found across the cohort — nothing to audit.")
    sys.exit(0)

print("\n=== SEQUENCED WELL COVERAGE ===")
print(df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0).to_string())
print(f"Total sequenced wells in Excel: {len(df)}")

pivot = coverage_pivot(df)
print("\n=== BY EXPERIMENT ===")
print(pivot.to_string())

conflicts = df[df["disposition_conflict"] != ""]
if not conflicts.empty:
    print(f"\n=== STALE DISPOSITIONS ({len(conflicts)}) — prune these from well_dispositions.csv ===")
    print(conflicts[["exp", "well", "status", "disposition_conflict"]].to_string(index=False))

# tables
audit_csv = TABLE_DIR / "sequenced_coverage_audit.csv"
df.to_csv(audit_csv, index=False)
print(f"\n  wrote {audit_csv.relative_to(RUN_DIR)}")

loss_map = df[["embryo_id", "exp", "well", "seq_code", "status",
               "exclusion_flags", "disposition", "note", "disposition_conflict"]]
loss_csv = TABLE_DIR / "embryo_loss_map.csv"
loss_map.to_csv(loss_csv, index=False)
print(f"  wrote {loss_csv.relative_to(RUN_DIR)}")

write_markdown(df, pivot, RUN_DIR / "MISSING_SEQUENCED_AUDIT.md")

# plots
plot_heatmap(pivot, AUDIT_PLOT_DIR / "sequenced_coverage_heatmap.png")
plot_stacked_bar(pivot, AUDIT_PLOT_DIR / "status_stacked_bar.png")
plot_loss_reasons_by_gene(df, AUDIT_PLOT_DIR / "loss_reasons_by_gene.png")
plot_wellgrids(df, WELLGRID_DIR)
print("\nDone.")
