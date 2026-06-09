"""
cep290 homozygous-conditional phenotype transfer.

Train only on cep290 homozygous-mutant reference embryos with directional phenotypes:
High_to_Low vs Low_to_High. Intermediate is merged into Low_to_High; Not Penetrant is excluded.

This answers the conditional question:
    given a cep290 homozygous mutant, is the phenotype High_to_Low or Low_to_High?

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/cep290_homo_low_to_high.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import prepare_reference, transfer_labels  # noqa: E402
from src.analyze.classification.label_transfer.core import MIN_BIN_EMBRYOS_PERBIN  # noqa: E402
from src.analyze.classification import run_classification  # noqa: E402
# run_classification is used solely for per-bin p-values (significance marking on plots)

import build_reference_and_transfer as T  # noqa: E402
import label_transfer_snapshots as S  # noqa: E402
from sequenced_focus_config import (  # noqa: E402
    HOMO_PHENOTYPE_ORDER,
    PHENOTYPE_ALIASES,
    PHENOTYPE_COLORS,
    STAGE_GRIDS,
    TARGET_HPF_WINDOWS,
)

OUT = RUN_DIR / "plots" / "cep290" / "cep290_homo_only_and_its_phenotypes"
SEQ_OUT = RUN_DIR / "plots" / "sequenced_focus" / "cep290" / "homozygous_focus"
TR = RUN_DIR / "transfer_results"

PHENO_ORDER = HOMO_PHENOTYPE_ORDER["cep290"]
DATASET_LABEL = "cep290"
STAGE_GRID = STAGE_GRIDS["cep290"]
FILE_PREFIX = "cep290_homo_low_to_high"

PHENO_COLORS = PHENOTYPE_COLORS["cep290"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RUN_DIR)}")


def _cep290_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, float]]:
    cfg = T.DATASETS["cep290"]
    queries = [e for e in cfg["queries"] if (T.B6 / f"df03_final_output_with_latents_{e}.csv").exists()]
    qpaths = [T.B6 / f"df03_final_output_with_latents_{e}.csv" for e in queries]
    feat = T.resolve_feature_cols([cfg["ref"], *qpaths])
    ref = T._load(cfg["ref"], feat, gene_hint="cep290")

    qparts = []
    start_age = {}
    for exp, path in zip(queries, qpaths):
        q = T._load(path, feat, gene_hint="cep290")
        q["query_experiment"] = exp
        qparts.append(q)
        sa = pd.read_csv(path, usecols=[T.GROUP_COL, "start_age_hpf"], low_memory=False)
        start_age.update(
            sa.dropna(subset=["start_age_hpf"])
            .groupby(T.GROUP_COL)["start_age_hpf"]
            .median()
            .to_dict()
        )
    qry = pd.concat(qparts, ignore_index=True)
    return ref, qry, feat, queries, start_age


def _homo_directional_reference(ref: pd.DataFrame) -> pd.DataFrame:
    r = ref.dropna(subset=[T.PHENO_COL, T.TIME_COL]).copy()
    r.loc[r[T.PHENO_COL] == "Intermediate", T.PHENO_COL] = "Low_to_High"
    r = r[(r[T.ZYG_COL] == "homozygous") & r[T.PHENO_COL].isin(PHENO_ORDER)].copy()
    return r


def _embryo_table(df: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    emb = df.groupby(T.GROUP_COL)[feat].mean().reset_index()
    label = df.groupby(T.GROUP_COL)[T.PHENO_COL].agg(lambda s: s.mode().iloc[0]).reset_index()
    hpf = df.groupby(T.GROUP_COL)[T.TIME_COL].median().reset_index()
    exp = df.groupby(T.GROUP_COL)["experiment_id"].first().reset_index()
    geno = df.groupby(T.GROUP_COL)[T.GENO_COL].first().reset_index()
    return emb.merge(label, on=T.GROUP_COL).merge(hpf, on=T.GROUP_COL).merge(exp, on=T.GROUP_COL).merge(geno, on=T.GROUP_COL)


def _target_cv_predictions(
    ref: pd.DataFrame,
    feat: list[str],
    targets: tuple[int, ...] | None = None,
    window: float = 2.0,
) -> pd.DataFrame:
    targets = tuple(TARGET_HPF_WINDOWS["cep290"] if targets is None else targets)
    base = prepare_reference(
        ref,
        feat,
        label_col=T.PHENO_COL,
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col="experiment_id",
    )
    pipe_template = base["final_model"]
    rows = []
    for target in targets:
        win = ref[(ref[T.TIME_COL] >= target - window) & (ref[T.TIME_COL] <= target + window)].copy()
        if win.empty:
            continue
        emb = _embryo_table(win, feat)
        if emb[T.PHENO_COL].nunique() < 2 or emb["experiment_id"].nunique() < 2:
            continue
        for exp in sorted(emb["experiment_id"].dropna().unique()):
            train = emb[emb["experiment_id"] != exp]
            test = emb[emb["experiment_id"] == exp]
            if train.empty or test.empty or train[T.PHENO_COL].nunique() < 2:
                continue
            pipe = clone(pipe_template)
            pipe.fit(train[feat].to_numpy(dtype=float), train[T.PHENO_COL].to_numpy())
            pred = pipe.predict(test[feat].to_numpy(dtype=float))
            for (_, row), pred_label in zip(test.iterrows(), pred):
                rows.append(
                    {
                        T.GROUP_COL: row[T.GROUP_COL],
                        "target_hpf": target,
                        "window_hpf": window,
                        "hpf": row[T.TIME_COL],
                        "experiment_id": row["experiment_id"],
                        "true_phenotype": row[T.PHENO_COL],
                        "predicted_phenotype": pred_label,
                    }
                )
    return pd.DataFrame(rows)


def _plot_target_confusion(cv: pd.DataFrame) -> None:
    if cv.empty:
        return
    targets = TARGET_HPF_WINDOWS["cep290"]
    fig, axes = plt.subplots(1, len(targets), figsize=(2.8 * len(targets), 3.4), squeeze=False)
    for ax, target in zip(axes[0], targets):
        sub = cv[cv["target_hpf"] == target]
        cm = confusion_matrix(sub["true_phenotype"], sub["predicted_phenotype"], labels=PHENO_ORDER)
        denom = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(PHENO_ORDER)))
        ax.set_xticklabels(PHENO_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(PHENO_ORDER)))
        ax.set_yticklabels(PHENO_ORDER if ax is axes[0][0] else [""] * len(PHENO_ORDER), fontsize=7)
        ax.set_title(f"{int(target)} hpf +/-2\nn={len(sub)}", fontsize=9)
        for i in range(len(PHENO_ORDER)):
            for j in range(len(PHENO_ORDER)):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=7)
    axes[0][0].set_ylabel("reference phenotype")
    fig.supxlabel("CV predicted phenotype")
    fig.suptitle("cep290 homo-only directional phenotype CV at target hpf windows", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, OUT / "cep290_homo_low_to_high_reference_confusion_target_hpf_pm2.png")


def _truth_group(row: pd.Series) -> str:
    genotype = str(row.get("true_genotype", ""))
    stratum = str(row.get("stratum", ""))
    if stratum == "AB" or genotype == "ab_wildtype":
        return "AB -> wildtype"
    if stratum == "wildtype_sibling" or genotype.endswith("_wildtype"):
        return "cep290_wildtype -> wildtype"
    if stratum == "heterozygous" or genotype.endswith("_heterozygous"):
        return "cep290_heterozygous -> heterozygous"
    if stratum == "homozygous" or genotype.endswith("_homozygous"):
        return "cep290_homozygous -> homozygous"
    return "unknown"


def _truth_group_order() -> list[str]:
    return [
        "AB -> wildtype",
        "cep290_wildtype -> wildtype",
        "cep290_heterozygous -> heterozygous",
        "cep290_homozygous -> homozygous",
    ]


def _run_pval_by_bin(ref: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Run run_classification on the full reference to get per-bin p-values.

    Binary comparison: Low_to_High (positive) vs High_to_Low (negative).
    Returns result.scores with columns time_bin_center, auroc_obs, pval.
    Used only for significance marking on the plot (dark border when p <= 0.05).
    """
    emb_ref = _embryo_table(ref, feat)
    if emb_ref[T.PHENO_COL].nunique() < 2 or len(emb_ref) < 6:
        return pd.DataFrame(columns=["time_bin_center", "auroc_obs", "pval"])
    result = run_classification(
        emb_ref,
        class_col=T.PHENO_COL,
        id_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        positive="Low_to_High",
        negative="High_to_Low",
        features={"emb": feat},
        bin_width=4.0,
        n_permutations=200,
        n_splits=5,
        verbose=False,
    )
    return result.scores[["time_bin_center", "auroc_obs", "pval"]].copy()


def _pval_for_bin(pval_scores: pd.DataFrame, bin_center: float) -> float | None:
    """Look up the permutation p-value nearest to a given bin center. Returns None if unavailable."""
    if pval_scores.empty:
        return None
    bin_centers = pval_scores["time_bin_center"].to_numpy()
    idx = int(np.argmin(np.abs(bin_centers - bin_center)))
    return float(pval_scores["pval"].iloc[idx])


def _short_plate(plate: str) -> str:
    return str(plate).replace("_cep290", "").replace("cep290_", "")


def _ordered_plates(df: pd.DataFrame) -> list[str]:
    return sorted(
        df["query_experiment"].dropna().astype(str).unique(),
        key=lambda p: (df.loc[df["query_experiment"].astype(str) == p, "stage"].dropna().min(), str(p)),
    )


def _plot_probability_spectrum(pred: pd.DataFrame, pval_scores: pd.DataFrame | None = None) -> None:
    """Sequenced embryos on the two-class homo-only phenotype probability axis.

    pval_scores: output of _run_pval_by_bin(). When provided, embryos whose
    time-bin model is statistically significant (p <= 0.05) get a darker border,
    making it easy to distinguish bins where the classifier actually has power.
    """
    PVAL_THRESHOLD = 0.05
    BORDER_SIGNIFICANT = ("black", 1.2)    # edgecolor, linewidth when p <= 0.05
    BORDER_DEFAULT     = ("black", 0.25)   # edgecolor, linewidth otherwise

    p = pred.dropna(subset=["stage"]).copy()
    if p.empty:
        return
    left, right = PHENO_ORDER
    prob_col = f"prob_{right}"
    if prob_col not in p.columns:
        return

    p["stage"] = p["stage"].astype(float).round().astype(int)
    stages = STAGE_GRID
    groups = [g for g in _truth_group_order() if (p["truth_group"] == g).any()]
    if not groups:
        return
    group_to_y = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    cmap = LinearSegmentedColormap.from_list(
        "phenotype_spectrum",
        [PHENO_COLORS[left], "#E6E6E6", PHENO_COLORS[right]],
        N=256,
    )

    fig, axes = plt.subplots(
        1, len(stages),
        figsize=(3.0 * len(stages) + 1.5, 1.25 + 0.62 * len(groups)),
        sharey=True, squeeze=False,
    )
    for ax, stage in zip(axes[0], stages):
        sub_stage = p[p["stage"] == stage].copy()

        # p-value for this stage's bin
        pval = _pval_for_bin(pval_scores, float(stage)) if pval_scores is not None else None
        significant = pval is not None and pval <= PVAL_THRESHOLD
        ec, lw = BORDER_SIGNIFICANT if significant else BORDER_DEFAULT
        pval_label = f"p={pval:.3f}" if pval is not None else ""
        sig_marker = "*" if significant else ""

        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        if sub_stage.empty:
            ax.text(0.5, 0.5, "no sequenced", ha="center", va="center",
                    fontsize=8, color="#777777", transform=ax.transAxes)
        ax.set_title(f"{stage} hpf{sig_marker}\n{pval_label}", fontsize=8)

        for group in groups:
            cell = sub_stage[sub_stage["truth_group"] == group]
            if cell.empty:
                continue
            y0 = group_to_y[group]
            y = y0 + rng.uniform(-0.18, 0.18, size=len(cell))
            x = cell[prob_col].astype(float).to_numpy()
            ax.scatter(
                x, y, c=x, cmap=cmap, vmin=0, vmax=1,
                s=36, edgecolors=ec, linewidths=lw, alpha=0.9,
            )
            ax.text(1.02, y0, f"n={len(cell)}", ha="left", va="center", fontsize=7,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_xlabel(f"{left}  <-  P({right})  ->  {right}", fontsize=8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=7)
    axes[0][0].set_yticks(range(len(groups)))
    axes[0][0].set_yticklabels(groups, fontsize=8)
    axes[0][0].set_ylabel("sequenced true genotype group", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(left=0.19, right=0.88, top=0.86, bottom=0.18, wspace=0.32)
    cax = fig.add_axes([0.91, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"P({right})", fontsize=8)
    note = "  (* = p≤0.05 per-bin permutation test)" if pval_scores is not None else ""
    fig.suptitle(
        f"{DATASET_LABEL} homo-only phenotype probability spectrum (sequenced){note}",
        fontsize=9,
    )
    _save(fig, SEQ_OUT / f"{FILE_PREFIX}_homo_only_probability_spectrum_sequenced.png")


def _plot_spectrum_with_accuracy(
    pred: pd.DataFrame,
    ref_cv: pd.DataFrame,
    pval_scores: pd.DataFrame | None = None,
) -> None:
    """4-column × 4-row comparison figure for choosing the best bottom-row design.

    Row 0  — query strip plot (same as existing spectrum, homo only)
    Row 1  — Option A: reference strip split by true class, colored correct/incorrect
    Row 2  — Option B: stacked accuracy bars per true class
    Row 3  — Option C: calibration scatter — mean predicted P per true class ± std

    All columns = 18 / 24 / 30 / 48 hpf.
    """
    left, right = PHENO_ORDER          # "High_to_Low", "Low_to_High"
    prob_col = f"prob_{right}"
    cmap = LinearSegmentedColormap.from_list(
        "phenotype_spectrum",
        [PHENO_COLORS[left], "#E6E6E6", PHENO_COLORS[right]],
        N=256,
    )
    stages = STAGE_GRID
    rng = np.random.default_rng(0)

    # ── query: homozygous embryos only ───────────────────────────────────────
    homo_group = "cep290_homozygous -> homozygous"
    qry = pred[pred["truth_group"] == homo_group].dropna(subset=["stage"]).copy()
    qry["stage"] = qry["stage"].astype(float).round().astype(int)

    # ref_cv: embryo_id, true_label, time_bin_center, p_global, p_perbin
    # p_perbin = P(High_to_Low); flip to get P(Low_to_High) for x-axis
    ref = ref_cv.copy()
    ref["p_Low_to_High"] = 1.0 - ref["p_perbin"]

    n_rows, n_cols = 4, len(stages)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    row_labels = [
        "Query\n(sequenced homo)",
        "Option A\nRef strip by true class\n(green=correct, red=wrong)",
        "Option B\nRef accuracy bars\nper true class",
        "Option C\nRef calibration\nmean P ± std per class",
    ]

    for col, stage in enumerate(stages):
        pval = _pval_for_bin(pval_scores, float(stage)) if pval_scores is not None else None
        sig = pval is not None and pval <= 0.05
        pval_str = f"p={pval:.3f}{'*' if sig else ''}" if pval is not None else ""
        axes[0][col].set_title(f"{stage} hpf\n{pval_str}", fontsize=9)

        # ── Row 0: query strip ────────────────────────────────────────────────
        ax = axes[0][col]
        sub = qry[qry["stage"] == stage]
        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        if not sub.empty:
            y = rng.uniform(-0.18, 0.18, size=len(sub))
            x = sub[prob_col].astype(float).to_numpy()
            ax.scatter(x, y, c=x, cmap=cmap, vmin=0, vmax=1,
                       s=42, edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3)
            ax.text(0.5, 0.42, f"n={len(sub)}", ha="center", va="bottom",
                    fontsize=8, transform=ax.transAxes)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.45, 0.45)
        ax.set_yticks([])
        ax.set_xlabel(f"P({right})", fontsize=8)

        # filter ref CV to this time bin (exact bin center match)
        sub_ref = ref[ref["time_bin_center"] == float(stage)]
        # fall back to ±2 hpf window if no exact match
        if sub_ref.empty:
            sub_ref = ref[np.abs(ref["time_bin_center"] - stage) <= 2.0]

        # ── Row 1: Option A — ref strip split by true class, colored by P ────
        # Low alpha so overlapping dots accumulate visually, revealing density
        ax = axes[1][col]
        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        class_y = {left: 0, right: 1}
        for ci, cls in enumerate([left, right]):
            sub_cls = sub_ref[sub_ref["true_label"] == cls]
            if sub_cls.empty:
                continue
            y0 = class_y[cls]
            yj = y0 + rng.uniform(-0.18, 0.18, size=len(sub_cls))
            xv = sub_cls["p_Low_to_High"].astype(float).to_numpy()
            ax.scatter(xv, yj, c=xv, cmap=cmap, vmin=0, vmax=1,
                       s=42, edgecolors="black", linewidths=0.3, alpha=0.35, zorder=3)
            ax.text(1.02, y0, f"n={len(sub_cls)}", ha="left", va="center",
                    fontsize=7, transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([left, right], fontsize=7)
        ax.set_xlabel(f"P({right})", fontsize=8)

        # ── Row 2: Option B — stacked accuracy bars ───────────────────────────
        ax = axes[2][col]
        correct_color = "#2ca02c"
        wrong_color   = "#d62728"
        for ci, cls in enumerate([left, right]):
            sub_cls = sub_ref[sub_ref["true_label"] == cls]
            if sub_cls.empty:
                ax.bar(ci, 0, color="#CCCCCC", width=0.6)
                continue
            # correct = prediction agrees with true label at 0.5 threshold
            correct = (
                ((cls == right) & (sub_cls["p_Low_to_High"] >= 0.5)) |
                ((cls == left)  & (sub_cls["p_Low_to_High"] < 0.5))
            )
            frac_corr = correct.mean()
            ax.bar(ci, frac_corr, color=correct_color, width=0.6)
            ax.bar(ci, 1 - frac_corr, bottom=frac_corr, color=wrong_color, width=0.6)
            ax.text(ci, 1.03, f"{frac_corr:.0%}\n(n={len(sub_cls)})",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.35)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([left, right], fontsize=7, rotation=20, ha="right")
        ax.set_ylabel("fraction correct" if col == 0 else "", fontsize=8)
        ax.axhline(0.5, color="#AAAAAA", lw=0.8, ls="--")

        # ── Row 3: Option C — violin plot per true class, clipped [0, 1] ───────
        ax = axes[3][col]
        ax.axhline(0.5, color="#AAAAAA", lw=0.8, ls="--", zorder=0)
        violin_data = []
        violin_pos  = []
        violin_colors = []
        for ci, cls in enumerate([left, right]):
            sub_cls = sub_ref[sub_ref["true_label"] == cls]
            if sub_cls.empty:
                continue
            p_vals = np.clip(sub_cls["p_Low_to_High"].astype(float).to_numpy(), 0.0, 1.0)
            if len(p_vals) < 2:
                ax.scatter([ci], p_vals, color=PHENO_COLORS[cls], s=50,
                           edgecolors="black", linewidths=0.6, zorder=3)
                ax.text(ci, p_vals[0] + 0.05, f"n={len(p_vals)}",
                        ha="center", va="bottom", fontsize=7)
                continue
            violin_data.append(p_vals)
            violin_pos.append(ci)
            violin_colors.append(PHENO_COLORS[cls])
        if violin_data:
            parts = ax.violinplot(violin_data, positions=violin_pos,
                                  showmedians=True, showextrema=False)
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_alpha(0.7)
                body.set_edgecolor("black")
                body.set_linewidth(0.6)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)
            for ci, cls, p_vals in zip(violin_pos, [left, right], violin_data):
                ax.text(ci, 1.04, f"n={len(p_vals)}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.02, 1.15)
        ax.set_xticks([0, 1] if len(violin_pos) == 2 else violin_pos)
        ax.set_xticklabels([left, right][:len(violin_pos)], fontsize=7,
                           rotation=20, ha="right")
        ax.set_ylabel(f"P({right})  [clipped 0–1]" if col == 0 else "", fontsize=8)

    # row labels on left margin
    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=8, labelpad=8)

    # shared colorbar for rows 0 and 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.72, 0.012, 0.20])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"P({right})", fontsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"], fontsize=7)

    # legend for row 2 (accuracy bars)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ca02c", label="correct"),
                       Patch(facecolor="#d62728", label="wrong")]
    fig.legend(handles=legend_elements, loc="upper right",
               bbox_to_anchor=(0.915, 0.50), fontsize=8, title="ref CV (row 2)")

    fig.suptitle(
        f"{DATASET_LABEL} homo-only spectrum — bottom row design comparison\n"
        f"Top: query sequenced homozygous  |  Rows 1–3: reference LOEO CV options",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.14, right=0.91, top=0.93, bottom=0.06,
                        hspace=0.55, wspace=0.35)
    _save(fig, SEQ_OUT / f"{FILE_PREFIX}_spectrum_bottom_row_options.png")


def _plot_minibars(pred: pd.DataFrame, *, homo_only: bool) -> None:
    p = pred.dropna(subset=["stage"]).copy()
    if homo_only:
        p = p[p["truth_group"] == "cep290_homozygous -> homozygous"].copy()
        groups = ["cep290_homozygous -> homozygous"]
        suffix = "homo_only"
        title = "cep290 homo-only model: sequenced homozygous predicted phenotype counts"
    else:
        groups = _truth_group_order()
        suffix = "all_true_genotypes"
        title = "cep290 homo-only model: sequenced predicted phenotype counts by true genotype"
    if p.empty:
        return
    p["stage"] = p["stage"].astype(float).round().astype(int)
    stages = STAGE_GRID
    plates = _ordered_plates(p)
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(3.4 + 3.7 * len(groups), 1.5 + 0.70 * len(plates)),
        squeeze=False,
    )
    bar_w = 0.20
    offsets = (np.arange(len(PHENO_ORDER)) - (len(PHENO_ORDER) - 1) / 2) * (bar_w * 1.25)
    for j, group in enumerate(groups):
        ax = axes[0][j]
        sub_group = p[p["truth_group"] == group]
        ax.set_xlim(-0.5, len(stages) - 0.5)
        ax.set_ylim(len(plates) - 0.5, -0.5)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels([_short_plate(x) for x in plates] if j == 0 else [""] * len(plates), fontsize=7)
        ax.set_title(group, fontsize=8)
        ax.set_xlabel("design stage", fontsize=8)
        ax.set_facecolor("#FAFAFA")
        ax.set_xticks(np.arange(-0.5, len(stages), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(plates), 1), minor=True)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, plate in enumerate(plates):
            for k, stage in enumerate(stages):
                cell = sub_group[(sub_group["query_experiment"].astype(str) == plate) & (sub_group["stage"] == stage)]
                if cell.empty:
                    continue
                counts = cell["predicted_label"].astype(str).value_counts()
                n = int(counts.sum())
                baseline = i + 0.36
                ax.text(k, i - 0.39, f"n={n}", ha="center", va="top", fontsize=5.8)
                for cls, off in zip(PHENO_ORDER, offsets):
                    count = int(counts.get(cls, 0))
                    if count == 0:
                        continue
                    height = (count / n) * 0.58
                    left = k + off - bar_w / 2
                    top = baseline - height
                    ax.add_patch(
                        plt.Rectangle(
                            (left, top),
                            bar_w,
                            height,
                            facecolor=PHENO_COLORS[cls],
                            edgecolor="black",
                            linewidth=0.25,
                        )
                    )
                    ax.text(k + off, top - 0.025, str(count), ha="center", va="bottom", fontsize=5.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHENO_COLORS[cls], edgecolor="black", linewidth=0.3)
        for cls in PHENO_ORDER
    ]
    fig.legend(handles, [PHENOTYPE_ALIASES[c] for c in PHENO_ORDER], loc="center right", bbox_to_anchor=(0.995, 0.5),
               fontsize=8, title="predicted phenotype")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.94, 0.94])
    _save(fig, SEQ_OUT / f"cep290_homo_low_to_high_predicted_phenotype_minibars_{suffix}.png")


def _plot_reference_quality_perbin(model: dict) -> None:
    """Precision and recall by time bin for a per-bin model's quality report.

    Mirrors the style of the existing reference quality timebin plot but saved
    alongside the sequenced focus output so it's easy to compare.
    """
    qr = model["quality_report"]
    classes = model["classes"]
    cfg = model["config"]
    bin_width = cfg["bin_width"]

    class_colors = {PHENO_ORDER[0]: PHENO_COLORS[PHENO_ORDER[0]],
                    PHENO_ORDER[1]: PHENO_COLORS[PHENO_ORDER[1]]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (metric_key, metric_title) in zip(axes, [("precision", "Precision"), ("recall", "Recall")]):
        for lbl in classes:
            rows = qr["by_timebin"].get(lbl, [])
            if not rows:
                continue
            bins_ = [r["time_bin"] + bin_width / 2 for r in rows]
            vals_ = [r[metric_key] for r in rows]
            n_    = [r["n_embryos"] for r in rows]
            color = class_colors.get(lbl, "#555555")
            ax.plot(bins_, vals_, "o-", color=color, lw=2, ms=5, label=lbl)
            for x, y, n in zip(bins_, vals_, n_):
                ax.text(x, y + 0.02, str(n), ha="center", va="bottom", fontsize=6, color=color)
        ax.axhline(0.5, ls=":", color="gray", lw=1, alpha=0.6, label="0.5 threshold")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(f"{cfg['time_col']} bin center (hpf)", fontsize=9)
        ax.set_ylabel(metric_title, fontsize=10)
        ax.set_title(metric_title, fontsize=11)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Per-bin model reference quality — {cfg['cv_strategy']}\n"
        f"bal_acc={qr['balanced_accuracy']:.3f}  macro_F1={qr['macro_f1']:.3f}  "
        f"n_embryos={qr['n_embryos_total']}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, SEQ_OUT / f"{FILE_PREFIX}_perbin_model_reference_quality_timebin.png")


def _ref_cv_probs(
    ref: pd.DataFrame,
    feat: list[str],
    model_global: dict,
    model_perbin: dict,
) -> pd.DataFrame:
    """LOEO CV probabilities for reference embryos, one row per (embryo, time-bin).

    Uses within-bin embryo aggregation — the same _aggregate_binned logic that
    core.py uses when fitting per-bin models — so each embryo contributes one
    prediction per bin it has images in (not one prediction per lifetime median).
    This means an embryo imaged from 20–80 hpf appears in multiple bins.

    Returns columns: embryo_id, true_label, time_bin_center, p_global, p_perbin.
    """
    from sklearn.base import clone
    from src.analyze.classification.engine.data_prep import _aggregate_binned
    from src.analyze.utils.binning import add_time_bins

    left = PHENO_ORDER[0]   # "High_to_Low"
    bin_width = model_perbin["config"]["bin_width"]
    global_pipe   = model_global["final_model"]
    perbin_global = model_perbin["final_model"]

    # Build within-bin embryo means from raw image rows (matching core.py training)
    ref_clean = ref.dropna(subset=[T.PHENO_COL]).copy()
    ref_binned = add_time_bins(ref_clean, time_col=T.TIME_COL,
                               bin_width=bin_width, bin_col="_time_bin")
    emb_binned = _aggregate_binned(
        ref_binned, id_col=T.GROUP_COL, feature_cols=feat,
        bin_col="_time_bin", bin_width=bin_width,
    )
    # attach modal label and experiment per (embryo, bin)
    modal_label = (
        ref_binned.groupby([T.GROUP_COL, "_time_bin"])[T.PHENO_COL]
        .agg(lambda s: s.mode().iloc[0]).reset_index()
    )
    exp_per_emb = ref_clean.groupby(T.GROUP_COL)["experiment_id"].first().reset_index()
    emb_binned = (
        emb_binned
        .merge(modal_label, on=[T.GROUP_COL, "_time_bin"], how="left")
        .merge(exp_per_emb, on=T.GROUP_COL, how="left")
        .dropna(subset=[T.PHENO_COL])
    )

    experiments = sorted(emb_binned["experiment_id"].dropna().unique())
    rows = []

    for held_out in experiments:
        train_all = emb_binned[emb_binned["experiment_id"] != held_out]
        test_all  = emb_binned[emb_binned["experiment_id"] == held_out]
        if train_all.empty or test_all.empty or train_all[T.PHENO_COL].nunique() < 2:
            continue

        # global CV: train on all-bin embryo means from training experiments
        # use the across-all-bins embryo mean as the global model training input
        train_global = (
            train_all.groupby(T.GROUP_COL)[feat].mean().reset_index()
            .merge(
                train_all.groupby(T.GROUP_COL)[T.PHENO_COL]
                .agg(lambda s: s.mode().iloc[0]).reset_index(),
                on=T.GROUP_COL,
            )
        )
        g_pipe = clone(global_pipe)
        g_pipe.fit(train_global[feat].to_numpy(dtype=float),
                   train_global[T.PHENO_COL].to_numpy())
        g_classes = list(g_pipe.classes_)
        g_col = g_classes.index(left) if left in g_classes else 0

        # score each held-out (embryo, bin) row
        for _, row in test_all.iterrows():
            bin_center = float(row["time_bin_center"])

            # per-bin CV: train only on training embryos in this same bin
            train_bin = train_all[train_all["time_bin_center"] == bin_center]
            if len(train_bin) >= MIN_BIN_EMBRYOS_PERBIN and train_bin[T.PHENO_COL].nunique() >= 2:
                pb_pipe = clone(perbin_global)
                pb_pipe.fit(train_bin[feat].to_numpy(dtype=float),
                            train_bin[T.PHENO_COL].to_numpy())
            else:
                pb_pipe = g_pipe  # fall back to global for sparse bins

            x = row[feat].to_numpy(dtype=float).reshape(1, -1)
            g_proba  = g_pipe.predict_proba(x)
            pb_proba = pb_pipe.predict_proba(x)
            pb_classes = list(pb_pipe.classes_)
            pb_col = pb_classes.index(left) if left in pb_classes else 0

            rows.append({
                "embryo_id":       row[T.GROUP_COL],
                "true_label":      row[T.PHENO_COL],
                "time_bin_center": bin_center,
                "p_global":        float(g_proba[0, g_col]),
                "p_perbin":        float(pb_proba[0, pb_col]),
            })

    return pd.DataFrame(rows)


def _scatter_panel(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    color_map: dict,
    stage: int | None = None,
    time_col: str = T.TIME_COL,
    window: float = 3.0,
    title: str = "",
    show_xy_labels: bool = True,
    show_legend: bool = False,
) -> None:
    """Draw one global-vs-perbin scatter panel, optionally filtered to a time window."""
    left  = PHENO_ORDER[0]
    right = PHENO_ORDER[1]
    cmap_diag = LinearSegmentedColormap.from_list(
        "diag", [PHENO_COLORS[right], "#E6E6E6", PHENO_COLORS[left]], N=256
    )

    sub = df.copy()
    if stage is not None:
        sub = sub[np.abs(sub[time_col].astype(float) - stage) <= window]

    ax.plot([0, 1], [0, 1], "--", color="#CCCCCC", lw=0.8, zorder=0)

    # draw colorbar strip along diagonal as background gradient
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap_diag,
              extent=[0, 1, 0, 1], alpha=0.12, zorder=0, origin="lower")

    for label in list(color_map.keys()):
        s = sub[sub[color_col] == label]
        if s.empty:
            continue
        ax.scatter(s[x_col], s[y_col],
                   c=color_map[label], edgecolors="black", linewidths=0.4,
                   s=38, alpha=0.85, label=label, zorder=3)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    n = len(sub)
    ax.set_title(f"{title}\nn={n}", fontsize=8)
    if show_xy_labels:
        ax.set_xlabel("Global", fontsize=7)
        ax.set_ylabel("Per-bin", fontsize=7)
    if show_legend:
        ax.legend(fontsize=6, markerscale=0.8, loc="upper left",
                  title="prediction / true label", title_fontsize=6)


def _plot_model_comparison(
    emb_global: pd.DataFrame,
    emb_perbin: pd.DataFrame,
    ref_cv: pd.DataFrame,
    seq_ids: pd.Index | None = None,
) -> None:
    """2-row × 4-column scatter grid: global P(High_to_Low) x-axis, per-bin y-axis.

    Top row   — query sequenced homozygous embryos, one panel per target stage.
    Bottom row — reference embryos (LOEO CV), one panel per target stage.
    Color = per-bin predicted phenotype (query) / true label (reference).
    Diagonal = perfect agreement. Off-diagonal = model disagreement.
    A colorbar strip along the diagonal encodes the probability gradient.
    """
    left  = PHENO_ORDER[0]   # "High_to_Low"
    right = PHENO_ORDER[1]   # "Low_to_High"
    homo_group = "cep290_homozygous -> homozygous"

    # ── query: sequenced homozygous embryos ───────────────────────────────────
    homo_ids = emb_perbin.loc[emb_perbin["truth_group"] == homo_group, "query_embryo_id"]
    if seq_ids is not None:
        homo_ids = homo_ids[homo_ids.isin(seq_ids)]

    prob_col = f"prob_{left}"
    qry_merged = (
        emb_global.loc[emb_global["query_embryo_id"].isin(homo_ids),
                       ["query_embryo_id", prob_col, "stage"]]
        .rename(columns={prob_col: "p_global"})
        .merge(
            emb_perbin.loc[emb_perbin["query_embryo_id"].isin(homo_ids),
                           ["query_embryo_id", prob_col, "predicted_label"]]
            .rename(columns={prob_col: "p_perbin"}),
            on="query_embryo_id", how="inner",
        )
    )

    stages = STAGE_GRID
    pheno_colors = {left: PHENO_COLORS[left], right: PHENO_COLORS[right]}

    fig, axes = plt.subplots(
        2, len(stages),
        figsize=(3.2 * len(stages), 6.5),
        sharex=True, sharey=True, squeeze=False,
    )

    for col, stage in enumerate(stages):
        # top row: query
        _scatter_panel(
            axes[0][col], qry_merged,
            x_col="p_global", y_col="p_perbin",
            color_col="predicted_label", color_map=pheno_colors,
            stage=stage, time_col="stage", window=0.5,
            title=f"{stage} hpf",
            show_xy_labels=(col == 0),
            show_legend=(col == 0),
        )
        # bottom row: reference CV
        _scatter_panel(
            axes[1][col], ref_cv,
            x_col="p_global", y_col="p_perbin",
            color_col="true_label", color_map=pheno_colors,
            stage=stage, time_col="time_bin_center", window=2.0,
            title=f"{stage} hpf  (ref CV)",
            show_xy_labels=(col == 0),
            show_legend=False,
        )

    # row labels
    axes[0][0].set_ylabel("Per-bin  P(High_to_Low)", fontsize=8)
    axes[1][0].set_ylabel("Per-bin  P(High_to_Low)", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Global  P(High_to_Low)", fontsize=8)

    # shared colorbar
    cmap_diag = LinearSegmentedColormap.from_list(
        "diag", [PHENO_COLORS[right], "#E6E6E6", PHENO_COLORS[left]], N=256
    )
    sm = plt.cm.ScalarMappable(cmap=cmap_diag, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"← {right}   |   {left} →", fontsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"], fontsize=7)

    fig.text(0.01, 0.73, "Query\n(sequenced\nhomozygous)", va="center",
             ha="left", fontsize=8, style="italic", rotation=90)
    fig.text(0.01, 0.28, "Reference\n(LOEO CV)", va="center",
             ha="left", fontsize=8, style="italic", rotation=90)

    fig.suptitle(
        f"{DATASET_LABEL} homo-only: global vs per-bin model comparison\n"
        f"x = global P(High_to_Low),  y = per-bin P(High_to_Low)   [0=Low_to_High, 1=High_to_Low]",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.91, top=0.88, bottom=0.08, hspace=0.35, wspace=0.25)
    _save(fig, SEQ_OUT / f"{FILE_PREFIX}_global_vs_perbin_model_comparison.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SEQ_OUT.mkdir(parents=True, exist_ok=True)
    TR.mkdir(exist_ok=True)
    ref_raw, qry, feat, queries, start_age = _cep290_inputs()
    ref = _homo_directional_reference(ref_raw)
    seqlk = S.build_sequenced_lookup(queries)

    print("cep290 homo-only directional phenotype transfer")
    print(f"  features={len(feat)} ref_rows={len(ref)} query_rows={len(qry)}")
    print("  embryo reference class counts:")
    print(_embryo_table(ref, feat)[T.PHENO_COL].value_counts().reindex(PHENO_ORDER).to_string())

    # per-bin p-values via run_classification (significance markers on the plot)
    print("  running run_classification on reference for per-bin p-values...")
    pval_scores = _run_pval_by_bin(ref, feat)
    if not pval_scores.empty:
        print("  per-bin AUROC / p-value (reference):")
        for _, row in pval_scores.iterrows():
            sig = "*" if row["pval"] <= 0.05 else ""
            print(f"    {row['time_bin_center']:.1f} hpf: AUROC={row['auroc_obs']:.3f}  p={row['pval']:.3f}{sig}")

    def _enrich(emb_raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        e = emb_raw.copy()
        e["dataset"] = "cep290"
        e["query_experiment"] = e["query_embryo_id"].map(meta["query_experiment"])
        e["true_genotype"] = e["query_embryo_id"].map(meta[T.GENO_COL])
        e["true_zygosity"] = e["query_embryo_id"].map(meta[T.ZYG_COL])
        e["stage"] = e["query_embryo_id"].map(start_age)
        e = S._tag(e, seqlk)
        e["truth_group"] = e.apply(_truth_group, axis=1)
        return e

    meta = qry.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)

    # ── global model ──────────────────────────────────────────────────────────
    model_global = prepare_reference(
        ref, feat,
        label_col=T.PHENO_COL, group_col=T.GROUP_COL, time_col=T.TIME_COL,
        cv_group_col="experiment_id", model_type="global",
    )
    emb_global = _enrich(transfer_labels(model_global, qry, skip_flagged=False)["embryo_predictions"], meta)

    # ── per-bin model ─────────────────────────────────────────────────────────
    model_perbin = prepare_reference(
        ref, feat,
        label_col=T.PHENO_COL, group_col=T.GROUP_COL, time_col=T.TIME_COL,
        cv_group_col="experiment_id", model_type="per_bin",
    )
    print(f"  per-bin models fitted: {len(model_perbin.get('bin_models', {}))}")
    emb_perbin = _enrich(transfer_labels(model_perbin, qry, skip_flagged=False)["embryo_predictions"], meta)

    # save per-bin predictions as the canonical output
    emb_perbin.to_csv(TR / "cep290_homo_low_to_high_predictions.csv", index=False)
    seq = emb_perbin[emb_perbin["sequenced"] > 0].copy()
    seq.to_csv(TR / "cep290_homo_low_to_high_sequenced_predictions.csv", index=False)
    seq_ids = seq["query_embryo_id"]

    # reference LOEO CV predictions for both models (used in comparison plot bottom row)
    print("  computing reference LOEO CV predictions for comparison plot...")
    ref_cv = _ref_cv_probs(ref, feat, model_global, model_perbin)

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_minibars(seq, homo_only=True)
    _plot_minibars(seq, homo_only=False)
    _plot_probability_spectrum(seq, pval_scores=pval_scores)
    _plot_spectrum_with_accuracy(seq, ref_cv, pval_scores=pval_scores)
    _plot_reference_quality_perbin(model_perbin)
    _plot_model_comparison(emb_global, emb_perbin, ref_cv=ref_cv, seq_ids=seq_ids)

    cv = _target_cv_predictions(ref, feat)
    cv.to_csv(TR / "cep290_homo_low_to_high_reference_cv_predictions_target_hpf_pm2.csv", index=False)
    _plot_target_confusion(cv)

    homo = seq[seq["truth_group"] == "cep290_homozygous -> homozygous"]
    if not homo.empty:
        print("\nSequenced true-homozygous split (per-bin model):")
        print(homo["predicted_label"].value_counts().reindex(PHENO_ORDER).fillna(0).astype(int).to_string())
    print(f"\nWrote CSVs under: {TR.relative_to(RUN_DIR)}/")
    print(f"Wrote reference plots under: {OUT.relative_to(RUN_DIR)}/")
    print(f"Wrote sequenced plots under: {SEQ_OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
