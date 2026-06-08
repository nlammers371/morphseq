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


def _short_plate(plate: str) -> str:
    return str(plate).replace("_cep290", "").replace("cep290_", "")


def _ordered_plates(df: pd.DataFrame) -> list[str]:
    return sorted(
        df["query_experiment"].dropna().astype(str).unique(),
        key=lambda p: (df.loc[df["query_experiment"].astype(str) == p, "stage"].dropna().min(), str(p)),
    )


def _plot_probability_spectrum(pred: pd.DataFrame) -> None:
    """Sequenced embryos on the two-class homo-only phenotype probability axis."""
    p = pred.dropna(subset=["stage"]).copy()
    if p.empty:
        return
    left, right = PHENO_ORDER
    prob_right = f"prob_{right}"
    if prob_right not in p.columns:
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
        1,
        len(stages),
        figsize=(3.0 * len(stages) + 1.5, 1.25 + 0.62 * len(groups)),
        sharey=True,
        squeeze=False,
    )
    for ax, stage in zip(axes[0], stages):
        sub_stage = p[p["stage"] == stage].copy()
        ax.axvspan(0.45, 0.55, color="#EEEEEE", zorder=0)
        if sub_stage.empty:
            ax.text(0.5, 0.5, "no sequenced", ha="center", va="center",
                    fontsize=8, color="#777777", transform=ax.transAxes)
        ax.axvline(0.5, color="#777777", lw=0.8, ls=":", zorder=1)
        for group in groups:
            cell = sub_stage[sub_stage["truth_group"] == group]
            if cell.empty:
                continue
            y0 = group_to_y[group]
            y = y0 + rng.uniform(-0.18, 0.18, size=len(cell))
            x = cell[prob_right].astype(float).to_numpy()
            ax.scatter(
                x,
                y,
                c=x,
                cmap=cmap,
                vmin=0,
                vmax=1,
                s=36,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.9,
            )
            ax.text(1.02, y0, f"n={len(cell)}", ha="left", va="center", fontsize=7,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_title(f"{stage} hpf", fontsize=9)
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
    fig.suptitle(f"{DATASET_LABEL} homo-only phenotype probability spectrum (sequenced)", fontsize=11)
    _save(fig, SEQ_OUT / f"{FILE_PREFIX}_homo_only_probability_spectrum_sequenced.png")


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

    model = prepare_reference(
        ref,
        feat,
        label_col=T.PHENO_COL,
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col="experiment_id",
    )
    result = transfer_labels(model, qry, skip_flagged=False)
    emb = result["embryo_predictions"].copy()
    meta = qry.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)
    emb["dataset"] = "cep290"
    emb["query_experiment"] = emb["query_embryo_id"].map(meta["query_experiment"])
    emb["true_genotype"] = emb["query_embryo_id"].map(meta[T.GENO_COL])
    emb["true_zygosity"] = emb["query_embryo_id"].map(meta[T.ZYG_COL])
    emb["stage"] = emb["query_embryo_id"].map(start_age)
    emb = S._tag(emb, seqlk)
    emb["truth_group"] = emb.apply(_truth_group, axis=1)
    emb.to_csv(TR / "cep290_homo_low_to_high_predictions.csv", index=False)

    seq = emb[emb["sequenced"] > 0].copy()
    seq.to_csv(TR / "cep290_homo_low_to_high_sequenced_predictions.csv", index=False)
    _plot_minibars(seq, homo_only=True)
    _plot_minibars(seq, homo_only=False)
    _plot_probability_spectrum(seq)

    cv = _target_cv_predictions(ref, feat)
    cv.to_csv(TR / "cep290_homo_low_to_high_reference_cv_predictions_target_hpf_pm2.csv", index=False)
    _plot_target_confusion(cv)

    homo = seq[seq["truth_group"] == "cep290_homozygous -> homozygous"]
    if not homo.empty:
        print("\nSequenced true-homozygous split:")
        print(homo["predicted_label"].value_counts().reindex(PHENO_ORDER).fillna(0).astype(int).to_string())
    print(f"\nWrote CSVs under: {TR.relative_to(RUN_DIR)}/")
    print(f"Wrote reference plots under: {OUT.relative_to(RUN_DIR)}/")
    print(f"Wrote sequenced plots under: {SEQ_OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
