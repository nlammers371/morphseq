"""
cep290 Phase A phenotype transfer audit: keep Not Penetrant.

This script is deliberately separate from make_plots.py. The general plot flow still makes the
older two-directional-class cep290 phenotype figures; this audit keeps the honest 3-class cep290
reference:

    High_to_Low / Low_to_High / Not Penetrant

Intermediate is merged into Low_to_High. Query phenotype has no ground truth, so query plots are
predicted distributions only. Reference-CV plots are real confusion diagnostics.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/cep290_phenotype_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import prepare_reference, transfer_labels  # noqa: E402

import build_reference_and_transfer as T  # noqa: E402
import label_transfer_snapshots as S  # noqa: E402

OUT = RUN_DIR / "plots" / "cep290_phaseA_3class"
TR = RUN_DIR / "transfer_results"

PHENO_ORDER = ["High_to_Low", "Low_to_High", "Not Penetrant"]
PHENO_COLORS = {
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
    "Not Penetrant": "#BBBBBB",
}
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous"]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RUN_DIR)}")


def _cep290_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict[str, float]]:
    cfg = T.DATASETS["cep290"]
    queries = [e for e in cfg["queries"] if (T.B6 / f"df03_final_output_with_latents_{e}.csv").exists()]
    if not queries:
        raise RuntimeError("No cep290 query build06 files found.")

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


def _cep290_pheno_reference(ref: pd.DataFrame) -> pd.DataFrame:
    r = ref.dropna(subset=[T.PHENO_COL, T.TIME_COL]).copy()
    r = r[~r[T.PHENO_COL].astype(str).isin(["unlabeled", "nan"])]
    r.loc[r[T.PHENO_COL] == "Intermediate", T.PHENO_COL] = "Low_to_High"
    r = r[r[T.PHENO_COL].isin(PHENO_ORDER)].copy()
    return r


def _embryo_reference_table(ref: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    emb = ref.groupby(T.GROUP_COL)[feat].mean().reset_index()
    label = ref.groupby(T.GROUP_COL)[T.PHENO_COL].agg(lambda s: s.mode().iloc[0]).reset_index()
    hpf = ref.groupby(T.GROUP_COL)[T.TIME_COL].median().reset_index()
    exp = ref.groupby(T.GROUP_COL)["experiment_id"].first().reset_index()
    return emb.merge(label, on=T.GROUP_COL).merge(hpf, on=T.GROUP_COL).merge(exp, on=T.GROUP_COL)


def _reference_cv_predictions(ref: pd.DataFrame, feat: list[str]) -> pd.DataFrame:
    """Recreate prepare_reference's embryo-level leave-one-experiment-out CV rows."""
    emb = _embryo_reference_table(ref, feat)
    X = emb[feat].to_numpy(dtype=float)
    y = emb[T.PHENO_COL].to_numpy()
    groups = emb["experiment_id"].to_numpy()

    # Use the same public helper for the fitted estimator shape, then pull its sklearn pipeline type.
    tmp_model = prepare_reference(
        ref,
        feat,
        label_col=T.PHENO_COL,
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col="experiment_id",
    )
    pipe = tmp_model["final_model"]
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=LeaveOneGroupOut())
    out = emb[[T.GROUP_COL, T.PHENO_COL, T.TIME_COL, "experiment_id"]].copy()
    out = out.rename(columns={T.PHENO_COL: "true_phenotype", T.TIME_COL: "hpf"})
    out["predicted_phenotype"] = y_pred
    return out


def _reference_cv_predictions_at_targets(
    ref: pd.DataFrame,
    feat: list[str],
    targets: tuple[int, ...] = (18, 24, 30, 48),
    window: float = 2.0,
) -> pd.DataFrame:
    """LOEO CV after aggregating only images within target +/- window hpf per embryo.

    This answers the narrower question: at exactly 18/24/30/48 hpf, how confused is the cep290
    phenotype reference? It avoids mixing 30 hpf and 48 hpf embryos into one broad confusion panel.
    """
    base_model = prepare_reference(
        ref,
        feat,
        label_col=T.PHENO_COL,
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col="experiment_id",
    )
    pipe_template = base_model["final_model"]
    rows = []
    for target in targets:
        win = ref[(ref[T.TIME_COL] >= target - window) & (ref[T.TIME_COL] <= target + window)].copy()
        if win.empty:
            continue
        emb = _embryo_reference_table(win, feat)
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


def _plot_confusion(df: pd.DataFrame, path: Path, title: str) -> None:
    labels = [c for c in PHENO_ORDER if c in set(df["true_phenotype"]) | set(df["predicted_phenotype"])]
    cm = confusion_matrix(df["true_phenotype"], df["predicted_phenotype"], labels=labels)
    denom = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)

    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * len(labels), 1.8 + 1.2 * len(labels)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("CV predicted phenotype")
    ax.set_ylabel("reference phenotype")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cmn[i, j]:.2f}\nn={cm[i, j]}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cmn[i, j] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def _stage_bins(values: pd.Series) -> pd.Series:
    bins = [0, 18, 24, 30, 48, 10_000]
    labels = ["<18", "18-24", "24-30", "30-48", "48+"]
    return pd.cut(values, bins=bins, labels=labels, right=False)


def _plot_confusion_by_stage(cv: pd.DataFrame) -> None:
    cv = cv.copy()
    cv["stage_window"] = _stage_bins(cv["hpf"])
    windows = [w for w in cv["stage_window"].cat.categories if (cv["stage_window"] == w).any()]
    labels = [c for c in PHENO_ORDER if c in set(cv["true_phenotype"]) | set(cv["predicted_phenotype"])]
    if not windows or not labels:
        return

    fig, axes = plt.subplots(1, len(windows), figsize=(2.8 * len(windows), 3.4), squeeze=False)
    for ax, win in zip(axes[0], windows):
        sub = cv[cv["stage_window"] == win]
        cm = confusion_matrix(sub["true_phenotype"], sub["predicted_phenotype"], labels=labels)
        denom = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0][0] else [""] * len(labels), fontsize=7)
        ax.set_title(f"{win} hpf\nn={len(sub)}", fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6)
    axes[0][0].set_ylabel("reference phenotype")
    fig.supxlabel("CV predicted phenotype", fontsize=9)
    fig.suptitle("cep290 3-class reference CV confusion by stage window", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, OUT / "cep290_phenotype_3class_reference_confusion_by_stage.png")


def _plot_confusion_by_target_hpf(cv: pd.DataFrame, path: Path) -> None:
    """Plot reference CV confusion for target hpf windows such as 18+/-2, 24+/-2, etc."""
    if cv.empty:
        return
    targets = sorted(cv["target_hpf"].dropna().unique())
    labels = [c for c in PHENO_ORDER if c in set(cv["true_phenotype"]) | set(cv["predicted_phenotype"])]
    if not targets or not labels:
        return
    fig, axes = plt.subplots(1, len(targets), figsize=(2.9 * len(targets), 3.5), squeeze=False)
    for ax, target in zip(axes[0], targets):
        sub = cv[cv["target_hpf"] == target]
        cm = confusion_matrix(sub["true_phenotype"], sub["predicted_phenotype"], labels=labels)
        denom = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0][0] else [""] * len(labels), fontsize=7)
        ax.set_title(f"{int(target)} hpf +/-2\nn={len(sub)}", fontsize=9)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]}", ha="center", va="center", fontsize=6)
    axes[0][0].set_ylabel("reference phenotype")
    fig.supxlabel("CV predicted phenotype", fontsize=9)
    fig.suptitle("cep290 3-class reference CV confusion at target hpf windows", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, path)


def _plot_predicted_over_stage(df: pd.DataFrame, path: Path, title: str, stage_col: str) -> None:
    d = df.dropna(subset=[stage_col]).copy()
    if d.empty:
        return
    d["stage"] = d[stage_col].astype(float).round().astype(int)
    stages = sorted(d["stage"].unique())
    classes = [c for c in PHENO_ORDER if (d["predicted_label"].astype(str) == c).any()]
    fig, ax = plt.subplots(figsize=(1.8 + 0.65 * len(stages), 5))
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    for cls in classes:
        vals = []
        for stage in stages:
            sub = d[d["stage"] == stage]
            vals.append((sub["predicted_label"].astype(str) == cls).sum() / len(sub) if len(sub) else 0.0)
        ax.bar(x, vals, bottom=bottom, color=PHENO_COLORS.get(cls, "#888"), edgecolor="white",
               linewidth=0.5, label=cls)
        bottom += np.array(vals)
    for i, stage in enumerate(stages):
        ax.text(i, 1.01, f"n={(d['stage'] == stage).sum()}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of query embryos")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    _save(fig, path)


def _attach_query_metadata(emb: pd.DataFrame, qry: pd.DataFrame, start_age: dict[str, float]) -> pd.DataFrame:
    meta = qry.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)
    out = emb.copy()
    out["dataset"] = "cep290"
    out["query_experiment"] = out["query_embryo_id"].map(meta["query_experiment"])
    out["true_genotype"] = out["query_embryo_id"].map(meta[T.GENO_COL])
    out["true_zygosity"] = out["query_embryo_id"].map(meta[T.ZYG_COL])
    out["predicted_stage_hpf"] = out["query_embryo_id"].map(meta[T.TIME_COL])
    out["stage"] = out["query_embryo_id"].map(start_age)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TR.mkdir(exist_ok=True)

    ref_raw, qry, feat, queries, start_age = _cep290_inputs()
    ref = _cep290_pheno_reference(ref_raw)
    seqlk = S.build_sequenced_lookup(queries)

    print("cep290 Phase A 3-class phenotype transfer")
    print(f"  features={len(feat)} ref_rows={len(ref)} query_rows={len(qry)}")
    print("  reference class counts:")
    print(ref.drop_duplicates(T.GROUP_COL)[T.PHENO_COL].value_counts().reindex(PHENO_ORDER).to_string())

    pheno_model = prepare_reference(
        ref,
        feat,
        label_col=T.PHENO_COL,
        group_col=T.GROUP_COL,
        time_col=T.TIME_COL,
        cv_group_col="experiment_id",
    )
    pheno_result = transfer_labels(pheno_model, qry, skip_flagged=False)
    pheno = _attach_query_metadata(pheno_result["embryo_predictions"], qry, start_age)
    pheno = S._tag(pheno, seqlk)
    pheno.to_csv(TR / "cep290_phenotype_3class_predictions.csv", index=False)

    geno = T.run_genotype_transfer("cep290", ref_raw, qry, feat)
    geno = _attach_query_metadata(geno, qry, start_age)
    geno = S._tag(geno, seqlk)
    joined = geno[
        [
            "query_embryo_id",
            "query_experiment",
            "stage",
            "predicted_stage_hpf",
            "true_genotype",
            "true_zygosity",
            "predicted_label",
            "correct",
            "sequenced",
            "stratum",
        ]
    ].rename(columns={"predicted_label": "predicted_zygosity"})
    joined = joined.merge(
        pheno[["query_embryo_id", "predicted_label", "top_probability"]].rename(
            columns={"predicted_label": "predicted_phenotype", "top_probability": "phenotype_top_probability"}
        ),
        on="query_embryo_id",
        how="left",
    )
    joined.to_csv(TR / "cep290_genotype_same_model_context.csv", index=False)

    cv = _reference_cv_predictions(ref, feat)
    cv.to_csv(TR / "cep290_phenotype_3class_reference_cv_predictions.csv", index=False)
    _plot_confusion(cv, OUT / "cep290_phenotype_3class_reference_confusion.png",
                    "cep290 3-class phenotype reference CV confusion")
    _plot_confusion_by_stage(cv)

    cv_target = _reference_cv_predictions_at_targets(ref, feat)
    cv_target.to_csv(TR / "cep290_phenotype_3class_reference_cv_predictions_target_hpf_pm2.csv",
                     index=False)
    _plot_confusion_by_target_hpf(
        cv_target,
        OUT / "cep290_phenotype_3class_reference_confusion_target_hpf_pm2.png",
    )
    _plot_predicted_over_stage(
        pheno,
        OUT / "cep290_phenotype_3class_query_predicted_by_stage.png",
        "cep290 query predicted phenotype by design stage\n(no query phenotype truth)",
        "stage",
    )
    _plot_predicted_over_stage(
        pheno[pheno["sequenced"] > 0],
        OUT / "cep290_phenotype_3class_sequenced_predicted_by_stage.png",
        "cep290 SEQUENCED query predicted phenotype by design stage\n(no query phenotype truth)",
        "stage",
    )

    homo = joined[joined["true_zygosity"] == "homozygous"]
    if not homo.empty:
        print("\nKnown-homozygous cep290 query embryos:")
        print(pd.crosstab(homo["predicted_zygosity"], homo["predicted_phenotype"]).to_string())

    print(f"\nWrote CSVs under: {TR.relative_to(RUN_DIR)}/")
    print(f"Wrote plots under: {OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
