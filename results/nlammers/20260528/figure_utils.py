"""Shared cache and plotting helpers for the 20260528 morphseq figure pass."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform


REPO_ROOT = Path("/Users/nick/Projects/repositories/morphseq")
DROPBOX_ROOT = Path("/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick")
MORPHSEQ_DROPBOX = DROPBOX_ROOT / "morphseq"
CACHE_DIR = Path("/Users/nick/Projects/data/morphseq/results/20260528")
FIG_DIR = CACHE_DIR / "figures_no19C_timepoint"

EXCLUDED_TEMPERATURES = (19.0,)
TEMP_CMAP = "RdBu_r"
TEMP_VMIN = 24
TEMP_CENTER = 28.5
TEMP_VMAX = 35
TIMEPOINT_MARKERS = {
    24.0: "o",
    30.0: "s",
    36.0: "^",
}
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 20260528


def temperature_norm() -> mpl.colors.Normalize:
    return mpl.colors.TwoSlopeNorm(vmin=TEMP_VMIN, vcenter=TEMP_CENTER, vmax=TEMP_VMAX)


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_light_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    ensure_dirs()
    for suffix in (".png", ".pdf"):
        fig.savefig(FIG_DIR / f"{name}{suffix}", dpi=300)


def temperature_scatter(ax, x, y, temp, **kwargs):
    sc = ax.scatter(
        x,
        y,
        c=temp,
        cmap=TEMP_CMAP,
        norm=temperature_norm(),
        edgecolor="black",
        linewidth=0.25,
        rasterized=True,
        **kwargs,
    )
    return sc


def drop_excluded_temperatures(df: pd.DataFrame, temperature_col: str = "temperature") -> pd.DataFrame:
    if temperature_col not in df.columns:
        return df.copy()
    temp = pd.to_numeric(df[temperature_col], errors="coerce")
    return df.loc[~temp.isin(EXCLUDED_TEMPERATURES)].copy()


def is_included_temperature(temp: float) -> bool:
    return float(temp) not in EXCLUDED_TEMPERATURES


def _marker_for_timepoint(timepoint) -> str:
    try:
        key = float(timepoint)
    except (TypeError, ValueError):
        return "o"
    return TIMEPOINT_MARKERS.get(key, "o")


def _timepoint_label(timepoint) -> str:
    try:
        return f"{float(timepoint):g} hpf"
    except (TypeError, ValueError):
        return str(timepoint)


def add_timepoint_legend(ax, timepoints, title: str = "collection") -> None:
    unique = sorted(pd.Series(timepoints).dropna().unique(), key=lambda v: float(v))
    handles = [
        Line2D(
            [0],
            [0],
            marker=_marker_for_timepoint(timepoint),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            markersize=6,
            label=_timepoint_label(timepoint),
        )
        for timepoint in unique
    ]
    if handles:
        ax.legend(handles=handles, title=title, frameon=False, loc="best", fontsize=8, title_fontsize=8)


def value_timepoint_scatter(
    ax,
    x,
    y,
    value,
    timepoint,
    *,
    z=None,
    cmap=TEMP_CMAP,
    vmin=None,
    vmax=None,
    norm=None,
    colorbar_label="",
    add_colorbar=True,
    add_legend=True,
    **kwargs,
):
    plot_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "value": pd.to_numeric(value, errors="coerce"),
            "timepoint": timepoint,
        }
    )
    if z is not None:
        plot_df["z"] = z
    plot_df = plot_df.dropna(subset=["x", "y", "value", "timepoint"])

    if norm is None and vmin is None:
        vmin = float(plot_df["value"].min())
    if norm is None and vmax is None:
        vmax = float(plot_df["value"].max())

    cmap_obj = plt.get_cmap(cmap)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scatter_kwargs = {
        "s": 38,
        "alpha": 0.9,
        "edgecolor": "black",
        "linewidth": 0.25,
        "rasterized": True,
        **kwargs,
    }
    for timepoint_value, group in plot_df.groupby("timepoint", sort=True):
        marker = _marker_for_timepoint(timepoint_value)
        if z is None:
            ax.scatter(
                group["x"],
                group["y"],
                c=group["value"],
                cmap=cmap_obj,
                norm=norm,
                marker=marker,
                **scatter_kwargs,
            )
        else:
            ax.scatter(
                group["x"],
                group["y"],
                group["z"],
                c=group["value"],
                cmap=cmap_obj,
                norm=norm,
                marker=marker,
                **scatter_kwargs,
            )

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array([])
    if add_colorbar:
        cb = plt.colorbar(mappable, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)
    if add_legend:
        add_timepoint_legend(ax, plot_df["timepoint"])
    return mappable


def temperature_timepoint_scatter(ax, x, y, temp, timepoint, *, z=None, **kwargs):
    kwargs.setdefault("add_colorbar", False)
    return value_timepoint_scatter(
        ax,
        x,
        y,
        temp,
        timepoint,
        z=z,
        cmap=TEMP_CMAP,
        norm=temperature_norm(),
        colorbar_label="temperature (C)",
        **kwargs,
    )


def bootstrap_mean_se(values: pd.Series, rng: np.random.Generator, n_bootstrap: int = BOOTSTRAP_N) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    return float(np.std(np.mean(samples, axis=1), ddof=1))


def bootstrap_std_se(values: pd.Series, rng: np.random.Generator, n_bootstrap: int = BOOTSTRAP_N) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    return float(np.std(np.std(samples, axis=1, ddof=1), ddof=1))


def timepoint_average_variability_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    *,
    temperature_col: str = "temperature",
    timepoint_col: str = "timepoint",
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    data = drop_excluded_temperatures(df, temperature_col)
    for temperature, temp_df in data.groupby(temperature_col, sort=True):
        groups = [
            pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            for _, group in temp_df.groupby(timepoint_col, sort=True)
        ]
        groups = [arr for arr in groups if arr.size >= 2]
        if not groups:
            continue
        observed = np.array([np.std(arr, ddof=1) for arr in groups], dtype=float)
        boot = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boot[i] = np.mean([np.std(rng.choice(arr, size=arr.size, replace=True), ddof=1) for arr in groups])
        rows.append(
            {
                "temperature": temperature,
                "variability_mean": float(np.mean(observed)),
                "variability_boot_se": float(np.std(boot, ddof=1)),
                "n_timepoints": len(groups),
                "n": int(sum(arr.size for arr in groups)),
                "n_bootstrap": n_bootstrap,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows).sort_values("temperature").reset_index(drop=True)


def plot_temperature_variability_bootstrap(summary: pd.DataFrame, ylabel: str, title: str | None = None):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.errorbar(
        summary["temperature"],
        summary["variability_mean"],
        yerr=summary["variability_boot_se"],
        fmt="o-",
        color="#333333",
        ecolor="#555555",
        elinewidth=0.9,
        capsize=2.5,
        markersize=5,
    )
    ax.set_xlabel("temperature (C)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def add_identity(ax, x=None, y=None, **kwargs):
    if x is None or y is None:
        lo, hi = ax.get_xlim()
        yo, yh = ax.get_ylim()
        mn, mx = min(lo, yo), max(hi, yh)
    else:
        vals = np.asarray(pd.concat([pd.Series(x), pd.Series(y)], ignore_index=True), dtype=float)
        vals = vals[np.isfinite(vals)]
        mn, mx = float(vals.min()), float(vals.max())
    pad = 0.03 * (mx - mn)
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], color="#555555", linestyle="--", linewidth=1)


def manifest_record(analysis_name, source_notebook_or_script, source_data_path, output_cache_file, status, notes):
    return {
        "analysis_name": analysis_name,
        "source_notebook_or_script": str(source_notebook_or_script),
        "source_data_path": str(source_data_path),
        "output_cache_file": str(output_cache_file),
        "status": status,
        "notes": notes,
    }


def copy_csv_if_exists(src: Path, dst: Path, manifest: list[dict], analysis_name: str, source_nb: str, notes: str = ""):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        status = "ok"
    else:
        status = "missing"
    manifest.append(manifest_record(analysis_name, source_nb, src, dst, status, notes))
    return status


def feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    if prefix == "V":
        return sorted([c for c in df.columns if c.startswith("V")], key=lambda c: int(c[1:]) if c[1:].isdigit() else c)
    return sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("_")[-1]) if c.split("_")[-1].isdigit() else c,
    )


def _offdiag_values(R):
    R = np.asarray(R, dtype=float)
    return R[~np.eye(R.shape[0], dtype=bool)]


def corr_metrics(R):
    vals = _offdiag_values(R)
    R0 = np.nan_to_num(np.asarray(R, dtype=float), nan=0.0)
    eigvals = np.clip(np.linalg.eigvalsh(R0), 0, None)
    eig_sum = eigvals.sum()
    if eig_sum > 0:
        q = eigvals / eig_sum
        effective_rank = float(np.exp(-np.sum(q * np.log(q + 1e-12))))
        eig_desc = np.sort(eigvals)[::-1]
        top1 = float(eig_desc[:1].sum() / eig_sum)
        top3 = float(eig_desc[:3].sum() / eig_sum)
        top5 = float(eig_desc[:5].sum() / eig_sum)
    else:
        effective_rank = top1 = top3 = top5 = np.nan
    return {
        "mean_abs_offdiag_corr": float(np.nanmean(np.abs(vals))),
        "rms_offdiag_corr": float(np.sqrt(np.nanmean(vals**2))),
        "effective_rank": effective_rank,
        "top1_eigen_fraction": top1,
        "top3_eigen_fraction": top3,
        "top5_eigen_fraction": top5,
    }


def compute_feature_cc_tables(df: pd.DataFrame, prefix: str, temperature_col: str = "temperature", min_pair_count: int = 8):
    cols = feature_columns(df, prefix)
    temps = np.asarray(sorted(pd.to_numeric(df[temperature_col], errors="coerce").dropna().unique()), dtype=float)
    cc_by_temp = {}
    counts = []
    for temp in temps:
        temp_df = df.loc[df[temperature_col].astype(float) == temp, cols].apply(pd.to_numeric, errors="coerce")
        cc_by_temp[temp] = temp_df.corr(method="pearson")
        bool_df = temp_df.notna()
        counts.append(bool_df.astype(int).T.dot(bool_df.astype(int)).to_numpy())
    stacked = np.stack(counts, axis=2)
    keep = np.max(np.min(stacked, axis=2), axis=0) >= min_pair_count
    kept_cols = np.asarray(cols)[keep]
    cc_by_temp = {temp: mat.loc[kept_cols, kept_cols] for temp, mat in cc_by_temp.items()}

    ref_temp = 28.5 if 28.5 in cc_by_temp else 28.0 if 28.0 in cc_by_temp else temps[len(temps) // 2]
    corr = cc_by_temp[ref_temp].fillna(0).copy()
    np.fill_diagonal(corr.values, 1)
    order = dendrogram(linkage(squareform(1 - corr, checks=False), method="average"), no_plot=True)["leaves"]

    rows = []
    flat_rows = []
    for temp, mat in cc_by_temp.items():
        ordered = mat.iloc[order, order]
        ordered.to_csv(CACHE_DIR / f"{prefix}_feature_cc_{temp:g}C.csv")
        vals = ordered.to_numpy()[np.tril_indices(ordered.shape[0], k=-1)]
        flat_rows.append(pd.DataFrame({"temperature": temp, "cc": vals[np.isfinite(vals)]}))
        rows.append({"temperature": temp, "n_features": ordered.shape[0], **corr_metrics(ordered)})
    flat = pd.concat(flat_rows, ignore_index=True) if flat_rows else pd.DataFrame()
    metrics = pd.DataFrame(rows).sort_values("temperature")
    flat.to_csv(CACHE_DIR / f"{prefix}_feature_cc_values.csv", index=False)
    metrics.to_csv(CACHE_DIR / f"{prefix}_feature_order_metrics.csv", index=False)
    return cc_by_temp, metrics, flat


def load_seq_latents(model_path: Path):
    latent_list = []
    col_sets = []
    for folder in sorted(model_path.glob("*C")):
        if not (folder / "latents.csv").exists():
            continue
        temp = float(folder.name[:-1])
        df = pd.read_csv(folder / "latents.csv", index_col=0)
        col_sets.append(set(df.columns))
        df["sample"] = df.index
        df["temperature"] = temp
        latent_list.append(df.reset_index(drop=True))
    if not latent_list:
        return pd.DataFrame()
    common = set.intersection(*col_sets)
    cols = sorted([c for c in common if c.startswith("V")], key=lambda c: int(c[1:]) if c[1:].isdigit() else c)
    return pd.concat([df.loc[:, ["sample", "temperature"] + cols] for df in latent_list], ignore_index=True)


def build_cache():
    ensure_dirs()
    manifest = []

    morph_base = MORPHSEQ_DROPBOX / "results" / "20250312" / "morph_latent_space"
    source_morph_nb = REPO_ROOT / "results" / "nlammers" / "20260504" / "hotfish_morph_analyses.ipynb"
    source_seq_nb = REPO_ROOT / "results" / "nlammers" / "20260504" / "hotfish2_stage_analysis_cell_type_coarse.ipynb"
    source_morph_cc_nb = REPO_ROOT / "results" / "nlammers" / "20260504" / "morph_feature_cc.ipynb"
    source_seq_cc_nb = REPO_ROOT / "results" / "nlammers" / "20260504" / "seq_feature_cc.ipynb"
    source_arr_nb = REPO_ROOT / "results" / "nlammers" / "20250319" / "hf2_seq_morph_arrhenius.ipynb"
    source_volcano_r = REPO_ROOT / "results" / "nlammers" / "20260504" / "hf_hooke_morph.R"

    for filename, analysis, source_nb in [
        ("hf_pca_morph_df.csv", "morph_hotfish_pca", source_morph_nb),
        ("hf_pca_morph_df_hooke.csv", "morph_hotfish_pca_hooke", source_morph_nb),
        ("ab_ref_pca_morph_df.csv", "morph_reference_trajectories", source_morph_nb),
        ("spline_morph_df.csv", "morph_reference_spline", source_morph_nb),
        ("spline_seq_df.csv", "seq_reference_spline", source_seq_nb),
        ("hf_seq_df.csv", "seq_hotfish_latent_pca", source_seq_nb),
        ("seq_to_morph_pca_pd.csv", "joint_141_morph_seq_stage", source_arr_nb),
    ]:
        copy_csv_if_exists(morph_base / filename, CACHE_DIR / filename, manifest, analysis, source_nb)

    joint_path = CACHE_DIR / "seq_to_morph_pca_pd.csv"
    hooke_morph_path = CACHE_DIR / "hf_pca_morph_df_hooke.csv"
    if joint_path.exists():
        joint = pd.read_csv(joint_path)
        if "embryo_id" not in joint.columns and "snip_id" in joint.columns:
            joint["embryo_id"] = joint["snip_id"].str.replace(r"_t\\d+$", "", regex=True)
        if hooke_morph_path.exists():
            hooke_morph = pd.read_csv(hooke_morph_path)
            keep = [c for c in ["snip_id", "sample", "morph_dist_spline", "morph_branch_flag", "nn_stage_hpf"] if c in hooke_morph.columns]
            joint = joint.merge(hooke_morph.loc[:, keep].drop_duplicates("snip_id"), on="snip_id", how="left")
        joint.to_csv(CACHE_DIR / "joint_141_morph_seq.csv", index=False)
        manifest.append(manifest_record("joint_141_morph_seq", source_arr_nb, joint_path, CACHE_DIR / "joint_141_morph_seq.csv", "ok", f"{joint.shape[0]} rows; expected 141 matched embryos"))
    else:
        manifest.append(manifest_record("joint_141_morph_seq", source_arr_nb, joint_path, CACHE_DIR / "joint_141_morph_seq.csv", "missing", "Could not locate seq_to_morph_pca_pd.csv"))

    if (CACHE_DIR / "hf_pca_morph_df.csv").exists() and (CACHE_DIR / "spline_morph_df.csv").exists():
        hf = pd.read_csv(CACHE_DIR / "hf_pca_morph_df.csv")
        spline = pd.read_csv(CACHE_DIR / "spline_morph_df.csv")
        pca_cols = [c for c in hf.columns if c.startswith("PCA_") and c.endswith("_bio")]
        if pca_cols:
            dist = distance_matrix(hf[pca_cols].to_numpy(), spline[pca_cols].to_numpy())
            nn = np.argmin(dist, axis=1)
            hf["morph_dist_spline"] = dist[np.arange(len(hf)), nn]
            if "stage_hpf" in spline.columns:
                hf["spline_stage_hpf"] = spline.loc[nn, "stage_hpf"].to_numpy()
            hf.to_csv(CACHE_DIR / "hf_pca_morph_df_with_spline_distance.csv", index=False)
            manifest.append(manifest_record("morph_distance_from_wt_spline", source_morph_nb, CACHE_DIR / "hf_pca_morph_df.csv", CACHE_DIR / "hf_pca_morph_df_with_spline_distance.csv", "ok", "Nearest-neighbor Euclidean distance in morph PCA coordinates"))

    hooke_path = MORPHSEQ_DROPBOX / "results" / "20250312" / "HF_hooke_regressions"
    seq_latents = load_seq_latents(hooke_path)
    if not seq_latents.empty:
        seq_latents.to_csv(CACHE_DIR / "seq_hooke_latents.csv", index=False)
        manifest.append(manifest_record("seq_hooke_latents", source_seq_cc_nb, hooke_path, CACHE_DIR / "seq_hooke_latents.csv", "ok", f"{seq_latents.shape[0]} rows"))
        compute_feature_cc_tables(seq_latents, "V")
        manifest.append(manifest_record("latent_cell_type_count_correlation", source_seq_cc_nb, hooke_path, CACHE_DIR / "V_feature_order_metrics.csv", "ok", "Recomputed from per-temperature Hooke latents.csv files"))
    else:
        manifest.append(manifest_record("seq_hooke_latents", source_seq_cc_nb, hooke_path, CACHE_DIR / "seq_hooke_latents.csv", "missing", "No per-temperature latents.csv files found"))

    train_fig = sorted((MORPHSEQ_DROPBOX / "training_data" / "20241107_ds" / "SeqVAE_z100_ne150_sweep_01_block01_iter030").glob("*/figures"))
    if train_fig:
        morph_df = pd.read_csv(train_fig[-1] / "embryo_stats_df.csv", index_col=0)
        hf_meta_path = CACHE_DIR / "hf_pca_morph_df.csv"
        if hf_meta_path.exists():
            meta = pd.read_csv(hf_meta_path).loc[:, ["snip_id", "temperature", "timepoint"]].drop_duplicates("snip_id")
            if "experiment_date" in morph_df.columns and "snip_id" in morph_df.columns:
                hf_morph = morph_df.loc[
                    morph_df["experiment_date"].isin(["20240813_24hpf", "20240813_30hpf", "20240813_36hpf"])
                ].copy()
                hf_morph = hf_morph.drop(columns=[c for c in ["temperature", "timepoint"] if c in hf_morph.columns])
                hf_morph = hf_morph.merge(meta, on="snip_id", how="inner")
                hf_morph.to_csv(CACHE_DIR / "hf_morph_latent_features.csv", index=False)
                manifest.append(manifest_record("latent_morph_variable_source", source_morph_cc_nb, train_fig[-1] / "embryo_stats_df.csv", CACHE_DIR / "hf_morph_latent_features.csv", "ok", f"{hf_morph.shape[0]} rows"))
                compute_feature_cc_tables(hf_morph, "z_mu_b")
                manifest.append(manifest_record("latent_morph_variable_correlation", source_morph_cc_nb, CACHE_DIR / "hf_morph_latent_features.csv", CACHE_DIR / "z_mu_b_feature_order_metrics.csv", "ok", "Recomputed from embryo_stats_df and hotfish temperature metadata"))
            else:
                manifest.append(manifest_record("latent_morph_variable_source", source_morph_cc_nb, train_fig[-1] / "embryo_stats_df.csv", CACHE_DIR / "hf_morph_latent_features.csv", "missing", "embryo_stats_df.csv lacks experiment_date or snip_id"))
    else:
        manifest.append(manifest_record("latent_morph_variable_source", source_morph_cc_nb, "training_data/20241107_ds/.../figures", CACHE_DIR / "hf_morph_latent_features.csv", "missing", "Training figures folder not found"))

    nn_dirs = [
        DROPBOX_ROOT / "slides" / "morphseq" / "20250521" / "nn_staging",
        DROPBOX_ROOT / "slides" / "morphseq" / "20250312" / "nn_staging",
    ]
    nn_dir = next((d for d in nn_dirs if d.exists()), None)
    if nn_dir:
        staging_dir = CACHE_DIR / "pairwise_staging_correlations"
        staging_dir.mkdir(exist_ok=True)
        for src in sorted(nn_dir.glob("*.csv")):
            shutil.copy2(src, staging_dir / src.name)
        manifest.append(manifest_record("pairwise_staging_correlation", source_seq_nb, nn_dir, staging_dir, "ok", f"Copied {len(list(staging_dir.glob('*.csv')))} CSV files"))
    else:
        manifest.append(manifest_record("pairwise_staging_correlation", source_seq_nb, nn_dirs, CACHE_DIR / "pairwise_staging_correlations", "missing", "No nn_staging CSV folder found"))

    volcano_src = MORPHSEQ_DROPBOX / "analysis" / "crossmodal" / "hotfish" / "hotfish_morph_ccs_contrast.csv"
    copy_csv_if_exists(volcano_src, CACHE_DIR / "hotfish_morph_ccs_contrast.csv", manifest, "curved_vs_normal_cell_type_volcano", source_volcano_r, "Expected output of hf_hooke_morph.R")

    pd.DataFrame(manifest).to_csv(CACHE_DIR / "source_manifest.csv", index=False)
    return pd.DataFrame(manifest)


def plot_heatmap_matrix(matrix, ax=None, title="", cmap="RdBu_r", vmin=-1, vmax=1):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
