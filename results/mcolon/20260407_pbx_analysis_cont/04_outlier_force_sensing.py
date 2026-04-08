from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_condensation_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.trajectory_condensation.condensation.forces.slice_outlier import build_slice_outlier_refs

from common import GENOTYPE_COLORS, condensation_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sense slice-outlier force activation on one PBX slice.")
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--include-wik-ab", action="store_true")
    parser.add_argument("--variant", choices=["shrunk", "raw"], default="raw")
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--time-bin-center", type=float, default=70.0)
    parser.add_argument("--strengths", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--cutoff-presets", nargs="+", default=["q99", "q97", "q95", "robust3"])
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--source", choices=["x0", "final"], default="x0")
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_dir = args.trajectory_dir or condensation_results_dir(
        variant=args.variant,
        include_wik_ab=bool(args.include_wik_ab),
        bin_width=float(args.bin_width),
        n_permutations=int(args.n_permutations),
    )
    output_dir = args.output_dir or trajectory_dir / "force_diagnostics" / f"outlier_force_sensing_{args.source}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(trajectory_dir / "condensed_positions.npz", allow_pickle=True)
    manifest = json.loads((trajectory_dir / "condensation_manifest.json").read_text())
    pairwise_df = pd.read_csv(manifest["input"])

    x0 = np.asarray(payload["x0"], dtype=float)
    positions = np.asarray(payload["positions"], dtype=float)
    mask = np.asarray(payload["mask"], dtype=bool)
    time_values = np.asarray(payload["time_values"], dtype=float)
    embryo_ids = payload["embryo_ids"].astype(str)
    labels = payload["labels"].astype(str)
    source_positions = x0 if args.source == "x0" else positions

    t_idx = pick_time_index(time_values, float(args.time_bin_center))
    slice_time = float(time_values[t_idx])
    obs = np.flatnonzero(mask[:, t_idx])
    if obs.size == 0:
        raise ValueError(f"No observed embryos at time {slice_time:g} hpf")

    summary_rows: list[dict[str, object]] = []
    all_rows: list[pd.DataFrame] = []
    tracked_ids: list[str] = []
    preset_defs = [parse_cutoff_preset(preset) for preset in args.cutoff_presets]
    for preset_name, cutoff_mode, cutoff_value in preset_defs:
        refs = build_slice_outlier_refs(
            x0,
            mask,
            cutoff_mode=cutoff_mode,
            quantile=cutoff_value if cutoff_mode == "quantile" else 0.99,
            robust_k=cutoff_value if cutoff_mode == "robust" else 3.0,
        )
        base_df = build_slice_table(
            positions=source_positions,
            refs=refs,
            mask=mask,
            t_idx=t_idx,
            time_value=slice_time,
            embryo_ids=embryo_ids,
            labels=labels,
            pairwise_df=pairwise_df,
            beta=float(args.beta),
            strength=1.0,
            cutoff_preset=preset_name,
        )
        if not tracked_ids:
            top_hits = base_df.sort_values("severity_ratio", ascending=False).head(int(args.top_k))
            tracked_ids = top_hits["embryo_id"].astype(str).tolist()

        for strength in [float(s) for s in args.strengths]:
            df = build_slice_table(
                positions=source_positions,
                refs=refs,
                mask=mask,
                t_idx=t_idx,
                time_value=slice_time,
                embryo_ids=embryo_ids,
                labels=labels,
                pairwise_df=pairwise_df,
                beta=float(args.beta),
                strength=strength,
                cutoff_preset=preset_name,
            )
            df["tracked"] = df["embryo_id"].isin(tracked_ids)
            df.to_csv(
                output_dir / f"slice_{format_strength(slice_time)}hpf_{preset_name}_strength_{format_strength(strength)}.csv",
                index=False,
            )
            all_rows.append(df.assign(outlier_strength=strength, cutoff_preset=preset_name))
            summary_rows.append(summarize_slice(df, slice_time=slice_time, strength=strength, cutoff_preset=preset_name))

    summary_df = pd.DataFrame(summary_rows)
    all_df = pd.concat(all_rows, ignore_index=True)
    tracked_df = all_df[all_df["tracked"]].copy()
    summary_df.to_csv(output_dir / "slice_force_summary.csv", index=False)
    tracked_df.to_csv(output_dir / "slice_force_tracked.csv", index=False)
    plot_slice_grid(all_df, tracked_ids, slice_time, output_dir / "slice_force_scatter_grid.png")
    plot_force_curves(all_df, slice_time, output_dir / "slice_force_curves.png")
    (output_dir / "slice_force_summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(output_dir)


def pick_time_index(time_values: np.ndarray, target: float) -> int:
    idx = int(np.argmin(np.abs(time_values - target)))
    return idx


def build_slice_table(
    *,
    positions: np.ndarray,
    refs,
    mask: np.ndarray,
    t_idx: int,
    time_value: float,
    embryo_ids: np.ndarray,
    labels: np.ndarray,
    pairwise_df: pd.DataFrame,
    beta: float,
    strength: float,
    cutoff_preset: str,
) -> pd.DataFrame:
    obs = np.flatnonzero(mask[:, t_idx])
    coords = positions[obs, t_idx, :]
    center = refs.centers[t_idx]
    cutoff_scale = float(refs.scale[t_idx])
    delta = coords - center
    dist = np.linalg.norm(delta, axis=1)
    severity = dist / max(cutoff_scale, 1e-12)
    logits = beta * (severity - 1.0)
    activation = np.log1p(np.exp(logits)) / beta
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    energy = strength * np.power(activation, 4)
    dE_du = strength * 4.0 * np.power(activation, 3) * sigmoid
    dE_dd = dE_du / max(cutoff_scale, 1e-12)

    metadata_cols = ["embryo_id", "time_bin_center", "experiment_id", "time_bin"]
    metadata = pairwise_df[metadata_cols].drop_duplicates(subset=["embryo_id", "time_bin_center"]).copy()

    df = pd.DataFrame(
        {
            "embryo_id": embryo_ids[obs].astype(str),
            "genotype": labels[obs].astype(str),
            "time_bin_center": float(time_value),
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
            "dist_to_slice_center": dist,
            "severity_ratio": severity,
            "u_logit": logits,
            "activation": activation,
            "point_energy": energy,
            "dE_du": dE_du,
            "dE_dd": dE_dd,
            "percentile_by_distance": pd.Series(dist).rank(pct=True, method="average").to_numpy() * 100.0,
            "cutoff_preset": cutoff_preset,
            "cutoff_mode": refs.cutoff_mode,
            "cutoff_value": refs.cutoff_value,
            "slice_cutoff_scale": cutoff_scale,
            "slice_center_dim1": center[0],
            "slice_center_dim2": center[1],
        }
    )
    df["time_index"] = t_idx
    df = df.merge(metadata, on=["embryo_id", "time_bin_center"], how="left")
    return df


def summarize_slice(df: pd.DataFrame, *, slice_time: float, strength: float, cutoff_preset: str) -> dict[str, object]:
    activated = df["u_logit"] > 0
    meaningful = df["activation"] > 0.05
    total_energy = float(df["point_energy"].sum())
    top_energy = float(df["point_energy"].max()) if len(df) else 0.0
    top_share = top_energy / total_energy if total_energy > 0 else 0.0
    return {
        "time_bin_center": slice_time,
        "cutoff_preset": cutoff_preset,
        "outlier_strength": strength,
        "n_embryos": int(len(df)),
        "cutoff_mode": str(df["cutoff_mode"].iloc[0]),
        "cutoff_value": float(df["cutoff_value"].iloc[0]),
        "slice_cutoff_scale": float(df["slice_cutoff_scale"].iloc[0]),
        "median_distance": float(df["dist_to_slice_center"].median()),
        "max_distance": float(df["dist_to_slice_center"].max()),
        "activated_fraction": float(activated.mean()),
        "meaningful_fraction": float(meaningful.mean()),
        "max_dE_dd": float(df["dE_dd"].max()),
        "median_dE_dd_activated": float(df.loc[activated, "dE_dd"].median()) if activated.any() else 0.0,
        "total_point_energy": total_energy,
        "top_embryo_energy_share": float(top_share),
        "top_embryo_id": str(df.sort_values("point_energy", ascending=False).iloc[0]["embryo_id"]),
    }


def plot_slice_grid(all_df: pd.DataFrame, tracked_ids: list[str], slice_time: float, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presets = list(dict.fromkeys(all_df["cutoff_preset"].tolist()))
    strengths = list(dict.fromkeys(all_df["outlier_strength"].tolist()))
    fig, axes = plt.subplots(
        len(presets),
        len(strengths),
        figsize=(4.4 * len(strengths), 4.0 * len(presets)),
        squeeze=False,
    )
    vmax = float(all_df["dE_dd"].max()) if len(all_df) else 1.0

    sc = None
    for row_idx, preset in enumerate(presets):
        for col_idx, strength in enumerate(strengths):
            ax = axes[row_idx, col_idx]
            df = all_df[(all_df["cutoff_preset"] == preset) & np.isclose(all_df["outlier_strength"], strength)].copy()
            sc = ax.scatter(
                df["dim1"],
                df["dim2"],
                c=df["dE_dd"],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                s=24,
                alpha=0.85,
                linewidths=0.0,
            )
            ax.scatter(
                df["slice_center_dim1"].iloc[0],
                df["slice_center_dim2"].iloc[0],
                marker="x",
                s=70,
                c="cyan",
                linewidths=1.5,
            )
            tracked = df[df["embryo_id"].isin(tracked_ids)]
            ax.scatter(
                tracked["dim1"],
                tracked["dim2"],
                facecolors="none",
                edgecolors="black",
                s=90,
                linewidths=1.0,
            )
            if row_idx == 0:
                ax.set_title(f"out={strength:g}")
            if col_idx == 0:
                ax.set_ylabel(f"{preset}\ndim 2")
            else:
                ax.set_ylabel("dim 2")
            ax.set_xlabel("dim 1")
            ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Radial outlier force dE/dd")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_force_curves(all_df: pd.DataFrame, slice_time: float, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presets = list(dict.fromkeys(all_df["cutoff_preset"].tolist()))
    strengths = list(dict.fromkeys(all_df["outlier_strength"].tolist()))
    fig, axes = plt.subplots(2, len(presets), figsize=(5.0 * len(presets), 8.0), squeeze=False)
    for col_idx, preset in enumerate(presets):
        for strength in strengths:
            df = all_df[(all_df["cutoff_preset"] == preset) & np.isclose(all_df["outlier_strength"], strength)].sort_values("dist_to_slice_center")
            axes[0, col_idx].plot(df["dist_to_slice_center"], df["point_energy"], lw=1.4, label=f"{strength:g}")
            axes[1, col_idx].plot(df["dist_to_slice_center"], df["dE_dd"], lw=1.4, label=f"{strength:g}")
        axes[0, col_idx].set_title(f"{slice_time:g} hpf | {preset}")
        axes[0, col_idx].set_xlabel("Distance to slice center")
        axes[0, col_idx].set_ylabel("Point outlier energy")
        axes[0, col_idx].grid(True, alpha=0.25)
        axes[1, col_idx].set_xlabel("Distance to slice center")
        axes[1, col_idx].set_ylabel("dE/dd")
        axes[1, col_idx].grid(True, alpha=0.25)
        axes[1, col_idx].legend(title="outlier_strength", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_cutoff_preset(preset: str) -> tuple[str, str, float]:
    p = str(preset).strip().lower()
    if p.startswith("q"):
        q = float(p[1:]) / 100.0
        return p, "quantile", q
    if p.startswith("robust"):
        suffix = p.replace("robust", "", 1)
        k = float(suffix) if suffix else 3.0
        return p, "robust", k
    raise ValueError(f"Unsupported cutoff preset: {preset!r}")


def format_strength(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
