"""
05_pbx_condensation.py
----------------------
Run trajectory condensation on PBX crispant vector data.

Pipeline:
  1. Load multiclass probability or pairwise margin vectors -> CosmologyData
  2. Initialize 2D positions via AlignedUMAP, NaN-aware UMAP, or PCA
  3. Run condensation with calibrated force defaults
  4. Save positions, metrics, and diagnostic figures
  5. Rank saved iterations and render top candidate graphs
  6. Optionally run a second-pass principal graph on the ranked candidates
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology import animation as tc_animation, schema, init_embedding, plotting
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.state import CondensationConfig
from trajectory_cosmology.condensation.engine.stopping import StoppingConfig
from trajectory_cosmology.iteration_ranking import (
    plot_iteration_scores,
    render_selected_iteration_bundle,
    save_ranking_outputs,
    score_saved_iterations,
)


GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PBX crispant trajectory condensation.")
    p.add_argument("--input", required=True, help="Path to multiclass probability or pairwise margin vector CSV.")
    p.add_argument("--input-type", choices=["auto", "multiclass", "pairwise"], default="auto")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init", choices=["umap", "pca"], default="umap")
    p.add_argument("--x0-path", help="Optional path to a saved x0_init.npz to reuse instead of recomputing init.")
    p.add_argument("--n-iter", type=int, default=500)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of ranked iterations to render as candidate graphs.")
    p.add_argument("--selection-objective", choices=["balanced"], default="balanced")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.init = "umap"
        args.n_iter = 100
        args.save_every = 10
        args.top_k = min(args.top_k, 3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    data = _load_cosmology_data(args.input, args.input_type)
    schema.validate(data, allow_feature_nans=bool(np.isnan(data.features[data.mask]).any()))

    n_e, t_count, k_count = data.features.shape
    print(f"  {n_e} embryos x {t_count} time bins x {k_count} features")
    print(f"  Observed entries: {data.mask.sum()} / {data.mask.size} ({100 * data.mask.mean():.1f}%)")
    print(f"  Time range: {data.time_values[0]:.0f}-{data.time_values[-1]:.0f} hpf")
    print(f"  Genotypes: {dict(zip(*np.unique(data.labels, return_counts=True)))}")

    if args.x0_path:
        print(f"\nLoading saved initialization: {args.x0_path}")
        x0_payload = np.load(args.x0_path)
        x0 = np.asarray(x0_payload["x0"], dtype=float)
        if x0.shape != (n_e, t_count, 2):
            raise ValueError(
                f"x0 shape mismatch: expected {(n_e, t_count, 2)}, found {tuple(x0.shape)}"
            )
    else:
        print(f"\nInitializing with {args.init.upper()} (seed={args.seed})...")
        if args.init == "umap":
            x0 = init_embedding.aligned_umap_init(
                data.features,
                data.mask,
                n_neighbors=15,
                min_dist=0.1,
                alignment_regularisation=1e-2,
                alignment_window_size=3,
                random_state=args.seed,
            )
        else:
            x0 = init_embedding.pca_init(data.features, data.mask, random_state=args.seed)

    print(f"  x0 shape: {x0.shape}  (NaN where not observed)")
    np.savez(output_dir / "x0_init.npz", x0=x0, time_values=data.time_values)
    print(f"  Saved: {output_dir / 'x0_init.npz'}")

    config = CondensationConfig(
        sigma=0.5,
        delta=3,
        epsilon_r=0.005,
        lambda_stretch=0.04,
        lambda_bend=0.04,
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        epsilon_void=0.014,
        k_attract=20,
        lr=1e-4,
        alpha=0.9,
        max_iter=args.n_iter,
    )
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    save_every = args.save_every if args.save_every > 0 else None
    print(f"\nRunning condensation ({args.n_iter} iterations, lr={config.lr:.0e}) ...")
    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=config,
        stopping=stopping,
        log_every=max(1, args.n_iter // 20),
        save_every=save_every,
        verbose=True,
    )
    print(f"\nDone. Converged: {result.converged} | Iterations: {result.n_iter}")

    pos_path = output_dir / "condensed_positions.npz"
    payload = {
        "positions": result.positions,
        "x0": x0,
        "mask": data.mask,
        "time_values": data.time_values,
        "embryo_ids": data.embryo_ids,
        "labels": data.labels,
    }
    if result.position_history is not None:
        payload["position_history"] = result.position_history
        payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(pos_path, **payload)
    print(f"Saved positions: {pos_path}  shape={result.positions.shape}")

    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(data.labels)}

    fig, _ = plotting.plot_trajectories(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX condensed trajectories ({args.init.upper()} init, {result.n_iter} iters)",
    )
    fig.savefig(output_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_trajectories(
        x0, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX initialization ({args.init.upper()})",
    )
    fig.savefig(output_dir / "plot_trajectories_init.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    snap_indices = np.linspace(0, t_count - 1, min(6, t_count), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        snapshot_times=snapshot_times,
        title=f"PBX panels ({args.init.upper()} init, {result.n_iter} iters)",
    )
    fig.savefig(output_dir / "plot_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX stacked 3D ({args.init.upper()} init)",
    )
    fig.savefig(output_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    _plot_metrics(
        metrics_df,
        output_dir / "plot_metrics.png",
        title=f"{args.init.upper()} init - sigma={config.sigma} delta={config.delta} lr={config.lr:.0e} k={config.k_attract}",
    )

    if result.position_history is not None:
        tc_animation.animate_rotation(
            result.positions, data.mask, data.time_values,
            labels=data.labels, color_map=color_map,
            output_path=output_dir / "rotation.gif",
            title=f"PBX condensed trajectories ({args.init.upper()} init, {result.n_iter} iters)",
        )
        tc_animation.animate_iterations(
            result.position_history, data.mask, data.time_values,
            iter_labels=result.snapshot_iters,
            labels=data.labels, color_map=color_map,
            output_path=output_dir / "iterations.gif",
            fps=4,
            title=f"PBX condensation progress ({args.init.upper()} init)",
        )

        _rank_and_render_candidates(
            output_dir=output_dir,
            metrics_df=metrics_df,
            position_history=result.position_history,
            snapshot_iters=list(result.snapshot_iters),
            final_positions=result.positions,
            final_iter=max(0, result.n_iter - 1),
            mask=data.mask,
            labels=data.labels,
            time_values=data.time_values,
            color_map=color_map,
            objective=args.selection_objective,
            top_k=args.top_k,
            init_name=args.init.upper(),
        )

    print(f"\nAll outputs saved to: {output_dir}")
    _print_final_summary(metrics_df)


def _rank_and_render_candidates(
    *,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    position_history: np.ndarray,
    snapshot_iters: list[int],
    final_positions: np.ndarray,
    final_iter: int,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    color_map: dict[str, str],
    objective: str,
    top_k: int,
    init_name: str,
) -> list[Path]:
    history = position_history
    iters = list(snapshot_iters)
    if not iters or iters[-1] != final_iter:
        history = np.concatenate([history, final_positions[None, ...]], axis=0)
        iters = iters + [int(final_iter)]

    ranking_dir = output_dir / "iteration_ranking"
    scores = score_saved_iterations(
        history,
        iters,
        mask,
        labels,
        time_values,
        metrics_df,
        objective=objective,
    )
    save_ranking_outputs(
        scores,
        output_dir=ranking_dir,
        config_payload={
            "selection_objective": objective,
            "top_k": top_k,
            "n_saved_iterations": len(iters),
            "density_metrics": ["density_knn_mean", "density_knn_cv"],
        },
    )
    plot_iteration_scores(
        scores,
        ranking_dir / "plot_iteration_scores.png",
        title=f"Iteration ranking ({objective})",
    )

    selected_root = output_dir / "selected_iterations"
    selected_dirs: list[Path] = []
    for _, row in scores.head(top_k).iterrows():
        iter_idx = int(row["iter"])
        snapshot_index = int(row["snapshot_index"])
        candidate_dir = selected_root / f"iter_{iter_idx:04d}_rank_{int(row['rank']):02d}"
        metadata = {
            "iter": iter_idx,
            "rank": int(row["rank"]),
            "geometry_score": float(row["geometry_score"]),
            "selection_score": float(row["geometry_score"]),
            "geometry_objective": objective,
            "selection_objective": objective,
            "is_final_iteration": bool(iter_idx == final_iter),
            "density_knn_mean": _float_or_none(row.get("density_knn_mean")),
            "density_knn_cv": _float_or_none(row.get("density_knn_cv")),
            "compactness_mean": _float_or_none(row.get("compactness_mean")),
            "centroid_separation_median": _float_or_none(row.get("centroid_separation_median")),
            "crowding_p10": _float_or_none(row.get("crowding_p10")),
            "centroid_shift_mean": _float_or_none(row.get("centroid_shift_mean")),
        }
        render_selected_iteration_bundle(
            positions=history[snapshot_index],
            mask=mask,
            time_values=time_values,
            labels=labels,
            color_map=color_map,
            output_dir=candidate_dir,
            title_prefix=f"PBX candidate ({init_name} init)",
            snapshot_iter=iter_idx,
            metadata=metadata,
        )
        tc_animation.animate_rotation(
            history[snapshot_index],
            mask,
            time_values,
            labels=labels,
            color_map=color_map,
            output_path=candidate_dir / "rotation.gif",
            title=f"PBX candidate rotation ({init_name} init, iter {iter_idx})",
        )
        selected_dirs.append(candidate_dir)

    scores.head(top_k).to_csv(selected_root / "top_k_candidates.csv", index=False)
    return selected_dirs


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(fval):
        return fval
    return None


def _plot_metrics(metrics_df: pd.DataFrame, output_path: Path, title: str = "") -> None:
    cols_labels = [
        ("energy_total", "Total energy"),
        ("disp_rms_rel", "RMS displacement (rel)"),
        ("energy_change_rel", "Energy change (rel)"),
        ("coherence_change_rel", "Coherence change (rel)"),
    ]
    available = [(c, l) for c, l in cols_labels if c in metrics_df.columns]
    if not available:
        return
    ncols = min(2, len(available))
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, available):
        ax.plot(metrics_df["iter"], metrics_df[col], lw=1.4)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(available):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _print_final_summary(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    last = metrics_df.iloc[-1]
    print("\n--- Final iteration summary ---")
    for col in ["energy_total", "disp_rms_rel", "disp_max_rel", "energy_change_rel", "coherence_change_rel"]:
        if col in last.index and not pd.isna(last[col]):
            print(f"  {col}: {last[col]:.6f}")


def _load_cosmology_data(input_path: str, input_type: str) -> schema.CosmologyData:
    if input_type == "multiclass":
        return schema.from_multiclass_csv(input_path, label_col="genotype")
    if input_type == "pairwise":
        return schema.from_pairwise_margin_csv(input_path, label_col="genotype")

    df = pd.read_csv(input_path, nrows=5)
    has_prob = any(c.startswith("p_") or c.startswith("pred_proba_") for c in df.columns)
    has_pairwise = any("_vs_" in c for c in df.columns)
    if has_pairwise and not has_prob:
        return schema.from_pairwise_margin_csv(input_path, label_col="genotype")
    if has_prob:
        return schema.from_multiclass_csv(input_path, label_col="genotype")
    raise ValueError(
        f"Could not infer input type for {input_path}. "
        "Expected probability columns ('p_*') or pairwise columns ('*_vs_*')."
    )


if __name__ == "__main__":
    main()
