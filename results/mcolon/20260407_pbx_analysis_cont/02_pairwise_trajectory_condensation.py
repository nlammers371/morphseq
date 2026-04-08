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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.trajectory_condensation import animation as tc_animation, init_embedding, plotting, schema
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation
from analyze.trajectory_condensation.iteration_ranking import (
    save_ranking_outputs,
    score_saved_iterations,
    select_evenly_distributed,
)
from analyze.trajectory_condensation.viz.iteration_choice_plots import (
    plot_iteration_scores,
    render_selected_iteration_bundle,
)

from common import GENOTYPE_COLORS, condensation_results_dir, pairwise_results_dir


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PBX condensation on combined all-pairs vectors.")
    parser.add_argument("--variant", choices=["shrunk", "raw"], default="raw")
    parser.add_argument("--include-wik-ab", action="store_true")
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init", choices=["umap", "pca"], default="umap")
    parser.add_argument("--x0-path", type=Path, default=None)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--selection-objective", choices=["balanced"], default="balanced")
    parser.add_argument("--epsilon-r", type=float, default=5e-4)
    parser.add_argument("--elastic-strength", type=float, default=16.0)
    parser.add_argument("--elastic-mix", type=float, default=0.25)
    parser.add_argument("--outlier-strength", type=float, default=16.0)
    parser.add_argument("--outlier-cutoff-preset", choices=["q95", "q97", "q99", "robust3"], default="robust3")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_cutoff_preset(name: str) -> tuple[str, float]:
    preset = str(name).strip().lower()
    if preset == "q95":
        return "quantile", 0.95
    if preset == "q97":
        return "quantile", 0.97
    if preset == "q99":
        return "quantile", 0.99
    if preset == "robust3":
        return "robust", 3.0
    raise ValueError(f"Unsupported outlier cutoff preset: {name!r}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_iter = 100
        args.save_every = 10
        args.top_k = min(args.top_k, 3)

    cutoff_mode, cutoff_value = parse_cutoff_preset(args.outlier_cutoff_preset)

    pairwise_dir = pairwise_results_dir(
        include_wik_ab=bool(args.include_wik_ab),
        bin_width=float(args.bin_width),
        n_permutations=int(args.n_permutations),
    )
    default_input = pairwise_dir / f"pairwise_{args.variant}_vectors.csv"
    input_path = args.input or default_input
    output_dir = args.output_dir or condensation_results_dir(
        variant=args.variant,
        include_wik_ab=bool(args.include_wik_ab),
        bin_width=float(args.bin_width),
        n_permutations=int(args.n_permutations),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    data = schema.from_pairwise_margin_csv(input_path, label_col="genotype")
    schema.validate(data, allow_feature_nans=True)

    if args.x0_path:
        x0_payload = np.load(args.x0_path)
        x0 = np.asarray(x0_payload["x0"], dtype=float)
    else:
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

    np.savez(output_dir / "x0_init.npz", x0=x0, time_values=data.time_values)

    config = CondensationConfig(
        sigma=0.5,
        temporal_cohere_window=3,
        epsilon_r=float(args.epsilon_r),
        elastic_strength=float(args.elastic_strength),
        elastic_mix=float(args.elastic_mix),
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        void_strength=0.014,
        outlier_strength=float(args.outlier_strength),
        outlier_cutoff_mode=cutoff_mode,
        outlier_cutoff_value=float(cutoff_value),
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=args.n_iter,
    )
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=config,
        stopping=stopping,
        log_every=max(1, args.n_iter // 20),
        save_every=args.save_every if args.save_every > 0 else None,
        verbose=True,
    )

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
    np.savez(output_dir / "condensed_positions.npz", **payload)

    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(data.labels)}

    fig, _ = plotting.plot_trajectories(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f"PBX combined {args.variant} trajectories",
    )
    fig.savefig(output_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_trajectories(
        x0,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f"PBX combined {args.variant} init",
    )
    fig.savefig(output_dir / "plot_trajectories_init.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    snap_indices = np.linspace(0, len(data.time_values) - 1, min(6, len(data.time_values)), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        snapshot_times=snapshot_times,
        title=f"PBX combined {args.variant} panels",
    )
    fig.savefig(output_dir / "plot_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f"PBX combined {args.variant} stacked 3D",
    )
    fig.savefig(output_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    _plot_metrics(metrics_df, output_dir / "plot_metrics.png")

    tc_animation.animate_rotation(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        output_path=output_dir / "rotation.gif",
        title=f"PBX combined {args.variant} rotation",
    )
    tc_animation.animate_init_final_rotation(
        x0,
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        output_path=output_dir / "init_vs_final_rotation.gif",
        title=f"PBX combined {args.variant} init vs final",
    )

    if result.position_history is not None:
        tc_animation.animate_iterations(
            result.position_history,
            data.mask,
            data.time_values,
            iter_labels=result.snapshot_iters,
            labels=data.labels,
            color_map=color_map,
            output_path=output_dir / "iterations.gif",
            fps=4,
            title=f"PBX combined {args.variant} iterations",
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

    manifest = {
        "input": str(input_path),
        "variant": args.variant,
        "include_wik_ab": bool(args.include_wik_ab),
        "bin_width": float(args.bin_width),
        "n_permutations": int(args.n_permutations),
        "n_iter": int(args.n_iter),
        "init_requested": args.init,
        "init_used": args.init,
        "canonical_representation": "pairwise_raw_vectors.csv" if not args.include_wik_ab else f"pairwise_{args.variant}_vectors.csv",
    }
    (output_dir / "condensation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(output_dir)


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
    scores = score_saved_iterations(history, iters, mask, labels, time_values, metrics_df, objective=objective)
    save_ranking_outputs(
        scores,
        output_dir=ranking_dir,
        config_payload={"selection_objective": objective, "top_k": top_k, "n_saved_iterations": len(iters)},
    )
    plot_iteration_scores(scores, ranking_dir / "plot_iteration_scores.png", title=f"Iteration ranking ({objective})")

    selected_root = output_dir / "selected_iterations"
    selected_dirs: list[Path] = []
    for _, row in select_evenly_distributed(scores, top_k).iterrows():
        iter_idx = int(row["iter"])
        snapshot_index = int(row["snapshot_index"])
        candidate_dir = selected_root / f"iter_{iter_idx:04d}_rank_{int(row['rank']):02d}"
        metadata = {
            "iter": iter_idx,
            "rank": int(row["rank"]),
            "geometry_score": float(row["geometry_score"]),
            "selection_objective": objective,
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
        selected_dirs.append(candidate_dir)
    return selected_dirs


def _plot_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
