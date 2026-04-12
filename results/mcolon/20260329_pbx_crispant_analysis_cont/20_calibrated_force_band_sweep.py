from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from temporal_sandbox import (
    TemporalRunConfig,
    make_crossing_bundles,
    make_stable_bundles,
    plot_temporal_run,
    run_temporal,
)
from trajectory_cosmology import plotting, schema
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.engine.stopping import StoppingConfig
from trajectory_cosmology.condensation.geometry_refs import estimate_geometry_refs
from trajectory_cosmology.condensation.state import CondensationConfig
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
    p = argparse.ArgumentParser(
        description="Calibrated condensation sweep using sigma_att=c_att*s_global and sigma_coh=c_coh*s_local."
    )
    p.add_argument("--mode", choices=["toy", "pbx", "both"], default="both")
    p.add_argument(
        "--output-dir",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/calibrated_force_band_sweep",
    )
    p.add_argument("--att-mults", default="0.3,0.5,0.7")
    p.add_argument("--coh-mults", default="0.1,0.3,0.4,0.5,0.6")
    p.add_argument("--n-iter-toy", type=int, default=300)
    p.add_argument("--n-iter-pbx", type=int, default=500)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument(
        "--input",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_pairwise_bin4_perm500/pairwise_raw_vectors.csv",
    )
    p.add_argument("--input-type", choices=["auto", "multiclass", "pairwise"], default="pairwise")
    p.add_argument(
        "--x0-path",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_raw_condensation_iter1000_ranked_umap/x0_init.npz",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        att_mults = [0.3, 0.5]
        coh_mults = [0.1, 0.5]
        args.n_iter_toy = min(args.n_iter_toy, 120)
        args.n_iter_pbx = min(args.n_iter_pbx, 150)
        args.save_every = min(args.save_every, 10)
        args.top_k = min(args.top_k, 2)
        args.log_every = min(args.log_every, 10)
    else:
        att_mults = [float(x) for x in args.att_mults.split(",") if x.strip()]
        coh_mults = [float(x) for x in args.coh_mults.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": args.mode,
        "att_mults": att_mults,
        "coh_mults": coh_mults,
        "k_attract": int(args.k_attract),
        "n_iter_toy": int(args.n_iter_toy),
        "n_iter_pbx": int(args.n_iter_pbx),
        "save_every": int(args.save_every),
        "top_k": int(args.top_k),
        "smoke": bool(args.smoke),
    }
    (output_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.mode in {"toy", "both"}:
        _run_toy_sweep(
            output_dir=output_dir / "toy",
            att_mults=att_mults,
            coh_mults=coh_mults,
            n_iter=args.n_iter_toy,
            k_attract=args.k_attract,
            seed=args.seed,
        )
    if args.mode in {"pbx", "both"}:
        _run_pbx_sweep(
            output_dir=output_dir / "pbx",
            input_path=args.input,
            input_type=args.input_type,
            x0_path=args.x0_path,
            att_mults=att_mults,
            coh_mults=coh_mults,
            n_iter=args.n_iter_pbx,
            save_every=args.save_every,
            top_k=args.top_k,
            log_every=args.log_every,
            k_attract=args.k_attract,
        )


def _run_toy_sweep(
    *,
    output_dir: Path,
    att_mults: list[float],
    coh_mults: list[float],
    n_iter: int,
    k_attract: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "stable_bundles": make_stable_bundles(n_per_cluster=40, n_time=10, random_seed=seed),
        "crossing_bundles": make_crossing_bundles(n_per_cluster=40, n_time=15, random_seed=seed),
    }
    rows: list[dict[str, Any]] = []
    for dataset_name, ds in datasets.items():
        for att_mult, coh_mult in product(att_mults, coh_mults):
            tag = _condition_tag(att_mult, coh_mult)
            run_dir = output_dir / dataset_name / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = TemporalRunConfig(
                sigma_frac=float(att_mult),
                temporal_cohere_bandwidth_mult=float(coh_mult),
                k_attract=int(k_attract),
                n_iter=int(n_iter),
                lr=1e-4,
                delta=3,
                fidelity_strength_mult=0.25,
                repulsion_strength_mult=0.005,
                stretch_strength_mult=0.04,
                bend_strength_mult=0.04,
                epsilon_void=0.014,
            )
            result = run_temporal(ds, cfg, save_snapshots=False, verbose=False)
            result.metrics_df.to_csv(run_dir / "metrics_history.csv", index=False)
            plot_temporal_run(result, run_dir)
            rows.append(
                {
                    "dataset": dataset_name,
                    "condition": tag,
                    "att_mult": float(att_mult),
                    "coh_mult": float(coh_mult),
                    "sep_ratio_final": float(result.final_metrics["sep_ratio_mean"]),
                    "coherence_selectivity_final": float(result.final_metrics["coherence_selectivity"]),
                    "collapse_score": float(result.collapse_score),
                    "within_bundle_spread_ratio": float(
                        result.final_metrics.get("within_bundle_spread_ratio", np.nan)
                    ),
                    "local_radius_ratio_p95": float(
                        result.final_metrics.get("local_radius_ratio_p95", np.nan)
                    ),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["dataset", "att_mult", "coh_mult"]).reset_index(drop=True)
    summary.to_csv(output_dir / "toy_summary.csv", index=False)
    _plot_toy_summary(summary, output_dir / "plot_toy_summary.png")


def _run_pbx_sweep(
    *,
    output_dir: Path,
    input_path: str,
    input_type: str,
    x0_path: str,
    att_mults: list[float],
    coh_mults: list[float],
    n_iter: int,
    save_every: int,
    top_k: int,
    log_every: int,
    k_attract: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_cosmology_data(input_path, input_type)
    schema.validate(data, allow_feature_nans=bool(np.isnan(data.features[data.mask]).any()))

    x0_payload = np.load(x0_path)
    x0 = np.asarray(x0_payload["x0"], dtype=float)
    if x0.shape != (data.features.shape[0], data.features.shape[1], 2):
        raise ValueError(
            f"x0 shape mismatch: expected {(data.features.shape[0], data.features.shape[1], 2)}, found {tuple(x0.shape)}"
        )
    refs = estimate_geometry_refs(x0, data.mask, k_local=5)
    (output_dir / "geometry_refs.json").write_text(
        json.dumps(
            {
                "s_local": float(refs.s_local),
                "s_step": float(refs.s_step),
                "s_bend": float(refs.s_bend),
                "s_global": float(refs.s_global),
            },
            indent=2,
        )
    )

    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(data.labels)}
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    rows: list[dict[str, Any]] = []
    for att_mult, coh_mult in product(att_mults, coh_mults):
        tag = _condition_tag(att_mult, coh_mult)
        run_dir = output_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = CondensationConfig(
            sigma=0.5,
            sigma_coh=None,
            attract_bandwidth_mult=float(att_mult),
            temporal_cohere_bandwidth_mult=float(coh_mult),
            temporal_cohere_mode="computed",
            delta=3,
            epsilon_r=0.005,
            lambda_stretch=0.04,
            lambda_bend=0.04,
            fidelity_init_strength=0.25,
            fidelity_half_life=_gamma_from_half_life_iters(70.0),
            epsilon_void=0.014,
            k_attract=int(k_attract),
            lr=1e-4,
            solver_momentum=0.9,
            solver_max_iter=int(n_iter),
        )
        result = run_condensation(
            x0=x0,
            mask=data.mask,
            config=cfg,
            stopping=stopping,
            log_every=log_every,
            save_every=save_every,
            verbose=True,
        )
        row = _save_run_bundle(
            run_dir=run_dir,
            name=tag,
            description=f"c_att={att_mult:.2f}, c_coh={coh_mult:.2f}",
            data=data,
            x0=x0,
            result=result,
            color_map=color_map,
            top_k=top_k,
            objective="balanced",
        )
        row.update(
            {
                "att_mult": float(att_mult),
                "coh_mult": float(coh_mult),
                "sigma_att_resolved": float(att_mult * refs.s_global),
                "sigma_coh_resolved": float(coh_mult * refs.s_local),
                "k_attract": int(k_attract),
            }
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["att_mult", "coh_mult"]).reset_index(drop=True)
    summary.to_csv(output_dir / "pbx_summary.csv", index=False)
    _plot_pbx_summary(summary, output_dir / "plot_pbx_summary.png")


def _save_run_bundle(
    *,
    run_dir: Path,
    name: str,
    description: str,
    data: schema.CosmologyData,
    x0: np.ndarray,
    result: Any,
    color_map: dict[str, str],
    top_k: int,
    objective: str,
) -> dict[str, Any]:
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
    np.savez(run_dir / "condensed_positions.npz", **payload)

    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    fig, _ = plotting.plot_trajectories(
        result.positions, data.mask, data.time_values, labels=data.labels, color_map=color_map, title=f"{name} | final"
    )
    fig.savefig(run_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_trajectories(
        x0, data.mask, data.time_values, labels=data.labels, color_map=color_map, title=f"{name} | init"
    )
    fig.savefig(run_dir / "plot_trajectories_init.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        result.positions, data.mask, data.time_values, labels=data.labels, color_map=color_map, title=f"{name} | stacked 3D"
    )
    fig.savefig(run_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ranking = pd.DataFrame()
    if result.position_history is not None:
        ranking = _rank_and_render_candidates(
            output_dir=run_dir,
            metrics_df=metrics_df,
            position_history=result.position_history,
            snapshot_iters=list(result.snapshot_iters),
            final_positions=result.positions,
            final_iter=max(0, result.n_iter - 1),
            mask=data.mask,
            labels=data.labels,
            time_values=data.time_values,
            color_map=color_map,
            objective=objective,
            top_k=top_k,
            title_prefix=name,
        )

    summary = {
        "name": name,
        "description": description,
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "final_energy_total": _last_metric(metrics_df, "energy_total"),
        "final_disp_rms_rel": _last_metric(metrics_df, "disp_rms_rel"),
        "final_disp_max_rel": _last_metric(metrics_df, "disp_max_rel"),
        "final_coherence_mean": _last_metric(metrics_df, "coherence_mean"),
        "final_gated_active_frac": _last_metric(metrics_df, "gated_active_frac"),
        "top_geometry_score": _float_or_none(ranking.iloc[0]["geometry_score"]) if not ranking.empty else None,
        "top_iter": int(ranking.iloc[0]["iter"]) if not ranking.empty else None,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


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
    title_prefix: str,
) -> pd.DataFrame:
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
        config_payload={"geometry_objective": objective, "top_k": top_k, "n_saved_iterations": len(iters)},
    )
    plot_iteration_scores(scores, ranking_dir / "plot_iteration_scores.png", title=f"{title_prefix} | geometry ranking")

    selected_root = output_dir / "selected_iterations"
    selected_root.mkdir(parents=True, exist_ok=True)
    for _, row in scores.head(top_k).iterrows():
        iter_idx = int(row["iter"])
        snapshot_index = int(row["snapshot_index"])
        candidate_dir = selected_root / f"iter_{iter_idx:04d}_rank_{int(row['rank']):02d}"
        render_selected_iteration_bundle(
            positions=history[snapshot_index],
            mask=mask,
            time_values=time_values,
            labels=labels,
            color_map=color_map,
            output_dir=candidate_dir,
            title_prefix=title_prefix,
            snapshot_iter=iter_idx,
            metadata={
                "iter": iter_idx,
                "rank": int(row["rank"]),
                "geometry_score": float(row["geometry_score"]),
                "selection_score": float(row["geometry_score"]),
                "geometry_objective": objective,
            },
        )
    scores.head(top_k).to_csv(selected_root / "top_k_candidates.csv", index=False)
    return scores


def _plot_toy_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    metrics = [
        ("sep_ratio_final", "Final separation ratio"),
        ("coherence_selectivity_final", "Final coherence selectivity"),
        ("collapse_score", "Collapse score"),
        ("local_radius_ratio_p95", "Local radius ratio p95"),
    ]
    cross = summary[summary["dataset"] == "crossing_bundles"]
    for ax, (metric, title) in zip(axes, metrics):
        pivot = cross.pivot(index="coh_mult", columns="att_mult", values=metric).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"crossing_bundles | {title}", fontsize=9)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel("c_att", fontsize=8)
        ax.set_ylabel("c_coh", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_pbx_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    metrics = [
        ("top_geometry_score", "Top geometry score"),
        ("final_disp_rms_rel", "Final RMS displacement"),
        ("final_coherence_mean", "Final coherence mean"),
        ("final_gated_active_frac", "Final gated active frac"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        pivot = summary.pivot(index="coh_mult", columns="att_mult", values=metric).sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel("c_att", fontsize=8)
        ax.set_ylabel("c_coh", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _condition_tag(att_mult: float, coh_mult: float) -> str:
    return f"catt_{str(att_mult).replace('.', 'p')}_ccoh_{str(coh_mult).replace('.', 'p')}"


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
    raise ValueError(f"Could not infer input type for {input_path}.")


def _last_metric(metrics_df: pd.DataFrame, col: str) -> float | None:
    if col not in metrics_df.columns or metrics_df.empty:
        return None
    return _float_or_none(metrics_df.iloc[-1][col])


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    main()
