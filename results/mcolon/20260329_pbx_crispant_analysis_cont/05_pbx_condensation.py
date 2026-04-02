"""
05_pbx_condensation.py
----------------------
Run trajectory condensation on PBX crispant vector data.

Pipeline:
  1. Load multiclass probability or pairwise margin vectors → CosmologyData
  2. Initialize 2D positions via AlignedUMAP (or PCA for fast smoke test)
  3. Run condensation with calibrated force defaults
  4. Save positions, metrics, and diagnostic figures

Calibrated defaults (from force_calibration_v1/FORCE_CALIBRATION_SUMMARY.md):
  lr=1e-4, n_iter=500
  fidelity_strength=0.25, fidelity_half_life_iters=70
  repulsion epsilon_r=0.005
  lambda_stretch=0.04, lambda_bend=0.04
  epsilon_void=0.014

Usage:
  # Smoke test (PCA init, 50 iters, ~1 min)
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/05_pbx_condensation.py \\
    --input results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
            phenotypic_positioning_multiclass_bin4_perm500/multiclass_probability_vectors.csv \\
    --output-dir /tmp/pbx_condensation_smoke --smoke

  # Full run (AlignedUMAP init, 500 iters)
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/05_pbx_condensation.py \\
    --input results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
            phenotypic_positioning_multiclass_bin4_perm500/multiclass_probability_vectors.csv \\
    --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
                 force_calibration_v1/pbx_condensation_v1
"""
from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Genotype color palette (colorblind-safe, consistent with project styling)
# ---------------------------------------------------------------------------

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",   # blue
    "wik_ab":                 "#808080",   # gray
    "pbx1b_crispant":         "#9467bd",   # purple
    "pbx4_crispant":          "#F7B267",   # amber
    "pbx1b_pbx4_crispant":    "#B2182B",   # crimson
}


def _gamma_from_half_life_iters(h: float) -> float:
    """gamma = 2^(-1/h) — internal per-iter fidelity retention."""
    return 2.0 ** (-1.0 / h)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PBX crispant trajectory condensation.")
    p.add_argument(
        "--input", required=True,
        help="Path to multiclass probability or pairwise margin vector CSV.",
    )
    p.add_argument(
        "--input-type", choices=["auto", "multiclass", "pairwise"], default="auto",
        help="Input schema. 'auto' infers from columns.",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write all outputs.",
    )
    p.add_argument(
        "--init", choices=["umap", "pca"], default="umap",
        help="Initialization method. 'pca' is faster for smoke tests.",
    )
    p.add_argument(
        "--n-iter", type=int, default=500,
        help="Number of condensation iterations.",
    )
    p.add_argument(
        "--save-every", type=int, default=25,
        help="Save position snapshots every N iters (for animation). 0 = disabled.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Fast smoke-test mode: PCA init, 50 iters, save-every=10.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.smoke:
        args.init = "pca"
        args.n_iter = 50
        args.save_every = 10

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading: {args.input}")
    data = _load_cosmology_data(args.input, args.input_type)
    schema.validate(data)

    N_e, T, K = data.features.shape
    print(f"  {N_e} embryos × {T} time bins × {K} features")
    print(f"  Observed entries: {data.mask.sum()} / {data.mask.size} "
          f"({100 * data.mask.mean():.1f}%)")
    print(f"  Time range: {data.time_values[0]:.0f}–{data.time_values[-1]:.0f} hpf")
    print(f"  Genotypes: {dict(zip(*np.unique(data.labels, return_counts=True)))}")

    # ------------------------------------------------------------------
    # 2. Initialize 2D embedding
    # ------------------------------------------------------------------
    print(f"\nInitializing with {args.init.upper()} (seed={args.seed})...")
    if args.init == "umap":
        x0 = init_embedding.aligned_umap_init(
            data.features, data.mask,
            n_neighbors=15,
            min_dist=0.1,
            alignment_regularisation=1e-2,
            alignment_window_size=3,
            random_state=args.seed,
        )
    else:
        x0 = init_embedding.pca_init(data.features, data.mask, random_state=args.seed)

    print(f"  x0 shape: {x0.shape}  (NaN where not observed)")

    # Save initialization so we can reload without re-running UMAP
    np.savez(output_dir / "x0_init.npz", x0=x0, time_values=data.time_values)
    print(f"  Saved: {output_dir / 'x0_init.npz'}")

    # ------------------------------------------------------------------
    # 3. Build condensation config with calibrated defaults
    # ------------------------------------------------------------------
    config = CondensationConfig(
        # Spatial / temporal
        sigma=0.5,
        delta=3,
        # Forces — calibrated on Y-benchmark at lr=1e-4, n_iter=500
        epsilon_r=0.005,                                           # repulsion
        lambda_stretch=0.04,                                       # elasticity stretch
        lambda_bend=0.04,                                          # elasticity bend
        fidelity_init_strength=0.25,                               # fidelity anchor strength
        fidelity_half_life=_gamma_from_half_life_iters(70.0),      # ≈ 0.99029
        epsilon_void=0.014,                                        # void proxy
        k_attract=20,
        # Optimization
        lr=1e-4,
        alpha=0.9,
        max_iter=args.n_iter,
    )

    # Disable early stopping so we always run the full n_iter budget
    # (forces need the full budget to accumulate)
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    # ------------------------------------------------------------------
    # 4. Run condensation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    # 5a. Positions
    pos_path = output_dir / "condensed_positions.npz"
    save_payload = {
        "positions": result.positions,
        "x0": x0,
        "mask": data.mask,
        "time_values": data.time_values,
        "embryo_ids": data.embryo_ids,
        "labels": data.labels,
    }
    if result.position_history is not None:
        save_payload["position_history"] = result.position_history
        save_payload["snapshot_iters"] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(pos_path, **save_payload)
    print(f"Saved positions: {pos_path}  shape={result.positions.shape}")

    # 5b. Metrics
    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(data.labels)}

    # 6a. Trajectory lines
    fig, _ = plotting.plot_trajectories(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX condensed trajectories ({args.init.upper()} init, {result.n_iter} iters)",
    )
    fig.savefig(output_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also plot init (x0) for comparison
    fig, _ = plotting.plot_trajectories(
        x0, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX initialization ({args.init.upper()})",
    )
    fig.savefig(output_dir / "plot_trajectories_init.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6b. Panel snapshots (6 time points spread across range)
    snap_indices = np.linspace(0, T - 1, min(6, T), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        snapshot_times=snapshot_times,
        title=f"PBX panels ({args.init.upper()} init, {result.n_iter} iters)",
    )
    fig.savefig(output_dir / "plot_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6c. Stacked 3D view
    fig, _ = plotting.plot_stacked_3d(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f"PBX stacked 3D ({args.init.upper()} init)",
    )
    fig.savefig(output_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6d. Convergence metrics
    _plot_metrics(metrics_df, output_dir / "plot_metrics.png",
                  title=f"{args.init.upper()} init — sigma={config.sigma} delta={config.delta} "
                        f"lr={config.lr:.0e} k={config.k_attract}")

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
            title=f"PBX condensation progress ({args.init.upper()} init)",
        )

    print(f"\nAll outputs saved to: {output_dir}")
    _print_final_summary(metrics_df)


def _plot_metrics(metrics_df: pd.DataFrame, output_path: Path, title: str = "") -> None:
    cols_labels = [
        ("energy_total",        "Total energy"),
        ("disp_rms_rel",        "RMS displacement (rel)"),
        ("energy_change_rel",   "Energy change (rel)"),
        ("coherence_change_rel","Coherence change (rel)"),
    ]
    available = [(c, l) for c, l in cols_labels if c in metrics_df.columns]
    n = len(available)
    if n == 0:
        return

    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (col, label) in zip(axes, available):
        ax.plot(metrics_df["iter"], metrics_df[col], lw=1.4)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.25)

    for ax in axes[n:]:
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
    for col in ["energy_total", "disp_rms_rel", "disp_max_rel",
                "energy_change_rel", "coherence_change_rel"]:
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
