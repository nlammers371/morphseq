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

from analyze.trajectory_condensation import viz as tc_viz
from analyze.trajectory_condensation.condensation import (
    CondensationConfig,
    StoppingConfig,
    describe_force_balance,
    run_condensation,
)

from common import GENOTYPE_COLORS, condensation_results_dir


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PBX condensation outliers and elasticity thresholds.")
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--include-wik-ab", action="store_true")
    parser.add_argument("--variant", choices=["shrunk", "raw"], default="raw")
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--elastic-sweep", action="store_true")
    parser.add_argument("--elastic-strengths", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["quadratic", "ratio_hinge", "quadratic_plus_outlier"],
        default=["quadratic", "ratio_hinge", "quadratic_plus_outlier"],
    )
    parser.add_argument("--outlier-sweep", action="store_true")
    parser.add_argument("--outlier-strengths", nargs="+", type=float, default=[4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--elastic-mix", type=float, default=0.25)
    parser.add_argument("--outlier-strength", type=float, default=16.0)
    parser.add_argument("--outlier-cutoff-preset", type=str, default="robust3")
    parser.add_argument("--epsilon-r", type=float, default=5e-4)
    parser.add_argument("--solver-max-iter-override", type=int, default=None)
    parser.add_argument("--track-top-k", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def baseline_config() -> CondensationConfig:
    return CondensationConfig(
        sigma=0.5,
        temporal_cohere_window=3,
        epsilon_r=0.005,
        lambda_stretch=0.04,
        lambda_bend=0.04,
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        epsilon_void=0.014,
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=500,
    )


def elastic_config(strength: float, mix: float, kernel: str, *, epsilon_r: float = 0.005) -> CondensationConfig:
    return CondensationConfig(
        sigma=0.5,
        temporal_cohere_window=3,
        epsilon_r=float(epsilon_r),
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        epsilon_void=0.014,
        attract_k=20,
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=500,
        elastic_strength=float(strength),
        elastic_mix=float(mix),
        elastic_kernel=str(kernel),
    )


def elastic_outlier_config(
    strength: float,
    mix: float,
    outlier_strength: float,
    *,
    epsilon_r: float = 0.005,
    cutoff_mode: str = "quantile",
    cutoff_value: float = 0.99,
) -> CondensationConfig:
    cfg = elastic_config(strength, mix, "quadratic", epsilon_r=epsilon_r)
    cfg.outlier_strength = float(outlier_strength)
    cfg.epsilon_outlier = float(outlier_strength)
    cfg.outlier_cutoff_mode = str(cutoff_mode)
    cfg.slice_outlier_cutoff_mode = str(cutoff_mode)
    cfg.outlier_cutoff_value = float(cutoff_value)
    cfg.slice_outlier_cutoff_value = float(cutoff_value)
    return cfg


def main() -> None:
    args = parse_args()
    trajectory_dir = args.trajectory_dir or condensation_results_dir(
        variant=args.variant,
        include_wik_ab=bool(args.include_wik_ab),
        bin_width=float(args.bin_width),
        n_permutations=int(args.n_permutations),
    )
    output_dir = args.output_dir or trajectory_dir / "force_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(trajectory_dir / "condensed_positions.npz", allow_pickle=True)
    manifest = json.loads((trajectory_dir / "condensation_manifest.json").read_text())
    pairwise_df = pd.read_csv(manifest["input"])

    x0 = np.asarray(payload["x0"], dtype=float)
    mask = np.asarray(payload["mask"], dtype=bool)
    time_values = np.asarray(payload["time_values"], dtype=float)
    embryo_ids = payload["embryo_ids"].astype(str)
    labels = payload["labels"].astype(str)
    baseline_positions = np.asarray(payload["positions"], dtype=float)

    baseline_audit = build_outlier_audit(
        positions=baseline_positions,
        x0=x0,
        mask=mask,
        time_values=time_values,
        embryo_ids=embryo_ids,
        labels=labels,
        pairwise_df=pairwise_df,
    )
    baseline_audit.to_csv(output_dir / "outlier_audit.csv", index=False)
    focus_df = focus_window(baseline_audit)
    focus_df.to_csv(output_dir / "outlier_audit_focus_60_80hpf.csv", index=False)
    baseline_summary = summarize_outliers(baseline_audit)
    (output_dir / "outlier_summary.json").write_text(json.dumps(baseline_summary, indent=2))
    cutoff_mode, cutoff_value = parse_cutoff_preset(str(args.outlier_cutoff_preset))

    if args.elastic_sweep:
        strengths = [float(x) for x in args.elastic_strengths]
        if args.smoke:
            strengths = strengths[:2]
        tracked_keys = select_tracked_keys(focus_df, top_k=int(args.track_top_k))
        sweep_results = run_elastic_sweep(
            strengths=strengths,
            methods=[str(x) for x in args.methods],
            mix=float(args.elastic_mix),
            outlier_strength=float(args.outlier_strength),
            epsilon_r=float(args.epsilon_r),
            outlier_cutoff_mode=cutoff_mode,
            outlier_cutoff_value=cutoff_value,
            x0=x0,
            mask=mask,
            time_values=time_values,
            embryo_ids=embryo_ids,
            labels=labels,
            pairwise_df=pairwise_df,
            baseline_positions=baseline_positions,
            baseline_audit=baseline_audit,
            output_dir=output_dir,
            smoke=bool(args.smoke),
            solver_max_iter_override=args.solver_max_iter_override,
            tracked_keys=tracked_keys,
        )
        (output_dir / "elastic_sweep_summary.json").write_text(json.dumps(sweep_results["summary"], indent=2))
        sweep_results["summary_df"].to_csv(output_dir / "elastic_sweep_summary.csv", index=False)
        sweep_results["tracked_df"].to_csv(output_dir / "tracked_embryo_sweep.csv", index=False)
        print(output_dir)
        return

    if args.outlier_sweep:
        strengths = [float(x) for x in args.outlier_strengths]
        if args.smoke:
            strengths = strengths[:2]
        tracked_keys = select_tracked_keys(focus_df, top_k=int(args.track_top_k))
        sweep_results = run_outlier_sweep(
            strengths=strengths,
            elastic_strength=float(args.elastic_strengths[0]),
            mix=float(args.elastic_mix),
            epsilon_r=float(args.epsilon_r),
            outlier_cutoff_mode=cutoff_mode,
            outlier_cutoff_value=cutoff_value,
            x0=x0,
            mask=mask,
            time_values=time_values,
            embryo_ids=embryo_ids,
            labels=labels,
            pairwise_df=pairwise_df,
            baseline_positions=baseline_positions,
            baseline_audit=baseline_audit,
            output_dir=output_dir,
            smoke=bool(args.smoke),
            solver_max_iter_override=args.solver_max_iter_override,
            tracked_keys=tracked_keys,
        )
        (output_dir / "outlier_sweep_summary.json").write_text(json.dumps(sweep_results["summary"], indent=2))
        sweep_results["summary_df"].to_csv(output_dir / "outlier_sweep_summary.csv", index=False)
        sweep_results["tracked_df"].to_csv(output_dir / "tracked_embryo_outlier_sweep.csv", index=False)
        print(output_dir)
        return

    print(output_dir)


def run_elastic_sweep(
    *,
    strengths: list[float],
    methods: list[str],
    mix: float,
    outlier_strength: float,
    epsilon_r: float,
    outlier_cutoff_mode: str,
    outlier_cutoff_value: float,
    x0: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    embryo_ids: np.ndarray,
    labels: np.ndarray,
    pairwise_df: pd.DataFrame,
    baseline_positions: np.ndarray,
    baseline_audit: pd.DataFrame,
    output_dir: Path,
    smoke: bool,
    solver_max_iter_override: int | None,
    tracked_keys: list[tuple[str, float]],
) -> dict[str, object]:
    run_dir = output_dir / "elastic_strength_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    method_builders = {
        "quadratic": lambda strength: elastic_config(strength, mix, "quadratic", epsilon_r=epsilon_r),
        "ratio_hinge": lambda strength: elastic_config(strength, mix, "ratio_hinge", epsilon_r=epsilon_r),
        "quadratic_plus_outlier": lambda strength: elastic_outlier_config(
            strength,
            mix,
            outlier_strength,
            epsilon_r=epsilon_r,
            cutoff_mode=outlier_cutoff_mode,
            cutoff_value=outlier_cutoff_value,
        ),
    }
    method_defs = [(name, method_builders[name]) for name in methods]

    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(labels)}
    baseline_run = tc_viz.RunDescriptor(
        positions=baseline_positions,
        mask=mask,
        time_values=time_values,
        labels=labels,
        embryo_ids=embryo_ids,
        color_map=color_map,
        title="baseline",
        x0=x0,
    )

    compare_grid: list[list[tc_viz.RunDescriptor]] = []
    summary_rows = [summarize_threshold_row("baseline", baseline_audit)]
    tracked_rows = build_tracked_rows("baseline", baseline_audit, tracked_keys)

    for method_name, config_builder in method_defs:
        row_runs: list[tc_viz.RunDescriptor] = []
        for strength in strengths:
            tag = f"{method_name}_elastic_{format_strength(strength)}"
            this_dir = run_dir / tag
            this_dir.mkdir(parents=True, exist_ok=True)

            config = config_builder(strength)
            if smoke:
                config.solver_max_iter = 50
                config.max_iter = 50
            elif solver_max_iter_override is not None:
                config.solver_max_iter = int(solver_max_iter_override)
                config.max_iter = int(solver_max_iter_override)

            balance = describe_force_balance(x0, mask, config)
            (this_dir / "force_balance.json").write_text(json.dumps(balance, indent=2))

            result = run_condensation(
                x0=x0,
                mask=mask,
                config=config,
                stopping=StoppingConfig(
                    disp_max_rel_threshold=None,
                    disp_rms_rel_threshold=None,
                    energy_change_rel_threshold=None,
                    coherence_change_rel_threshold=None,
                ),
                log_every=max(1, config.max_iter // 10),
                save_every=None,
                verbose=True,
            )

            np.savez(
                this_dir / "condensed_positions.npz",
                positions=result.positions,
                x0=x0,
                mask=mask,
                time_values=time_values,
                embryo_ids=embryo_ids,
                labels=labels,
            )
            audit_df = build_outlier_audit(
                positions=result.positions,
                x0=x0,
                mask=mask,
                time_values=time_values,
                embryo_ids=embryo_ids,
                labels=labels,
                pairwise_df=pairwise_df,
            )
            audit_df.to_csv(this_dir / "outlier_audit.csv", index=False)
            descriptor = tc_viz.RunDescriptor(
                positions=result.positions,
                mask=mask,
                time_values=time_values,
                labels=labels,
                embryo_ids=embryo_ids,
                color_map=color_map,
                title=f"{method_name} | {strength:g}",
                x0=x0,
            )
            row_runs.append(descriptor)
            run_name = f"{method_name} | elastic={strength:g}"
            summary_rows.append(summarize_threshold_row(run_name, audit_df))
            tracked_rows.extend(build_tracked_rows(run_name, audit_df, tracked_keys))
            tc_viz.render_run(descriptor, this_dir, title_prefix=f"PBX {tag}", skip_animations=True)
        compare_grid.append(row_runs)

    summary_df = pd.DataFrame(summary_rows)
    tracked_df = pd.DataFrame(tracked_rows)
    tc_viz.compare_run_grid(
        compare_grid,
        mode="trajectories",
        output_path=run_dir / "compare_trajectories.png",
    )
    tc_viz.compare_run_grid(
        compare_grid,
        mode="stacked_3d",
        output_path=run_dir / "compare_stacked_3d.png",
    )
    plot_sweep_thresholds(summary_df, run_dir / "elastic_sweep_thresholds.png")

    threshold_hit = summary_df[summary_df["focus_count_z_gt_10"] <= 0]
    first_hit = None if threshold_hit.empty else str(threshold_hit.iloc[0]["run_name"])
    summary = {
        "methods": [name for name, _ in method_defs],
        "strengths": strengths,
        "elastic_mix": mix,
        "epsilon_r": epsilon_r,
        "outlier_strength": outlier_strength,
        "outlier_cutoff_mode": outlier_cutoff_mode,
        "outlier_cutoff_value": outlier_cutoff_value,
        "tracked_keys": [{"embryo_id": eid, "time_bin_center": t} for eid, t in tracked_keys],
        "first_focus_z10_clear": first_hit,
        "rows": summary_df.to_dict(orient="records"),
    }
    return {"summary": summary, "summary_df": summary_df, "tracked_df": tracked_df}


def run_outlier_sweep(
    *,
    strengths: list[float],
    elastic_strength: float,
    mix: float,
    epsilon_r: float,
    outlier_cutoff_mode: str,
    outlier_cutoff_value: float,
    x0: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    embryo_ids: np.ndarray,
    labels: np.ndarray,
    pairwise_df: pd.DataFrame,
    baseline_positions: np.ndarray,
    baseline_audit: pd.DataFrame,
    output_dir: Path,
    smoke: bool,
    solver_max_iter_override: int | None,
    tracked_keys: list[tuple[str, float]],
) -> dict[str, object]:
    run_dir = output_dir / "outlier_strength_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    color_map = {g: GENOTYPE_COLORS.get(g, "#555555") for g in np.unique(labels)}
    compare_runs: list[tc_viz.RunDescriptor] = []
    summary_rows = [summarize_threshold_row("baseline", baseline_audit)]
    tracked_rows = build_tracked_rows("baseline", baseline_audit, tracked_keys)

    for strength in strengths:
        tag = f"quadratic_plus_outlier_outlier_{format_strength(strength)}"
        this_dir = run_dir / tag
        this_dir.mkdir(parents=True, exist_ok=True)

        config = elastic_outlier_config(
            elastic_strength,
            mix,
            strength,
            epsilon_r=epsilon_r,
            cutoff_mode=outlier_cutoff_mode,
            cutoff_value=outlier_cutoff_value,
        )
        if smoke:
            config.solver_max_iter = 50
            config.max_iter = 50
        elif solver_max_iter_override is not None:
            config.solver_max_iter = int(solver_max_iter_override)
            config.max_iter = int(solver_max_iter_override)

        balance = describe_force_balance(x0, mask, config)
        (this_dir / "force_balance.json").write_text(json.dumps(balance, indent=2))

        result = run_condensation(
            x0=x0,
            mask=mask,
            config=config,
            stopping=StoppingConfig(
                disp_max_rel_threshold=None,
                disp_rms_rel_threshold=None,
                energy_change_rel_threshold=None,
                coherence_change_rel_threshold=None,
            ),
            log_every=max(1, config.max_iter // 10),
            save_every=None,
            verbose=True,
        )

        np.savez(
            this_dir / "condensed_positions.npz",
            positions=result.positions,
            x0=x0,
            mask=mask,
            time_values=time_values,
            embryo_ids=embryo_ids,
            labels=labels,
        )
        audit_df = build_outlier_audit(
            positions=result.positions,
            x0=x0,
            mask=mask,
            time_values=time_values,
            embryo_ids=embryo_ids,
            labels=labels,
            pairwise_df=pairwise_df,
        )
        audit_df.to_csv(this_dir / "outlier_audit.csv", index=False)
        descriptor = tc_viz.RunDescriptor(
            positions=result.positions,
            mask=mask,
            time_values=time_values,
            labels=labels,
            embryo_ids=embryo_ids,
            color_map=color_map,
            title=f"quadratic+outlier | out={strength:g}",
            x0=x0,
        )
        compare_runs.append(descriptor)
        run_name = f"quadratic+outlier | outlier={strength:g}"
        summary_rows.append(summarize_threshold_row(run_name, audit_df))
        tracked_rows.extend(build_tracked_rows(run_name, audit_df, tracked_keys))
        tc_viz.render_run(descriptor, this_dir, title_prefix=f"PBX {tag}", skip_animations=True)

    summary_df = pd.DataFrame(summary_rows)
    tracked_df = pd.DataFrame(tracked_rows)
    tc_viz.compare_runs(
        compare_runs,
        mode="trajectories",
        output_path=run_dir / "compare_trajectories.png",
    )
    tc_viz.compare_runs(
        compare_runs,
        mode="stacked_3d",
        output_path=run_dir / "compare_stacked_3d.png",
    )
    plot_sweep_thresholds(summary_df, run_dir / "outlier_sweep_thresholds.png")

    threshold_hit = summary_df[summary_df["focus_count_z_gt_10"] <= 0]
    first_hit = None if threshold_hit.empty else str(threshold_hit.iloc[0]["run_name"])
    summary = {
        "method": "quadratic_plus_outlier",
        "elastic_strength": elastic_strength,
        "elastic_mix": mix,
        "epsilon_r": epsilon_r,
        "outlier_strengths": strengths,
        "outlier_cutoff_mode": outlier_cutoff_mode,
        "outlier_cutoff_value": outlier_cutoff_value,
        "tracked_keys": [{"embryo_id": eid, "time_bin_center": t} for eid, t in tracked_keys],
        "first_focus_z10_clear": first_hit,
        "rows": summary_df.to_dict(orient="records"),
    }
    return {"summary": summary, "summary_df": summary_df, "tracked_df": tracked_df}


def build_outlier_audit(
    *,
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    embryo_ids: np.ndarray,
    labels: np.ndarray,
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | None]] = []
    metadata_cols = ["embryo_id", "time_bin_center", "experiment_id", "time_bin"]
    metadata = pairwise_df[metadata_cols].drop_duplicates(subset=["embryo_id", "time_bin_center"]).copy()
    for t_idx, time_value in enumerate(time_values):
        obs = np.flatnonzero(mask[:, t_idx])
        if obs.size == 0:
            continue
        coords = positions[obs, t_idx, :]
        init_coords = x0[obs, t_idx, :]
        centroid = coords.mean(axis=0)
        dist = np.linalg.norm(coords - centroid, axis=1)
        median = float(np.median(dist))
        mad = float(np.median(np.abs(dist - median)))
        z = (dist - median) / max(1.4826 * mad, 1e-9)
        init_disp = np.linalg.norm(coords - init_coords, axis=1)
        for offset, embryo_idx in enumerate(obs):
            step_length = np.nan
            bend = np.nan
            if t_idx > 0 and mask[embryo_idx, t_idx - 1]:
                step_length = float(np.linalg.norm(positions[embryo_idx, t_idx, :] - positions[embryo_idx, t_idx - 1, :]))
            if 0 < t_idx < len(time_values) - 1 and mask[embryo_idx, t_idx - 1] and mask[embryo_idx, t_idx + 1]:
                bend = float(
                    np.linalg.norm(
                        positions[embryo_idx, t_idx + 1, :]
                        - 2.0 * positions[embryo_idx, t_idx, :]
                        + positions[embryo_idx, t_idx - 1, :]
                    )
                )
            rows.append(
                {
                    "embryo_id": str(embryo_ids[embryo_idx]),
                    "genotype": str(labels[embryo_idx]),
                    "time_bin_center": float(time_value),
                    "distance_from_slice_centroid": float(dist[offset]),
                    "distance_zscore_robust": float(z[offset]),
                    "init_to_final_displacement": float(init_disp[offset]),
                    "step_length": step_length,
                    "bend_magnitude": bend,
                }
            )
    audit_df = pd.DataFrame(rows)
    audit_df = audit_df.merge(metadata, on=["embryo_id", "time_bin_center"], how="left")
    audit_df["outlier_rank"] = audit_df["distance_zscore_robust"].rank(method="dense", ascending=False).astype(int)
    return audit_df.sort_values(["outlier_rank", "time_bin_center", "embryo_id"]).reset_index(drop=True)


def focus_window(audit_df: pd.DataFrame) -> pd.DataFrame:
    return audit_df[
        ((audit_df["time_bin_center"] >= 60.0) & (audit_df["time_bin_center"] <= 80.0))
        | np.isclose(audit_df["time_bin_center"], 80.0)
    ].copy()


def select_tracked_keys(focus_df: pd.DataFrame, top_k: int) -> list[tuple[str, float]]:
    keys = []
    for _, row in focus_df.head(top_k).iterrows():
        key = (str(row["embryo_id"]), float(row["time_bin_center"]))
        if key not in keys:
            keys.append(key)
    return keys


def build_tracked_rows(run_name: str, audit_df: pd.DataFrame, tracked_keys: list[tuple[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for embryo_id, time_value in tracked_keys:
        hit = audit_df[(audit_df["embryo_id"] == embryo_id) & np.isclose(audit_df["time_bin_center"], time_value)]
        if hit.empty:
            rows.append({"run_name": run_name, "embryo_id": embryo_id, "time_bin_center": time_value, "present": False})
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "run_name": run_name,
                "embryo_id": embryo_id,
                "time_bin_center": float(time_value),
                "present": True,
                "genotype": str(row["genotype"]),
                "experiment_id": str(row["experiment_id"]),
                "distance_zscore_robust": float(row["distance_zscore_robust"]),
                "distance_from_slice_centroid": float(row["distance_from_slice_centroid"]),
                "init_to_final_displacement": float(row["init_to_final_displacement"]),
                "step_length": float(row["step_length"]) if pd.notna(row["step_length"]) else np.nan,
                "bend_magnitude": float(row["bend_magnitude"]) if pd.notna(row["bend_magnitude"]) else np.nan,
            }
        )
    return rows


def summarize_outliers(audit_df: pd.DataFrame) -> dict[str, object]:
    genotype_summary = (
        audit_df.groupby("genotype")["distance_zscore_robust"]
        .agg(["count", "max", "median"])
        .sort_values("max", ascending=False)
        .reset_index()
    )
    focus = focus_window(audit_df)
    return {
        "top_20_outliers": audit_df.head(20).to_dict(orient="records"),
        "focus_60_80hpf_top_20": focus.head(20).to_dict(orient="records"),
        "by_genotype": genotype_summary.to_dict(orient="records"),
    }


def summarize_threshold_row(run_name: str, audit_df: pd.DataFrame) -> dict[str, object]:
    focus = focus_window(audit_df)
    return {
        "run_name": run_name,
        "max_z": float(audit_df["distance_zscore_robust"].max()),
        "focus_max_z": float(focus["distance_zscore_robust"].max()),
        "count_z_gt_10": int((audit_df["distance_zscore_robust"] > 10).sum()),
        "focus_count_z_gt_10": int((focus["distance_zscore_robust"] > 10).sum()),
        "focus_count_z_gt_8": int((focus["distance_zscore_robust"] > 8).sum()),
        "inj_ctrl_focus_max": float(focus.loc[focus["genotype"] == "inj_ctrl", "distance_zscore_robust"].max()),
        "double_focus_max": float(focus.loc[focus["genotype"] == "pbx1b_pbx4_crispant", "distance_zscore_robust"].max()),
    }


def plot_sweep_thresholds(summary_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plotted = summary_df[summary_df["run_name"] != "baseline"].copy()
    if plotted.empty:
        return
    x = np.arange(len(plotted))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(x, plotted["focus_max_z"], marker="o", lw=1.5)
    axes[0].set_xticks(x, plotted["run_name"], rotation=25, ha="right")
    axes[0].set_ylabel("Focus max robust z-score")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, plotted["focus_count_z_gt_10"], marker="o", lw=1.5, label="z > 10")
    axes[1].plot(x, plotted["focus_count_z_gt_8"], marker="o", lw=1.5, label="z > 8")
    axes[1].set_xticks(x, plotted["run_name"], rotation=25, ha="right")
    axes[1].set_ylabel("Focus outlier count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_strength(value: float) -> str:
    return str(value).replace(".", "p")


def parse_cutoff_preset(preset: str) -> tuple[str, float]:
    p = str(preset).strip().lower()
    if p.startswith("q"):
        return "quantile", float(p[1:]) / 100.0
    if p.startswith("robust"):
        suffix = p.replace("robust", "", 1)
        return "robust", float(suffix) if suffix else 3.0
    raise ValueError(f"Unsupported outlier cutoff preset: {preset!r}")


if __name__ == "__main__":
    main()
