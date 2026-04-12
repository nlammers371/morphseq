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

from analyze.trajectory_condensation import CondensationConfig, force_snapshot, force_target_table
from common import condensation_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit per-force gradient balance on selected PBX IDs.")
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--variant", choices=["shrunk", "raw"], default="raw")
    parser.add_argument("--baseline-audit-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--time-bin-center", type=float, default=70.0)
    parser.add_argument("--track-top-k", type=int, default=8)
    parser.add_argument("--source", choices=["x0", "final"], default="x0")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--compare-robust3", action="store_true")
    parser.add_argument("--robust3-outlier-strength", type=float, default=2.0)
    parser.add_argument("--robust3-epsilon-r", type=float, default=0.005)
    return parser.parse_args()


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


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


def elastic_outlier_config(
    strength: float,
    mix: float,
    outlier_strength: float,
    *,
    epsilon_r: float = 0.005,
    cutoff_mode: str = "quantile",
    cutoff_value: float = 0.99,
) -> CondensationConfig:
    cfg = CondensationConfig(
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
        elastic_kernel="quadratic",
        outlier_strength=float(outlier_strength),
        outlier_cutoff_mode=str(cutoff_mode),
        outlier_cutoff_value=float(cutoff_value),
    )
    return cfg


def main() -> None:
    args = parse_args()
    trajectory_dir = args.trajectory_dir or condensation_results_dir(
        variant=args.variant,
        include_wik_ab=False,
        bin_width=4.0,
        n_permutations=500,
    )
    baseline_audit_dir = args.baseline_audit_dir or trajectory_dir / "force_diagnostics"
    output_dir = args.output_dir or trajectory_dir / "force_diagnostics" / f"force_vector_audit_{args.source}"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(trajectory_dir / "condensed_positions.npz", allow_pickle=True)
    x0 = np.asarray(payload["x0"], dtype=float)
    positions = np.asarray(payload["positions"], dtype=float)
    mask = np.asarray(payload["mask"], dtype=bool)
    time_values = np.asarray(payload["time_values"], dtype=float)
    ids = payload["embryo_ids"].astype(str)
    labels = payload["labels"].astype(str)
    state_positions = x0 if args.source == "x0" else positions

    audit_df = pd.read_csv(baseline_audit_dir / "outlier_audit_focus_60_80hpf.csv")
    audit_df = audit_df[np.isclose(audit_df["time_bin_center"], float(args.time_bin_center))].copy()
    tracked_keys = []
    for _, row in audit_df.head(int(args.track_top_k)).iterrows():
        key = (str(row["embryo_id"]), float(row["time_bin_center"]))
        if key not in tracked_keys:
            tracked_keys.append(key)

    configs: list[tuple[str, CondensationConfig]] = [("baseline", baseline_config())]
    if args.compare_robust3:
        configs.append(
            (
                "robust3_outlier",
                elastic_outlier_config(
                    0.5,
                    0.5,
                    float(args.robust3_outlier_strength),
                    epsilon_r=float(args.robust3_epsilon_r),
                    cutoff_mode="robust",
                    cutoff_value=3.0,
                ),
            )
        )

    all_rows = []
    for name, config in configs:
        snap = force_snapshot(
            positions=state_positions,
            x0=x0,
            mask=mask,
            config=config,
            iteration=int(args.iteration),
        )
        table = force_target_table(
            snapshot=snap,
            positions=state_positions,
            mask=mask,
            time_values=time_values,
            ids=ids,
            labels=labels,
            config=config,
            targets=tracked_keys,
        )
        table.insert(0, "run_name", name)
        table.to_csv(output_dir / f"{name}_force_vectors.csv", index=False)
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_dir / "force_vectors_combined.csv", index=False)
    summary = summarize_force_vectors(combined)
    summary.to_csv(output_dir / "force_vectors_summary.csv", index=False)
    print(output_dir)


def summarize_force_vectors(df: pd.DataFrame) -> pd.DataFrame:
    force_names = ["attract", "repel", "void", "elastic", "fidelity", "scale", "outlier", "total"]
    rows: list[dict[str, object]] = []
    for run_name, part in df.groupby("run_name"):
        row: dict[str, object] = {"run_name": str(run_name), "n_targets": int(len(part))}
        for force_name in force_names:
            col = f"{force_name}_grad_norm"
            if col not in part.columns:
                continue
            row[f"{force_name}_grad_norm_median"] = float(part[col].median())
            row[f"{force_name}_grad_norm_max"] = float(part[col].max())
            row[f"{force_name}_step_steady_median"] = float(part[f"{force_name}_step_steady"].median())
            row[f"{force_name}_step_steady_max"] = float(part[f"{force_name}_step_steady"].max())
            row[f"{force_name}_radial_inward_median"] = float(part[f"{force_name}_grad_radial_inward"].median())
            row[f"{force_name}_step_steady_inward_median"] = float(part[f"{force_name}_step_steady_inward"].median())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
