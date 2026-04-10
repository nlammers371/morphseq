from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_subset_condensation_cache"
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

from analyze.trajectory_condensation import schema, viz as tc_viz
from analyze.trajectory_condensation.condensation import CondensationConfig, StoppingConfig, run_condensation

from common import GENOTYPE_COLORS


META_COLS = {"embryo_id", "time_bin_center", "genotype", "experiment_id", "time_bin"}


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run condensation from saved subset initializations.")
    parser.add_argument("--pairwise-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=["raw", "shrunk"], default="raw")
    parser.add_argument("--subset-root", type=Path, default=None)
    parser.add_argument("--subset-root-raw", type=Path, default=None)
    parser.add_argument("--subset-root-shrunk", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", default=["all_5class", "all_except_wikab", "all_except_injctrl"])
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--epsilon-r", type=float, default=5e-4)
    parser.add_argument("--elastic-strength", type=float, default=16.0)
    parser.add_argument("--elastic-mix", type=float, default=0.25)
    parser.add_argument("--outlier-strength", type=float, default=16.0)
    parser.add_argument("--outlier-cutoff-preset", choices=["q95", "q97", "q99", "robust3"], default="robust3")
    parser.add_argument("--skip-animations", action="store_true")
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


def _resolve_subset_root(args: argparse.Namespace) -> Path:
    if args.subset_root is not None:
        return args.subset_root
    if args.variant == "raw" and args.subset_root_raw is not None:
        return args.subset_root_raw
    if args.variant == "shrunk" and args.subset_root_shrunk is not None:
        return args.subset_root_shrunk
    raise ValueError("Provide --subset-root or the variant-specific subset root.")


def _build_data(input_path: Path, embryo_ids: np.ndarray) -> schema.CondensationData:
    df = pd.read_csv(input_path)
    df = df[df["embryo_id"].astype(str).isin(set(map(str, embryo_ids.tolist())))].copy()
    feature_cols = [c for c in df.columns if c not in META_COLS]
    df = df[["embryo_id", "time_bin_center", "genotype", *feature_cols]].copy()
    data = schema._build_canonical(
        df,
        feature_cols,
        embryo_col="embryo_id",
        time_col="time_bin_center",
        label_col="genotype",
        allow_feature_nans=True,
    )
    schema.validate(data, allow_feature_nans=True)
    if list(map(str, data.embryo_ids.tolist())) != list(map(str, embryo_ids.tolist())):
        raise ValueError("Subset embryo ordering mismatch with x0_init.npz")
    return data


def _run_case(*, input_path: Path, x0_path: Path, output_dir: Path, config: CondensationConfig, save_every: int, skip_animations: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x0_payload = np.load(x0_path, allow_pickle=True)
    x0 = np.asarray(x0_payload["x0"], dtype=float)
    embryo_ids = np.asarray(x0_payload["embryo_ids"], dtype=object)
    data = _build_data(input_path, embryo_ids)

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
        log_every=max(1, int(config.solver_max_iter) // 20),
        save_every=int(save_every) if int(save_every) > 0 else None,
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
    pd.DataFrame(result.metrics_history).to_csv(output_dir / "metrics.csv", index=False)

    run = tc_viz.load_run(output_dir / "condensed_positions.npz", title=output_dir.name, color_map=GENOTYPE_COLORS)
    tc_viz.render_run(run, output_dir, title_prefix=output_dir.name, skip_animations=skip_animations)


def main() -> None:
    args = parse_args()
    subset_root = _resolve_subset_root(args)
    cutoff_mode, cutoff_value = parse_cutoff_preset(args.outlier_cutoff_preset)
    input_path = args.pairwise_root / f"pairwise_{args.variant}_vectors.csv"
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
        solver_max_iter=int(args.n_iter),
    )

    for subset in args.subsets:
        x0_path = subset_root / subset / "x0_init.npz"
        output_dir = args.output_root / args.variant / subset
        _run_case(
            input_path=input_path,
            x0_path=x0_path,
            output_dir=output_dir,
            config=config,
            save_every=int(args.save_every),
            skip_animations=bool(args.skip_animations),
        )
        print(output_dir)


if __name__ == "__main__":
    main()
