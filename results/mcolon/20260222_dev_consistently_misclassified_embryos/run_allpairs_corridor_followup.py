from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})


def _norm_label(x: str) -> str:
    y = re.sub(r"[^a-zA-Z0-9]+", "_", str(x).strip().lower())
    y = re.sub(r"_+", "_", y).strip("_")
    return y


def _pair_label(src: str, dst: str) -> str:
    return f"{src}->{dst}"


def _load_discovery_pair_file(path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if df.empty:
        stem = path.stem.replace("rolling_destination_confusion_", "").replace("_5hpf", "")
        return stem.replace("__to__", "->"), df, pd.DataFrame()

    src = str(df["dest_source_class"].iloc[0])
    dst = str(df["dest_target_class"].iloc[0])
    pair = _pair_label(src, dst)

    valid = df[df["dest_confusion_valid_test"].fillna(False)].copy()
    if valid.empty:
        return pair, df, pd.DataFrame()

    w = (
        valid.groupby(["window_center_hpf", "window_start_hpf", "window_end_hpf"], as_index=False)
        .agg(
            source_support=("dest_confusion_window_source_support", "first"),
            hits_total=("dest_confusion_window_hits_total", "first"),
            rate_obs=("dest_confusion_window_rate_obs", "first"),
            min_q_global=("qval_dest_confusion_global_perm", "min"),
            mean_q_global=("qval_dest_confusion_global_perm", "mean"),
            max_z=("dest_confusion_z", "max"),
            n_valid_rows=("embryo_id", "size"),
            n_sig_rows=("is_dest_confusion_significant_global_perm", "sum"),
        )
        .sort_values(["window_center_hpf"])
        .reset_index(drop=True)
    )
    w["pair"] = pair
    return pair, df, w


def _build_discovery_tables(summary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(summary_dir.glob("rolling_destination_confusion_*_5hpf.csv"))
    if not files:
        raise FileNotFoundError(f"No 5hpf destination confusion files found in {summary_dir}")

    window_rows: list[pd.DataFrame] = []
    pair_rows: list[dict[str, object]] = []

    for f in files:
        pair, raw_df, w = _load_discovery_pair_file(f)
        if not w.empty:
            window_rows.append(w)
            best = w.sort_values(["min_q_global", "window_center_hpf"]).iloc[0]
            min_q = float(w["min_q_global"].min())
            n_q10 = int((w["min_q_global"] <= 0.10).sum())
            n_q20 = int((w["min_q_global"] <= 0.20).sum())
            n_z2 = int((w["max_z"] >= 2.0).sum())
            n_z3 = int((w["max_z"] >= 3.0).sum())
            window_count = int(len(w))
            valid_tests = int(raw_df["dest_confusion_valid_test"].fillna(False).sum())
            global_sig_rows = int(raw_df["is_dest_confusion_significant_global_perm"].fillna(False).sum())

            # Discovery-oriented score: prioritize low q and persistence over windows.
            score = (
                -math.log10(max(min_q, 1e-12))
                + 1.25 * n_q10
                + 0.60 * n_q20
                + 0.02 * window_count
                + 0.0005 * int(best["source_support"])
            )

            pair_rows.append(
                {
                    "pair": pair,
                    "valid_tests": valid_tests,
                    "window_count_valid": window_count,
                    "min_q_global": min_q,
                    "n_global_sig_rows": global_sig_rows,
                    "n_windows_q_le_0p10": n_q10,
                    "n_windows_q_le_0p20": n_q20,
                    "n_windows_z_ge_2": n_z2,
                    "n_windows_z_ge_3": n_z3,
                    "best_window_hpf": float(best["window_center_hpf"]),
                    "best_window_rate": float(best["rate_obs"]),
                    "best_window_source_support": int(best["source_support"]),
                    "best_window_hits": int(best["hits_total"]),
                    "discovery_persistence_score": float(score),
                }
            )
        else:
            pair_rows.append(
                {
                    "pair": pair,
                    "valid_tests": int(raw_df["dest_confusion_valid_test"].fillna(False).sum()) if not raw_df.empty else 0,
                    "window_count_valid": 0,
                    "min_q_global": np.nan,
                    "n_global_sig_rows": int(raw_df["is_dest_confusion_significant_global_perm"].fillna(False).sum()) if not raw_df.empty else 0,
                    "n_windows_q_le_0p10": 0,
                    "n_windows_q_le_0p20": 0,
                    "n_windows_z_ge_2": 0,
                    "n_windows_z_ge_3": 0,
                    "best_window_hpf": np.nan,
                    "best_window_rate": np.nan,
                    "best_window_source_support": 0,
                    "best_window_hits": 0,
                    "discovery_persistence_score": -np.inf,
                }
            )

    window_df = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    rank_df = pd.DataFrame(pair_rows)
    rank_df = rank_df.sort_values(
        ["discovery_persistence_score", "min_q_global", "n_windows_q_le_0p20", "valid_tests", "pair"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    rank_df["discovery_rank"] = np.arange(1, len(rank_df) + 1)

    return window_df, rank_df


def _select_top_pairs(rank_df: pd.DataFrame, top_n: int) -> list[str]:
    keep = rank_df.copy()
    keep = keep[keep["window_count_valid"] > 0]
    keep = keep[keep["best_window_rate"].fillna(0.0) > 0.0]
    keep = keep[np.isfinite(keep["min_q_global"]) ]

    # Discovery candidate gate: prefer pairs with at least some non-trivial calibrated signal.
    keep_signal = keep[keep["min_q_global"] < 1.0].copy()
    if not keep_signal.empty:
        keep = keep_signal

    if keep.empty:
        raise ValueError("No candidate pairs available after filtering for valid discovery signal.")

    keep = keep.sort_values(
        ["min_q_global", "n_windows_q_le_0p20", "n_windows_q_le_0p10", "valid_tests", "best_window_source_support", "pair"],
        ascending=[True, False, False, False, False, True],
    )
    return keep["pair"].head(int(top_n)).astype(str).tolist()


def _run_confirmatory(
    *,
    python_exe: str,
    run_script: Path,
    run_id: str,
    pairs: list[str],
    permutations: int,
    bootstrap_iters: int,
    q_threshold: float,
    min_source_rows: int,
    min_window_support: int,
    driver_bootstrap_threshold: float,
) -> None:
    pair_arg = ",".join(pairs)
    cmd = [
        python_exe,
        str(run_script),
        "--run-id",
        run_id,
        "--stages",
        "hard",
        "--wrong-rate-n-permutations",
        str(permutations),
        "--rolling-window-hpf",
        "5",
        "--destination-pairs",
        pair_arg,
        "--destination-rolling-window-hpf",
        "15",
        "--destination-n-permutations",
        str(permutations),
        "--destination-q-threshold",
        str(q_threshold),
        "--destination-min-source-rows-per-embryo",
        str(min_source_rows),
        "--destination-min-window-source-support",
        str(min_window_support),
        "--destination-bootstrap-iters",
        str(bootstrap_iters),
        "--destination-driver-bootstrap-threshold",
        str(driver_bootstrap_threshold),
    ]
    env = dict(os.environ)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = "src" + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = "src"
    subprocess.run(cmd, check=True, env=env)


def _load_raw_for_morphology(raw_csv: Path, time_col: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, low_memory=False)
    use_mask = _as_bool(raw["use_embryo_flag"]) if "use_embryo_flag" in raw.columns else pd.Series(True, index=raw.index)
    dead_mask = _as_bool(raw["dead_flag2"]) if "dead_flag2" in raw.columns else pd.Series(False, index=raw.index)

    req = ["embryo_id", time_col, "baseline_deviation_um", "total_length_um"]
    raw = raw[use_mask & (~dead_mask)].copy()
    raw = raw.dropna(subset=req)

    if "true_class" in raw.columns:
        raw["true_class_src"] = raw["true_class"].astype(str)
    elif "phenotype" in raw.columns:
        raw["true_class_src"] = raw["phenotype"].astype(str)
    else:
        raw["true_class_src"] = ""

    raw["true_class_norm"] = raw["true_class_src"].map(_norm_label)
    raw["embryo_id"] = raw["embryo_id"].astype(str)
    raw[time_col] = raw[time_col].astype(float)
    return raw


def _characterize_surviving_corridors(confirm_summary_dir: Path, raw_csv: Path, time_col: str) -> dict[str, Path]:
    files = sorted(confirm_summary_dir.glob("rolling_destination_confusion_*_15hpf.csv"))
    if not files:
        raise FileNotFoundError(f"No confirmatory destination files found in {confirm_summary_dir}")

    raw = _load_raw_for_morphology(raw_csv, time_col=time_col)

    window_summary_rows: list[dict[str, object]] = []
    embryo_rows: list[dict[str, object]] = []

    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        src = str(df["dest_source_class"].iloc[0])
        dst = str(df["dest_target_class"].iloc[0])
        pair = _pair_label(src, dst)
        src_norm = _norm_label(src)

        sig = df[df["is_dest_confusion_significant_global_perm"].fillna(False)].copy()
        if sig.empty:
            continue

        valid = df[df["dest_confusion_valid_test"].fillna(False)].copy()
        if valid.empty:
            continue

        wagg = (
            valid.groupby(["window_center_hpf", "window_start_hpf", "window_end_hpf"], as_index=False)
            .agg(
                source_support=("dest_confusion_window_source_support", "first"),
                hits_total=("dest_confusion_window_hits_total", "first"),
                rate_obs=("dest_confusion_window_rate_obs", "first"),
                min_q_global=("qval_dest_confusion_global_perm", "min"),
                n_sig_rows=("is_dest_confusion_significant_global_perm", "sum"),
                n_valid_rows=("embryo_id", "size"),
            )
            .sort_values("window_center_hpf")
        )

        sig_windows = sorted(sig["window_center_hpf"].astype(float).unique().tolist())
        for wc in sig_windows:
            win = wagg[wagg["window_center_hpf"].astype(float) == float(wc)]
            if win.empty:
                continue
            wr = win.iloc[0]

            ws = float(wr["window_start_hpf"])
            we = float(wr["window_end_hpf"])
            sig_rows = sig[sig["window_center_hpf"].astype(float) == float(wc)].copy()
            sig_embryos = sorted(sig_rows["embryo_id"].astype(str).unique().tolist())

            window_summary_rows.append(
                {
                    "pair": pair,
                    "source_class": src,
                    "target_class": dst,
                    "window_center_hpf": float(wc),
                    "window_start_hpf": ws,
                    "window_end_hpf": we,
                    "source_support": int(wr["source_support"]),
                    "hits_total": int(wr["hits_total"]),
                    "rate_obs": float(wr["rate_obs"]),
                    "min_q_global": float(wr["min_q_global"]),
                    "n_sig_rows": int(wr["n_sig_rows"]),
                    "n_sig_embryos": int(len(sig_embryos)),
                }
            )

            raw_w = raw[(raw[time_col] >= ws) & (raw[time_col] <= we)].copy()
            src_w = raw_w[raw_w["true_class_norm"] == src_norm].copy()
            if src_w.empty:
                continue

            cohort_stats = {
                "baseline_mu": float(src_w["baseline_deviation_um"].mean()),
                "baseline_sd": float(src_w["baseline_deviation_um"].std(ddof=0)),
                "length_mu": float(src_w["total_length_um"].mean()),
                "length_sd": float(src_w["total_length_um"].std(ddof=0)),
            }

            emb = raw_w[raw_w["embryo_id"].isin(sig_embryos)].copy()
            if emb.empty:
                continue
            eagg = (
                emb.groupby("embryo_id", as_index=False)
                .agg(
                    n_points=("embryo_id", "size"),
                    baseline_deviation_um_mean=("baseline_deviation_um", "mean"),
                    total_length_um_mean=("total_length_um", "mean"),
                )
            )
            sig_count = sig_rows.groupby("embryo_id").size().to_dict()

            bsd = cohort_stats["baseline_sd"] if cohort_stats["baseline_sd"] > 0 else np.nan
            lsd = cohort_stats["length_sd"] if cohort_stats["length_sd"] > 0 else np.nan

            for _, r in eagg.iterrows():
                embryo_id = str(r["embryo_id"])
                bmean = float(r["baseline_deviation_um_mean"])
                lmean = float(r["total_length_um_mean"])
                embryo_rows.append(
                    {
                        "pair": pair,
                        "source_class": src,
                        "target_class": dst,
                        "embryo_id": embryo_id,
                        "window_center_hpf": float(wc),
                        "window_start_hpf": ws,
                        "window_end_hpf": we,
                        "n_points": int(r["n_points"]),
                        "n_sig_rows_for_embryo_window": int(sig_count.get(embryo_id, 0)),
                        "baseline_deviation_um_mean": bmean,
                        "total_length_um_mean": lmean,
                        "source_cohort_baseline_mu": cohort_stats["baseline_mu"],
                        "source_cohort_baseline_sd": cohort_stats["baseline_sd"],
                        "source_cohort_length_mu": cohort_stats["length_mu"],
                        "source_cohort_length_sd": cohort_stats["length_sd"],
                        "baseline_deviation_z_vs_source": float((bmean - cohort_stats["baseline_mu"]) / bsd) if np.isfinite(bsd) else np.nan,
                        "total_length_z_vs_source": float((lmean - cohort_stats["length_mu"]) / lsd) if np.isfinite(lsd) else np.nan,
                    }
                )

    window_df = pd.DataFrame(window_summary_rows)
    embryo_df = pd.DataFrame(embryo_rows)

    paths: dict[str, Path] = {}
    window_path = confirm_summary_dir / "corridor_morphology_window_summary.csv"
    window_df.to_csv(window_path, index=False)
    paths["window_summary"] = window_path

    embryo_path = confirm_summary_dir / "corridor_morphology_candidate_embryos.csv"
    embryo_df.to_csv(embryo_path, index=False)
    paths["candidate_embryos"] = embryo_path

    if not embryo_df.empty:
        agg = (
            embryo_df.groupby(["pair", "embryo_id"], as_index=False)
            .agg(
                n_windows=("window_center_hpf", "nunique"),
                n_points_total=("n_points", "sum"),
                mean_baseline_z_vs_source=("baseline_deviation_z_vs_source", "mean"),
                mean_length_z_vs_source=("total_length_z_vs_source", "mean"),
                max_sig_rows_for_embryo_window=("n_sig_rows_for_embryo_window", "max"),
            )
            .sort_values(["pair", "n_windows", "max_sig_rows_for_embryo_window"], ascending=[True, False, False])
        )
    else:
        agg = pd.DataFrame(
            columns=[
                "pair",
                "embryo_id",
                "n_windows",
                "n_points_total",
                "mean_baseline_z_vs_source",
                "mean_length_z_vs_source",
                "max_sig_rows_for_embryo_window",
            ]
        )

    agg_path = confirm_summary_dir / "corridor_morphology_candidate_embryos_agg.csv"
    agg.to_csv(agg_path, index=False)
    paths["candidate_embryos_agg"] = agg_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="All-pairs corridor follow-up: discovery ranking, confirmatory rerun, morphology tables")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/mcolon/20260222_dev_consistently_misclassified_embryos/real_runs/trajectory_stages"),
    )
    parser.add_argument("--discovery-run-id", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--confirm-run-id", type=str, default="")
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path("results/mcolon/20260222_dev_consistently_misclassified_embryos/run_cep290_stage_geometry.py"),
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path(
            "results/mcolon/20260213_subtle_phenotype_methods/input_data/experiments/"
            "cep290_20251229/input_core.csv"
        ),
    )
    parser.add_argument("--time-col", type=str, default="predicted_stage_hpf")

    parser.add_argument("--confirm-permutations", type=int, default=5000)
    parser.add_argument("--confirm-bootstrap-iters", type=int, default=5000)
    parser.add_argument("--confirm-q-threshold", type=float, default=0.10)
    parser.add_argument("--confirm-min-source-rows", type=int, default=3)
    parser.add_argument("--confirm-min-window-support", type=int, default=20)
    parser.add_argument("--confirm-driver-bootstrap-threshold", type=float, default=0.80)

    args = parser.parse_args()

    discovery_dir = args.output_root / args.discovery_run_id
    discovery_summary = discovery_dir / "summary"
    if not discovery_summary.exists():
        raise FileNotFoundError(f"Discovery summary directory not found: {discovery_summary}")

    window_df, rank_df = _build_discovery_tables(discovery_summary)
    window_scores_path = discovery_summary / "all_pairs_window_scores.csv"
    ranking_path = discovery_summary / "all_pairs_corridor_ranking.csv"
    window_df.to_csv(window_scores_path, index=False)
    rank_df.to_csv(ranking_path, index=False)

    top_pairs = _select_top_pairs(rank_df, top_n=int(args.top_n))

    confirm_run_id = (
        args.confirm_run_id.strip()
        if args.confirm_run_id.strip()
        else f"{args.discovery_run_id}__confirm_top{len(top_pairs)}_roll15_perm{int(args.confirm_permutations)}_boot{int(args.confirm_bootstrap_iters)}"
    )

    _run_confirmatory(
        python_exe=sys.executable,
        run_script=args.run_script,
        run_id=confirm_run_id,
        pairs=top_pairs,
        permutations=int(args.confirm_permutations),
        bootstrap_iters=int(args.confirm_bootstrap_iters),
        q_threshold=float(args.confirm_q_threshold),
        min_source_rows=int(args.confirm_min_source_rows),
        min_window_support=int(args.confirm_min_window_support),
        driver_bootstrap_threshold=float(args.confirm_driver_bootstrap_threshold),
    )

    confirm_summary = args.output_root / confirm_run_id / "summary"
    morph_paths = _characterize_surviving_corridors(
        confirm_summary_dir=confirm_summary,
        raw_csv=args.raw_csv,
        time_col=args.time_col,
    )

    meta = {
        "discovery_run_id": args.discovery_run_id,
        "discovery_summary": str(discovery_summary),
        "window_scores_csv": str(window_scores_path),
        "ranking_csv": str(ranking_path),
        "top_pairs_selected": top_pairs,
        "confirm_run_id": confirm_run_id,
        "confirm_summary": str(confirm_summary),
        "morphology_outputs": {k: str(v) for k, v in morph_paths.items()},
        "confirm_settings": {
            "permutations": int(args.confirm_permutations),
            "bootstrap_iters": int(args.confirm_bootstrap_iters),
            "q_threshold": float(args.confirm_q_threshold),
            "min_source_rows": int(args.confirm_min_source_rows),
            "min_window_support": int(args.confirm_min_window_support),
            "driver_bootstrap_threshold": float(args.confirm_driver_bootstrap_threshold),
        },
    }
    meta_path = discovery_summary / "all_pairs_followup_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    print(f"Wrote discovery window scores: {window_scores_path}")
    print(f"Wrote corridor ranking: {ranking_path}")
    print(f"Selected top pairs: {', '.join(top_pairs)}")
    print(f"Confirmatory run: {confirm_run_id}")
    print(f"Confirmatory summary: {confirm_summary}")
    for k, v in morph_paths.items():
        print(f"Morphology output ({k}): {v}")
    print(f"Follow-up metadata: {meta_path}")


if __name__ == "__main__":
    main()
