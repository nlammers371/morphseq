from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_qc_flag_examples_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from attrition_qc.config import (  # noqa: E402
    BUILD04_DIR,
    DEFAULT_BIN_WIDTH,
    DEFAULT_EXAMPLE_SUBDIR,
    DEFAULT_EXAMPLES_PER_GENOTYPE,
    DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
    DEFAULT_TARGET_FLAGS,
    DEFAULT_TIME_COL,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_VIDEO_WINDOW_HPF,
    EXPERIMENT_IDS,
    FIGURES_BASE,
    RESULTS_BASE,
    SNIP_ROOT,
)
from attrition_qc.examples import (  # noqa: E402
    ExampleRenderConfig,
    build_flag_example_frame_summary,
    build_flag_example_summary,
    load_build04_frame_qc_dataframe,
    render_flag_example_media,
    select_flag_example_frames,
    select_flag_examples,
    summarize_flag_examples,
)
from attrition_qc.io import ensure_output_dirs, save_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotated QC-flag example videos for PBX attrition audit.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_BASE / "embryo_attrition_qc_audit" / DEFAULT_EXAMPLE_SUBDIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_BASE / "embryo_attrition_qc_audit" / DEFAULT_EXAMPLE_SUBDIR)
    parser.add_argument("--build04-dir", type=Path, default=BUILD04_DIR)
    parser.add_argument("--snip-root", type=Path, default=SNIP_ROOT)
    parser.add_argument("--experiment-ids", nargs="+", default=EXPERIMENT_IDS)
    parser.add_argument("--target-flags", nargs="+", default=DEFAULT_TARGET_FLAGS)
    parser.add_argument("--selection-unit", choices=["frame", "bin"], default="frame")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    parser.add_argument("--examples-per-genotype", type=int, default=DEFAULT_EXAMPLES_PER_GENOTYPE)
    parser.add_argument("--window-hpf", type=float, default=DEFAULT_VIDEO_WINDOW_HPF)
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--n-frames-out", type=int, default=DEFAULT_VIDEO_FRAMES)
    parser.add_argument(
        "--require-target-family-only",
        action="store_true",
        help="Only export anchors where the requested canonical QC family is the only active canonical QC family at that selection unit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.results_dir, args.figures_dir)
    videos_dir = args.figures_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frame-level build04 QC data ...")
    frame_df = load_build04_frame_qc_dataframe(
        build_dir=args.build04_dir,
        experiment_ids=list(args.experiment_ids),
        genotype_map=DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
    )
    print(f"  {len(frame_df)} frame rows")

    print("Computing aggregate flag summaries ...")
    summary_df = summarize_flag_examples(
        frame_df,
        bin_width=float(args.bin_width),
        time_col=args.time_col,
        target_flags=list(args.target_flags),
    )
    summary_df.to_csv(args.results_dir / "qc_flag_rate_summary.csv", index=False)

    render_config = ExampleRenderConfig(
        bin_width=float(args.bin_width),
        time_col=args.time_col,
        window_hpf=float(args.window_hpf),
        fps=int(args.fps),
        n_frames_out=int(args.n_frames_out),
    )

    manifest_rows: list[pd.DataFrame] = []
    print("Selecting and rendering examples ...")
    for target_flag in args.target_flags:
        print(f"  Target flag: {target_flag}")
        if args.selection_unit == "frame":
            flag_summary = build_flag_example_frame_summary(
                frame_df,
                target_flag=str(target_flag),
                bin_width=float(args.bin_width),
                time_col=args.time_col,
            )
        else:
            flag_summary = build_flag_example_summary(
                frame_df,
                target_flag=str(target_flag),
                bin_width=float(args.bin_width),
                time_col=args.time_col,
            )
        if not flag_summary.empty:
            flag_summary["target_flag"] = str(target_flag)
        candidate_name = f"{target_flag}_{'candidate_frames' if args.selection_unit == 'frame' else 'candidate_bins'}.csv"
        flag_summary.to_csv(args.results_dir / candidate_name, index=False)
        if args.selection_unit == "frame":
            selected = select_flag_example_frames(
                flag_summary,
                max_examples_per_genotype=int(args.examples_per_genotype),
                require_target_family_only_in_frame=bool(args.require_target_family_only),
            )
        else:
            selected = select_flag_examples(
                flag_summary,
                max_examples_per_genotype=int(args.examples_per_genotype),
                require_target_family_only_in_bin=bool(args.require_target_family_only),
            )
        if selected.empty:
            print("    no eligible examples after alive + excluded filtering")
            continue
        selected["target_flag"] = str(target_flag)
        media_rows: list[dict[str, object]] = []
        flag_video_dir = videos_dir / str(target_flag)
        flag_video_dir.mkdir(parents=True, exist_ok=True)
        for _, row in selected.iterrows():
            media = render_flag_example_media(
                row,
                frame_df,
                target_flag=str(target_flag),
                output_dir=flag_video_dir,
                snip_root=args.snip_root,
                render_config=render_config,
            )
            media_rows.append(media)
        selected = pd.concat([selected.reset_index(drop=True), pd.DataFrame(media_rows)], axis=1)
        selected.to_csv(args.results_dir / f"{target_flag}_selected_examples.csv", index=False)
        manifest_rows.append(selected)
        print(f"    rendered {len(selected)} examples")

    selected_all = pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame()
    if not selected_all.empty:
        selected_all.to_csv(args.results_dir / "selected_examples_manifest.csv", index=False)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "videos_dir": str(videos_dir),
        "build04_dir": str(args.build04_dir),
        "snip_root": str(args.snip_root),
        "experiment_ids": list(args.experiment_ids),
        "target_flags": list(args.target_flags),
        "bin_width": float(args.bin_width),
        "time_col": args.time_col,
        "examples_per_genotype": int(args.examples_per_genotype),
        "window_hpf": float(args.window_hpf),
        "fps": int(args.fps),
        "n_frames_out": int(args.n_frames_out),
        "selection_logic": {
            "selection_unit": args.selection_unit,
            "candidate": (
                "single flagged frame/snip_id"
                if args.selection_unit == "frame"
                else "embryo_id x time_bin with target_flag present at least once"
            ),
            "eligible": (
                "alive frames only and use_embryo_flag == False"
                if args.selection_unit == "frame"
                else "alive bins only and included_in_bin == False"
            ),
            "require_target_family_only": bool(args.require_target_family_only),
            "ranking": (
                "prefer target_family_only_in_frame, then earliest flagged frame"
                if args.selection_unit == "frame"
                else "prefer target_family_only_in_bin, then only_target_canonical, then higher target_frame_fraction, then more target/excluded frames"
            ),
            "balance": "up to N examples per genotype per flag",
        },
        "outputs": {
            "qc_flag_rate_summary.csv": "Per-genotype/time aggregate counts for requested target flags",
            "<flag>_candidate_frames.csv or <flag>_candidate_bins.csv": "All selection-unit candidates carrying the target flag",
            "<flag>_selected_examples.csv": "Selected examples plus rendered media paths",
            "selected_examples_manifest.csv": "Combined selected examples across target flags",
        },
    }
    save_manifest(args.results_dir / "qc_flag_examples_manifest.json", manifest)

    print("Done.")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
