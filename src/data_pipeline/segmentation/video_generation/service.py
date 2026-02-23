"""Function-based service API for segmentation eval-video rendering."""

from __future__ import annotations

from pathlib import Path

from .results_adapter import list_videos
from .video_generator import VideoGenerator


def generate_eval_videos_for_experiment(
    *,
    results_json_path: Path,
    experiment_id: str,
    output_dir: Path,
    source_format: str = "auto",
    images_root: Path | None = None,
    policy: str = "all",
    video_ids: list[str] | None = None,
    max_videos: int | None = None,
    suffix: str = "_eval",
    show_bbox: bool = False,
    show_mask: bool = True,
    show_metrics: bool = True,
    show_qc_flags: bool = False,
    show_labels: bool = True,
    verbose: bool = True,
) -> dict[str, bool]:
    """Render eval videos for one experiment using a function-based API.

    Parameters
    ----------
    policy:
      - `all`: render all videos discovered under experiment
      - `first_n`: render first `max_videos` discovered videos
      - `explicit`: render only `video_ids`
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if policy not in {"all", "first_n", "explicit"}:
        raise ValueError(f"Unsupported video policy: {policy}")

    if policy == "explicit":
        if not video_ids:
            raise ValueError("`video_ids` is required when policy='explicit'")
        selected = list(video_ids)
    else:
        discovered = list_videos(
            results_json_path,
            source_format=source_format,
            experiment_id=experiment_id,
        )
        if policy == "first_n":
            n = max_videos if max_videos is not None else len(discovered)
            selected = discovered[: max(0, int(n))]
        else:
            selected = discovered

    vg = VideoGenerator()
    results: dict[str, bool] = {}
    for video_id in selected:
        out_path = output_dir / f"{video_id}{suffix}.mp4"
        ok = vg.create_eval_video_from_results(
            results_json_path=results_json_path,
            experiment_id=experiment_id,
            video_id=video_id,
            output_video_path=out_path,
            source_format=source_format,
            images_root=images_root,
            show_bbox=show_bbox,
            show_mask=show_mask,
            show_metrics=show_metrics,
            show_qc_flags=show_qc_flags,
            show_labels=show_labels,
            verbose=verbose,
        )
        results[video_id] = ok

    return results
