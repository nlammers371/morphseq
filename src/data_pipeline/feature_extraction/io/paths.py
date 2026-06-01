"""
Feature output path helpers.

The feature pipeline writes one directory per experiment and then one
subdirectory per feature family so the outputs stay readable and easy to
target from Snakemake.
"""

from __future__ import annotations

from pathlib import Path


FEATURES_ROOT_NAME = "computed_features"

FEATURE_OUTPUT_FILENAMES = {
    "mask_geometry": "mask_geometry_metrics.csv",
    "curvature_metrics": "curvature_metrics.csv",
    "pose_kinematics": "pose_kinematics_metrics.csv",
    "fraction_alive": "fraction_alive.csv",
    "stage_predictions": "stage_predictions.csv",
    "consolidated": "consolidated_snip_features.csv",
}


def experiment_features_root(data_root: Path, experiment_id: str) -> Path:
    return Path(data_root) / FEATURES_ROOT_NAME / str(experiment_id)


def feature_output_dir(data_root: Path, experiment_id: str, feature_name: str) -> Path:
    return experiment_features_root(data_root, experiment_id) / feature_name


def feature_output_path(
    data_root: Path,
    experiment_id: str,
    feature_name: str,
    filename: str | None = None,
) -> Path:
    resolved_filename = filename or FEATURE_OUTPUT_FILENAMES.get(feature_name)
    if resolved_filename is None:
        raise KeyError(
            f"No default filename registered for feature '{feature_name}'. "
            "Pass filename explicitly."
        )
    return feature_output_dir(data_root, experiment_id, feature_name) / resolved_filename


def consolidated_features_path(data_root: Path, experiment_id: str) -> Path:
    return feature_output_path(data_root, experiment_id, "consolidated")


def analysis_ready_path(data_root: Path, experiment_id: str) -> Path:
    return Path(data_root) / "analysis_ready" / str(experiment_id) / "analysis_ready.csv"


def feature_sentinel_path(table_path: Path) -> Path:
    return table_path.with_suffix(table_path.suffix + ".validated")


def schema_sidecar_path(table_path: Path) -> Path:
    return table_path.with_suffix("").with_suffix(".schema.json")
