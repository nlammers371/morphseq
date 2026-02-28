"""Task entrypoints used by Snakemake rules."""

from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from data_pipeline.metadata_ingest.plate.plate_processing import process_plate_layout
from data_pipeline.metadata_ingest.scope.keyence_scope_metadata import extract_keyence_scope_metadata
from data_pipeline.metadata_ingest.scope.yx1_scope_metadata import extract_yx1_scope_metadata
from data_pipeline.metadata_ingest.mapping.series_well_mapper_keyence import map_series_to_wells_keyence
from data_pipeline.metadata_ingest.mapping.series_well_mapper_yx1 import map_series_to_wells_yx1
from data_pipeline.metadata_ingest.mapping.apply_series_mapping import apply_series_mapping
from data_pipeline.metadata_ingest.mapping.align_scope_plate import align_scope_and_plate_metadata
from data_pipeline.metadata_ingest.stitched_index.materialize_stitched_images import materialize_stitched_images
from data_pipeline.segmentation_and_tracking.pipelines.segmentation_and_tracking import run_segmentation_and_tracking


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_selected_wells(csv: str | None) -> list[str]:
    if not csv:
        return []
    return [part.strip() for part in csv.split(",") if part.strip()]


def cmd_normalize_plate(args: argparse.Namespace) -> None:
    process_plate_layout(
        input_file=args.input_file,
        experiment_id=args.experiment,
        output_csv=args.output_csv,
    )


def cmd_extract_scope(args: argparse.Namespace) -> None:
    if args.microscope == "YX1":
        extract_yx1_scope_metadata(
            raw_data_dir=args.raw_images_dir,
            output_csv=args.output_csv,
            experiment_id=args.experiment,
        )
    elif args.microscope == "Keyence":
        extract_keyence_scope_metadata(
            raw_data_dir=args.raw_images_parent,
            experiment_id=args.experiment,
            output_csv=args.output_csv,
        )
    else:
        raise ValueError(f"Unsupported microscope: {args.microscope}")


def cmd_map_series(args: argparse.Namespace) -> None:
    if args.microscope == "YX1":
        nd2_files = sorted(args.raw_images_dir.glob("*.nd2"))
        nd2_path = nd2_files[0] if nd2_files else None
        map_series_to_wells_yx1(
            plate_metadata_csv=args.plate_csv,
            scope_metadata_csv=args.scope_csv,
            output_mapping_csv=args.output_mapping_csv,
            output_provenance_json=args.output_provenance_json,
            nd2_path=nd2_path,
            use_xy_reference=True,
            ref_xy_csv=args.ref_xy_csv,
            max_distance_um=args.max_distance_um,
        )
    elif args.microscope == "Keyence":
        map_series_to_wells_keyence(
            raw_data_dir=args.raw_images_parent,
            plate_metadata_csv=args.plate_csv,
            scope_metadata_csv=args.scope_csv,
            output_mapping_csv=args.output_mapping_csv,
            output_provenance_json=args.output_provenance_json,
        )
    else:
        raise ValueError(f"Unsupported microscope: {args.microscope}")


def cmd_apply_series(args: argparse.Namespace) -> None:
    apply_series_mapping(
        scope_metadata_csv=args.scope_csv,
        mapping_csv=args.mapping_csv,
        output_csv=args.output_csv,
        experiment_id=args.experiment,
        selected_wells=_parse_selected_wells(args.selected_wells),
    )


def cmd_align(args: argparse.Namespace) -> None:
    align_scope_and_plate_metadata(
        plate_metadata_csv=args.plate_csv,
        scope_metadata_csv=args.scope_csv,
        output_csv=args.output_csv,
    )


def cmd_materialize_stitched(args: argparse.Namespace) -> None:
    materialize_stitched_images(
        experiment=args.experiment,
        microscope=args.microscope,
        raw_images_dir=args.raw_images_dir,
        scope_csv=args.scope_csv,
        mapping_csv=args.mapping_csv,
        output_root=args.output_root,
        output_stitched_index_csv=args.output_stitched_index_csv,
        selected_wells=_parse_selected_wells(args.selected_wells),
        overwrite=_parse_bool(args.overwrite),
        output_image_extension=args.output_image_extension,
        device_preference=args.device_preference,
        keyence_projection_method=args.keyence_projection_method,
        keyence_ff_filter_res_um=args.keyence_ff_filter_res_um,
        done_flag=args.done_flag,
    )


def cmd_segmentation_and_tracking(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config_yaml).read_text()) or {}
    run_segmentation_and_tracking(
        frame_manifest_csv=args.frame_manifest_csv,
        experiment_id=args.experiment,
        well_id=args.well_id,
        output_root=args.output_root,
        pipeline_config=cfg,
        device=args.device,
        run_id=args.run_id,
        verbose=_parse_bool(args.verbose),
    )


def cmd_snip_processing(args: argparse.Namespace) -> None:
    from data_pipeline.snip_processing.pipelines.snip_processing import run_snip_processing_well

    cfg = yaml.safe_load(Path(args.config_yaml).read_text()) or {}

    output_root = Path(args.output_root)
    exp = str(args.experiment)
    well = str(args.well_id)

    frame_manifest_csv = Path(args.frame_manifest_csv)
    segmentation_tracking_csv = Path(args.segmentation_tracking_csv)

    run_snip_processing_well(
        output_root=output_root,
        experiment_id=exp,
        well_id=well,
        frame_manifest_csv=frame_manifest_csv,
        segmentation_tracking_csv=segmentation_tracking_csv,
        pipeline_config=cfg,
        verbose=_parse_bool(args.verbose),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("normalize-plate")
    p_norm.add_argument("--input-file", type=Path, required=True)
    p_norm.add_argument("--experiment", required=True)
    p_norm.add_argument("--output-csv", type=Path, required=True)
    p_norm.set_defaults(func=cmd_normalize_plate)

    p_scope = sub.add_parser("extract-scope")
    p_scope.add_argument("--raw-images-dir", type=Path, required=True)
    p_scope.add_argument("--experiment", required=True)
    p_scope.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_scope.add_argument("--output-csv", type=Path, required=True)
    p_scope.set_defaults(func=cmd_extract_scope)

    p_map = sub.add_parser("map-series")
    p_map.add_argument("--experiment", required=True)
    p_map.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_map.add_argument("--plate-csv", type=Path, required=True)
    p_map.add_argument("--scope-csv", type=Path, required=True)
    p_map.add_argument("--raw-images-dir", type=Path, required=True)
    p_map.add_argument("--output-mapping-csv", type=Path, required=True)
    p_map.add_argument("--output-provenance-json", type=Path, required=True)
    p_map.add_argument("--ref-xy-csv", type=Path, default=None)
    p_map.add_argument("--max-distance-um", type=float, default=4500.0)
    p_map.set_defaults(func=cmd_map_series)

    p_apply = sub.add_parser("apply-series")
    p_apply.add_argument("--experiment", required=True)
    p_apply.add_argument("--scope-csv", type=Path, required=True)
    p_apply.add_argument("--mapping-csv", type=Path, required=True)
    p_apply.add_argument("--output-csv", type=Path, required=True)
    p_apply.add_argument("--selected-wells", default="")
    p_apply.set_defaults(func=cmd_apply_series)

    p_align = sub.add_parser("align")
    p_align.add_argument("--plate-csv", type=Path, required=True)
    p_align.add_argument("--scope-csv", type=Path, required=True)
    p_align.add_argument("--output-csv", type=Path, required=True)
    p_align.set_defaults(func=cmd_align)

    p_mat = sub.add_parser("materialize-stitched")
    p_mat.add_argument("--experiment", required=True)
    p_mat.add_argument("--microscope", choices=["YX1", "Keyence"], required=True)
    p_mat.add_argument("--raw-images-dir", type=Path, required=True)
    p_mat.add_argument("--scope-csv", type=Path, required=True)
    p_mat.add_argument("--mapping-csv", type=Path, required=False)
    p_mat.add_argument("--output-root", type=Path, required=True)
    p_mat.add_argument("--output-stitched-index-csv", type=Path, required=True)
    p_mat.add_argument("--selected-wells", default="")
    p_mat.add_argument("--output-image-extension", default="jpg")
    p_mat.add_argument("--device-preference", default="cuda")
    p_mat.add_argument("--keyence-projection-method", default="log")
    p_mat.add_argument("--keyence-ff-filter-res-um", type=float, default=3.0)
    p_mat.add_argument("--overwrite", default="false")
    p_mat.add_argument("--done-flag", type=Path, required=False)
    p_mat.set_defaults(func=cmd_materialize_stitched)

    p_sat = sub.add_parser("segmentation-and-tracking")
    p_sat.add_argument("--frame-manifest-csv", type=Path, required=True)
    p_sat.add_argument("--experiment", required=True)
    p_sat.add_argument("--well-id", required=True)
    p_sat.add_argument("--output-root", type=Path, required=True)
    p_sat.add_argument("--config-yaml", type=Path, required=True)
    p_sat.add_argument("--device", default="cuda")
    p_sat.add_argument("--run-id", default=None)
    p_sat.add_argument("--verbose", default="false")
    p_sat.set_defaults(func=cmd_segmentation_and_tracking)

    p_snip = sub.add_parser("snip-processing")
    p_snip.add_argument("--experiment", required=True)
    p_snip.add_argument("--well-id", required=True)
    p_snip.add_argument("--output-root", type=Path, required=True)
    p_snip.add_argument("--frame-manifest-csv", type=Path, required=True)
    p_snip.add_argument("--segmentation-tracking-csv", type=Path, required=True)
    p_snip.add_argument("--config-yaml", type=Path, required=True)
    p_snip.add_argument("--verbose", default="false")
    p_snip.set_defaults(func=cmd_snip_processing)

    return parser


def _normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    # Keyence extraction/mapping helpers expect raw_data_root parent and experiment_id.
    if hasattr(args, "raw_images_dir") and isinstance(args.raw_images_dir, Path):
        raw_images_dir = args.raw_images_dir
        if getattr(args, "microscope", None) == "Keyence":
            args.raw_images_parent = raw_images_dir.parent
        else:
            args.raw_images_parent = raw_images_dir
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = _normalize_paths(args)
    args.func(args)


if __name__ == "__main__":
    main()
