"""Phase 5 rules: stage_predictions per-well shards + experiment merge + validation."""


rule compute_stage_predictions_well:
    input:
        plate_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".plate_metadata.validated",
        plate_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv",
        frame_manifest_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_manifest.validated",
        frame_manifest_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_manifest.csv",
        seg_well_validated=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / ".segment_and_track.validated",
        seg_tracking_csv=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / "segmentation_tracking.csv",
    output:
        validated_flag=DATA_ROOT / "computed_features" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / ".stage_predictions.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        experiment=lambda wc: wc.experiment,
        well_id=lambda wc: wc.well_id,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.feature_extraction.pipelines.compute_stage_predictions '
            '--output-root "{params.output_root}" '
            '--experiment "{params.experiment}" --well-id "{params.well_id}" '
            '--segmentation-tracking-csv "{input.seg_tracking_csv}" '
            '--frame-manifest-csv "{input.frame_manifest_csv}" '
            '--plate-metadata-csv "{input.plate_csv}"'
        )


rule merge_stage_predictions:
    input:
        per_well_validated=lambda wc: expand(
            DATA_ROOT / "computed_features" / wc.experiment / "per_well" / "{well_id}" / "contracts" / ".stage_predictions.validated",
            well_id=selected_well_ids_for_experiment(wc.experiment),
        )
    output:
        merged_parquet=DATA_ROOT / "computed_features" / "{experiment}" / "contracts" / "stage_predictions.parquet",
        merged_csv=DATA_ROOT / "computed_features" / "{experiment}" / "contracts" / "stage_predictions.csv",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        experiment=lambda wc: wc.experiment,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.feature_extraction.pipelines.merge_stage_predictions '
            '--output-root "{params.output_root}" --experiment "{params.experiment}"'
        )


rule validate_stage_predictions:
    input:
        merged_parquet=DATA_ROOT / "computed_features" / "{experiment}" / "contracts" / "stage_predictions.parquet",
    output:
        validated_flag=DATA_ROOT / "computed_features" / "{experiment}" / "contracts" / ".stage_predictions.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.feature_extraction.pipelines.validate_stage_predictions '
            '--input "{input.merged_parquet}" --output-flag "{output.validated_flag}"'
        )

