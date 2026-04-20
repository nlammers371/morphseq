"""segmentation_and_tracking rules: per-well shards + experiment merge + validation."""


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _selected_wells_csv(experiment: str) -> str:
    wells = selected_wells_for_experiment(experiment)
    return ",".join(wells)


def _sat_device() -> str:
    return str(config.get("segmentation_and_tracking", {}).get("device", "cuda"))


def _sat_channel() -> str:
    return str(config.get("segmentation_and_tracking", {}).get("channel_id", "BF"))


def _scope_dir(experiment: str):
    return EXPERIMENT_METADATA_DIR / experiment / "scope" / microscope_for_experiment(experiment).lower()


rule segment_and_track_well:
    input:
        physical_mapping_validated=lambda wc: _scope_dir(wc.experiment) / ".physical_well_mapping.validated",
        frame_contract_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_contract.validated",
        frame_contract_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_contract.csv",
    output:
        seg_tracking_csv=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / "segmentation_tracking.csv",
        validated_flag=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / ".segment_and_track.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        config_yaml=WORKFLOW_DIR / "config.yaml",
        device=lambda wc: _sat_device(),
        channel_id=lambda wc: _sat_channel(),
        experiment=lambda wc: wc.experiment,
        well_id=lambda wc: wc.well_id,
    resources:
        gpu=lambda wc: 1 if _sat_device().lower().startswith("cuda") else 0
    shell:
        (
            'TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'segmentation-and-tracking '
            '--frame-contract-csv "{input.frame_contract_csv}" '
            '--experiment "{params.experiment}" '
            '--well-id "{params.well_id}" '
            '--output-root "{params.output_root}" '
            '--config-yaml "{params.config_yaml}" '
            '--device "{params.device}" '
            '--run-id "{wildcards.experiment}_{wildcards.well_id}"'
        )


rule merge_segmentation_and_tracking_contracts:
    input:
        per_well_validated=lambda wc: expand(
            DATA_ROOT / "segmentation_and_tracking" / wc.experiment / "per_well" / "{well_id}" / "contracts" / ".segment_and_track.validated",
            well_id=selected_well_ids_for_experiment(wc.experiment),
        )
    output:
        frame_detections=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "frame_detections.parquet",
        seed_selection=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "seed_selection.parquet",
        embryo_tracks=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "embryo_track_instances.parquet",
        embryo_masks=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "embryo_mask_rle.parquet",
        segmentation_tracking=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "segmentation_tracking.csv",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        experiment=lambda wc: wc.experiment,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.segmentation_and_tracking.pipelines.merge_segmentation_and_tracking_contracts '
            '--experiment "{params.experiment}" --output-root "{params.output_root}"'
        )


rule validate_segmentation_and_tracking:
    input:
        segmentation_tracking=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / "segmentation_tracking.csv",
    output:
        validated_flag=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "contracts" / ".segmentation_tracking.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.segmentation_and_tracking.pipelines.validate_segmentation_and_tracking '
            '--input-csv "{input.segmentation_tracking}" --output-flag "{output.validated_flag}"'
        )
