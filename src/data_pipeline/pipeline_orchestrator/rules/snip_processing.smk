"""snip_processing rules: per-well shards + experiment merge + validation."""


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _snip_write_views() -> bool:
    return _as_bool(config.get("snip_processing", {}).get("write_views", True))


rule snip_processing_well:
    input:
        frame_contract_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_contract.validated",
        frame_contract_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_contract.csv",
        seg_well_validated=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / ".segment_and_track.validated",
        seg_tracking_csv=DATA_ROOT / "segmentation_and_tracking" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / "segmentation_tracking.csv",
    output:
        validated_flag=DATA_ROOT / "processed_snips" / "{experiment}" / "per_well" / "{well_id}" / "contracts" / ".snip_processing.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        config_yaml=WORKFLOW_DIR / "config.yaml",
        experiment=lambda wc: wc.experiment,
        well_id=lambda wc: wc.well_id,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'snip-processing '
            '--experiment "{params.experiment}" '
            '--well-id "{params.well_id}" '
            '--output-root "{params.output_root}" '
            '--frame-contract-csv "{input.frame_contract_csv}" '
            '--segmentation-tracking-csv "{input.seg_tracking_csv}" '
            '--config-yaml "{params.config_yaml}" '
            '--verbose "false"'
        )


rule merge_snip_manifests:
    input:
        per_well_validated=lambda wc: expand(
            DATA_ROOT / "processed_snips" / wc.experiment / "per_well" / "{well_id}" / "contracts" / ".snip_processing.validated",
            # Snip processing shards are keyed by experiment-qualified well_id (e.g. 20240418_A01).
            well_id=selected_well_ids_for_experiment(wc.experiment),
        )
    output:
        manifest_parquet=DATA_ROOT / "processed_snips" / "{experiment}" / "contracts" / "snip_manifest.parquet",
        manifest_csv=DATA_ROOT / "processed_snips" / "{experiment}" / "contracts" / "snip_manifest.csv",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        output_root=DATA_ROOT,
        experiment=lambda wc: wc.experiment,
        write_views=lambda wc: str(_snip_write_views()).lower(),
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.snip_processing.pipelines.merge_snip_manifests '
            '--experiment "{params.experiment}" --output-root "{params.output_root}" '
            '--write-views "{params.write_views}"'
        )


rule validate_snip_manifest:
    input:
        manifest_csv=DATA_ROOT / "processed_snips" / "{experiment}" / "contracts" / "snip_manifest.csv",
    output:
        validated_flag=DATA_ROOT / "processed_snips" / "{experiment}" / "contracts" / ".snip_manifest.validated",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.snip_processing.pipelines.validate_snip_manifest '
            '--input "{input.manifest_csv}" --output-flag "{output.validated_flag}"'
        )
