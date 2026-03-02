"""Frame-contract rules: stitched image materialization and validation."""


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _materialize_overwrite_enabled() -> bool:
    if _as_bool(config.get("overwrite_all", False)):
        return True
    if config.get("overwrite_materialize_stitched_images") is not None:
        return _as_bool(config.get("overwrite_materialize_stitched_images"))
    step_val = config.get("overwrite_steps", {}).get("materialize_stitched_images")
    if step_val is not None:
        return _as_bool(step_val)
    if config.get("frame_contracts", {}).get("overwrite") is not None:
        return _as_bool(config.get("frame_contracts", {}).get("overwrite"))
    return False


def _selected_wells_csv(experiment: str) -> str:
    wells = selected_wells_for_experiment(experiment)
    return ",".join(wells)


def _microscope(experiment: str) -> str:
    return microscope_for_experiment(experiment)


def _scope_dir(experiment: str) -> Path:
    return EXPERIMENT_METADATA_DIR / experiment / "scope" / _microscope(experiment).lower()


rule materialize_stitched_images:
    input:
        scope_csv=lambda wc: _scope_dir(wc.experiment) / "scope_metadata_mapped.csv",
        mapping_csv=lambda wc: _scope_dir(wc.experiment) / "series_well_mapping.csv",
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / _microscope(wc.experiment) / wc.experiment
    output:
        stitched_index_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "stitched_image_index.csv",
        done_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".materialize_stitched_images.done"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment,
        microscope=lambda wc: _microscope(wc.experiment),
        selected_wells=lambda wc: _selected_wells_csv(wc.experiment),
        output_root=BUILT_IMAGE_DATA_DIR,
        output_image_extension=lambda wc: (
            config.get("frame_contracts", {}).get("output_image_extension") or "jpg"
        ),
        device_preference=lambda wc: (
            config.get("frame_contracts", {}).get(_microscope(wc.experiment).lower(), {}).get("device_preference")
            or config.get("frame_contracts", {}).get("yx1", {}).get("device_preference")
            or "cuda"
        ),
        keyence_projection_method=lambda wc: (
            config.get("frame_contracts", {}).get("keyence", {}).get("projection_method")
            or "log"
        ),
        keyence_ff_filter_res_um=lambda wc: (
            config.get("frame_contracts", {}).get("keyence", {}).get("ff_filter_res_um")
            or 3.0
        ),
        overwrite=lambda wc: str(_materialize_overwrite_enabled()).lower()
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'materialize-stitched --experiment "{params.experiment}" --microscope "{params.microscope}" '
            '--raw-images-dir "{input.raw_images_dir}" --scope-csv "{input.scope_csv}" '
            '--mapping-csv "{input.mapping_csv}" --output-root "{params.output_root}" '
            '--output-stitched-index-csv "{output.stitched_index_csv}" --selected-wells "{params.selected_wells}" '
            '--output-image-extension "{params.output_image_extension}" '
            '--device-preference "{params.device_preference}" '
            '--keyence-projection-method "{params.keyence_projection_method}" '
            '--keyence-ff-filter-res-um "{params.keyence_ff_filter_res_um}" '
            '--overwrite "{params.overwrite}" '
            '--done-flag "{output.done_flag}"'
        )


rule validate_stitched_image_index:
    input:
        stitched_index_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "stitched_image_index.csv"
    output:
        validation_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".stitched_image_index.validated"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.stitched_index.validate_stitched_image_index '
            '--input-csv "{input.stitched_index_csv}" --output-flag "{output.validation_flag}"'
        )


rule build_frame_manifest:
    input:
        stitched_index_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "stitched_image_index.csv",
        stitched_index_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".stitched_image_index.validated",
        scope_metadata_csv=lambda wc: _scope_dir(wc.experiment) / "scope_metadata_mapped.csv"
    output:
        frame_manifest_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_manifest.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.frame_manifest.build_frame_manifest '
            '--stitched-index-csv "{input.stitched_index_csv}" '
            '--scope-metadata-csv "{input.scope_metadata_csv}" '
            '--output-csv "{output.frame_manifest_csv}"'
        )


rule validate_frame_manifest:
    input:
        frame_manifest_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_manifest.csv"
    output:
        validation_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_manifest.validated"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.frame_manifest.validate_frame_manifest '
            '--input-csv "{input.frame_manifest_csv}" --output-flag "{output.validation_flag}"'
        )


# Optional convenience join for notebooks/debugging (plate fields duplicated onto per-frame rows).
rule attach_plate_annotations_to_frames:
    input:
        frame_manifest_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_manifest.validated",
        frame_manifest_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_manifest.csv",
        plate_validated=EXPERIMENT_METADATA_DIR / "{experiment}" / ".plate_metadata.validated",
        plate_metadata_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv"
    output:
        annotated_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "frame_manifest_with_plate_annotations.csv",
        annotated_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".frame_manifest_with_plate_annotations.validated"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.frame_manifest.attach_plate_annotations '
            '--frame-manifest-csv "{input.frame_manifest_csv}" --plate-metadata-csv "{input.plate_metadata_csv}" '
            '--output-csv "{output.annotated_csv}" --output-flag "{output.annotated_flag}"'
        )
