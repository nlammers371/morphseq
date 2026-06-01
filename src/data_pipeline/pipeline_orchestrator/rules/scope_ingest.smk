"""Scope ingestion rules: microscope extraction + plate-free physical well mapping + optional plate alignment."""


def _selected_wells_csv(experiment: str) -> str:
    wells = selected_wells_for_experiment(experiment)
    return ",".join(wells)


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _scope_ingest_cfg() -> dict:
    return dict(config.get("scope_ingest", {}) or {})


def _allow_unmapped_wells() -> bool:
    # Explicit opt-in for alternative workflows only; default path must validate wells.
    return _as_bool(_scope_ingest_cfg().get("allow_unmapped_wells", False))


def _scope_dir(experiment: str) -> str:
    # One microscope per experiment; scope artifacts live under scope/{microscope}/
    return microscope_for_experiment(experiment).lower()


rule normalize_plate_metadata:
    input:
        plate_file=lambda wc: PLATE_METADATA_DIR / f"{wc.experiment}_well_metadata.xlsx"
    output:
        plate_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'normalize-plate --input-file "{input.plate_file}" --experiment "{params.experiment}" '
            '--output-csv "{output.plate_csv}"'
        )


rule validate_plate_metadata:
    input:
        plate_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv"
    output:
        validated_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".plate_metadata.validated"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.plate.validate_plate_metadata '
            '--input-csv "{input.plate_csv}" --output-flag "{output.validated_flag}"'
        )


rule extract_scope_metadata_yx1:
    input:
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / "YX1" / wc.experiment
    output:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "scope_series_metadata_raw.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'extract-scope --raw-images-dir "{input.raw_images_dir}" --experiment "{params.experiment}" '
            '--microscope "YX1" --output-csv "{output.scope_csv}"'
        )


rule extract_scope_metadata_keyence:
    input:
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / "Keyence" / wc.experiment
    output:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "scope_series_metadata_raw.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'extract-scope --raw-images-dir "{input.raw_images_dir}" --experiment "{params.experiment}" '
            '--microscope "Keyence" --output-csv "{output.scope_csv}"'
        )


rule map_series_to_wells_yx1:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "scope_series_metadata_raw.csv",
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / "YX1" / wc.experiment
    output:
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "series_well_mapping.csv",
        provenance_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "mapping_provenance.json",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment,
        ref_xy_csv=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("ref_xy_csv", ""),
        row_y_tol_um=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("row_y_tol_um", 1200.0),
        col_x_tol_um=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("col_x_tol_um", 1200.0),
        dx_cv_tol=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("dx_cv_tol", 0.15),
        dy_cv_tol=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("dy_cv_tol", 0.15),
        max_distance_um=lambda wc: (_scope_ingest_cfg().get("yx1_mapping", {}) or {}).get("max_distance_um", 4500.0),
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'map-series --experiment "{params.experiment}" --microscope "YX1" '
            '--scope-csv "{input.scope_csv}" '
            '--raw-images-dir "{input.raw_images_dir}" --output-mapping-csv "{output.mapping_csv}" '
            '--output-provenance-json "{output.provenance_json}" --ref-xy-csv "{params.ref_xy_csv}" '
            '--max-distance-um "{params.max_distance_um}" '
            ' --row-y-tol-um "{params.row_y_tol_um}" --col-x-tol-um "{params.col_x_tol_um}"'
            ' --dx-cv-tol "{params.dx_cv_tol}" --dy-cv-tol "{params.dy_cv_tol}"'
        )


rule map_series_to_wells_keyence:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "scope_series_metadata_raw.csv",
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / "Keyence" / wc.experiment
    output:
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "series_well_mapping.csv",
        provenance_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "mapping_provenance.json",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'map-series --experiment "{params.experiment}" --microscope "Keyence" '
            '--scope-csv "{input.scope_csv}" '
            '--raw-images-dir "{input.raw_images_dir}" --output-mapping-csv "{output.mapping_csv}" '
            '--output-provenance-json "{output.provenance_json}"'
        )


rule apply_series_mapping_yx1:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "scope_series_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "series_well_mapping.csv",
    output:
        mapped_scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "scope_series_metadata_mapped.csv",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment,
        selected_wells=lambda wc: _selected_wells_csv(wc.experiment)
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'apply-series --experiment "{params.experiment}" --scope-csv "{input.scope_csv}" '
            '--mapping-csv "{input.mapping_csv}" --output-csv "{output.mapped_scope_csv}" '
            '--selected-wells "{params.selected_wells}"'
        )


rule apply_series_mapping_keyence:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "scope_series_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "series_well_mapping.csv",
    output:
        mapped_scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "scope_series_metadata_mapped.csv",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        experiment=lambda wc: wc.experiment,
        selected_wells=lambda wc: _selected_wells_csv(wc.experiment)
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'apply-series --experiment "{params.experiment}" --scope-csv "{input.scope_csv}" '
            '--mapping-csv "{input.mapping_csv}" --output-csv "{output.mapped_scope_csv}" '
            '--selected-wells "{params.selected_wells}"'
        )


rule validate_physical_well_mapping_yx1:
    """Validate that physical series->well mapping is complete and canonical."""
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "scope_series_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "series_well_mapping.csv",
    output:
        validated_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / ".physical_well_mapping.validated",
        diagnostics_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "yx1" / "physical_well_mapping_diagnostics.json",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.scope.shared.validate_physical_well_mapping '
            '--scope-metadata-csv "{input.scope_csv}" --mapping-csv "{input.mapping_csv}" '
            '--output-flag "{output.validated_flag}" --diagnostics-json "{output.diagnostics_json}"'
        )


rule validate_physical_well_mapping_keyence:
    """Validate that physical series->well mapping is complete and canonical."""
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "scope_series_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "series_well_mapping.csv",
    output:
        validated_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / ".physical_well_mapping.validated",
        diagnostics_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope" / "keyence" / "physical_well_mapping_diagnostics.json",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.scope.shared.validate_physical_well_mapping '
            '--scope-metadata-csv "{input.scope_csv}" --mapping-csv "{input.mapping_csv}" '
            '--output-flag "{output.validated_flag}" --diagnostics-json "{output.diagnostics_json}"'
        )

