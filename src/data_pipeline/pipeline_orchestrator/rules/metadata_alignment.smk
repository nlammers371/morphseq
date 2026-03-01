"""Metadata-alignment rules: extraction, mapping, and alignment."""


def _selected_wells_csv(experiment: str) -> str:
    wells = selected_wells_for_experiment(experiment)
    return ",".join(wells)


def _microscope(experiment: str) -> str:
    return microscope_for_experiment(experiment)

def _allow_unmapped_wells() -> bool:
    return str(config.get("metadata_alignment", {}).get("allow_unmapped_wells", False)).strip().lower() in {"1", "true", "yes", "y", "on"}


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


rule extract_scope_metadata:
    input:
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / _microscope(wc.experiment) / wc.experiment
    output:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_raw.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        microscope=lambda wc: _microscope(wc.experiment),
        experiment=lambda wc: wc.experiment
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'extract-scope --raw-images-dir "{input.raw_images_dir}" --experiment "{params.experiment}" '
            '--microscope "{params.microscope}" --output-csv "{output.scope_csv}"'
        )


rule map_series_to_wells:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_raw.csv",
        raw_images_dir=lambda wc: RAW_IMAGES_DIR / _microscope(wc.experiment) / wc.experiment
    output:
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "series_well_mapping.csv",
        provenance_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "mapping_provenance.json"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        microscope=lambda wc: _microscope(wc.experiment),
        experiment=lambda wc: wc.experiment,
        allow_unmapped=lambda wc: str(_allow_unmapped_wells()).lower(),
        ref_xy_csv=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("ref_xy_csv")
            or config.get("phase1", {}).get("yx1_mapping", {}).get("ref_xy_csv", "")
        ),
        row_y_tol_um=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("row_y_tol_um", 1200.0)
        ),
        col_x_tol_um=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("col_x_tol_um", 1200.0)
        ),
        dx_cv_tol=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("dx_cv_tol", 0.15)
        ),
        dy_cv_tol=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("dy_cv_tol", 0.15)
        ),
        max_distance_um=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("max_distance_um")
            or config.get("phase1", {}).get("yx1_mapping", {}).get("max_distance_um", 4500.0)
        )
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'map-series --experiment "{params.experiment}" --microscope "{params.microscope}" '
            '--scope-csv "{input.scope_csv}" '
            '--raw-images-dir "{input.raw_images_dir}" --output-mapping-csv "{output.mapping_csv}" '
            '--output-provenance-json "{output.provenance_json}" --ref-xy-csv "{params.ref_xy_csv}" '
            '--max-distance-um "{params.max_distance_um}" '
            '--allow-unmapped-wells "{params.allow_unmapped}"'
            ' --row-y-tol-um "{params.row_y_tol_um}" --col-x-tol-um "{params.col_x_tol_um}"'
            ' --dx-cv-tol "{params.dx_cv_tol}" --dy-cv-tol "{params.dy_cv_tol}"'
        )


rule apply_series_mapping:
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "series_well_mapping.csv"
    output:
        mapped_scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_mapped.csv"
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

rule validate_physical_well_mapping:
    """Validate that physical series->well mapping is complete and canonical before Phase 3."""
    input:
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_raw.csv",
        mapping_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "series_well_mapping.csv",
    output:
        validated_flag=EXPERIMENT_METADATA_DIR / "{experiment}" / ".physical_well_mapping.validated",
        diagnostics_json=EXPERIMENT_METADATA_DIR / "{experiment}" / "physical_well_mapping_diagnostics.json",
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT,
        allow_unmapped=lambda wc: str(_allow_unmapped_wells()).lower(),
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.metadata_ingest.mapping.validate_physical_well_mapping '
            '--scope-metadata-csv "{input.scope_csv}" --mapping-csv "{input.mapping_csv}" '
            '--output-flag "{output.validated_flag}" --diagnostics-json "{output.diagnostics_json}" '
            '--allow-unmapped-wells "{params.allow_unmapped}"'
        )


rule align_scope_and_plate:
    input:
        plate_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv",
        scope_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_metadata_mapped.csv"
    output:
        aligned_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "scope_and_plate_metadata.csv"
    params:
        python=PYTHON_EXE,
        pythonpath=SRC_ROOT
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'align --plate-csv "{input.plate_csv}" --scope-csv "{input.scope_csv}" '
            '--output-csv "{output.aligned_csv}"'
        )
