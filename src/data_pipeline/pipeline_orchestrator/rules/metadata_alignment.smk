"""Metadata-alignment rules: extraction, mapping, and alignment."""


def _selected_wells_csv(experiment: str) -> str:
    wells = selected_wells_for_experiment(experiment)
    return ",".join(wells)


def _microscope(experiment: str) -> str:
    return microscope_for_experiment(experiment)


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
        plate_csv=EXPERIMENT_METADATA_DIR / "{experiment}" / "plate_metadata.csv",
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
        ref_xy_csv=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("ref_xy_csv")
            or config.get("phase1", {}).get("yx1_mapping", {}).get("ref_xy_csv", "")
        ),
        max_distance_um=lambda wc: (
            config.get("metadata_alignment", {}).get("yx1_mapping", {}).get("max_distance_um")
            or config.get("phase1", {}).get("yx1_mapping", {}).get("max_distance_um", 4500.0)
        )
    shell:
        (
            'PYTHONPATH="{params.pythonpath}" "{params.python}" -m data_pipeline.pipeline_orchestrator.tasks '
            'map-series --experiment "{params.experiment}" --microscope "{params.microscope}" '
            '--plate-csv "{input.plate_csv}" --scope-csv "{input.scope_csv}" '
            '--raw-images-dir "{input.raw_images_dir}" --output-mapping-csv "{output.mapping_csv}" '
            '--output-provenance-json "{output.provenance_json}" --ref-xy-csv "{params.ref_xy_csv}" '
            '--max-distance-um "{params.max_distance_um}"'
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
