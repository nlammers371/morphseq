# Quality control (per experiment)

rule compute_segmentation_qc:
    input:
        segmentation_tracking = SEGMENTATION_DIR / "{experiment}" / "contracts" / "segmentation_tracking.csv",
        segmentation_tracking_done = SEGMENTATION_DIR / "{experiment}" / "contracts" / ".segmentation_tracking_merged.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "segmentation_qc" / "segmentation_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "segmentation_qc" / "segmentation_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_segmentation_qc \
          --segmentation-tracking-csv "{input.segmentation_tracking}" \
          --output-csv "{output.csv}"
        """


rule compute_viability_qc:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "viability_qc" / "viability_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "viability_qc" / "viability_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_viability_qc \
          --fraction-alive-csv "{input.features}" \
          --features-csv "{input.features}" \
          --output-csv "{output.csv}"
        """


rule compute_death_detection:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "death_detection" / "death_detection_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "death_detection" / "death_detection_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_death_detection           --fraction-alive-csv "{input.features}"           --features-csv "{input.features}"           --output-csv "{output.csv}"
        """


rule compute_surface_area_qc:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
        sa_reference = PROJECT_ROOT / "metadata" / "sa_reference_curves.csv",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "surface_area_qc" / "surface_area_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "surface_area_qc" / "surface_area_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_surface_area_qc \
          --features-csv "{input.features}" \
          --sa-reference-csv "{input.sa_reference}" \
          --output-csv "{output.csv}"
        """


rule compute_auxiliary_mask_qc:
    input:
        auxiliary_masks = AUXILIARY_MASKS_DIR / "{experiment}" / "contracts" / "auxiliary_masks.csv",
        auxiliary_masks_done = AUXILIARY_MASKS_DIR / "{experiment}" / "contracts" / ".auxiliary_masks.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "auxiliary_mask_qc" / "auxiliary_mask_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "auxiliary_mask_qc" / "auxiliary_mask_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_auxiliary_mask_qc \
          --auxiliary-masks-csv "{input.auxiliary_masks}" \
          --output-csv "{output.csv}"
        """


rule compute_focus_qc:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "focus_qc" / "focus_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "focus_qc" / "focus_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_focus_qc \
          --features-csv "{input.features}" \
          --output-csv "{output.csv}"
        """


rule compute_motion_qc:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "motion_qc" / "motion_qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "motion_qc" / "motion_qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.compute_motion_qc \
          --features-csv "{input.features}" \
          --output-csv "{output.csv}"
        """


rule consolidate_qc:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
        segmentation_qc = QUALITY_CONTROL_DIR / "{experiment}" / "segmentation_qc" / "segmentation_qc_flags.csv",
        segmentation_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "segmentation_qc" / "segmentation_qc_flags.csv.validated",
        viability_qc = QUALITY_CONTROL_DIR / "{experiment}" / "viability_qc" / "viability_qc_flags.csv",
        viability_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "viability_qc" / "viability_qc_flags.csv.validated",
        death_detection = QUALITY_CONTROL_DIR / "{experiment}" / "death_detection" / "death_detection_flags.csv",
        death_detection_done = QUALITY_CONTROL_DIR / "{experiment}" / "death_detection" / "death_detection_flags.csv.validated",
        surface_area_qc = QUALITY_CONTROL_DIR / "{experiment}" / "surface_area_qc" / "surface_area_qc_flags.csv",
        surface_area_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "surface_area_qc" / "surface_area_qc_flags.csv.validated",
        auxiliary_mask_qc = QUALITY_CONTROL_DIR / "{experiment}" / "auxiliary_mask_qc" / "auxiliary_mask_qc_flags.csv",
        auxiliary_mask_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "auxiliary_mask_qc" / "auxiliary_mask_qc_flags.csv.validated",
        focus_qc = QUALITY_CONTROL_DIR / "{experiment}" / "focus_qc" / "focus_qc_flags.csv",
        focus_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "focus_qc" / "focus_qc_flags.csv.validated",
        motion_qc = QUALITY_CONTROL_DIR / "{experiment}" / "motion_qc" / "motion_qc_flags.csv",
        motion_qc_done = QUALITY_CONTROL_DIR / "{experiment}" / "motion_qc" / "motion_qc_flags.csv.validated",
    output:
        csv = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / "qc_flags.csv",
        done = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / "qc_flags.csv.validated",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.quality_control.entrypoints.consolidate_qc \
          --features-csv "{input.features}" \
          --segmentation-qc-csv "{input.segmentation_qc}" \
          --viability-qc-csv "{input.viability_qc}" \
          --death-detection-csv "{input.death_detection}" \
          --surface-area-qc-csv "{input.surface_area_qc}" \
          --auxiliary-mask-qc-csv "{input.auxiliary_mask_qc}" \
          --focus-qc-csv "{input.focus_qc}" \
          --motion-qc-csv "{input.motion_qc}" \
          --output-csv "{output.csv}"
        """


rule assemble_analysis_ready:
    input:
        features = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv",
        features_done = FEATURES_DIR / "{experiment}" / "consolidated" / "consolidated_snip_features.csv.validated",
        qc_flags = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / "qc_flags.csv",
        qc_flags_done = QUALITY_CONTROL_DIR / "{experiment}" / "consolidated" / "qc_flags.csv.validated",
    output:
        csv = ANALYSIS_READY_DIR / "{experiment}" / "analysis_ready.csv",
        done = ANALYSIS_READY_DIR / "{experiment}" / ".analysis_ready.validated",
        schema = ANALYSIS_READY_DIR / "{experiment}" / "analysis_ready.schema.json",
    shell:
        """
        PYTHONPATH="{PROJECT_ROOT}:{PROJECT_ROOT}/src" "{PYTHON}" -m data_pipeline.analysis_ready.entrypoints.assemble_analysis_ready \
          --features-csv "{input.features}" \
          --qc-flags-csv "{input.qc_flags}" \
          --output-csv "{output.csv}" \
          --output-schema-json "{output.schema}"
        """
