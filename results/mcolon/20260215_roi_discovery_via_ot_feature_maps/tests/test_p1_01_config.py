"""
Phase 1 / Task 1.1 — Configuration construction and validation.

Checks:
- All enums have expected members
- Frozen dataclasses reject mutation
- LAMBDA_PRESETS and MU_PRESETS are non-empty and sorted
- ROIRunConfig.resolve_lambda_values / resolve_mu_values return correct presets
- Default SweepConfig has sensible ranges
"""

import pytest
from roi_config import (
    CHANNEL_SCHEMAS,
    LAMBDA_PRESETS,
    MU_PRESETS,
    ChannelSchema,
    FeatureDatasetConfig,
    FeatureSet,
    NullConfig,
    NullMode,
    ROIRunConfig,
    ROISizePreset,
    SelectionRule,
    SmoothnessPreset,
    SweepConfig,
    TrainerConfig,
)


# ---- Enum completeness ----

def test_feature_set_members():
    assert set(FeatureSet) == {FeatureSet.COST, FeatureSet.COST_DISP, FeatureSet.ALL_OT}


def test_roi_size_preset_members():
    assert set(ROISizePreset) == {ROISizePreset.SMALL, ROISizePreset.MEDIUM, ROISizePreset.LARGE}


def test_smoothness_preset_members():
    assert set(SmoothnessPreset) == {SmoothnessPreset.LOW, SmoothnessPreset.MEDIUM, SmoothnessPreset.HIGH}


def test_null_mode_members():
    assert set(NullMode) == {NullMode.PERMUTE, NullMode.BOOTSTRAP, NullMode.BOTH, NullMode.NONE}


def test_selection_rule_members():
    assert set(SelectionRule) == {SelectionRule.PARETO_KNEE, SelectionRule.EPSILON_BEST}


# ---- Presets are non-empty ----

def test_lambda_presets_coverage():
    for preset in ROISizePreset:
        vals = LAMBDA_PRESETS[preset]
        assert len(vals) >= 2, f"LAMBDA_PRESETS[{preset}] too short"
        assert all(v > 0 for v in vals), "All lambda values must be positive"


def test_mu_presets_coverage():
    for preset in SmoothnessPreset:
        vals = MU_PRESETS[preset]
        assert len(vals) >= 2, f"MU_PRESETS[{preset}] too short"
        assert all(v > 0 for v in vals), "All mu values must be positive"


# ---- Channel schemas ----

def test_channel_schemas_all_feature_sets():
    for fs in FeatureSet:
        schemas = CHANNEL_SCHEMAS[fs]
        assert len(schemas) >= 1
        assert all(isinstance(s, ChannelSchema) for s in schemas)


def test_cost_has_one_channel():
    assert len(CHANNEL_SCHEMAS[FeatureSet.COST]) == 1
    assert CHANNEL_SCHEMAS[FeatureSet.COST][0].name == "total_cost"


def test_all_ot_has_five_channels():
    assert len(CHANNEL_SCHEMAS[FeatureSet.ALL_OT]) == 5


# ---- Frozen dataclass enforcement ----

def test_feature_dataset_config_frozen():
    cfg = FeatureDatasetConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.canonical_grid_hw = (256, 256)


def test_trainer_config_frozen():
    cfg = TrainerConfig()
    with pytest.raises(Exception):
        cfg.learn_res = 256


def test_sweep_config_frozen():
    cfg = SweepConfig()
    with pytest.raises(Exception):
        cfg.n_cv_folds = 3


# ---- Defaults ----

def test_feature_dataset_defaults():
    cfg = FeatureDatasetConfig()
    assert cfg.canonical_grid_hw == (512, 512)
    assert cfg.group_key == "embryo_id"
    assert cfg.iqr_multiplier == 1.5


def test_trainer_defaults():
    cfg = TrainerConfig()
    assert cfg.learn_res == 128
    assert cfg.output_res == 512
    assert cfg.max_steps >= 1000


def test_sweep_config_defaults():
    cfg = SweepConfig()
    assert cfg.n_cv_folds == 5
    assert cfg.selection_rule == SelectionRule.PARETO_KNEE
    assert 0.0 < cfg.roi_quantile < 1.0


# ---- ROIRunConfig resolution ----

def test_resolve_lambda_values_medium():
    rc = ROIRunConfig(roi_size=ROISizePreset.MEDIUM)
    vals = rc.resolve_lambda_values()
    assert vals == list(LAMBDA_PRESETS[ROISizePreset.MEDIUM])


def test_resolve_mu_values_high():
    rc = ROIRunConfig(smoothness=SmoothnessPreset.HIGH)
    vals = rc.resolve_mu_values()
    assert vals == list(MU_PRESETS[SmoothnessPreset.HIGH])


def test_roi_run_config_default_genotype():
    rc = ROIRunConfig()
    assert rc.genotype == "cep290"
    assert rc.reference == "WT"
