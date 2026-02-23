"""
Configuration dataclasses for ROI discovery via OT feature maps.

Follows the same frozen-dataclass pattern as UOTConfig in
src/analyze/utils/optimal_transport/config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class FeatureSet(str, Enum):
    """Which OT-derived channels to use as input features."""
    COST = "cost"
    COST_DISP = "cost+disp"
    ALL_OT = "all_ot"


class Phase0FeatureSet(str, Enum):
    """Phase 0 channel sets (1D S-bin analysis, not 2D weight maps)."""
    V0_COST = "v0_cost"          # C=1: cost_density only
    V1_DYNAMICS = "v1_dynamics"  # C=5: cost + disp_u + disp_v + disp_mag + delta_mass


class ROISizePreset(str, Enum):
    """Maps to λ (L1 penalty) grid presets."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SmoothnessPreset(str, Enum):
    """Maps to μ (TV penalty) grid presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NullMode(str, Enum):
    PERMUTE = "permute"
    BOOTSTRAP = "bootstrap"
    BOTH = "both"
    NONE = "none"


class SelectionRule(str, Enum):
    """Deterministic selection rule for (λ,μ) from sweep."""
    PARETO_KNEE = "pareto_knee"       # Option A: knee on Pareto front
    EPSILON_BEST = "epsilon_best"     # Option B: smallest complexity within ε of best AUROC


# ---------------------------------------------------------------------------
# λ/μ presets — these are *starting points*, not universal truths.
# Biology-dependent tuning will refine them.
# ---------------------------------------------------------------------------

LAMBDA_PRESETS = {
    ROISizePreset.SMALL:  [1e-2, 3e-2, 1e-1, 3e-1],
    ROISizePreset.MEDIUM: [1e-3, 3e-3, 1e-2, 3e-2],
    ROISizePreset.LARGE:  [1e-4, 3e-4, 1e-3, 3e-3],
}

MU_PRESETS = {
    SmoothnessPreset.LOW:    [1e-4, 1e-3, 1e-2],
    SmoothnessPreset.MEDIUM: [1e-3, 1e-2, 1e-1],
    SmoothnessPreset.HIGH:   [1e-2, 1e-1, 1.0],
}


@dataclass(frozen=True)
class FeatureDatasetConfig:
    """Configuration for the on-disk FeatureDataset contract."""
    canonical_grid_hw: Tuple[int, int] = (256, 576)
    chunk_size_n: int = 8
    compression: str = "zstd"
    compression_level: int = 3
    iqr_multiplier: float = 2.0       # for QC outlier filter on total_cost_C
    group_key: str = "embryo_id"      # MANDATORY: prevents leakage in CV splits


@dataclass(frozen=True)
class ChannelSchema:
    """Describes a single feature channel in the dataset."""
    name: str
    definition: str
    units: str


# Phase 0 channel schemas
PHASE0_CHANNEL_SCHEMAS = {
    Phase0FeatureSet.V0_COST: [
        ChannelSchema("cost_density", "Per-pixel OT transport cost (ref→target)", "cost_units"),
    ],
    Phase0FeatureSet.V1_DYNAMICS: [
        ChannelSchema("cost_density", "Per-pixel OT transport cost (ref→target)", "cost_units"),
        ChannelSchema("disp_u", "Displacement field x-component (ref→target)", "um"),
        ChannelSchema("disp_v", "Displacement field y-component (ref→target)", "um"),
        ChannelSchema("disp_mag", "Displacement magnitude sqrt(u^2+v^2)", "um"),
        ChannelSchema("delta_mass", "Unbalanced OT mass difference (created-destroyed)", "mass_units"),
    ],
}

# Phase 0 S-bin feature columns by channel set
PHASE0_SBIN_FEATURES = {
    Phase0FeatureSet.V0_COST: ["cost_mean"],
    Phase0FeatureSet.V1_DYNAMICS: [
        "cost_mean", "disp_mag_mean", "disp_par_mean", "disp_perp_mean",
    ],
}


# Standard channel schemas for OT-derived features (Phase 1 2D ROI)
CHANNEL_SCHEMAS = {
    FeatureSet.COST: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
    ],
    FeatureSet.COST_DISP: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
        ChannelSchema("displacement_y", "Barycentric displacement (y)", "um"),
        ChannelSchema("displacement_x", "Barycentric displacement (x)", "um"),
    ],
    FeatureSet.ALL_OT: [
        ChannelSchema("total_cost", "Per-pixel OT transport cost", "cost_units"),
        ChannelSchema("displacement_y", "Barycentric displacement (y)", "um"),
        ChannelSchema("displacement_x", "Barycentric displacement (x)", "um"),
        ChannelSchema("mass_created", "Mass creation per pixel", "mass_units"),
        ChannelSchema("mass_destroyed", "Mass destruction per pixel", "mass_units"),
    ],
}


@dataclass(frozen=True)
class TrainerConfig:
    """JAX trainer configuration."""
    learn_res: int = 128
    output_res: int = 512
    learning_rate: float = 1e-2
    max_steps: int = 2000
    convergence_tol: float = 1e-6
    log_every: int = 100
    random_seed: int = 42


@dataclass(frozen=True)
class SweepConfig:
    """λ/μ sweep configuration."""
    lambda_values: Tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    mu_values: Tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    n_cv_folds: int = 5
    selection_rule: SelectionRule = SelectionRule.PARETO_KNEE
    # For PARETO_KNEE: beta controls knee sensitivity
    pareto_beta: float = 1.0
    # For EPSILON_BEST: ε tolerance on AUROC
    epsilon_auroc: float = 0.02
    # ROI extraction
    roi_quantile: float = 0.9    # threshold |w| at this quantile


@dataclass(frozen=True)
class NullConfig:
    """Null distribution + stability configuration."""
    null_mode: NullMode = NullMode.BOTH
    n_permute: int = 100
    n_boot: int = 200
    boot_roi_quantile: float = 0.9
    random_seed: int = 42


@dataclass
class ROIRunConfig:
    """Top-level run configuration combining all sub-configs."""
    genotype: str = "cep290"
    reference: str = "WT"
    features: FeatureSet = FeatureSet.COST
    roi_size: ROISizePreset = ROISizePreset.MEDIUM
    smoothness: SmoothnessPreset = SmoothnessPreset.MEDIUM

    dataset: FeatureDatasetConfig = field(default_factory=FeatureDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    nulls: NullConfig = field(default_factory=NullConfig)

    out_dir: Optional[str] = None

    def resolve_lambda_values(self) -> List[float]:
        """Get λ values from preset or sweep config."""
        return list(LAMBDA_PRESETS.get(self.roi_size, self.sweep.lambda_values))

    def resolve_mu_values(self) -> List[float]:
        """Get μ values from preset or sweep config."""
        return list(MU_PRESETS.get(self.smoothness, self.sweep.mu_values))


# ---------------------------------------------------------------------------
# Phase 0 configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase0SCoordinateConfig:
    """Configuration for S coordinate computation."""
    centerline_method: str = "geodesic"   # 'geodesic' or 'pca'
    orient_head_to_tail: bool = True      # S=0 head, S=1 tail
    bspline_smoothing: float = 5.0
    random_seed: int = 42


@dataclass(frozen=True)
class Phase0SBinConfig:
    """Configuration for S-bin discretization."""
    K: int = 10                           # number of bins (also run K=20 as robustness check)
    K_robustness: int = 20                # secondary K for robustness


@dataclass(frozen=True)
class Phase0ClassificationConfig:
    """Configuration for Phase 0 AUROC + logistic classification."""
    n_cv_folds: int = 5
    group_key: str = "embryo_id"
    random_seed: int = 42


class Phase0IntervalSelectionRule(str, Enum):
    """Selection rule for best S-interval."""
    PARSIMONY = "parsimony"  # smallest interval within ε of best AUROC
    PENALIZED = "penalized"  # maximize AUROC - gamma*(len/K)


@dataclass(frozen=True)
class Phase0IntervalConfig:
    """Configuration for 1D interval search on S."""
    selection_rule: Phase0IntervalSelectionRule = Phase0IntervalSelectionRule.PARSIMONY
    epsilon_auroc: float = 0.02       # for PARSIMONY: tolerance from best
    gamma_penalty: float = 0.01       # for PENALIZED: length penalty weight
    min_interval_bins: int = 1
    max_interval_bins: Optional[int] = None  # None = K


@dataclass(frozen=True)
class Phase0NullConfig:
    """Configuration for Phase 0 null + stability tests."""
    n_permute: int = 200
    n_boot: int = 200
    random_seed: int = 42


@dataclass
class Phase0RunConfig:
    """Top-level Phase 0 run configuration."""
    genotype: str = "cep290"
    reference: str = "WT"
    stage_window: Tuple[float, float] = (0.0, 2.0)  # one 2 hpf bin

    feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST
    dataset: FeatureDatasetConfig = field(default_factory=FeatureDatasetConfig)
    s_coord: Phase0SCoordinateConfig = field(default_factory=Phase0SCoordinateConfig)
    s_bins: Phase0SBinConfig = field(default_factory=Phase0SBinConfig)
    classification: Phase0ClassificationConfig = field(default_factory=Phase0ClassificationConfig)
    interval: Phase0IntervalConfig = field(default_factory=Phase0IntervalConfig)
    nulls: Phase0NullConfig = field(default_factory=Phase0NullConfig)

    out_dir: Optional[str] = None

    # Visualization smoothing (display only, not for stats)
    viz_sigma_grid: Tuple[float, ...] = (1.0, 2.0, 4.0)
    quiver_stride: int = 8


__all__ = [
    "FeatureSet",
    "Phase0FeatureSet",
    "ROISizePreset",
    "SmoothnessPreset",
    "NullMode",
    "SelectionRule",
    "FeatureDatasetConfig",
    "ChannelSchema",
    "CHANNEL_SCHEMAS",
    "PHASE0_CHANNEL_SCHEMAS",
    "PHASE0_SBIN_FEATURES",
    "TrainerConfig",
    "SweepConfig",
    "NullConfig",
    "ROIRunConfig",
    "LAMBDA_PRESETS",
    "MU_PRESETS",
    "Phase0SCoordinateConfig",
    "Phase0SBinConfig",
    "Phase0ClassificationConfig",
    "Phase0IntervalSelectionRule",
    "Phase0IntervalConfig",
    "Phase0NullConfig",
    "Phase0RunConfig",
]
