"""Target expansion helpers for pipeline orchestration."""

from __future__ import annotations

from typing import Dict, List

DEFAULT_SMOKE_EXPERIMENTS = ["20240418", "20240509_24hpf"]
DEFAULT_EXPERIMENT_MICROSCOPES = {
    "20240418": "YX1",
    "20240509_24hpf": "Keyence",
}
DEFAULT_EXPERIMENT_WELLS = {
    "20240418": ["A01", "C01"],
    "20240509_24hpf": ["A04", "B04"],
}


def get_experiments(config: dict) -> List[str]:
    """Return configured experiments in deterministic order."""
    experiments = config.get("experiments") or DEFAULT_SMOKE_EXPERIMENTS
    return [str(exp) for exp in experiments]


def get_experiment_microscopes(config: dict) -> Dict[str, str]:
    """Return microscope assignment for each experiment."""
    configured = dict(config.get("experiment_microscopes") or {})
    result = dict(DEFAULT_EXPERIMENT_MICROSCOPES)
    result.update({str(k): str(v) for k, v in configured.items()})
    return result


def get_experiment_wells(config: dict) -> Dict[str, List[str]]:
    """Return selected wells per experiment for smoke runs."""
    configured = dict(config.get("experiment_wells") or {})
    result = {exp: list(wells) for exp, wells in DEFAULT_EXPERIMENT_WELLS.items()}
    for exp, wells in configured.items():
        result[str(exp)] = [str(well) for well in wells]
    return result


def get_selected_wells(config: dict, experiment: str) -> List[str]:
    """Return selected wells for a specific experiment."""
    wells = get_experiment_wells(config)
    return wells.get(experiment, [])


def get_experiments_for_microscope(config: dict, microscope_id: str) -> List[str]:
    """Return experiments matching the requested microscope."""
    experiments = get_experiments(config)
    microscopes = get_experiment_microscopes(config)
    return [exp for exp in experiments if microscopes.get(exp, "") == microscope_id]
