"""Top-level analyze package exports."""

import importlib

__all__ = [
    'plot_experiment_time_coverage',
    'experiment_hpf_coverage',
    'longest_interval_where',
    'plot_hpf_overlap_quick',
]


_HPF_EXPORTS = {
    'plot_experiment_time_coverage',
    'experiment_hpf_coverage',
    'longest_interval_where',
    'plot_hpf_overlap_quick',
}


def __getattr__(name: str):
    if name in _HPF_EXPORTS:
        module = importlib.import_module(f"{__name__}.viz.hpf_coverage")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
