import json

import numpy as np
import pytest

from analyze.utils.resampling.lightweight_numpy_resampling import run_lite


def _kernel(rng: np.random.Generator, n_iters: int):
    obs = np.array([0.2, 0.4], dtype=float)
    draws = rng.random((n_iters, 2))
    exceed = (draws >= obs).sum(axis=0)
    mean = draws.mean(axis=0)
    std = draws.std(axis=0)
    pval = (exceed + 1) / (n_iters + 1)
    z = (obs - mean) / (std + 1e-9)
    return {
        "stat_name": "toy",
        "exceed_count": exceed,
        "null_mean": mean,
        "null_std": std,
        "obs_stat": obs,
        "pval": pval,
        "z_score": z,
    }


def test_run_lite_deterministic_given_seed():
    run1 = run_lite(test_name="t", n_iters=100, seed=42, spec={"a": 1}, kernel=_kernel)
    run2 = run_lite(test_name="t", n_iters=100, seed=42, spec={"a": 1}, kernel=_kernel)
    assert np.allclose(run1.summary["null_mean"], run2.summary["null_mean"])
    assert np.array_equal(run1.summary["exceed_count"], run2.summary["exceed_count"])


def test_collect_samples_requires_kernel_samples():
    with pytest.raises(ValueError):
        run_lite(test_name="t", n_iters=10, seed=1, spec={}, kernel=_kernel, collect_samples=True)


def test_spec_json_roundtrip():
    spec = {"type": "permute_labels", "within": "time_bin"}
    run = run_lite(test_name="t", n_iters=10, seed=1, spec=spec, kernel=_kernel)
    payload = json.loads(json.dumps(run.spec))
    assert payload == spec
