"""
Streaming reducers for resampling.

Reducers accumulate iteration results without storing the full null
distribution. They run in the **parent process only** — workers return
scalar results, and the parent calls ``update()`` in deterministic
iteration order.

PR1 ships ``PermutationReducer`` (scalar-only, exact streaming p-value
with reservoir sampling). ``BootstrapReducer`` is deferred.
"""

from typing import Optional

import numpy as np
from numpy.random import SeedSequence


class PermutationReducer:
    """Exact streaming p-value without storing full null. Scalar outputs only.

    Runs in parent process only. Workers return scalars; parent calls
    ``update()`` in deterministic iteration order.

    Parameters
    ----------
    alternative : str, default="two-sided"
        Tail of the test: ``"greater"`` | ``"less"`` | ``"two-sided"``.
        Can be overridden in ``init()`` by the engine.
    reservoir_size : int, default=200
        Number of null samples to retain via reservoir sampling (Algorithm R).

    Notes
    -----
    Two-sided extremeness uses ``abs(val) >= abs(observed)``, NOT comparison
    to the running mean. This is order-independent and correct for
    sign-symmetric statistics. If your statistic is not symmetric around 0,
    choose ``"greater"`` or ``"less"`` explicitly.
    """

    def __init__(
        self,
        *,
        alternative: str = "two-sided",
        reservoir_size: int = 200,
    ):
        self.alternative = alternative
        self.reservoir_size = reservoir_size
        self._reservoir_rng: Optional[np.random.Generator] = None

        # State (populated after init())
        self.observed: Optional[float] = None
        self.count_extreme: int = 0
        self.count_total: int = 0
        self.running_mean: float = 0.0
        self.running_m2: float = 0.0
        self.reservoir: list = []

    def init(
        self,
        observed: float,
        alternative: Optional[str] = None,
        *,
        seedseq: Optional[SeedSequence] = None,
    ):
        """Called by the engine after computing the observed statistic.

        Parameters
        ----------
        observed : float
            Observed test statistic value.
        alternative : str, optional
            Override the tail direction. If ``None``, keep constructor value.
        seedseq : SeedSequence, optional
            From ``run()``'s 3-way split. Seeds the reservoir RNG.
        """
        self.observed = float(observed)
        if alternative is not None:
            self.alternative = alternative
        if seedseq is not None:
            self._reservoir_rng = np.random.default_rng(seedseq)

    def update(self, sample_value: float):
        """Ingest one iteration result. Called in deterministic order."""
        self.count_total += 1
        val = float(sample_value)

        # Extremeness check
        if self.alternative == "greater":
            if val >= self.observed:
                self.count_extreme += 1
        elif self.alternative == "less":
            if val <= self.observed:
                self.count_extreme += 1
        elif self.alternative == "two-sided":
            if abs(val) >= abs(self.observed):
                self.count_extreme += 1

        # Welford running mean / variance
        delta = val - self.running_mean
        self.running_mean += delta / self.count_total
        self.running_m2 += delta * (val - self.running_mean)

        # Reservoir sampling (Algorithm R)
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(val)
        else:
            j = int(self._reservoir_rng.integers(self.count_total))
            if j < self.reservoir_size:
                self.reservoir[j] = val

    def finalize(self) -> dict:
        """Return accumulated state as a plain dict."""
        B = self.count_total
        return {
            "pvalue": (1 + self.count_extreme) / (B + 1) if B > 0 else 1.0,
            "null_mean": self.running_mean,
            "null_std": (self.running_m2 / B) ** 0.5 if B > 0 else 0.0,
            "reservoir": np.array(self.reservoir),
            "n_permutations": B,
            "alternative": self.alternative,
        }
