import unittest

import numpy as np
import pandas as pd


try:
    import matplotlib  # noqa: F401
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


if MPL_AVAILABLE:
    from analyze.difference_detection import penetrance_threshold as pt


@unittest.skipIf(not MPL_AVAILABLE, "matplotlib not available")
class PenetranceThresholdTests(unittest.TestCase):
    def test_compute_iqr_bounds_raises_on_empty(self):
        df = pd.DataFrame({"metric": [np.nan, np.nan]})
        with self.assertRaisesRegex(ValueError, "no valid values"):
            pt.compute_iqr_bounds(df, "metric")

    def test_compute_iqr_bounds_basic(self):
        df = pd.DataFrame({"metric": [0.0, 1.0, 2.0, 3.0]})
        bounds = pt.compute_iqr_bounds(df, "metric")
        self.assertEqual(bounds["n_samples"], 4)
        self.assertLessEqual(bounds["low"], bounds["high"])


if __name__ == "__main__":
    unittest.main()
