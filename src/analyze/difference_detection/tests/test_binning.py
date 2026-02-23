import unittest

import numpy as np
import pandas as pd

from analyze.utils import binning


class BinningTests(unittest.TestCase):
    def test_add_time_bins_handles_nan(self):
        df = pd.DataFrame({"predicted_stage_hpf": [1.2, np.nan, 5.9]})
        out = binning.add_time_bins(df, bin_width=2.0)

        self.assertEqual(out["time_bin"].iloc[0], 0)
        self.assertTrue(pd.isna(out["time_bin"].iloc[1]))
        self.assertEqual(out["time_bin"].iloc[2], 4)

    def test_add_time_bins_preserves_int_dtype_without_nan(self):
        df = pd.DataFrame({"predicted_stage_hpf": [0.1, 2.0, 3.9]})
        out = binning.add_time_bins(df, bin_width=2.0)

        self.assertEqual(out["time_bin"].tolist(), [0, 2, 2])
        self.assertIn(out["time_bin"].dtype.kind, ("i", "u"))


if __name__ == "__main__":
    unittest.main()
