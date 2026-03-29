from __future__ import annotations

import pandas as pd

from src.build.qc import determine_use_embryo_flag


def test_no_yolk_flag_is_informational_only():
    df = pd.DataFrame(
        {
            "dead_flag": [False, False],
            "dead_flag2": [False, False],
            "sa_outlier_flag": [False, False],
            "sam2_qc_flag": [False, False],
            "frame_flag": [False, True],
            "no_yolk_flag": [True, True],
        }
    )

    got = determine_use_embryo_flag(df).tolist()

    assert got == [True, False]
