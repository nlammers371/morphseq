from __future__ import annotations

import pandas as pd

from ._shared import align_to_universe


def compute_motion_qc_flags(snip_universe_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"snip_id": snip_universe_df["snip_id"].astype(str)})
    out["motion_flag"] = False
    aligned = align_to_universe(snip_universe_df, out, "motion_qc")
    return aligned[["snip_id", "motion_flag"]]


compute_motion_qc = compute_motion_qc_flags
