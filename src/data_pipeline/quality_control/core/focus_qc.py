from __future__ import annotations

import pandas as pd

from ._shared import align_to_universe


def compute_focus_qc_flags(snip_universe_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"snip_id": snip_universe_df["snip_id"].astype(str)})
    out["focus_flag"] = False
    aligned = align_to_universe(snip_universe_df, out, "focus_qc")
    return aligned[["snip_id", "focus_flag"]]


compute_focus_qc = compute_focus_qc_flags
