from __future__ import annotations

import pandas as pd

from ._shared import align_to_universe


def compute_auxiliary_mask_qc_flags(
    auxiliary_masks_df: pd.DataFrame | None,
    snip_universe_df: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame({"snip_id": snip_universe_df["snip_id"].astype(str)})
    out["yolk_flag"] = False
    out["bubble_flag"] = False
    if auxiliary_masks_df is not None and "snip_id" in auxiliary_masks_df.columns:
        # Validate only the snip universe coverage. This stub does not inspect masks.
        observed = set(auxiliary_masks_df["snip_id"].astype(str).tolist())
        expected = set(out["snip_id"].tolist())
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                "auxiliary_mask_qc: snip_id set mismatch "
                f"(missing={missing}, extra={extra})"
            )
    aligned = align_to_universe(snip_universe_df, out, "auxiliary_mask_qc")
    return aligned[["snip_id", "yolk_flag", "bubble_flag"]]


compute_auxiliary_mask_qc = compute_auxiliary_mask_qc_flags
