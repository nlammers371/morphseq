"""Make a bootstrap-SE version of the tempo/noise comparison figure."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd

from figure_utils import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CACHE_DIR,
    add_identity,
    bootstrap_std_se,
    drop_excluded_temperatures,
    savefig,
    set_light_style,
    temperature_timepoint_scatter,
)

import numpy as np


def build_bootstrap_summary(joint: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (temperature, timepoint), group in joint.groupby(["temperature", "timepoint"], sort=True):
        rows.append(
            {
                "temperature": temperature,
                "timepoint": timepoint,
                "morph_stage_mean": group["mdl_stage_hpf"].mean(),
                "morph_stage_std": group["mdl_stage_hpf"].std(ddof=1),
                "morph_stage_std_boot_se": bootstrap_std_se(group["mdl_stage_hpf"], rng),
                "seq_stage_mean": group["pseudostage"].mean(),
                "seq_stage_std": group["pseudostage"].std(ddof=1),
                "seq_stage_std_boot_se": bootstrap_std_se(group["pseudostage"], rng),
                "n": group["snip_id"].count(),
                "n_bootstrap": BOOTSTRAP_N,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_light_style()
    joint = drop_excluded_temperatures(pd.read_csv(CACHE_DIR / "joint_141_morph_seq.csv"))
    cohort = build_bootstrap_summary(joint)
    cohort.to_csv(CACHE_DIR / "tempo_noise_cohort_summary_bootstrap_se_no19C.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.errorbar(
        cohort["morph_stage_std"],
        cohort["seq_stage_std"],
        xerr=cohort["morph_stage_std_boot_se"],
        yerr=cohort["seq_stage_std_boot_se"],
        fmt="none",
        ecolor="#444444",
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.75,
        zorder=1,
    )
    temperature_timepoint_scatter(
        ax,
        cohort["morph_stage_std"],
        cohort["seq_stage_std"],
        cohort["temperature"],
        cohort["timepoint"],
        s=55,
        zorder=2,
    )
    add_identity(ax, cohort["morph_stage_std"], cohort["seq_stage_std"])
    ax.set_xlabel("morphology stage variability (sd hpf)")
    ax.set_ylabel("sequence stage variability (sd hpf)")
    ax.set_title("Staging variability with bootstrap SE")
    savefig(fig, "01b_staging_variability_comparison_bootstrap_se")
    plt.close(fig)


if __name__ == "__main__":
    main()
