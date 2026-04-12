from __future__ import annotations

import matplotlib.pyplot as plt


PLOT_FIGSIZE_IN = (7.6, 6.9)
PLOT_DPI = 120
TICK_LABELSIZE = 18
AXIS_LABELSIZE = 21
TITLE_SIZE = 23
SUBPLOTS = {
    "left": 0.12,
    "right": 0.98,
    "bottom": 0.12,
    "top": 0.90,
}


def apply_transition_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": PLOT_DPI,
            "savefig.dpi": PLOT_DPI,
            "xtick.labelsize": TICK_LABELSIZE,
            "ytick.labelsize": TICK_LABELSIZE,
            "axes.labelsize": AXIS_LABELSIZE,
            "axes.titlesize": TITLE_SIZE,
        }
    )


def style_transition_figure(fig) -> None:
    apply_transition_rcparams()
    fig.set_dpi(PLOT_DPI)
    fig.set_size_inches(*PLOT_FIGSIZE_IN, forward=True)
    fig.subplots_adjust(**SUBPLOTS)
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
        ax.xaxis.label.set_size(AXIS_LABELSIZE)
        ax.yaxis.label.set_size(AXIS_LABELSIZE)
        ax.title.set_size(TITLE_SIZE)
