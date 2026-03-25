from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG_THRESHOLD = 0.01


def apply_nwdb_axis_overrides(ax: plt.Axes) -> None:
    ax.set_xlim((24.0, 120.0))


def apply_nwdb_legend(ax: plt.Axes, *, outside: bool) -> None:
    if outside:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=10,
            frameon=False,
        )
    else:
        ax.legend(loc="upper left", fontsize=10, frameon=False)


def save_figure(
    fig: plt.Figure,
    out_png: Path,
    out_pdf: Path,
    *,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
    use_tight_layout: bool = True,
    close: bool = True,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        if tight_layout_rect is not None:
            fig.tight_layout(rect=tight_layout_rect)
        else:
            fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    if close:
        plt.close(fig)
