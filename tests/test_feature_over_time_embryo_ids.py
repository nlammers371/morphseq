import pandas as pd

from analyze.viz.plotting import IdTraceStyle, plot_feature_over_time


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "embryo_id": ["emb1", "emb1", "emb2", "emb2"],
            "predicted_stage_hpf": [24.0, 26.0, 24.0, 26.0],
            "metric": [0.2, 0.4, 0.3, 0.5],
            "group": ["wt", "wt", "wt", "wt"],
        }
    )


def test_plot_feature_over_time_include_ids_filters_individual_traces():
    fig = plot_feature_over_time(
        _sample_df(),
        features="metric",
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="group",
        show_individual=True,
        show_trend=False,
        backend="matplotlib",
        include_ids=["emb1"],
    )

    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [24.0, 26.0]


def test_plot_feature_over_time_highlight_ids_adds_emphasized_trace():
    fig = plot_feature_over_time(
        _sample_df(),
        features="metric",
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="group",
        show_individual=True,
        show_trend=False,
        backend="matplotlib",
        highlight_ids=["emb2"],
        id_style_lookup={
            "emb2": IdTraceStyle(color="#ff0000", width=5.0, alpha=1.0),
        },
    )

    ax = fig.axes[0]
    assert len(ax.lines) == 3
    widths = [float(line.get_linewidth()) for line in ax.lines]
    colors = [str(line.get_color()) for line in ax.lines]
    assert 5.0 in widths
    assert "#ff0000" in colors
