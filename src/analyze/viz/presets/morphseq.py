"""
MorphSeq-specific color presets.

These are project palettes, not generic viz defaults. They are defined as
explicit ColorPreset objects so provenance stays visible at the call site.
"""

from __future__ import annotations

from analyze.viz.styling import build_color_preset


PBX_TALK = build_color_preset(
    {
        "wik_ab": "#1f77b4",
        "inj_ctrl": "#7f7f7f",
        "pbx1b_crispant": "#9467bd",
        "pbx4_crispant": "#d62728",
        "pbx1b_pbx4_crispant": "#b2182b",
    },
    order=[
        "wik_ab",
        "inj_ctrl",
        "pbx1b_crispant",
        "pbx4_crispant",
        "pbx1b_pbx4_crispant",
    ],
    fill="error",
    name="morphseq/pbx_talk",
    source=__file__,
)
