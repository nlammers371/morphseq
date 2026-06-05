"""
NWDB-specific color presets.

These are project palettes, not generic viz defaults. They are defined as
explicit ColorPreset objects so provenance stays visible at the call site.
"""

from __future__ import annotations

from analyze.viz.styling import build_color_preset


NWDB_PHENOTYPE_TALK = build_color_preset(
    {
        "High_to_Low": "#E76FA2",
        "Low_to_High": "#2FB7B0",
        "Not Penetrant": "#3A3A3A",
    },
    order=[
        "High_to_Low",
        "Low_to_High",
        "Not Penetrant",
    ],
    fill="error",
    name="nwdb/phenotype_talk",
    source=__file__,
)
