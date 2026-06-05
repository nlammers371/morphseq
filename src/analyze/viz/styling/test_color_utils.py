import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.environ.get("MORPHSEQ_REPO_ROOT")
if not REPO_ROOT:
    REPO_ROOT = str(Path(__file__).resolve().parents[4])

SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyze.trajectory_analysis.viz.styling import get_color_for_genotype as legacy_get_color_for_genotype
from analyze.viz.styling import (
    ColorPreset,
    build_genotype_color_lookup,
    get_color_for_genotype,
    get_known_genotype_color,
    resolve_color_lookup,
)


def test_known_genotype_default_colors_are_used_before_palette():
    colors = resolve_color_lookup(
        ["pbx4", "inj_ctrl", "cep290_homozygous", "mystery_label"],
        enforce_distinct=True,
        warn_on_collision=False,
    )

    assert colors["pbx4"] == "#d62728"
    assert colors["inj_ctrl"] == "#7f7f7f"
    assert colors["cep290_homozygous"] == "#b2182b"
    assert colors["mystery_label"] not in {"#d62728", "#7f7f7f", "#b2182b"}


def test_default_genotype_duplicates_are_reassigned_when_distinct_is_enabled():
    colors = resolve_color_lookup(
        ["pbx4", "pbx1b_pbx4", "other_group"],
        enforce_distinct=True,
        warn_on_collision=False,
    )

    assert colors["pbx4"] == "#d62728"
    assert colors["pbx1b_pbx4"] != "#d62728"
    assert len(set(colors.values())) == 3


def test_explicit_override_duplicates_are_reassigned_when_distinct_is_enabled():
    colors = resolve_color_lookup(
        ["pbx4", "pbx1b_pbx4", "other_group"],
        color_lookup={"pbx4": "#ff0000", "pbx1b_pbx4": "#ff0000"},
        enforce_distinct=True,
        warn_on_collision=False,
    )

    assert colors["pbx4"] == "#ff0000"
    assert colors["pbx1b_pbx4"] != "#ff0000"
    assert len(set(colors.values())) == 3


def test_duplicates_can_still_be_preserved_when_distinct_is_disabled():
    colors = resolve_color_lookup(
        ["pbx4", "pbx1b_pbx4", "other_group"],
        color_lookup={"pbx4": "#ff0000", "pbx1b_pbx4": "#ff0000"},
        enforce_distinct=False,
        warn_on_collision=False,
    )

    assert colors["pbx4"] == "#ff0000"
    assert colors["pbx1b_pbx4"] == "#ff0000"
    assert colors["other_group"] != "#ff0000"


def test_batch_genotype_lookup_breaks_suffix_collisions_for_crispants():
    colors = build_genotype_color_lookup(
        ["pbx2b_crispant", "pbx4_crispant", "tfap2a_crispant", "inj_ctrl"],
        warn_on_collision=False,
    )

    assert len(set(colors.values())) == 4
    assert colors["inj_ctrl"] == "#7f7f7f"


def test_known_genotype_color_returns_none_for_non_genotype_values():
    assert get_known_genotype_color("totally_unrelated_category") is None


def test_trajectory_helper_delegates_to_shared_genotype_color_logic():
    assert legacy_get_color_for_genotype("pbx4") == get_color_for_genotype("pbx4")
    assert legacy_get_color_for_genotype("cep290_homozygous") == get_color_for_genotype("cep290_homozygous")


def test_color_preset_object_is_resolved_with_explicit_colors():
    preset = ColorPreset(
        colors={"inj_ctrl": "#7f7f7f", "pbx4": "#d62728"},
        order=["inj_ctrl", "pbx4"],
        fill="error",
    )

    colors = resolve_color_lookup(
        ["pbx4", "inj_ctrl"],
        color_preset=preset,
        warn_on_collision=False,
    )

    assert colors["inj_ctrl"] == "#7f7f7f"
    assert colors["pbx4"] == "#d62728"


def test_color_preset_respects_label_map_for_display_keys():
    preset = ColorPreset(
        colors={"pbx1b_crispant": "#9467bd"},
        order=["pbx1b_crispant"],
        fill="error",
    )

    colors = resolve_color_lookup(
        ["pbx1b_crispant"],
        color_preset=preset,
        label_map={"pbx1b_crispant": "pbx1b"},
        warn_on_collision=False,
    )

    assert colors["pbx1b"] == "#9467bd"


def test_color_preset_fill_error_raises_for_missing_labels():
    preset = ColorPreset(colors={"inj_ctrl": "#7f7f7f"}, fill="error")

    with pytest.raises(KeyError):
        resolve_color_lookup(
            ["inj_ctrl", "pbx4"],
            color_preset=preset,
            warn_on_collision=False,
        )
