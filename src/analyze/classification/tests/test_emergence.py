"""
Tests for analyze.classification.emergence

Fixture: PBX 5-class onset matrix (auroc=none, p_sep=0.05, subsequent_frac=0.40)

                  1b+4   pbx4   pbx1b  ctrl   wik
pbx1b_pbx4         -     62      22     22     22
pbx4              62      -      82     22     22
pbx1b             22     82       -     26     30
inj_ctrl          22     22      26      -    NaN
wik_ab            22     22      30    NaN      -
"""
import math

import numpy as np
import pandas as pd
import pytest

from analyze.classification.emergence import (
    EmergenceBlock,
    EmergenceScore,
    EmergenceTimeline,
    ReferenceValidation,
    ResolutionNode,
    build_emergence_timeline,
    compute_emergence_scores,
    form_emergence_blocks,
    resolve_block,
    validate_reference,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSES = ["pbx1b_pbx4_crispant", "pbx4_crispant", "pbx1b_crispant", "inj_ctrl", "wik_ab"]
NAN = float("nan")


def make_pbx_onset_matrix() -> pd.DataFrame:
    """Build the PBX onset matrix from known production values."""
    data = {
        "pbx1b_pbx4_crispant": [NAN,  62,   22,   22,   22],
        "pbx4_crispant":        [62,   NAN,  82,   22,   22],
        "pbx1b_crispant":       [22,   82,   NAN,  26,   30],
        "inj_ctrl":             [22,   22,   26,   NAN,  NAN],
        "wik_ab":               [22,   22,   30,   NAN,  NAN],
    }
    return pd.DataFrame(data, index=CLASSES)


@pytest.fixture
def pbx_onset() -> pd.DataFrame:
    return make_pbx_onset_matrix()


# ---------------------------------------------------------------------------
# Step 1: validate_reference
# ---------------------------------------------------------------------------

class TestValidateReference:
    def test_valid(self, pbx_onset):
        """inj_ctrl vs wik_ab = NaN → valid, coherence=1.0"""
        rv = validate_reference(pbx_onset, ["inj_ctrl", "wik_ab"])
        assert rv.status == "valid"
        assert rv.coherence_score == 1.0
        assert rv.offending_pairs == []
        assert rv.n_internal_pairs == 1

    def test_singleton(self, pbx_onset):
        """Single-member reference → valid, n_internal_pairs=0, coherence=1.0"""
        rv = validate_reference(pbx_onset, ["inj_ctrl"])
        assert rv.status == "valid"
        assert rv.coherence_score == 1.0
        assert rv.n_internal_pairs == 0

    def test_ambiguous(self):
        """3-member reference where 1 of 3 pairs is finite → coherence=2/3 >= 0.5 → ambiguous."""
        # ref = {a, b, c}: pairs (a,b)=NaN, (a,c)=NaN, (b,c)=10 → coherence=2/3
        data = {
            "a": [NAN, NAN,  NAN,  NAN],
            "b": [NAN, NAN,  10.0, NAN],
            "c": [NAN, 10.0, NAN,  NAN],
            "d": [NAN, NAN,  NAN,  NAN],
        }
        mat = pd.DataFrame(data, index=["a", "b", "c", "d"])
        rv = validate_reference(mat, ["a", "b", "c"])
        assert rv.status == "ambiguous"
        assert len(rv.offending_pairs) == 1
        assert rv.offending_pairs[0][2] == pytest.approx(10.0)
        assert rv.n_internal_pairs == 3
        assert rv.coherence_score == pytest.approx(2 / 3)

    def test_invalid(self):
        """All internal pairs finite → coherence=0 → invalid"""
        # Construct a 3-class matrix where all pairs finite
        data = {
            "a": [NAN, 10.0, 12.0],
            "b": [10.0, NAN, 14.0],
            "c": [12.0, 14.0, NAN],
        }
        mat = pd.DataFrame(data, index=["a", "b", "c"])
        rv = validate_reference(mat, ["a", "b", "c"])
        assert rv.status == "invalid"
        assert rv.coherence_score == pytest.approx(0.0)
        assert len(rv.offending_pairs) == 3


# ---------------------------------------------------------------------------
# Step 2: compute_emergence_scores
# ---------------------------------------------------------------------------

class TestComputeEmergenceScores:
    def test_emergence_scores(self, pbx_onset):
        ref = ["inj_ctrl", "wik_ab"]
        scores = compute_emergence_scores(pbx_onset, ref)
        score_dict = {s.class_name: s for s in scores}

        # pbx1b_pbx4_crispant: onset(inj_ctrl)=22, onset(wik_ab)=22 → median=22
        s = score_dict["pbx1b_pbx4_crispant"]
        assert s.emergence_time == pytest.approx(22.0)
        assert s.emergence_min == pytest.approx(22.0)
        assert s.emergence_max == pytest.approx(22.0)
        assert s.n_resolved_refs == 2
        assert s.n_total_refs == 2

        # pbx4_crispant: onset(inj_ctrl)=22, onset(wik_ab)=22 → median=22
        s = score_dict["pbx4_crispant"]
        assert s.emergence_time == pytest.approx(22.0)

        # pbx1b_crispant: onset(inj_ctrl)=26, onset(wik_ab)=30 → median=28
        s = score_dict["pbx1b_crispant"]
        assert s.emergence_time == pytest.approx(28.0)
        assert s.emergence_min == pytest.approx(26.0)
        assert s.emergence_max == pytest.approx(30.0)

    def test_nan_class(self):
        """Class with no finite onset to reference → emergence_time=NaN"""
        data = {
            "mut": [NAN, NAN],
            "ctrl": [NAN, NAN],
        }
        mat = pd.DataFrame(data, index=["mut", "ctrl"])
        scores = compute_emergence_scores(mat, ["ctrl"])
        assert len(scores) == 1
        assert math.isnan(scores[0].emergence_time)
        assert scores[0].n_resolved_refs == 0

    def test_per_ref_onsets(self, pbx_onset):
        """per_ref_onsets populated correctly."""
        scores = compute_emergence_scores(pbx_onset, ["inj_ctrl", "wik_ab"])
        score_dict = {s.class_name: s for s in scores}
        s = score_dict["pbx1b_crispant"]
        assert s.per_ref_onsets["inj_ctrl"] == pytest.approx(26.0)
        assert s.per_ref_onsets["wik_ab"] == pytest.approx(30.0)

    def test_sorted_nan_last(self, pbx_onset):
        """NaN emergence scores come last in the returned list."""
        # Override one row to produce NaN emergence
        mat = pbx_onset.copy()
        mat.loc["pbx4_crispant", "inj_ctrl"] = NAN
        mat.loc["pbx4_crispant", "wik_ab"] = NAN
        mat.loc["inj_ctrl", "pbx4_crispant"] = NAN
        mat.loc["wik_ab", "pbx4_crispant"] = NAN
        scores = compute_emergence_scores(mat, ["inj_ctrl", "wik_ab"])
        for i in range(len(scores) - 1):
            if math.isnan(scores[i].emergence_time):
                assert math.isnan(scores[i + 1].emergence_time), \
                    "NaN scores should all be at the end"


# ---------------------------------------------------------------------------
# Step 3: form_emergence_blocks
# ---------------------------------------------------------------------------

class TestFormEmergenceBlocks:
    def test_same_bin_grouped(self, pbx_onset):
        """pbx1b_pbx4 and pbx4 both emerge at 22 → same bin; pbx1b at 28 → separate"""
        scores = compute_emergence_scores(pbx_onset, ["inj_ctrl", "wik_ab"])
        blocks = form_emergence_blocks(scores, bin_width=4.0)

        # Should have 2 finite blocks (bin_key=20 and bin_key=28)
        finite_blocks = [b for b in blocks if math.isfinite(b.emergence_time)]
        assert len(finite_blocks) == 2

        # First block: pbx1b_pbx4 and pbx4 (both emergence=22, bin_key=20)
        b0 = finite_blocks[0]
        assert set(b0.members) == {"pbx1b_pbx4_crispant", "pbx4_crispant"}
        assert b0.bin_key == pytest.approx(20.0)
        assert b0.emergence_time == pytest.approx(22.0)  # raw median, NOT 20.0

        # Second block: pbx1b (emergence=28, bin_key=28)
        b1 = finite_blocks[1]
        assert set(b1.members) == {"pbx1b_crispant"}
        assert b1.bin_key == pytest.approx(28.0)
        assert b1.emergence_time == pytest.approx(28.0)

    def test_display_times_not_floored(self, pbx_onset):
        """Block emergence_time should be raw median (22.0), not bin_key (20.0)."""
        scores = compute_emergence_scores(pbx_onset, ["inj_ctrl", "wik_ab"])
        blocks = form_emergence_blocks(scores, bin_width=4.0)
        finite_blocks = [b for b in blocks if math.isfinite(b.emergence_time)]
        b0 = finite_blocks[0]
        # Raw emergence = 22; bin_key = 20; display should be 22 not 20
        assert b0.emergence_time == pytest.approx(22.0)
        assert b0.bin_key == pytest.approx(20.0)
        assert b0.emergence_time != b0.bin_key

    def test_nan_block_at_end(self):
        """Classes with NaN emergence → appear in last block."""
        scores = [
            EmergenceScore("a", 10.0, 10.0, 10.0, 1, 1, {}),
            EmergenceScore("b", NAN, NAN, NAN, 0, 1, {}),
        ]
        blocks = form_emergence_blocks(scores, bin_width=4.0)
        assert len(blocks) == 2
        nan_block = blocks[-1]
        assert "b" in nan_block.members
        assert math.isnan(nan_block.emergence_time)

    def test_stable_block_ids(self, pbx_onset):
        """Block IDs assigned in order, starting from 0."""
        scores = compute_emergence_scores(pbx_onset, ["inj_ctrl", "wik_ab"])
        blocks = form_emergence_blocks(scores, bin_width=4.0)
        for i, b in enumerate(blocks):
            assert b.block_id == i


# ---------------------------------------------------------------------------
# Step 4: resolve_block
# ---------------------------------------------------------------------------

class TestResolveBlock:
    def test_singleton(self, pbx_onset):
        """Single member → singleton leaf."""
        node = resolve_block(["inj_ctrl"], pbx_onset)
        assert node.members == ["inj_ctrl"]
        assert node.split_time is None
        assert node.children == []
        assert not node.unresolved

    def test_pair_resolves(self, pbx_onset):
        """pbx1b_pbx4 vs pbx4 → split at 62."""
        node = resolve_block(
            ["pbx1b_pbx4_crispant", "pbx4_crispant"], pbx_onset
        )
        assert node.split_time == pytest.approx(62.0)
        assert len(node.children) == 2
        assert not node.unresolved

    def test_pair_unresolved(self):
        """Pair with only NaN between them → unresolved composite."""
        data = {"a": [NAN, NAN], "b": [NAN, NAN]}
        mat = pd.DataFrame(data, index=["a", "b"])
        node = resolve_block(["a", "b"], mat)
        assert node.unresolved
        assert node.split_time is None
        assert node.children == []

    def test_triple_best_bipartition(self):
        """3-class block: find best bipartition by cross_median."""
        # Classes: x, y, z
        # x vs y = 50, x vs z = 80, y vs z = 50
        # Best split: {x,y} | {z} → cross_median=avg(80,50)=65
        # OR {x} | {y,z} → cross_median=avg(50,80)=65 → tie
        # OR {y} | {x,z} → cross_median=avg(50,80)=65 → tie
        # Support: all 1.0 (1/1 per pair)
        # internal_finite: {x,y}|{z}: internal={x,y}=50→1; {x}|{y,z}: internal={y,z}=50→1
        # All tied, so any valid split is fine — just check it runs and splits
        data = {
            "x": [NAN, 50.0, 80.0],
            "y": [50.0, NAN, 50.0],
            "z": [80.0, 50.0, NAN],
        }
        mat = pd.DataFrame(data, index=["x", "y", "z"])
        node = resolve_block(["x", "y", "z"], mat)
        assert not node.unresolved
        assert node.split_time is not None
        assert len(node.children) == 2

    def test_split_acceptance_low_support(self):
        """Only 1 of 4 cross-pairs finite → support=0.25 < 0.5 → unresolved."""
        # 2x2 cross: a-c, a-d, b-c, b-d
        # Only a-c = 40, rest NaN → cross_support = 1/4 = 0.25
        data = {
            "a": [NAN, NAN, 40.0, NAN],
            "b": [NAN, NAN, NAN,  NAN],
            "c": [40.0, NAN, NAN, NAN],
            "d": [NAN, NAN, NAN, NAN],
        }
        mat = pd.DataFrame(data, index=["a", "b", "c", "d"])
        node = resolve_block(["a", "b", "c", "d"], mat, min_cross_support=0.5)
        assert node.unresolved

    def test_split_acceptance_high_support(self):
        """3 of 4 cross-pairs finite → support=0.75 >= 0.5 → split accepted."""
        data = {
            "a": [NAN, NAN, 40.0, 45.0],
            "b": [NAN, NAN, 42.0, NAN],
            "c": [40.0, 42.0, NAN, NAN],
            "d": [45.0, NAN, NAN, NAN],
        }
        mat = pd.DataFrame(data, index=["a", "b", "c", "d"])
        node = resolve_block(["a", "b", "c", "d"], mat, min_cross_support=0.5)
        # {a,b} | {c,d}: cross = 40, 45, 42, NaN → 3/4 finite → accepted
        assert not node.unresolved
        assert node.split_time is not None


# ---------------------------------------------------------------------------
# Step 5: build_emergence_timeline (full pipeline)
# ---------------------------------------------------------------------------

class TestBuildEmergenceTimeline:
    def test_full_pipeline(self, pbx_onset):
        """End-to-end with reference={inj_ctrl, wik_ab}."""
        tl = build_emergence_timeline(
            pbx_onset,
            ["inj_ctrl", "wik_ab"],
            bin_width=4.0,
        )
        assert tl.reference == ["inj_ctrl", "wik_ab"]
        assert tl.reference_validation.status == "valid"

        # 2 finite blocks + no NaN block
        finite_blocks = [b for b in tl.blocks if math.isfinite(b.emergence_time)]
        assert len(finite_blocks) == 2

        # First block contains {pbx1b_pbx4, pbx4}
        b0 = finite_blocks[0]
        assert set(b0.members) == {"pbx1b_pbx4_crispant", "pbx4_crispant"}
        # Resolution: split at 62
        res0 = tl.block_resolutions[b0.block_id]
        assert res0.split_time == pytest.approx(62.0)
        assert len(res0.children) == 2

        # Second block: pbx1b singleton
        b1 = finite_blocks[1]
        assert b1.members == ["pbx1b_crispant"]
        res1 = tl.block_resolutions[b1.block_id]
        assert res1.split_time is None
        assert not res1.unresolved

    def test_pipeline_single_ref(self, pbx_onset):
        """Reference = {inj_ctrl} only → valid singleton reference."""
        tl = build_emergence_timeline(pbx_onset, ["inj_ctrl"])
        assert tl.reference_validation.status == "valid"
        assert tl.reference_validation.n_internal_pairs == 0

        # Non-ref classes: pbx1b_pbx4, pbx4, pbx1b, wik_ab
        non_ref = [c for c in tl.all_classes if c not in {"inj_ctrl"}]
        score_names = [s.class_name for s in tl.scores]
        for c in non_ref:
            assert c in score_names

    def test_invalid_reference_still_builds(self, pbx_onset):
        """Reference = {inj_ctrl, pbx1b_crispant} → invalid (1 pair, all finite).
        Timeline should still be built, just flagged."""
        tl = build_emergence_timeline(pbx_onset, ["inj_ctrl", "pbx1b_crispant"])
        # 1 pair total, all finite → coherence=0 → invalid
        assert tl.reference_validation.status == "invalid"
        assert len(tl.reference_validation.offending_pairs) >= 1
        # Still builds timeline despite invalid reference
        assert len(tl.blocks) > 0

    def test_block_resolutions_keyed_by_block_id(self, pbx_onset):
        """block_resolutions keys match block_ids."""
        tl = build_emergence_timeline(pbx_onset, ["inj_ctrl", "wik_ab"])
        for block in tl.blocks:
            assert block.block_id in tl.block_resolutions

    def test_all_classes_present(self, pbx_onset):
        """all_classes matches the onset_matrix index."""
        tl = build_emergence_timeline(pbx_onset, ["inj_ctrl", "wik_ab"])
        assert set(tl.all_classes) == set(CLASSES)

    def test_scores_sorted_nan_last(self, pbx_onset):
        """Scores are sorted by emergence_time with NaN last."""
        tl = build_emergence_timeline(pbx_onset, ["inj_ctrl", "wik_ab"])
        times = [s.emergence_time for s in tl.scores]
        finite_times = [t for t in times if math.isfinite(t)]
        # All finite times should be non-decreasing
        assert finite_times == sorted(finite_times)
        # NaN should only appear at end
        seen_nan = False
        for t in times:
            if math.isnan(t):
                seen_nan = True
            elif seen_nan:
                pytest.fail("Finite time after NaN in scores list")
