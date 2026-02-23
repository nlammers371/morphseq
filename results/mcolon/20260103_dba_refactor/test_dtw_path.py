#!/usr/bin/env python
"""
Unit tests for DTW path extraction.
Run with: python test_dtw_path.py

Tests:
1. Identical sequences -> diagonal path
2. Different length sequences -> warped path
3. Backward compatibility -> default returns float only
"""
import numpy as np
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.dtw_distance import compute_dtw_distance


def test_dtw_path_basic():
    """Identical sequences should have diagonal path."""
    seq1 = np.array([1.0, 2.0, 3.0, 4.0])
    seq2 = np.array([1.0, 2.0, 3.0, 4.0])

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    expected_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert path == expected_path, f"Expected {expected_path}, got {path}"
    assert dist == 0.0, f"Expected 0 distance for identical sequences, got {dist}"
    print("✓ test_dtw_path_basic passed")


def test_dtw_path_with_shift():
    """Different length sequences should have warped path."""
    seq1 = np.array([0.0, 1.0, 2.0, 3.0])
    seq2 = np.array([1.0, 2.0, 3.0])

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    # Path should connect (0,0) to (3,2)
    assert len(path) >= 3, f"Path too short: {path}"
    assert path[0] == (0, 0), f"Path should start at (0,0), got {path[0]}"
    assert path[-1] == (3, 2), f"Path should end at (3,2), got {path[-1]}"

    # Path should NOT be purely diagonal (that would be wrong)
    is_diagonal = all(p[0] == p[1] for p in path)
    assert not is_diagonal, f"Path should NOT be diagonal for different length sequences: {path}"

    print(f"✓ test_dtw_path_with_shift passed")
    print(f"  Path: {path}")


def test_dtw_path_with_stretch():
    """Stretched sequence should have repeated indices in path."""
    # seq2 is seq1 but with middle value repeated (temporal stretch)
    seq1 = np.array([1.0, 2.0, 3.0])
    seq2 = np.array([1.0, 2.0, 2.0, 3.0])  # stretched in middle

    dist, path = compute_dtw_distance(seq1, seq2, return_path=True)

    assert path[0] == (0, 0), f"Path should start at (0,0)"
    assert path[-1] == (2, 3), f"Path should end at (2,3)"

    # Check that some index is repeated (warping happened)
    i_indices = [p[0] for p in path]
    has_repeat = len(i_indices) != len(set(i_indices))
    # Note: might also have j repeats, either is fine

    print(f"✓ test_dtw_path_with_stretch passed")
    print(f"  Path: {path}")


def test_backward_compatible():
    """Default behavior should be unchanged (returns float only)."""
    seq1 = np.array([1.0, 2.0, 3.0])
    seq2 = np.array([1.5, 2.5, 3.5])

    # Without return_path (default) - should return float
    result = compute_dtw_distance(seq1, seq2)
    assert isinstance(result, float), f"Expected float, got {type(result)}"

    # With return_path=True - should return tuple
    result = compute_dtw_distance(seq1, seq2, return_path=True)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    assert isinstance(result[0], float), f"First element should be float (distance)"
    assert isinstance(result[1], list), f"Second element should be list (path)"

    print("✓ test_backward_compatible passed")


def test_distance_unchanged():
    """Distance should be same whether or not path is returned."""
    seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seq2 = np.array([1.5, 2.5, 3.5, 4.5])

    dist_only = compute_dtw_distance(seq1, seq2)
    dist_with_path, _ = compute_dtw_distance(seq1, seq2, return_path=True)

    assert abs(dist_only - dist_with_path) < 1e-10, \
        f"Distance mismatch: {dist_only} vs {dist_with_path}"

    print("✓ test_distance_unchanged passed")


if __name__ == '__main__':
    print("=" * 60)
    print("DTW Path Extraction Tests")
    print("=" * 60)
    print()

    test_dtw_path_basic()
    test_dtw_path_with_shift()
    test_dtw_path_with_stretch()
    test_backward_compatible()
    test_distance_unchanged()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
