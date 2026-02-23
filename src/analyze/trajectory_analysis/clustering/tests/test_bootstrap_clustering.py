"""
Property-based tests for bootstrap clustering (hierarchical + kmedoids).

Tests verify structural contracts (return types, key presence, value ranges)
rather than exact values, since the RNG changed from np.random.seed to
SeedSequence-based generators.
"""

import numpy as np
import pytest

from analyze.trajectory_analysis.clustering.bootstrap_clustering import (
    run_bootstrap_hierarchical,
    run_bootstrap_kmedoids,
    compute_consensus_labels,
    compute_coassociation_matrix,
    coassociation_to_distance,
    bootstrap_projection_assignments_from_distance,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_distance_matrix(n: int, k: int, seed: int = 0) -> np.ndarray:
    """Build a synthetic distance matrix with k well-separated clusters."""
    rng = np.random.default_rng(seed)
    # Create cluster centers far apart, then add noise
    centers = rng.uniform(0, 100, size=(k, 2)) * 10
    labels = np.repeat(np.arange(k), (n + k - 1) // k)[:n]
    points = centers[labels] + rng.normal(0, 1, size=(n, 2))
    # Euclidean distance matrix
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(points))
    return D, labels


@pytest.fixture
def small_data():
    """Small dataset: 20 points, k=2."""
    D, _ = _make_distance_matrix(20, 2, seed=123)
    ids = [f"emb_{i:02d}" for i in range(20)]
    return D, 2, ids


@pytest.fixture
def medium_data():
    """Medium dataset: 30 points, k=3."""
    D, _ = _make_distance_matrix(30, 3, seed=456)
    ids = [f"emb_{i:02d}" for i in range(30)]
    return D, 3, ids


# ── Hierarchical clustering ──────────────────────────────────────────

class TestRunBootstrapHierarchical:

    def test_return_type_is_dict(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        assert isinstance(result, dict)

    def test_required_keys(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        expected_keys = {
            'embryo_ids', 'reference_labels', 'bootstrap_results',
            'n_clusters', 'distance_matrix', 'n_samples'
        }
        assert expected_keys.issubset(result.keys())

    def test_embryo_ids_preserved(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        assert result['embryo_ids'] == ids

    def test_n_clusters_matches(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        assert result['n_clusters'] == k

    def test_n_samples_matches(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        assert result['n_samples'] == len(ids)

    def test_reference_labels_shape(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        assert result['reference_labels'].shape == (len(ids),)

    def test_reference_labels_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        labels = result['reference_labels']
        assert set(labels).issubset(set(range(k)))

    def test_bootstrap_results_length(self, small_data):
        D, k, ids = small_data
        n_boot = 10
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=n_boot, random_state=42)
        # May be <= n_boot if some iterations failed, but should be > 0
        assert 0 < len(result['bootstrap_results']) <= n_boot

    def test_bootstrap_result_keys(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert 'labels' in br
            assert 'indices' in br
            assert 'silhouette' in br

    def test_bootstrap_labels_shape(self, small_data):
        D, k, ids = small_data
        n = len(ids)
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert br['labels'].shape == (n,)

    def test_bootstrap_labels_values(self, small_data):
        """Labels should be -1 (unsampled) or in [0, k)."""
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            unique_vals = set(br['labels'])
            assert unique_vals.issubset(set(range(k)) | {-1})

    def test_indices_within_bounds(self, small_data):
        D, k, ids = small_data
        n = len(ids)
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert np.all(br['indices'] >= 0)
            assert np.all(br['indices'] < n)

    def test_indices_are_sorted(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert np.all(br['indices'][:-1] <= br['indices'][1:])

    def test_indices_match_labels(self, small_data):
        """Sampled indices should have non-negative labels; unsampled should be -1."""
        D, k, ids = small_data
        n = len(ids)
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            sampled = set(br['indices'])
            for i in range(n):
                if i in sampled:
                    assert br['labels'][i] >= 0, f"Sampled index {i} has label -1"
                else:
                    assert br['labels'][i] == -1, f"Unsampled index {i} has label {br['labels'][i]}"

    def test_silhouette_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            s = br['silhouette']
            assert np.isnan(s) or (-1.0 <= s <= 1.0)

    def test_frac_controls_subsample_size(self, small_data):
        D, k, ids = small_data
        n = len(ids)
        frac = 0.5
        result = run_bootstrap_hierarchical(
            D, k, ids, n_bootstrap=5, frac=frac, random_state=42
        )
        expected_size = max(int(np.ceil(frac * n)), 1)
        for br in result['bootstrap_results']:
            assert len(br['indices']) == expected_size

    def test_different_seeds_different_results(self, small_data):
        D, k, ids = small_data
        r1 = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        r2 = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=99)
        # At least one bootstrap iteration should differ
        any_differ = False
        for b1, b2 in zip(r1['bootstrap_results'], r2['bootstrap_results']):
            if not np.array_equal(b1['indices'], b2['indices']):
                any_differ = True
                break
        assert any_differ, "Different seeds should produce different subsamples"

    def test_same_seed_same_results(self, small_data):
        D, k, ids = small_data
        r1 = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        r2 = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        for b1, b2 in zip(r1['bootstrap_results'], r2['bootstrap_results']):
            np.testing.assert_array_equal(b1['labels'], b2['labels'])
            np.testing.assert_array_equal(b1['indices'], b2['indices'])

    def test_distance_matrix_preserved(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=5, random_state=42)
        np.testing.assert_array_equal(result['distance_matrix'], D)

    def test_medium_k3(self, medium_data):
        """Smoke test with k=3 on 30 points."""
        D, k, ids = medium_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=7)
        assert result['n_clusters'] == 3
        assert len(result['bootstrap_results']) > 0


# ── K-medoids clustering ─────────────────────────────────────────────

class TestRunBootstrapKmedoids:

    @pytest.fixture(autouse=True)
    def _check_kmedoids(self):
        try:
            from sklearn_extra.cluster import KMedoids  # noqa: F401
        except ImportError:
            pytest.skip("sklearn-extra not installed")

    def test_return_type_is_dict(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        assert isinstance(result, dict)

    def test_required_keys(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        expected_keys = {
            'embryo_ids', 'reference_labels', 'bootstrap_results',
            'n_clusters', 'medoid_indices', 'distance_matrix', 'n_samples'
        }
        assert expected_keys.issubset(result.keys())

    def test_medoid_indices_valid(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        medoids = result['medoid_indices']
        assert len(medoids) == k
        assert np.all(medoids >= 0)
        assert np.all(medoids < len(ids))

    def test_bootstrap_result_keys(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert 'labels' in br
            assert 'indices' in br
            assert 'silhouette' in br

    def test_bootstrap_labels_shape(self, small_data):
        D, k, ids = small_data
        n = len(ids)
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert br['labels'].shape == (n,)

    def test_indices_within_bounds(self, small_data):
        D, k, ids = small_data
        n = len(ids)
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            assert np.all(br['indices'] >= 0)
            assert np.all(br['indices'] < n)

    def test_silhouette_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        for br in result['bootstrap_results']:
            s = br['silhouette']
            assert np.isnan(s) or (-1.0 <= s <= 1.0)

    def test_same_seed_same_results(self, small_data):
        D, k, ids = small_data
        r1 = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        r2 = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=5, random_state=42)
        for b1, b2 in zip(r1['bootstrap_results'], r2['bootstrap_results']):
            np.testing.assert_array_equal(b1['labels'], b2['labels'])
            np.testing.assert_array_equal(b1['indices'], b2['indices'])

    def test_different_seeds_different_results(self, small_data):
        D, k, ids = small_data
        r1 = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=10, random_state=42)
        r2 = run_bootstrap_kmedoids(D, k, ids, n_bootstrap=10, random_state=99)
        any_differ = False
        for b1, b2 in zip(r1['bootstrap_results'], r2['bootstrap_results']):
            if not np.array_equal(b1['indices'], b2['indices']):
                any_differ = True
                break
        assert any_differ


# ── Consensus labels ─────────────────────────────────────────────────

class TestComputeConsensusLabels:

    def test_returns_ndarray(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        consensus = compute_consensus_labels(result)
        assert isinstance(consensus, np.ndarray)

    def test_correct_shape(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        consensus = compute_consensus_labels(result)
        assert consensus.shape == (len(ids),)

    def test_labels_in_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        consensus = compute_consensus_labels(result)
        assert set(consensus).issubset(set(range(k)))


# ── Co-association matrix ────────────────────────────────────────────

class TestCoassociationMatrix:

    def test_shape(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        assert M.shape == (len(ids), len(ids))

    def test_diagonal_is_one(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        np.testing.assert_array_equal(np.diag(M), np.ones(len(ids)))

    def test_symmetric(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_values_in_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0)


# ── Coassociation to distance ───────────────────────────────────────

class TestCoassociationToDistance:

    def test_distance_range(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        D_consensus = coassociation_to_distance(M)
        assert np.all(D_consensus >= 0.0)
        assert np.all(D_consensus <= 1.0)

    def test_diagonal_is_zero(self, small_data):
        D, k, ids = small_data
        result = run_bootstrap_hierarchical(D, k, ids, n_bootstrap=10, random_state=42)
        M = compute_coassociation_matrix(result, verbose=False)
        D_consensus = coassociation_to_distance(M)
        np.testing.assert_array_equal(np.diag(D_consensus), np.zeros(len(ids)))


# ── Bootstrap projection from distance ───────────────────────────────

class TestBootstrapProjectionFromDistance:
    """Tests for bootstrap_projection_assignments_from_distance.

    Uses a mock for the projection function to avoid heavy DTW deps.
    """

    @pytest.fixture
    def projection_data(self):
        """Create synthetic cross-distance matrix and cluster map."""
        rng = np.random.default_rng(42)
        n_source = 15
        n_ref = 10
        D_cross = rng.uniform(0, 10, size=(n_source, n_ref))
        source_ids = [f"src_{i:02d}" for i in range(n_source)]
        ref_ids = [f"ref_{i:02d}" for i in range(n_ref)]
        # 2 clusters in reference
        ref_cluster_map = {ref_ids[i]: i % 2 for i in range(n_ref)}
        return D_cross, source_ids, ref_ids, ref_cluster_map

    def test_return_type_is_dict(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        assert isinstance(result, dict)

    def test_required_keys(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        expected_keys = {
            'embryo_ids', 'reference_labels', 'bootstrap_results',
            'n_clusters', 'n_samples'
        }
        assert expected_keys.issubset(result.keys())

    def test_n_samples(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        assert result['n_samples'] == len(src_ids)

    def test_n_clusters(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        assert result['n_clusters'] == 2

    def test_bootstrap_results_length(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        n_boot = 8
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=n_boot, random_state=42
        )
        assert 0 < len(result['bootstrap_results']) <= n_boot

    def test_bootstrap_result_keys_source(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            bootstrap_on="source", n_bootstrap=5, random_state=42
        )
        for br in result['bootstrap_results']:
            assert 'labels' in br
            assert 'indices' in br

    def test_bootstrap_result_keys_reference(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            bootstrap_on="reference", n_bootstrap=5, random_state=42
        )
        for br in result['bootstrap_results']:
            assert 'labels' in br
            assert 'indices' in br

    def test_labels_shape(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        n = len(src_ids)
        result = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        for br in result['bootstrap_results']:
            assert br['labels'].shape == (n,)

    def test_same_seed_same_results(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        r1 = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        r2 = bootstrap_projection_assignments_from_distance(
            D_cross, src_ids, ref_ids, ref_map,
            n_bootstrap=5, random_state=42
        )
        for b1, b2 in zip(r1['bootstrap_results'], r2['bootstrap_results']):
            np.testing.assert_array_equal(b1['labels'], b2['labels'])
            np.testing.assert_array_equal(b1['indices'], b2['indices'])

    def test_invalid_bootstrap_on_raises(self, projection_data):
        D_cross, src_ids, ref_ids, ref_map = projection_data
        with pytest.raises(ValueError, match="bootstrap_on"):
            bootstrap_projection_assignments_from_distance(
                D_cross, src_ids, ref_ids, ref_map,
                bootstrap_on="invalid", n_bootstrap=5, random_state=42
            )
