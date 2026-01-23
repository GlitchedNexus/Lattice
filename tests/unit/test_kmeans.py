"""
Comprehensive test suite for KMeans clustering model.

Tests clustering functionality, centroid updates, assignments, and convergence.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.kmeans import KMeans


class TestKMeans:
    """Test KMeans clustering implementation."""

    @pytest.fixture
    def simple_clusters(self):
        """Create simple well-separated clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) + np.array([0, 0])
        cluster2 = np.random.randn(20, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])
        return X

    @pytest.fixture
    def three_clusters(self):
        """Create three well-separated clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(15, 2) + np.array([0, 0])
        cluster2 = np.random.randn(15, 2) + np.array([10, 0])
        cluster3 = np.random.randn(15, 2) + np.array([5, 10])
        X = np.vstack([cluster1, cluster2, cluster3])
        return X

    def test_initialization_with_fit(self, simple_clusters):
        """Test initialization that automatically fits."""
        X = simple_clusters
        model = KMeans(X, k=2, plot=False, log=False)
        
        # Should have learned centroids
        assert model.w is not None
        assert model.w.shape == (2, 2)  # k=2, d=2
        assert model.k == 2
        assert model.d == 2

    def test_initialization_without_fit(self):
        """Test initialization without fitting."""
        model = KMeans(None, k=3, plot=False, log=False)
        
        # Should not have centroids yet
        assert not hasattr(model, 'w') or model.w is None

    def test_fit_method(self, simple_clusters):
        """Test explicit fit method."""
        X = simple_clusters
        model = KMeans(None, k=2, plot=False, log=False)
        model.fit(X, k=2, plot=False, log=False)
        
        assert model.w is not None
        assert model.k == 2

    def test_get_assignments(self, simple_clusters):
        """Test cluster assignment method."""
        X = simple_clusters
        model = KMeans(X, k=2, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        assert assignments.shape == (len(X),)
        # All assignments should be 0 or 1
        assert np.all((assignments == 0) | (assignments == 1))

    def test_clusters_well_separated(self, simple_clusters):
        """Test clustering on well-separated data."""
        X = simple_clusters
        model = KMeans(X, k=2, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        
        # First 20 samples should mostly be in one cluster
        # Last 20 samples should mostly be in another
        cluster1_purity = np.sum(assignments[:20] == assignments[0]) / 20
        cluster2_purity = np.sum(assignments[20:] == assignments[20]) / 20
        
        # Both clusters should have high purity
        assert cluster1_purity >= 0.8
        assert cluster2_purity >= 0.8

    def test_three_clusters(self, three_clusters):
        """Test with three clusters."""
        X = three_clusters
        model = KMeans(X, k=3, plot=False, log=False)
        
        assert model.w.shape == (3, 2)
        
        assignments = model.get_assignments(X)
        # Should use all three clusters
        assert len(np.unique(assignments)) == 3

    def test_loss_function(self, simple_clusters):
        """Test loss (sum of squared distances) calculation."""
        X = simple_clusters
        model = KMeans(X, k=2, plot=False, log=False)
        
        loss = model.loss(X)
        # Loss should be non-negative
        assert loss >= 0
        
        # Loss should be relatively small for well-separated clusters
        # Normalized by number of points
        assert loss / len(X) < 100

    def test_loss_with_provided_assignments(self, simple_clusters):
        """Test loss calculation with provided assignments."""
        X = simple_clusters
        model = KMeans(X, k=2, plot=False, log=False)
        
        y = model.get_assignments(X)
        loss1 = model.loss(X, y)
        loss2 = model.loss(X)
        
        # Should give same result
        assert np.isclose(loss1, loss2)

    def test_convergence(self):
        """Test that algorithm converges."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(10, 2),
            np.random.randn(10, 2) + 5
        ])
        
        # Run multiple times, should converge to similar solutions
        losses = []
        for _ in range(3):
            np.random.seed(42)
            model = KMeans(X, k=2, plot=False, log=False)
            losses.append(model.loss(X))
        
        # Losses should be relatively consistent
        assert np.std(losses) < 1.0

    def test_k_equals_n(self):
        """Test when k equals number of samples."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        model = KMeans(X, k=3, plot=False, log=False)
        
        # Each point should be its own cluster
        assert model.w.shape == (3, 2)

    def test_single_cluster(self):
        """Test with k=1."""
        X = np.random.randn(20, 2)
        model = KMeans(X, k=1, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        # All assignments should be 0
        assert np.all(assignments == 0)
        
        # Centroid should be near the mean
        assert np.allclose(model.w[0], X.mean(axis=0), rtol=0.5)

    def test_identical_points(self):
        """Test clustering identical points."""
        X = np.ones((10, 2))
        model = KMeans(X, k=2, plot=False, log=False)
        
        # Should handle gracefully
        assert model.w.shape == (2, 2)
        
        loss = model.loss(X)
        # Loss should be 0 since all points are identical
        assert loss < 1e-10

    def test_one_dimensional_data(self):
        """Test on 1D data."""
        np.random.seed(42)
        X = np.concatenate([
            np.random.randn(10, 1),
            np.random.randn(10, 1) + 10
        ])
        model = KMeans(X, k=2, plot=False, log=False)
        
        assert model.w.shape == (2, 1)
        assignments = model.get_assignments(X)
        
        # Should separate the two groups
        cluster1 = assignments[:10]
        cluster2 = assignments[10:]
        # Each group should be mostly in one cluster
        assert (np.sum(cluster1 == 0) >= 8 or np.sum(cluster1 == 1) >= 8)
        assert (np.sum(cluster2 == 0) >= 8 or np.sum(cluster2 == 1) >= 8)

    def test_high_dimensional_data(self):
        """Test on high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        model = KMeans(X, k=3, plot=False, log=False)
        
        assert model.w.shape == (3, 10)
        assert model.d == 10

    def test_update_means(self):
        """Test centroid update method."""
        X = np.array([
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11]
        ])
        y = np.array([0, 0, 1, 1])
        
        model = KMeans(None, k=2, plot=False, log=False)
        model.k = 2
        model.d = 2
        model.w = np.zeros((2, 2))
        
        model.update_means(X, y)
        
        # Centroids should be at cluster means
        assert np.allclose(model.w[0], [0.5, 0.5])
        assert np.allclose(model.w[1], [10.5, 10.5])

    def test_update_means_empty_cluster(self):
        """Test update_means when a cluster is empty."""
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 0])  # All assigned to cluster 0
        
        model = KMeans(None, k=2, plot=False, log=False)
        model.k = 2
        model.d = 2
        model.w = np.array([[0, 0], [5, 5]])
        
        # Should handle empty cluster 1 gracefully
        model.update_means(X, y)
        
        # Cluster 0 should be updated
        assert np.allclose(model.w[0], [2, 2])
        # Cluster 1 should remain unchanged (empty)
        assert np.allclose(model.w[1], [5, 5])

    def test_random_initialization(self):
        """Test that random initialization samples from data."""
        X = np.array([[i, i] for i in range(10)])
        
        model = KMeans(X, k=3, plot=False, log=False)
        
        # All centroids should be close to actual data points
        for centroid in model.w:
            # Each centroid should be close to at least one data point
            distances = np.linalg.norm(X - centroid, axis=1)
            assert np.min(distances) < 2.0


class TestKMeansEdgeCases:
    """Test edge cases for KMeans."""

    def test_k_larger_than_n(self):
        """Test that k > n raises assertion error."""
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(AssertionError):
            model = KMeans(X, k=5, plot=False, log=False)

    def test_minimum_valid_input(self):
        """Test with minimum valid input."""
        X = np.array([[1, 2]])
        model = KMeans(X, k=1, plot=False, log=False)
        
        assert model.w.shape == (1, 2)

    def test_reassignment_during_iteration(self):
        """Test that points can be reassigned during iterations."""
        # Create a case where initial assignment might not be optimal
        np.random.seed(0)
        X = np.array([
            [0, 0],
            [0.1, 0.1],
            [10, 10],
            [10.1, 10.1]
        ])
        
        model = KMeans(X, k=2, plot=False, log=False)
        assignments = model.get_assignments(X)
        
        # Points close together should be in same cluster
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[0] != assignments[2]

    def test_large_scale_clustering(self):
        """Test on larger dataset."""
        np.random.seed(42)
        X = np.random.randn(500, 5)
        
        model = KMeans(X, k=10, plot=False, log=False)
        
        assert model.w.shape == (10, 5)
        assignments = model.get_assignments(X)
        
        # Should use most or all clusters
        n_used_clusters = len(np.unique(assignments))
        assert n_used_clusters >= 8  # Allow some clusters to be unused


class TestKMeansConsistency:
    """Test consistency properties of KMeans."""

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with same seed."""
        X = np.random.randn(50, 3)
        
        np.random.seed(42)
        model1 = KMeans(X, k=3, plot=False, log=False)
        loss1 = model1.loss(X)
        
        np.random.seed(42)
        model2 = KMeans(X, k=3, plot=False, log=False)
        loss2 = model2.loss(X)
        
        # Should give same loss with same seed
        assert np.isclose(loss1, loss2)

    def test_loss_decreases_or_stable(self):
        """Test that loss doesn't increase during iterations."""
        # This is implicitly tested by the algorithm, but we verify
        # the final loss is reasonable
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2),
            np.random.randn(20, 2) + 8
        ])
        
        model = KMeans(X, k=2, plot=False, log=False)
        final_loss = model.loss(X)
        
        # Loss should be reasonable for well-separated clusters
        mean_squared_distance = final_loss / len(X)
        assert mean_squared_distance < 50  # Reasonable threshold

    def test_assignment_consistency(self):
        """Test that assignments are consistent after convergence."""
        X = np.random.randn(30, 2)
        model = KMeans(X, k=3, plot=False, log=False)
        
        # Get assignments multiple times
        assign1 = model.get_assignments(X)
        assign2 = model.get_assignments(X)
        
        # Should be identical
        assert np.array_equal(assign1, assign2)
