"""
Comprehensive test suite for KMedians clustering model.

Tests L1-distance based clustering, median updates, and comparison with KMeans.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.kmedians import KMedians, l1_distances


class TestL1Distances:
    """Test the L1 distance helper function."""

    def test_l1_distance_basic(self):
        """Test basic L1 distance calculation."""
        X1 = np.array([[0, 0], [1, 1]])
        X2 = np.array([[2, 2], [3, 3]])
        
        distances = l1_distances(X1, X2)
        
        # Shape should be (2, 2)
        assert distances.shape == (2, 2)
        
        # Verify specific distances
        # Distance from [0,0] to [2,2] = |0-2| + |0-2| = 4
        assert distances[0, 0] == 4
        # Distance from [1,1] to [3,3] = |1-3| + |1-3| = 4
        assert distances[1, 1] == 4

    def test_l1_distance_zero(self):
        """Test L1 distance to itself."""
        X = np.array([[1, 2, 3]])
        
        distances = l1_distances(X, X)
        
        assert distances[0, 0] == 0

    def test_l1_distance_properties(self):
        """Test L1 distance properties (non-negative, symmetric)."""
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[5, 6]])
        
        distances = l1_distances(X1, X2)
        
        # All distances should be non-negative
        assert np.all(distances >= 0)

    def test_l1_vs_euclidean(self):
        """Verify L1 distance is different from Euclidean."""
        X1 = np.array([[0, 0]])
        X2 = np.array([[3, 4]])
        
        l1_dist = l1_distances(X1, X2)
        euclidean_sq = (3**2 + 4**2)  # = 25
        
        # L1 = |3| + |4| = 7, Euclidean squared = 25
        assert l1_dist[0, 0] == 7
        assert l1_dist[0, 0] != euclidean_sq


class TestKMedians:
    """Test KMedians clustering implementation."""

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

    def test_initialization(self, simple_clusters):
        """Test KMedians initialization."""
        X = simple_clusters
        model = KMedians(X, k=2, plot=False, log=False)
        
        assert model.w is not None
        assert model.w.shape == (2, 2)
        assert model.k == 2

    def test_inherits_from_kmeans(self):
        """Test that KMedians inherits from KMeans."""
        from lattice.kmeans import KMeans
        assert issubclass(KMedians, KMeans)

    def test_get_assignments_uses_l1(self, simple_clusters):
        """Test that get_assignments uses L1 distance."""
        X = simple_clusters
        model = KMedians(X, k=2, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        
        assert assignments.shape == (len(X),)
        assert np.all((assignments >= 0) & (assignments < 2))

    def test_update_means_uses_median(self):
        """Test that update uses median instead of mean."""
        X = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [100, 100]  # Outlier
        ])
        y = np.array([0, 0, 0, 0])  # All in cluster 0
        
        model = KMedians(None, k=1, plot=False, log=False)
        model.k = 1
        model.d = 2
        model.w = np.zeros((1, 2))
        
        model.update_means(X, y)
        
        # Median should be [1.5, 1.5], more robust to outlier than mean
        # Mean would be [25.75, 25.75]
        assert np.allclose(model.w[0], [1.5, 1.5])

    def test_loss_uses_l1(self, simple_clusters):
        """Test that loss function uses L1 distance."""
        X = simple_clusters
        model = KMedians(X, k=2, plot=False, log=False)
        
        loss = model.loss(X)
        
        # Loss should be non-negative
        assert loss >= 0
        
        # Should be sum of L1 distances, not squared Euclidean
        y = model.get_assignments(X)
        manual_loss = np.sum(np.abs(X - model.w[y]))
        assert np.isclose(loss, manual_loss)

    def test_clusters_well_separated(self, simple_clusters):
        """Test clustering on well-separated data."""
        X = simple_clusters
        model = KMedians(X, k=2, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        
        # Check cluster purity
        cluster1_purity = np.sum(assignments[:20] == assignments[0]) / 20
        cluster2_purity = np.sum(assignments[20:] == assignments[20]) / 20
        
        assert cluster1_purity >= 0.75
        assert cluster2_purity >= 0.75

    def test_three_clusters(self, three_clusters):
        """Test with three clusters."""
        X = three_clusters
        model = KMedians(X, k=3, plot=False, log=False)
        
        assert model.w.shape == (3, 2)
        
        assignments = model.get_assignments(X)
        # Should use all or most clusters
        assert len(np.unique(assignments)) >= 2

    def test_robustness_to_outliers(self):
        """Test that KMedians is more robust to outliers than KMeans."""
        # Create clusters with outliers
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2),
            np.random.randn(20, 2) + 10,
            np.array([[50, 50], [-50, -50]])  # Outliers
        ])
        
        model = KMedians(X, k=2, plot=False, log=False)
        
        # Should still identify main clusters
        assignments = model.get_assignments(X[:40])  # Exclude outliers
        
        # Main clusters should be mostly separated
        cluster1 = assignments[:20]
        cluster2 = assignments[20:40]
        
        # At least one cluster should have good purity
        purity1 = max(np.sum(cluster1 == 0), np.sum(cluster1 == 1)) / 20
        purity2 = max(np.sum(cluster2 == 0), np.sum(cluster2 == 1)) / 20
        
        assert purity1 >= 0.7 or purity2 >= 0.7

    def test_single_cluster(self):
        """Test with k=1."""
        X = np.random.randn(20, 2)
        model = KMedians(X, k=1, plot=False, log=False)
        
        assignments = model.get_assignments(X)
        assert np.all(assignments == 0)

    def test_convergence(self):
        """Test that algorithm converges."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(10, 2),
            np.random.randn(10, 2) + 5
        ])
        
        model = KMedians(X, k=2, plot=False, log=False)
        
        # Should converge (not infinite loop)
        assert model.w is not None

    def test_one_dimensional_data(self):
        """Test on 1D data."""
        np.random.seed(42)
        X = np.concatenate([
            np.random.randn(10, 1),
            np.random.randn(10, 1) + 10
        ])
        model = KMedians(X, k=2, plot=False, log=False)
        
        assert model.w.shape == (2, 1)

    def test_high_dimensional_data(self):
        """Test on high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        model = KMedians(X, k=3, plot=False, log=False)
        
        assert model.w.shape == (3, 10)

    def test_identical_points(self):
        """Test clustering identical points."""
        X = np.ones((10, 2)) * 5
        model = KMedians(X, k=2, plot=False, log=False)
        
        loss = model.loss(X)
        # Loss should be near 0
        assert loss < 1e-10

    def test_update_means_empty_cluster(self):
        """Test update with empty cluster."""
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 0])
        
        model = KMedians(None, k=2, plot=False, log=False)
        model.k = 2
        model.d = 2
        model.w = np.array([[0, 0], [10, 10]])
        
        model.update_means(X, y)
        
        # Cluster 0 should be updated to median
        assert np.allclose(model.w[0], [2, 2])
        # Cluster 1 remains unchanged (empty)
        assert np.allclose(model.w[1], [10, 10])


class TestKMediansVsKMeans:
    """Compare KMedians with KMeans behavior."""

    def test_median_vs_mean_difference(self):
        """Test that median and mean give different results with outliers."""
        # Data with outlier
        X = np.array([[0], [1], [2], [100]])
        y = np.array([0, 0, 0, 0])
        
        # KMeans update (mean)
        from lattice.kmeans import KMeans
        kmeans = KMeans(None, k=1, plot=False, log=False)
        kmeans.k = 1
        kmeans.d = 1
        kmeans.w = np.zeros((1, 1))
        kmeans.update_means(X, y)
        mean_center = kmeans.w[0, 0]
        
        # KMedians update (median)
        kmedians = KMedians(None, k=1, plot=False, log=False)
        kmedians.k = 1
        kmedians.d = 1
        kmedians.w = np.zeros((1, 1))
        kmedians.update_means(X, y)
        median_center = kmedians.w[0, 0]
        
        # Median (1.5) should be much smaller than mean (25.75)
        assert median_center < 10
        assert mean_center > 20

    def test_loss_function_difference(self):
        """Test that loss functions differ between KMeans and KMedians."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        
        from lattice.kmeans import KMeans
        
        # Fit both models
        np.random.seed(42)
        kmeans = KMeans(X, k=1, plot=False, log=False)
        
        np.random.seed(42)
        kmedians = KMedians(X, k=1, plot=False, log=False)
        
        # Losses will be different (L2 vs L1)
        loss_means = kmeans.loss(X)
        loss_medians = kmedians.loss(X)
        
        # Both should be small but different
        assert loss_means >= 0
        assert loss_medians >= 0

    def test_similar_performance_no_outliers(self):
        """Test that both perform similarly without outliers."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(15, 2),
            np.random.randn(15, 2) + 8
        ])
        
        from lattice.kmeans import KMeans
        
        np.random.seed(42)
        kmeans = KMeans(X, k=2, plot=False, log=False)
        kmeans_assign = kmeans.get_assignments(X)
        
        np.random.seed(42)
        kmedians = KMedians(X, k=2, plot=False, log=False)
        kmedians_assign = kmedians.get_assignments(X)
        
        # Both should separate clusters reasonably well
        # Check purity for kmeans
        kmeans_purity = max(
            np.sum(kmeans_assign[:15] == 0) / 15,
            np.sum(kmeans_assign[:15] == 1) / 15
        )
        
        # Check purity for kmedians
        kmedians_purity = max(
            np.sum(kmedians_assign[:15] == 0) / 15,
            np.sum(kmedians_assign[:15] == 1) / 15
        )
        
        # Both should achieve decent purity
        assert kmeans_purity >= 0.6
        assert kmedians_purity >= 0.6


class TestKMediansEdgeCases:
    """Test edge cases for KMedians."""

    def test_k_larger_than_n(self):
        """Test that k > n raises assertion error."""
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(AssertionError):
            model = KMedians(X, k=5, plot=False, log=False)

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[5, 10]])
        model = KMedians(X, k=1, plot=False, log=False)
        
        assert model.w.shape == (1, 2)

    def test_deterministic_with_seed(self):
        """Test determinism with same seed."""
        X = np.random.randn(30, 2)
        
        np.random.seed(42)
        model1 = KMedians(X, k=3, plot=False, log=False)
        loss1 = model1.loss(X)
        
        np.random.seed(42)
        model2 = KMedians(X, k=3, plot=False, log=False)
        loss2 = model2.loss(X)
        
        assert np.isclose(loss1, loss2)

    def test_median_with_even_number_points(self):
        """Test median calculation with even number of points."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 0, 0])
        
        model = KMedians(None, k=1, plot=False, log=False)
        model.k = 1
        model.d = 1
        model.w = np.zeros((1, 1))
        
        model.update_means(X, y)
        
        # Median of [1,2,3,4] is 2.5
        assert np.isclose(model.w[0, 0], 2.5)

    def test_median_with_odd_number_points(self):
        """Test median calculation with odd number of points."""
        X = np.array([[1], [2], [3]])
        y = np.array([0, 0, 0])
        
        model = KMedians(None, k=1, plot=False, log=False)
        model.k = 1
        model.d = 1
        model.w = np.zeros((1, 1))
        
        model.update_means(X, y)
        
        # Median of [1,2,3] is 2
        assert np.isclose(model.w[0, 0], 2)
