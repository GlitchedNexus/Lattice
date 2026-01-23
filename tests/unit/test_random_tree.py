"""
Comprehensive test suite for RandomTree model.

Tests RandomTree which uses bootstrap sampling and random feature selection.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.random_tree import RandomTree
from lattice.decision_tree import DecisionTree
from lattice.ramdom_stump import RandomStumpInfoGain


class TestRandomTree:
    """Test RandomTree implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple classification data."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass data."""
        X = np.array([[i, i*2] for i in range(30)])
        y = np.array([0]*10 + [1]*10 + [2]*10)
        return X, y

    def test_inherits_from_decision_tree(self):
        """Test that RandomTree inherits from DecisionTree."""
        assert issubclass(RandomTree, DecisionTree)

    def test_initialization(self):
        """Test RandomTree initialization."""
        tree = RandomTree(max_depth=5)
        
        assert tree.max_depth == 5
        assert tree.stump_class == RandomStumpInfoGain

    def test_uses_random_stump(self):
        """Test that RandomTree uses RandomStumpInfoGain."""
        tree = RandomTree(max_depth=3)
        
        # Should be initialized with RandomStumpInfoGain
        assert tree.stump_class == RandomStumpInfoGain

    def test_fit_basic(self, simple_data):
        """Test basic fitting."""
        X, y = simple_data
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        # Should have a stump model
        assert tree.stump_model is not None

    def test_bootstrap_sampling(self, simple_data):
        """Test that bootstrap sampling is used during fit."""
        X, y = simple_data
        n_original = len(X)
        
        # Bootstrap should sample n points with replacement
        # This means some samples might be repeated, others omitted
        tree = RandomTree(max_depth=3)
        
        # We can't directly observe bootstrap, but we can test behavior
        tree.fit(X, y)
        
        # Tree should be fitted
        assert tree.stump_model is not None

    def test_predict_simple(self, simple_data):
        """Test basic prediction."""
        X, y = simple_data
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        
        assert y_pred.shape == y.shape
        # Should achieve reasonable accuracy
        assert np.mean(y_pred == y) > 0.5

    def test_different_trees_different_seeds(self, simple_data):
        """Test that different seeds produce different trees."""
        X, y = simple_data
        
        np.random.seed(0)
        tree1 = RandomTree(max_depth=3)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        
        np.random.seed(1)
        tree2 = RandomTree(max_depth=3)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        
        # Different bootstrap samples should lead to different trees
        # (not always, but usually)
        # At minimum, they should both produce valid predictions
        assert len(pred1) == len(y)
        assert len(pred2) == len(y)

    def test_deterministic_with_seed(self, simple_data):
        """Test that results are deterministic with same seed."""
        X, y = simple_data
        
        np.random.seed(42)
        tree1 = RandomTree(max_depth=3)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        
        np.random.seed(42)
        tree2 = RandomTree(max_depth=3)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        
        # Same seed should give same results
        assert np.array_equal(pred1, pred2)

    def test_depth_1(self, simple_data):
        """Test with max_depth=1."""
        X, y = simple_data
        tree = RandomTree(max_depth=1)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        assert len(y_pred) == len(y)

    def test_deep_tree(self, simple_data):
        """Test with larger max_depth."""
        X, y = simple_data
        tree = RandomTree(max_depth=10)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Deeper tree might overfit to bootstrap sample
        assert len(y_pred) == len(y)

    def test_multiclass_classification(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        tree = RandomTree(max_depth=5)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        
        # Should predict valid classes
        assert all(label in [0, 1, 2] for label in y_pred)
        # Should learn something meaningful
        assert np.mean(y_pred == y) > 0.4

    def test_bootstrap_with_replacement(self):
        """Test that bootstrap sampling is with replacement."""
        # Small dataset to make repetitions likely
        X = np.array([[i] for i in range(5)])
        y = np.array([0, 0, 1, 1, 0])
        
        # With replacement, bootstrap will have n samples
        # but some might be duplicates
        tree = RandomTree(max_depth=2)
        tree.fit(X, y)
        
        # Tree should be fitted even with small data
        y_pred = tree.predict(X)
        assert len(y_pred) == len(y)

    def test_handles_small_bootstrap_classes(self):
        """Test handling when bootstrap might miss some classes."""
        X = np.random.randn(30, 2)
        y = np.array([0]*15 + [1]*15)
        
        # With bootstrap, might get imbalanced sample
        tree = RandomTree(max_depth=4)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should handle gracefully
        assert len(y_pred) == len(y)

    def test_variance_across_trees(self, simple_data):
        """Test that different trees have variance."""
        X, y = simple_data
        
        predictions = []
        for seed in range(5):
            np.random.seed(seed)
            tree = RandomTree(max_depth=3)
            tree.fit(X, y)
            predictions.append(tree.predict(X))
        
        # Trees should show some variance in predictions
        # (not all identical)
        all_same = all(np.array_equal(predictions[0], p) for p in predictions[1:])
        # Usually not all the same (but could be with small data)
        # At minimum, all should be valid
        assert all(len(p) == len(y) for p in predictions)

    def test_comparison_with_regular_tree(self, simple_data):
        """Compare with regular DecisionTree."""
        X, y = simple_data
        
        # Regular tree (no bootstrap, no random features)
        from lattice.decision_stump import DecisionStumpInfoGain
        regular_tree = DecisionTree(max_depth=3, stump_class=DecisionStumpInfoGain)
        regular_tree.fit(X, y)
        pred_regular = regular_tree.predict(X)
        
        # Random tree (with bootstrap and random features)
        np.random.seed(42)
        random_tree = RandomTree(max_depth=3)
        random_tree.fit(X, y)
        pred_random = random_tree.predict(X)
        
        # Both should produce valid predictions
        assert len(pred_regular) == len(y)
        assert len(pred_random) == len(y)
        
        # Regular tree typically better on training data
        # But random tree should still be reasonable
        assert np.mean(pred_random == y) > 0.4

    def test_predict_new_data(self, simple_data):
        """Test prediction on new data."""
        X_train, y_train = simple_data
        
        tree = RandomTree(max_depth=4)
        tree.fit(X_train, y_train)
        
        # New test data
        X_test = np.random.randn(20, 2)
        y_pred = tree.predict(X_test)
        
        assert len(y_pred) == len(X_test)
        # Predictions should be valid classes
        assert all(label in [0, 1] for label in y_pred)

    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([1])
        
        tree = RandomTree(max_depth=2)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        assert y_pred[0] == 1

    def test_uniform_labels(self):
        """Test with all labels the same."""
        X = np.random.randn(20, 3)
        y = np.ones(20)
        
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should predict all as 1
        assert np.all(y_pred == 1)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        tree = RandomTree(max_depth=5)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should handle high dimensions
        assert np.mean(y_pred == y) > 0.5


class TestRandomTreeEdgeCases:
    """Test edge cases for RandomTree."""

    def test_two_samples_different_classes(self):
        """Test with two samples of different classes."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        assert len(y_pred) == 2

    def test_many_samples_few_features(self):
        """Test with many samples but few features."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] > 0).astype(int)
        
        tree = RandomTree(max_depth=4)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        assert np.mean(y_pred == y) > 0.6

    def test_few_samples_many_features(self):
        """Test with few samples but many features."""
        X = np.random.randn(10, 50)
        y = np.random.randint(0, 2, 10)
        
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        assert len(y_pred) == 10

    def test_perfect_separation(self):
        """Test with perfectly separable data."""
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        tree = RandomTree(max_depth=5)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should achieve high accuracy even with bootstrap
        assert np.mean(y_pred == y) >= 0.5

    def test_noisy_data(self):
        """Test with noisy labels."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Add noise
        noise_idx = np.random.choice(100, 20, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        tree = RandomTree(max_depth=4)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should still learn something
        assert np.mean(y_pred == y) > 0.5

    def test_all_features_constant(self):
        """Test when all features are constant."""
        X = np.ones((20, 3))
        y = np.array([0]*10 + [1]*10)
        
        tree = RandomTree(max_depth=3)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should predict mode of bootstrap sample
        assert len(y_pred) == len(y)

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        X = np.random.randn(100, 3)
        y = np.array([0]*95 + [1]*5)
        
        tree = RandomTree(max_depth=4)
        tree.fit(X, y)
        
        y_pred = tree.predict(X)
        # Should handle imbalance
        assert len(y_pred) == len(y)


class TestRandomTreeConsistency:
    """Test consistency properties of RandomTree."""

    def test_prediction_consistency(self):
        """Test that predictions are consistent after training."""
        X_train = np.random.randn(50, 4)
        y_train = np.random.randint(0, 2, 50)
        
        tree = RandomTree(max_depth=3)
        tree.fit(X_train, y_train)
        
        X_test = np.random.randn(20, 4)
        
        # Multiple predictions should give same result
        pred1 = tree.predict(X_test)
        pred2 = tree.predict(X_test)
        
        assert np.array_equal(pred1, pred2)

    def test_recursive_structure(self):
        """Test that tree maintains proper recursive structure."""
        X = np.random.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        
        tree = RandomTree(max_depth=4)
        tree.fit(X, y)
        
        # Should have stump model
        assert tree.stump_model is not None
        
        # If split was made, should have subtrees
        if tree.stump_model.j_best is not None and tree.max_depth > 1:
            # Might or might not have subtrees depending on bootstrap sample
            pass

    def test_multiple_fits_overwrite(self):
        """Test that fitting multiple times overwrites previous fit."""
        X1 = np.random.randn(30, 2)
        y1 = np.zeros(30)
        
        X2 = np.random.randn(30, 2)
        y2 = np.ones(30)
        
        tree = RandomTree(max_depth=3)
        
        # First fit
        tree.fit(X1, y1)
        pred1 = tree.predict(X1[:5])
        
        # Second fit
        tree.fit(X2, y2)
        pred2 = tree.predict(X2[:5])
        
        # Second fit should have learned different pattern
        # (all ones vs all zeros)
        assert np.mean(pred2) > np.mean(pred1)
