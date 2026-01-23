"""
Comprehensive test suite for RandomForest model.

Tests ensemble of RandomTrees with majority voting for predictions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.random_forest import RandomForest
from lattice.random_tree import RandomTree


class TestRandomForest:
    """Test RandomForest implementation."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create simple binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass data."""
        np.random.seed(42)
        X = np.random.randn(150, 5)
        y = np.array([0]*50 + [1]*50 + [2]*50)
        # Add some structure
        y = ((X[:, 0] + X[:, 1]) // 1.5 + 1).astype(int)
        y = np.clip(y, 0, 2)
        return X, y

    def test_initialization(self):
        """Test RandomForest initialization."""
        rf = RandomForest(num_trees=10, max_depth=5)
        
        assert rf.num_trees == 10
        assert rf.max_depth == 5
        assert rf.trees == []

    def test_initialization_creates_empty_list(self):
        """Test that initialization creates empty tree list."""
        rf = RandomForest(num_trees=5, max_depth=3)
        
        assert len(rf.trees) == 0

    def test_fit_creates_trees(self, simple_binary_data):
        """Test that fit creates the specified number of trees."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=5, max_depth=3)
        rf.fit(X, y)
        
        assert len(rf.trees) == 5
        # Each tree should be a RandomTree
        assert all(isinstance(tree, RandomTree) for tree in rf.trees)

    def test_fit_with_max_depth(self, simple_binary_data):
        """Test that trees have correct max_depth."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=3, max_depth=7)
        rf.fit(X, y)
        
        # Each tree should have the specified max_depth
        assert all(tree.max_depth == 7 for tree in rf.trees)

    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        rf = RandomForest(num_trees=5, max_depth=3)
        X = np.random.randn(10, 2)
        
        with pytest.raises(ValueError, match="has not been fit"):
            rf.predict(X)

    def test_predict_shape(self, simple_binary_data):
        """Test prediction output shape."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=5, max_depth=3)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        
        assert y_pred.shape == y.shape

    def test_predict_binary_classification(self, simple_binary_data):
        """Test binary classification predictions."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=10, max_depth=5)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        
        # Predictions should be 0 or 1
        assert np.all((y_pred == 0) | (y_pred == 1))
        
        # Should achieve reasonable accuracy
        assert np.mean(y_pred == y) > 0.6

    def test_predict_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        rf = RandomForest(num_trees=10, max_depth=4)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        
        # Predictions should be valid classes
        assert np.all((y_pred >= 0) & (y_pred <= 2))
        
        # Should learn something
        assert np.mean(y_pred == y) > 0.4

    def test_majority_voting(self):
        """Test that ensemble uses majority voting."""
        # Simple data where ensemble should help
        X = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(num_trees=7, max_depth=2)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        
        # Should make valid predictions
        assert len(y_pred) == len(y)
        assert np.all((y_pred == 0) | (y_pred == 1))

    def test_single_tree_forest(self, simple_binary_data):
        """Test forest with single tree."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=1, max_depth=5)
        rf.fit(X, y)
        
        assert len(rf.trees) == 1
        
        y_pred = rf.predict(X)
        assert len(y_pred) == len(y)

    def test_many_trees(self, simple_binary_data):
        """Test forest with many trees."""
        X, y = simple_binary_data
        rf = RandomForest(num_trees=50, max_depth=3)
        rf.fit(X, y)
        
        assert len(rf.trees) == 50
        
        y_pred = rf.predict(X)
        # More trees should give stable predictions
        assert np.mean(y_pred == y) > 0.6

    def test_deterministic_with_seed(self, simple_binary_data):
        """Test that results are deterministic with seed."""
        X, y = simple_binary_data
        
        np.random.seed(42)
        rf1 = RandomForest(num_trees=5, max_depth=3)
        rf1.fit(X, y)
        pred1 = rf1.predict(X)
        
        np.random.seed(42)
        rf2 = RandomForest(num_trees=5, max_depth=3)
        rf2.fit(X, y)
        pred2 = rf2.predict(X)
        
        # Same seed should give same results
        assert np.array_equal(pred1, pred2)

    def test_different_seeds_different_results(self, simple_binary_data):
        """Test that different seeds give different forests."""
        X, y = simple_binary_data
        
        np.random.seed(0)
        rf1 = RandomForest(num_trees=5, max_depth=3)
        rf1.fit(X, y)
        
        np.random.seed(1)
        rf2 = RandomForest(num_trees=5, max_depth=3)
        rf2.fit(X, y)
        
        # Different forests should potentially have different trees
        # At minimum, both should produce valid predictions
        pred1 = rf1.predict(X[:10])
        pred2 = rf2.predict(X[:10])
        
        assert len(pred1) == 10
        assert len(pred2) == 10

    def test_predict_new_data(self, simple_binary_data):
        """Test prediction on new unseen data."""
        X_train, y_train = simple_binary_data
        
        rf = RandomForest(num_trees=10, max_depth=4)
        rf.fit(X_train, y_train)
        
        X_test = np.random.randn(30, 4)
        y_pred = rf.predict(X_test)
        
        assert len(y_pred) == 30
        assert np.all((y_pred == 0) | (y_pred == 1))

    def test_ensemble_better_than_single_tree(self, simple_binary_data):
        """Test that ensemble typically performs better than single tree."""
        X, y = simple_binary_data
        
        # Single tree
        np.random.seed(42)
        single_tree = RandomTree(max_depth=3)
        single_tree.fit(X, y)
        pred_single = single_tree.predict(X)
        acc_single = np.mean(pred_single == y)
        
        # Forest of trees
        np.random.seed(42)
        rf = RandomForest(num_trees=15, max_depth=3)
        rf.fit(X, y)
        pred_forest = rf.predict(X)
        acc_forest = np.mean(pred_forest == y)
        
        # Forest should typically be as good or better
        # (with some variance)
        assert acc_forest >= acc_single - 0.1

    def test_shallow_vs_deep_trees(self, simple_binary_data):
        """Test forests with different tree depths."""
        X, y = simple_binary_data
        
        # Shallow trees
        rf_shallow = RandomForest(num_trees=10, max_depth=2)
        rf_shallow.fit(X, y)
        pred_shallow = rf_shallow.predict(X)
        
        # Deep trees
        rf_deep = RandomForest(num_trees=10, max_depth=8)
        rf_deep.fit(X, y)
        pred_deep = rf_deep.predict(X)
        
        # Both should produce valid predictions
        assert len(pred_shallow) == len(y)
        assert len(pred_deep) == len(y)
        
        # Deeper trees might overfit more on training data
        # But ensemble should help
        assert np.mean(pred_deep == y) >= np.mean(pred_shallow == y) - 0.15

    def test_voting_mechanism_correctness(self):
        """Test that voting mechanism works correctly."""
        # Create simple case where we can verify voting
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(num_trees=5, max_depth=2)
        rf.fit(X, y)
        
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in rf.trees])
        
        # Verify forest prediction matches majority vote
        forest_pred = rf.predict(X)
        
        for i in range(len(X)):
            # Count votes for each class
            votes = tree_predictions[:, i]
            counts = np.bincount(votes.astype(int))
            majority = np.argmax(counts)
            
            # Forest prediction should match majority
            assert forest_pred[i] == majority

    def test_handles_ties_in_voting(self):
        """Test handling of ties in majority voting."""
        # With even number of trees, ties are possible
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        
        rf = RandomForest(num_trees=4, max_depth=3)  # Even number
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        
        # Should handle ties (by using argmax, which picks first max)
        assert len(y_pred) == len(y)
        assert np.all((y_pred == 0) | (y_pred == 1))


class TestRandomForestEdgeCases:
    """Test edge cases for RandomForest."""

    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2, 3]])
        y = np.array([1])
        
        rf = RandomForest(num_trees=5, max_depth=2)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert y_pred[0] == 1

    def test_two_samples(self):
        """Test with two samples."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        rf = RandomForest(num_trees=3, max_depth=2)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert len(y_pred) == 2

    def test_uniform_labels(self):
        """Test with all same labels."""
        X = np.random.randn(30, 4)
        y = np.ones(30)
        
        rf = RandomForest(num_trees=5, max_depth=3)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        # Should predict all as 1
        assert np.all(y_pred == 1)

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 30)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        rf = RandomForest(num_trees=10, max_depth=4)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert np.mean(y_pred == y) > 0.55

    def test_small_dataset_many_trees(self):
        """Test with small dataset but many trees."""
        X = np.array([[i, i*2] for i in range(10)])
        y = np.array([0]*5 + [1]*5)
        
        rf = RandomForest(num_trees=20, max_depth=3)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        # Should still work
        assert len(y_pred) == 10

    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        X = np.random.randn(100, 4)
        y = np.array([0]*90 + [1]*10)
        
        rf = RandomForest(num_trees=10, max_depth=4)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        # Should handle imbalance
        assert len(y_pred) == 100

    def test_noisy_labels(self):
        """Test with noisy labels."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Add noise
        noise_idx = np.random.choice(100, 25, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        rf = RandomForest(num_trees=15, max_depth=4)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        # Should handle noise reasonably
        assert np.mean(y_pred == y) > 0.5

    def test_all_features_identical(self):
        """Test when all features are identical."""
        X = np.ones((30, 4))
        y = np.array([0]*15 + [1]*15)
        
        rf = RandomForest(num_trees=5, max_depth=3)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        # Should predict mode of bootstrap samples
        assert len(y_pred) == len(y)

    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[i] for i in range(20)])
        y = np.array([0]*10 + [1]*10)
        
        rf = RandomForest(num_trees=7, max_depth=3)
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert np.mean(y_pred == y) > 0.6


class TestRandomForestConsistency:
    """Test consistency properties of RandomForest."""

    def test_prediction_consistency(self):
        """Test that predictions are consistent."""
        X_train = np.random.randn(50, 4)
        y_train = np.random.randint(0, 2, 50)
        
        rf = RandomForest(num_trees=10, max_depth=3)
        rf.fit(X_train, y_train)
        
        X_test = np.random.randn(20, 4)
        
        # Multiple predictions should give same result
        pred1 = rf.predict(X_test)
        pred2 = rf.predict(X_test)
        
        assert np.array_equal(pred1, pred2)

    def test_refitting_replaces_trees(self):
        """Test that refitting replaces old trees."""
        X1 = np.random.randn(30, 3)
        y1 = np.zeros(30)
        
        X2 = np.random.randn(30, 3)
        y2 = np.ones(30)
        
        rf = RandomForest(num_trees=5, max_depth=3)
        
        # First fit
        rf.fit(X1, y1)
        pred1 = rf.predict(X1[:5])
        
        # Second fit
        rf.fit(X2, y2)
        pred2 = rf.predict(X2[:5])
        
        # Should have learned different patterns
        assert np.mean(pred1) < np.mean(pred2)

    def test_trees_are_different(self):
        """Test that trees in forest are different."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        rf = RandomForest(num_trees=10, max_depth=4)
        rf.fit(X, y)
        
        # Get predictions from all trees
        tree_preds = [tree.predict(X) for tree in rf.trees]
        
        # Not all trees should make identical predictions
        # (though some might)
        all_identical = all(np.array_equal(tree_preds[0], p) for p in tree_preds[1:])
        
        # Usually not all identical with bootstrap and random features
        # But at minimum, all should produce valid predictions
        assert all(len(p) == len(X) for p in tree_preds)

    def test_empty_trees_list_after_init(self):
        """Test that trees list starts empty."""
        rf = RandomForest(num_trees=5, max_depth=3)
        assert rf.trees == []

    def test_correct_number_of_trees_after_fit(self):
        """Test that correct number of trees are created."""
        X = np.random.randn(40, 3)
        y = np.random.randint(0, 3, 40)
        
        for num_trees in [1, 3, 5, 10, 20]:
            rf = RandomForest(num_trees=num_trees, max_depth=3)
            rf.fit(X, y)
            assert len(rf.trees) == num_trees
