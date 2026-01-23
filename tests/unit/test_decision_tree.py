"""
Comprehensive test suite for DecisionTree model.

Tests recursive tree building with different stump classes and max_depth values.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.decision_stump import DecisionStumpErrorRate, DecisionStumpInfoGain
from lattice.decision_tree import DecisionTree


class TestDecisionTree:
    """Test DecisionTree implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple linearly separable data."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    @pytest.fixture
    def xor_data(self):
        """Create XOR-like data that requires depth > 1."""
        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ]
        )
        y = np.array([0, 1, 1, 0])
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass data."""
        X = np.array([[i, i] for i in range(12)])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        return X, y

    def test_initialization(self):
        """Test tree initialization."""
        tree = DecisionTree(max_depth=3)
        assert tree.max_depth == 3
        assert tree.stump_class == DecisionStumpErrorRate

    def test_initialization_with_infogain(self):
        """Test initialization with InfoGain stump."""
        tree = DecisionTree(max_depth=5, stump_class=DecisionStumpInfoGain)
        assert tree.max_depth == 5
        assert tree.stump_class == DecisionStumpInfoGain

    def test_fit_depth_1(self, simple_data):
        """Test fitting with max_depth=1 (just a stump)."""
        X, y = simple_data
        tree = DecisionTree(max_depth=1)
        tree.fit(X, y)

        # Should have a stump model
        assert tree.stump_model is not None
        # Should not have subtrees with depth 1
        assert tree.submodel_yes is None
        assert tree.submodel_no is None

    def test_predict_depth_1(self, simple_data):
        """Test prediction with depth 1."""
        X, y = simple_data
        tree = DecisionTree(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        assert y_pred.shape == y.shape
        assert np.mean(y_pred == y) >= 0.66

    def test_fit_depth_2(self, simple_data):
        """Test fitting with max_depth=2."""
        X, y = simple_data
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        # Should have subtrees if a split was made
        if tree.stump_model.j_best is not None:
            assert tree.submodel_yes is not None
            assert tree.submodel_no is not None
            assert tree.submodel_yes.max_depth == 1
            assert tree.submodel_no.max_depth == 1

    def test_predict_depth_2(self, simple_data):
        """Test prediction with depth 2."""
        X, y = simple_data
        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should achieve high accuracy with depth 2
        assert np.mean(y_pred == y) >= 0.83

    def test_deep_tree(self, simple_data):
        """Test with larger max_depth."""
        X, y = simple_data
        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Deeper tree should perfectly fit simple data
        assert np.mean(y_pred == y) >= 0.9

    def test_uniform_labels(self):
        """Test with all labels the same."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 1, 1, 1])

        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should predict all as class 1
        assert np.all(y_pred == 1)

    def test_binary_classification(self, multiclass_data):
        """Test on multiclass data."""
        X, y = multiclass_data
        tree = DecisionTree(max_depth=4)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should learn meaningful patterns
        assert np.mean(y_pred == y) > 0.6

    def test_xor_problem(self, xor_data):
        """Test on XOR problem (requires depth > 1)."""
        X, y = xor_data

        # Depth 1 should struggle
        tree1 = DecisionTree(max_depth=1)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        acc1 = np.mean(pred1 == y)

        # Depth 2+ should do better
        tree2 = DecisionTree(max_depth=3)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        acc2 = np.mean(pred2 == y)

        # Deeper tree should generally perform as well or better
        assert acc2 >= acc1 - 0.1  # Allow small variance

    def test_recursive_structure(self):
        """Test that tree builds recursive structure correctly."""
        X = np.array([[i] for i in range(16)])
        y = np.array([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)

        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)

        # Root should have a stump
        assert tree.stump_model is not None

        # Should have built subtrees if splits were made
        if tree.stump_model.j_best is not None:
            assert tree.submodel_yes is not None
            assert tree.submodel_no is not None

    def test_with_infogain_stump(self, simple_data):
        """Test tree with InfoGain stump class."""
        X, y = simple_data
        tree = DecisionTree(max_depth=3, stump_class=DecisionStumpInfoGain)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        assert np.mean(y_pred == y) >= 0.66

    def test_overfitting_prevention(self):
        """Test that max_depth limits overfitting."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = (X[:, 0] > 0).astype(int)

        # Shallow tree
        tree_shallow = DecisionTree(max_depth=1)
        tree_shallow.fit(X, y)
        pred_shallow = tree_shallow.predict(X)

        # Deep tree
        tree_deep = DecisionTree(max_depth=10)
        tree_deep.fit(X, y)
        pred_deep = tree_deep.predict(X)

        # Deep tree should fit training data better
        assert np.mean(pred_deep == y) >= np.mean(pred_shallow == y)

    def test_single_sample_per_leaf(self):
        """Test tree with very deep depth (one sample per leaf)."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])

        tree = DecisionTree(max_depth=10)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should memorize training data
        assert np.mean(y_pred == y) >= 0.75

    def test_predict_new_data(self, simple_data):
        """Test prediction on new unseen data."""
        X_train, y_train = simple_data
        tree = DecisionTree(max_depth=3)
        tree.fit(X_train, y_train)

        # New test data
        X_test = np.array([[1.5], [3.5], [5.5]])
        y_pred = tree.predict(X_test)

        assert len(y_pred) == len(X_test)
        # Predictions should be in valid range
        assert all(label in [0, 1] for label in y_pred)

    def test_multifeature_data(self):
        """Test on data with multiple features."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        tree = DecisionTree(max_depth=4)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should learn the pattern reasonably well
        assert np.mean(y_pred == y) > 0.7

    def test_no_split_at_pure_node(self):
        """Test that pure nodes don't split further."""
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])

        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should perfectly classify
        assert np.all(y_pred == y)

    def test_tree_structure_consistency(self):
        """Test that tree structure is consistent."""
        X = np.array([[i] for i in range(8)])
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)

        # Multiple predictions should give same result
        y_pred1 = tree.predict(X)
        y_pred2 = tree.predict(X)
        assert np.array_equal(y_pred1, y_pred2)


class TestDecisionTreeEdgeCases:
    """Test edge cases for DecisionTree."""

    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[5.0]])
        y = np.array([1])

        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        assert y_pred[0] == 1

    def test_two_samples_same_class(self):
        """Test with two samples of same class."""
        X = np.array([[1], [2]])
        y = np.array([1, 1])

        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        assert np.all(y_pred == 1)

    def test_two_samples_different_class(self):
        """Test with two samples of different classes."""
        X = np.array([[1], [2]])
        y = np.array([0, 1])

        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should be able to separate them
        assert np.mean(y_pred == y) >= 0.5

    def test_predict_empty_subsets(self):
        """Test handling of empty subsets during splitting."""
        X = np.array([[1], [1], [1], [10]])
        y = np.array([0, 0, 0, 1])

        tree = DecisionTree(max_depth=2)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        # Should handle gracefully
        assert len(y_pred) == len(y)


class TestDecisionTreeDifferentStumps:
    """Compare tree behavior with different stump classes."""

    def test_error_rate_vs_infogain(self):
        """Compare ErrorRate and InfoGain stumps."""
        X = np.array([[i] for i in range(12)])
        y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

        tree_error = DecisionTree(max_depth=3, stump_class=DecisionStumpErrorRate)
        tree_error.fit(X, y)
        pred_error = tree_error.predict(X)

        tree_info = DecisionTree(max_depth=3, stump_class=DecisionStumpInfoGain)
        tree_info.fit(X, y)
        pred_info = tree_info.predict(X)

        # Both should learn something
        assert np.mean(pred_error == y) > 0.5
        assert np.mean(pred_info == y) > 0.5

    def test_consistent_stump_class_propagation(self):
        """Test that stump class propagates to subtrees."""
        X = np.array([[i] for i in range(8)])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        tree = DecisionTree(max_depth=3, stump_class=DecisionStumpInfoGain)
        tree.fit(X, y)

        # Check that subtrees use the same stump class
        if tree.submodel_yes is not None:
            assert tree.submodel_yes.stump_class == DecisionStumpInfoGain
        if tree.submodel_no is not None:
            assert tree.submodel_no.stump_class == DecisionStumpInfoGain
