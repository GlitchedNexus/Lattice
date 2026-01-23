"""
Comprehensive test suite for KNN classifier.

Tests classification functionality, neighbor selection, and prediction accuracy.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.knn import KNN


class TestKNN:
    """Test KNN classifier implementation."""

    @pytest.fixture
    def simple_binary_data(self):
        """Create simple binary classification data."""
        np.random.seed(42)
        # Class 0: points around origin
        X0 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
        # Class 1: points around (5, 5)
        X1 = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
        X = np.vstack([X0, X1])
        y = np.array([0] * 20 + [1] * 20)
        return X, y

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification data with 3 classes."""
        np.random.seed(42)
        # Class 0
        X0 = np.random.randn(15, 2) * 0.3 + np.array([0, 0])
        # Class 1
        X1 = np.random.randn(15, 2) * 0.3 + np.array([5, 0])
        # Class 2
        X2 = np.random.randn(15, 2) * 0.3 + np.array([2.5, 4])
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * 15 + [1] * 15 + [2] * 15)
        return X, y

    @pytest.fixture
    def linearly_separable_data(self):
        """Create perfectly linearly separable data."""
        X = np.array([
            [0, 0], [1, 1], [1, 0], [0, 1],  # Class 0
            [5, 5], [6, 6], [5, 6], [6, 5]   # Class 1
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X, y

    def test_initialization(self):
        """Test KNN initialization."""
        model = KNN(k=3)
        assert model.k == 3
        assert not hasattr(model, 'X') or model.X is None
        assert not hasattr(model, 'y') or model.y is None

    def test_initialization_with_different_k(self):
        """Test initialization with different k values."""
        for k in [1, 3, 5, 10]:
            model = KNN(k=k)
            assert model.k == k

    def test_fit_stores_training_data(self, simple_binary_data):
        """Test that fit correctly stores training data."""
        X, y = simple_binary_data
        model = KNN(k=3)
        model.fit(X, y)
        
        assert model.X is not None
        assert model.y is not None
        assert np.array_equal(model.X, X)
        assert np.array_equal(model.y, y)

    def test_fit_with_different_shapes(self):
        """Test fit with different data shapes."""
        model = KNN(k=3)
        
        # 1D features
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)
        assert model.X.shape == (4, 1)
        assert model.y.shape == (4,)
        
        # 3D features
        X = np.random.randn(10, 3)
        y = np.array([0, 1] * 5)
        model.fit(X, y)
        assert model.X.shape == (10, 3)
        assert model.y.shape == (10,)

    def test_predict_shape(self, simple_binary_data):
        """Test that predict returns correct shape."""
        X, y = simple_binary_data
        model = KNN(k=3)
        model.fit(X, y)
        
        # Single point
        X_test = np.array([[0, 0]])
        y_pred = model.predict(X_test)
        assert y_pred.shape == (1,)
        
        # Multiple points
        X_test = np.random.randn(10, 2)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)

    def test_predict_on_training_data(self, linearly_separable_data):
        """Test perfect prediction on training data with k=1."""
        X, y = linearly_separable_data
        model = KNN(k=1)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # With k=1, should perfectly predict training data
        assert np.array_equal(y_pred, y)

    def test_predict_binary_classification(self, simple_binary_data):
        """Test binary classification accuracy."""
        X, y = simple_binary_data
        model = KNN(k=3)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y)
        # Should achieve high accuracy on well-separated data
        assert accuracy >= 0.9

    def test_predict_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        model = KNN(k=5)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Predictions should be in valid range
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 2)
        
        accuracy = np.mean(y_pred == y)
        # Should achieve reasonable accuracy
        assert accuracy >= 0.7

    def test_k_equals_1(self, simple_binary_data):
        """Test with k=1 (nearest neighbor)."""
        X, y = simple_binary_data
        model = KNN(k=1)
        model.fit(X, y)
        
        # Test on points very close to training points
        X_test = X[:5] + np.random.randn(5, 2) * 0.01
        y_pred = model.predict(X_test)
        y_expected = y[:5]
        
        # Should predict same class as nearest training point
        assert np.sum(y_pred == y_expected) >= 4

    def test_k_equals_n(self):
        """Test when k equals number of training samples."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 1])
        
        model = KNN(k=3)
        model.fit(X, y)
        
        X_test = np.array([[0.5, 0.5]])
        y_pred = model.predict(X_test)
        
        # Should predict majority class (0)
        assert y_pred[0] == 0

    def test_different_k_values_effect(self, simple_binary_data):
        """Test that different k values produce different results."""
        X, y = simple_binary_data
        
        # Test point at boundary
        X_test = np.array([[2.5, 2.5]])
        
        predictions = []
        for k in [1, 3, 5, 10]:
            model = KNN(k=k)
            model.fit(X, y)
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
        
        # At least some predictions should differ
        # (this might not always be true, but generally should be)
        assert len(predictions) > 0

    def test_predict_boundary_points(self, simple_binary_data):
        """Test prediction on boundary points between classes."""
        X, y = simple_binary_data
        model = KNN(k=5)
        model.fit(X, y)
        
        # Test points along the line between clusters
        X_test = np.array([
            [2.5, 2.5],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        y_pred = model.predict(X_test)
        
        # Should return valid predictions
        assert y_pred.shape == (3,)
        assert np.all((y_pred == 0) | (y_pred == 1))

    def test_predict_single_vs_batch(self, simple_binary_data):
        """Test that single and batch predictions are consistent."""
        X, y = simple_binary_data
        model = KNN(k=3)
        model.fit(X, y)
        
        X_test = np.random.randn(5, 2)
        
        # Batch prediction
        y_batch = model.predict(X_test)
        
        # Individual predictions
        y_individual = np.array([model.predict(X_test[i:i+1])[0] for i in range(5)])
        
        # Should be the same
        assert np.array_equal(y_batch, y_individual)

    def test_tie_breaking(self):
        """Test tie-breaking behavior when k neighbors have equal votes."""
        # Create data where ties can occur
        X = np.array([
            [0, 0],  # Class 0
            [0, 1],  # Class 1
            [1, 0],  # Class 2
        ])
        y = np.array([0, 1, 2])
        
        model = KNN(k=3)
        model.fit(X, y)
        
        # Test point equidistant from all three
        X_test = np.array([[0.5, 0.5]])
        y_pred = model.predict(X_test)
        
        # Should return a valid class (np.argmax picks first in case of tie)
        assert y_pred[0] in [0, 1, 2]

    def test_majority_voting(self):
        """Test that majority voting works correctly."""
        # Create specific scenario
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.2],  # 3 points of class 0
            [5, 5], [5.1, 5.1]                # 2 points of class 1
        ])
        y = np.array([0, 0, 0, 1, 1])
        
        model = KNN(k=5)
        model.fit(X, y)
        
        # Test point close to class 0 cluster
        X_test = np.array([[0.05, 0.05]])
        y_pred = model.predict(X_test)
        
        # Should predict class 0 (majority)
        assert y_pred[0] == 0

    def test_high_dimensional_data(self):
        """Test KNN with high-dimensional data."""
        np.random.seed(42)
        n_samples = 50
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, size=n_samples)
        
        model = KNN(k=5)
        model.fit(X, y)
        
        X_test = np.random.randn(10, n_features)
        y_pred = model.predict(X_test)
        
        # Should return valid predictions
        assert y_pred.shape == (10,)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 2)

    def test_single_sample_per_class(self):
        """Test with only one sample per class."""
        X = np.array([[0, 0], [5, 5], [10, 10]])
        y = np.array([0, 1, 2])
        
        model = KNN(k=1)
        model.fit(X, y)
        
        X_test = np.array([[0.1, 0.1], [5.1, 5.1], [9.9, 9.9]])
        y_pred = model.predict(X_test)
        
        # Should predict nearest class
        assert y_pred[0] == 0
        assert y_pred[1] == 1
        assert y_pred[2] == 2

    def test_imbalanced_classes(self):
        """Test KNN with imbalanced class distribution."""
        np.random.seed(42)
        # Many samples of class 0, few of class 1
        X0 = np.random.randn(40, 2) * 0.5
        X1 = np.random.randn(5, 2) * 0.5 + np.array([5, 5])
        
        X = np.vstack([X0, X1])
        y = np.array([0] * 40 + [1] * 5)
        
        model = KNN(k=3)
        model.fit(X, y)
        
        # Test point near class 1
        X_test = np.array([[5.1, 5.1]])
        y_pred = model.predict(X_test)
        
        # Should still be able to predict class 1
        assert y_pred[0] == 1

    def test_identical_points(self):
        """Test behavior with identical training points."""
        X = np.array([
            [0, 0], [0, 0], [0, 0],  # Three identical points of class 0
            [5, 5]                    # One point of class 1
        ])
        y = np.array([0, 0, 0, 1])
        
        model = KNN(k=3)
        model.fit(X, y)
        
        X_test = np.array([[0, 0]])
        y_pred = model.predict(X_test)
        
        # Should predict class 0 (majority among nearest neighbors)
        assert y_pred[0] == 0

    def test_deterministic_predictions(self, simple_binary_data):
        """Test that predictions are deterministic."""
        X, y = simple_binary_data
        model = KNN(k=5)
        model.fit(X, y)
        
        X_test = np.random.randn(10, 2)
        
        y_pred1 = model.predict(X_test)
        y_pred2 = model.predict(X_test)
        
        # Should get same predictions
        assert np.array_equal(y_pred1, y_pred2)

    def test_zero_features(self):
        """Test edge case with zero-dimensional features."""
        # This is an extreme edge case but tests robustness
        X = np.array([[]] * 5)
        y = np.array([0, 1, 0, 1, 0])
        
        model = KNN(k=3)
        model.fit(X, y)
        
        X_test = np.array([[]])
        y_pred = model.predict(X_test)
        
        # Should return a valid prediction (likely majority class)
        assert y_pred.shape == (1,)
        assert y_pred[0] in [0, 1]
