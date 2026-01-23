"""
Comprehensive test suite for DecisionStump models.

Tests both DecisionStumpErrorRate and DecisionStumpInfoGain implementations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.decision_stump import DecisionStumpErrorRate, DecisionStumpInfoGain, entropy


class TestEntropy:
    """Test the entropy helper function."""

    def test_entropy_uniform(self):
        """Entropy should be maximized for uniform distribution."""
        p = np.array([0.5, 0.5])
        expected = np.log(2)  # Maximum entropy for binary
        assert np.isclose(entropy(p), expected)

    def test_entropy_deterministic(self):
        """Entropy should be 0 for deterministic distribution."""
        p = np.array([1.0, 0.0])
        assert np.isclose(entropy(p), 0.0)

    def test_entropy_multiclass(self):
        """Test entropy for multiclass distribution."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        expected = np.log(4)  # log(4) for uniform 4-class
        assert np.isclose(entropy(p), expected)

    def test_entropy_handles_zero(self):
        """Entropy should handle zero probabilities without NaN."""
        p = np.array([0.7, 0.3, 0.0, 0.0])
        result = entropy(p)
        assert not np.isnan(result)
        assert result >= 0


class TestDecisionStumpErrorRate:
    """Test DecisionStumpErrorRate implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple linearly separable data."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        return X, y

    @pytest.fixture
    def multifeature_data(self):
        """Create data with multiple features."""
        X = np.array([
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
            [3, 3],
            [3, 4],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    def test_fit_simple(self, simple_data):
        """Test fitting on simple data."""
        X, y = simple_data
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        # Should find a split
        assert model.j_best is not None
        assert model.t_best is not None
        assert model.y_hat_yes is not None
        assert model.y_hat_no is not None

    def test_predict_simple(self, simple_data):
        """Test prediction on simple data."""
        X, y = simple_data
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        # Should get perfect or near-perfect accuracy on simple data
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.75

    def test_uniform_labels(self):
        """Test behavior when all labels are the same."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1, 1, 1, 1])
        
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        # Should not split when all labels are the same
        assert model.j_best is None
        assert model.t_best is None
        
        # Predictions should all be the mode
        y_pred = model.predict(X)
        assert np.all(y_pred == 1)

    def test_binary_classification(self, multifeature_data):
        """Test binary classification on multifeature data."""
        X, y = multifeature_data
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should learn something meaningful
        assert np.mean(y_pred == y) > 0.5

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        # Check that we predict valid class labels
        assert all(label in [0, 1, 2] for label in y_pred)

    def test_predict_without_fit(self):
        """Test prediction before fitting."""
        model = DecisionStumpErrorRate()
        X = np.array([[1], [2], [3]])
        
        # Should handle gracefully (all None attributes)
        y_pred = model.predict(X)
        assert len(y_pred) == len(X)

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1, 2]])
        y = np.array([1])
        
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred[0] == 1

    def test_feature_selection(self, multifeature_data):
        """Test that the best feature is selected."""
        X, y = multifeature_data
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        
        # Should select one of the features
        assert 0 <= model.j_best < X.shape[1]


class TestDecisionStumpInfoGain:
    """Test DecisionStumpInfoGain implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple linearly separable data."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    @pytest.fixture
    def multifeature_data(self):
        """Create data with multiple features."""
        np.random.seed(42)
        X = np.array([
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1],
        ])
        y = np.array([0, 0, 1, 1, 1])
        return X, y

    def test_fit_simple(self, simple_data):
        """Test fitting with information gain."""
        X, y = simple_data
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        assert model.j_best is not None
        assert model.t_best is not None

    def test_predict_simple(self, simple_data):
        """Test prediction with information gain."""
        X, y = simple_data
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        # Should achieve good accuracy
        assert np.mean(y_pred == y) >= 0.66

    def test_uniform_labels(self):
        """Test with uniform labels (no split needed)."""
        X = np.array([[1], [2], [3]])
        y = np.array([1, 1, 1])
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        # Should not split
        assert model.j_best is None
        
        y_pred = model.predict(X)
        assert np.all(y_pred == 1)

    def test_split_features_parameter(self, multifeature_data):
        """Test the split_features parameter."""
        X, y = multifeature_data
        
        # Test with only first feature
        model = DecisionStumpInfoGain()
        model.fit(X, y, split_features=[0])
        
        # Should only consider feature 0
        assert model.j_best in [None, 0]

    def test_multiclass_infogain(self):
        """Test multiclass with information gain."""
        X = np.array([[i] for i in range(9)])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        # Should learn meaningful splits
        assert np.mean(y_pred == y) > 0.33

    def test_perfect_split(self):
        """Test data with perfect information gain split."""
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should achieve perfect accuracy
        assert np.all(y_pred == y)

    def test_entropy_based_split(self):
        """Test that splits are based on entropy reduction."""
        # Create data where one feature has better information gain
        X = np.array([
            [1, 10],
            [2, 10],
            [10, 1],
            [11, 1],
        ])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        # Both features should work, but one might be preferred
        assert model.j_best in [0, 1]
        
        y_pred = model.predict(X)
        assert np.all(y_pred == y)

    def test_no_information_gain(self):
        """Test when no split provides information gain."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 1, 0, 1])  # Alternating pattern
        
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        
        # Should still make some split or return mode
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)


class TestDecisionStumpComparison:
    """Compare behavior of ErrorRate vs InfoGain stumps."""

    def test_both_stumps_same_perfect_data(self):
        """Both should perform similarly on perfectly separable data."""
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        model_error = DecisionStumpErrorRate()
        model_error.fit(X, y)
        pred_error = model_error.predict(X)
        
        model_info = DecisionStumpInfoGain()
        model_info.fit(X, y)
        pred_info = model_info.predict(X)
        
        # Both should achieve perfect accuracy
        assert np.all(pred_error == y)
        assert np.all(pred_info == y)

    def test_stumps_on_noisy_data(self):
        """Test both stumps on noisy data."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Add some noise
        noise_idx = np.random.choice(50, 5, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        model_error = DecisionStumpErrorRate()
        model_error.fit(X, y)
        pred_error = model_error.predict(X)
        
        model_info = DecisionStumpInfoGain()
        model_info.fit(X, y)
        pred_info = model_info.predict(X)
        
        # Both should learn something
        assert np.mean(pred_error == y) > 0.6
        assert np.mean(pred_info == y) > 0.6
