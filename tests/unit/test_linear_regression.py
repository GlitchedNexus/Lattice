"""
Comprehensive test suite for Linear Regression models.

Tests LeastSquares and LeastSquaresBias implementations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.linear_regression import LeastSquares, LeastSquaresBias


class TestLeastSquares:
    """Test LeastSquares (no bias term) implementation."""

    @pytest.fixture
    def simple_linear_data(self):
        """Create simple linear relationship without bias."""
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        # y = 2*x (no bias)
        y = 2 * X.flatten()
        return X, y

    @pytest.fixture
    def multifeature_data(self):
        """Create data with multiple features."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        # y = 2*x1 + 3*x2 - 1*x3
        w_true = np.array([2, 3, -1])
        y = X @ w_true
        return X, y, w_true

    def test_initialization_with_fit(self, simple_linear_data):
        """Test initialization that automatically fits."""
        X, y = simple_linear_data
        model = LeastSquares(X, y)
        
        assert model.w is not None
        assert len(model.w) == X.shape[1]

    def test_initialization_without_fit(self):
        """Test initialization without fitting."""
        model = LeastSquares()
        
        assert model.w is None

    def test_fit_simple(self, simple_linear_data):
        """Test fitting on simple data."""
        X, y = simple_linear_data
        model = LeastSquares()
        model.fit(X, y)
        
        assert model.w is not None
        # Should learn weight close to 2
        assert np.allclose(model.w[0], 2, atol=0.1)

    def test_predict_simple(self, simple_linear_data):
        """Test prediction on simple data."""
        X, y = simple_linear_data
        model = LeastSquares(X, y)
        
        y_pred = model.predict(X)
        
        # Should predict close to actual values
        assert np.allclose(y_pred, y, atol=0.1)

    def test_predict_without_fit_raises_error(self):
        """Test that prediction without fitting raises error."""
        model = LeastSquares()
        X = np.array([[1], [2], [3]])
        
        with pytest.raises(RuntimeError, match="You must fit the model first"):
            model.predict(X)

    def test_multifeature_regression(self, multifeature_data):
        """Test regression with multiple features."""
        X, y, w_true = multifeature_data
        model = LeastSquares(X, y)
        
        # Should recover true weights
        assert np.allclose(model.w, w_true, atol=0.1)

    def test_perfect_fit(self):
        """Test perfect fit on exact linear data."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = LeastSquares(X, y)
        y_pred = model.predict(X)
        
        # Should be exact fit
        assert np.allclose(y_pred, y)

    def test_multicollinearity_handling(self):
        """Test behavior with multicollinear features."""
        # Create perfectly correlated features
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        y = np.array([1, 2, 3, 4, 5])
        
        # This might be singular or near-singular
        # The model should either handle it or raise an error
        try:
            model = LeastSquares(X, y)
            y_pred = model.predict(X)
            # If it works, predictions should still be reasonable
            assert y_pred.shape == y.shape
        except np.linalg.LinAlgError:
            # It's acceptable to raise an error for singular matrices
            pass

    def test_overdetermined_system(self):
        """Test with more samples than features (typical case)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        w_true = np.array([1, -1, 2, -2, 0.5])
        y = X @ w_true + np.random.randn(100) * 0.1  # Add small noise
        
        model = LeastSquares(X, y)
        
        # Should approximate true weights
        assert np.allclose(model.w, w_true, atol=0.5)

    def test_underdetermined_system(self):
        """Test with fewer samples than features."""
        X = np.random.randn(3, 5)
        y = np.array([1, 2, 3])
        
        # Underdetermined system - may have issues
        try:
            model = LeastSquares(X, y)
            # If it fits, check shape
            assert model.w.shape == (5,)
        except np.linalg.LinAlgError:
            # Acceptable to fail on underdetermined system
            pass

    def test_negative_weights(self):
        """Test learning negative weights."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([-2, -4, -6, -8])
        
        model = LeastSquares(X, y)
        
        # Should learn negative weight
        assert model.w[0] < 0
        assert np.allclose(model.w[0], -2, atol=0.1)

    def test_zero_weights(self):
        """Test with features that don't affect output."""
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
        y = np.array([2, 4, 6, 8])
        
        model = LeastSquares(X, y)
        
        # First weight should be ~2, second should be ~0
        assert np.allclose(model.w[0], 2, atol=0.1)
        assert np.allclose(model.w[1], 0, atol=0.1)

    def test_prediction_shape(self):
        """Test that predictions have correct shape."""
        X_train = np.random.randn(20, 3)
        y_train = np.random.randn(20)
        
        model = LeastSquares(X_train, y_train)
        
        X_test = np.random.randn(10, 3)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (10,)


class TestLeastSquaresBias:
    """Test LeastSquaresBias (with bias term) implementation."""

    @pytest.fixture
    def simple_data_with_bias(self):
        """Create simple linear data with bias."""
        np.random.seed(42)
        X = np.array([[1], [2], [3], [4], [5]])
        # y = 2*x + 3 (bias = 3)
        y = 2 * X.flatten() + 3
        return X, y

    @pytest.fixture
    def multifeature_data_with_bias(self):
        """Create multifeature data with bias."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        w_true = np.array([2, 3, -1])
        b_true = 5.0
        y = X @ w_true + b_true
        return X, y, w_true, b_true

    def test_initialization_with_fit(self, simple_data_with_bias):
        """Test initialization with automatic fitting."""
        X, y = simple_data_with_bias
        model = LeastSquaresBias(X, y)
        
        assert model.w is not None
        assert model.b is not None

    def test_initialization_without_fit(self):
        """Test initialization without fitting."""
        model = LeastSquaresBias()
        
        assert model.w is None
        assert model.b is None

    def test_fit_simple_with_bias(self, simple_data_with_bias):
        """Test fitting on simple data with bias."""
        X, y = simple_data_with_bias
        model = LeastSquaresBias()
        model.fit(X, y)
        
        # Should learn weight close to 2 and bias close to 3
        assert np.allclose(model.w[0], 2, atol=0.1)
        assert np.allclose(model.b, 3, atol=0.1)

    def test_predict_with_bias(self, simple_data_with_bias):
        """Test prediction with bias term."""
        X, y = simple_data_with_bias
        model = LeastSquaresBias(X, y)
        
        y_pred = model.predict(X)
        
        # Should predict close to actual values
        assert np.allclose(y_pred, y, atol=0.1)

    def test_multifeature_with_bias(self, multifeature_data_with_bias):
        """Test multifeature regression with bias."""
        X, y, w_true, b_true = multifeature_data_with_bias
        model = LeastSquaresBias(X, y)
        
        # Should recover true weights and bias
        assert np.allclose(model.w, w_true, atol=0.1)
        assert np.allclose(model.b, b_true, atol=0.1)

    def test_zero_bias(self):
        """Test learning with zero bias."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # y = 2*x + 0
        
        model = LeastSquaresBias(X, y)
        
        # Bias should be close to 0
        assert np.allclose(model.b, 0, atol=0.5)
        assert np.allclose(model.w[0], 2, atol=0.1)

    def test_negative_bias(self):
        """Test learning negative bias."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([-1, 1, 3, 5])  # y = 2*x - 3
        
        model = LeastSquaresBias(X, y)
        
        # Should learn negative bias
        assert model.b < 0
        assert np.allclose(model.b, -3, atol=0.5)

    def test_bias_only_model(self):
        """Test when only bias is needed (constant output)."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([5, 5, 5, 5])  # Constant output
        
        model = LeastSquaresBias(X, y)
        
        # Weight should be ~0, bias should be ~5
        assert np.allclose(model.w[0], 0, atol=0.5)
        assert np.allclose(model.b, 5, atol=0.1)

    def test_prediction_shape_with_bias(self):
        """Test prediction shape with bias."""
        X_train = np.random.randn(20, 3)
        y_train = np.random.randn(20)
        
        model = LeastSquaresBias(X_train, y_train)
        
        X_test = np.random.randn(10, 3)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (10,)

    def test_perfect_fit_with_bias(self):
        """Test perfect fit on exact linear data with bias."""
        X = np.array([[1], [2], [3]])
        y = np.array([5, 7, 9])  # y = 2*x + 3
        
        model = LeastSquaresBias(X, y)
        y_pred = model.predict(X)
        
        # Should be exact fit
        assert np.allclose(y_pred, y)

    def test_overdetermined_with_bias(self):
        """Test overdetermined system with bias."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        w_true = np.array([1, -1, 2, -2, 0.5])
        b_true = 3.0
        y = X @ w_true + b_true + np.random.randn(100) * 0.1
        
        model = LeastSquaresBias(X, y)
        
        # Should approximate true parameters
        assert np.allclose(model.w, w_true, atol=0.5)
        assert np.allclose(model.b, b_true, atol=0.5)


class TestLeastSquaresVsLeastSquaresBias:
    """Compare LeastSquares and LeastSquaresBias behavior."""

    def test_bias_necessary(self):
        """Test case where bias term is necessary."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 5, 7, 9])  # y = 2*x + 1
        
        # Without bias
        model_no_bias = LeastSquares(X, y)
        pred_no_bias = model_no_bias.predict(X)
        error_no_bias = np.mean((pred_no_bias - y) ** 2)
        
        # With bias
        model_with_bias = LeastSquaresBias(X, y)
        pred_with_bias = model_with_bias.predict(X)
        error_with_bias = np.mean((pred_with_bias - y) ** 2)
        
        # With bias should be much better
        assert error_with_bias < error_no_bias

    def test_no_bias_needed(self):
        """Test case where bias is not needed."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # y = 2*x (no bias)
        
        # Without bias
        model_no_bias = LeastSquares(X, y)
        pred_no_bias = model_no_bias.predict(X)
        
        # With bias
        model_with_bias = LeastSquaresBias(X, y)
        pred_with_bias = model_with_bias.predict(X)
        
        # Both should perform similarly
        assert np.allclose(pred_no_bias, pred_with_bias, atol=0.5)

    def test_bias_equals_intercept(self):
        """Test that bias term acts as intercept."""
        X = np.array([[0], [1], [2]])
        y = np.array([5, 7, 9])  # y = 2*x + 5
        
        model = LeastSquaresBias(X, y)
        
        # At X=0, prediction should equal bias
        X_zero = np.array([[0]])
        pred_at_zero = model.predict(X_zero)
        
        assert np.allclose(pred_at_zero[0], model.b, atol=0.1)


class TestLinearRegressionEdgeCases:
    """Test edge cases for linear regression models."""

    def test_single_sample_no_bias(self):
        """Test with single sample (no bias)."""
        X = np.array([[2]])
        y = np.array([4])
        
        model = LeastSquares(X, y)
        
        # Should fit perfectly
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y)

    def test_single_sample_with_bias(self):
        """Test with single sample (with bias)."""
        X = np.array([[2]])
        y = np.array([5])
        
        # Single sample is underdetermined for bias model
        # But should still work
        model = LeastSquaresBias(X, y)
        y_pred = model.predict(X)
        
        # Prediction should match training point
        assert np.allclose(y_pred, y, atol=0.1)

    def test_two_samples(self):
        """Test with two samples."""
        X = np.array([[1], [2]])
        y = np.array([3, 5])
        
        model = LeastSquaresBias(X, y)
        y_pred = model.predict(X)
        
        # Should fit perfectly (2 samples, 2 parameters)
        assert np.allclose(y_pred, y)

    def test_all_same_input(self):
        """Test with all same input values."""
        X = np.array([[1], [1], [1], [1]])
        y = np.array([2, 3, 4, 5])
        
        # This should result in singular matrix for no-bias model
        try:
            model = LeastSquares(X, y)
        except np.linalg.LinAlgError:
            pass  # Expected

        # Bias model should work (predicts mean of y)
        model_bias = LeastSquaresBias(X, y)
        y_pred = model_bias.predict(X)
        
        # Should predict close to mean
        assert np.allclose(y_pred, np.mean(y), atol=1.0)

    def test_large_scale_regression(self):
        """Test on larger dataset."""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        w_true = np.random.randn(10)
        b_true = 2.5
        y = X @ w_true + b_true + np.random.randn(1000) * 0.5
        
        model = LeastSquaresBias(X, y)
        
        # Should approximate well
        assert np.allclose(model.w, w_true, atol=0.5)
        assert np.allclose(model.b, b_true, atol=0.5)

    def test_numerical_stability(self):
        """Test numerical stability with different scales."""
        X = np.array([[1000], [2000], [3000], [4000]])
        y = np.array([2000, 4000, 6000, 8000])
        
        model = LeastSquaresBias(X, y)
        y_pred = model.predict(X)
        
        # Should still work despite large values
        assert np.allclose(y_pred, y, rtol=0.01)
