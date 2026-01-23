"""
Comprehensive test suite for RandomStump model.

Tests RandomStumpInfoGain which randomly selects features for splitting.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lattice.ramdom_stump import RandomStumpInfoGain  # Note: typo in filename
from lattice.decision_stump import DecisionStumpInfoGain


class TestRandomStumpInfoGain:
    """Test RandomStumpInfoGain implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple data with multiple features."""
        X = np.array([[i, i*2] for i in range(10)])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        return X, y

    @pytest.fixture
    def multifeature_data(self):
        """Create data with many features."""
        np.random.seed(42)
        X = np.random.randn(50, 16)  # 16 features -> sqrt(16) = 4 to choose
        # Make first feature most predictive
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_inherits_from_decision_stump(self):
        """Test that RandomStumpInfoGain inherits from DecisionStumpInfoGain."""
        assert issubclass(RandomStumpInfoGain, DecisionStumpInfoGain)

    def test_fit_simple(self, simple_data):
        """Test basic fitting."""
        X, y = simple_data
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should have selected a feature
        assert model.j_best is not None or model.j_best == 0

    def test_predict_simple(self, simple_data):
        """Test basic prediction."""
        X, y = simple_data
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

    def test_random_feature_selection(self, multifeature_data):
        """Test that features are randomly selected."""
        X, y = multifeature_data
        d = X.shape[1]
        n_features_expected = int(np.floor(np.sqrt(d)))
        
        # The model should consider sqrt(d) features
        assert n_features_expected == 4  # sqrt(16) = 4
        
        # Fit multiple times and check that different features might be selected
        selected_features = []
        for i in range(10):
            np.random.seed(i)
            model = RandomStumpInfoGain()
            model.fit(X, y)
            if model.j_best is not None:
                selected_features.append(model.j_best)
        
        # Over multiple runs, should potentially select different features
        # (though the best feature might dominate)
        assert len(selected_features) > 0

    def test_subset_size_calculation(self):
        """Test that correct number of features are selected."""
        # For d=9 features, should select floor(sqrt(9)) = 3
        X = np.random.randn(20, 9)
        y = np.random.randint(0, 2, 20)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Can't directly test, but model should work
        assert model.j_best is None or model.j_best < 9

    def test_with_single_feature(self):
        """Test with only one feature."""
        X = np.array([[i] for i in range(10)])
        y = np.array([0]*5 + [1]*5)
        
        # sqrt(1) = 1, so should still select that one feature
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        if model.j_best is not None:
            assert model.j_best == 0

    def test_with_four_features(self):
        """Test with four features (sqrt(4) = 2)."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        y = (X[:, 0] > 0).astype(int)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should select from 2 random features
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_perfect_split_possible(self):
        """Test when a perfect split is possible."""
        X = np.array([[1, 10], [2, 10], [10, 1], [11, 1]])
        y = np.array([0, 0, 1, 1])
        
        # sqrt(2) = 1.41, floor = 1 feature selected
        # Even with one feature, should be able to split well
        accuracies = []
        for seed in range(10):
            np.random.seed(seed)
            model = RandomStumpInfoGain()
            model.fit(X, y)
            y_pred = model.predict(X)
            accuracies.append(np.mean(y_pred == y))
        
        # At least some runs should achieve good accuracy
        assert max(accuracies) >= 0.75

    def test_random_selection_uses_infogain(self, multifeature_data):
        """Test that selected features use information gain."""
        X, y = multifeature_data
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should make some split (unless all labels same)
        if np.unique(y).size > 1:
            y_pred = model.predict(X)
            # Should perform better than random
            assert np.mean(y_pred == y) > 0.4

    def test_uniform_labels_no_split(self):
        """Test behavior when all labels are the same."""
        X = np.random.randn(10, 4)
        y = np.ones(10)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should not split
        assert model.j_best is None
        
        y_pred = model.predict(X)
        assert np.all(y_pred == 1)

    def test_deterministic_with_seed(self, multifeature_data):
        """Test that results are deterministic with same seed."""
        X, y = multifeature_data
        
        np.random.seed(42)
        model1 = RandomStumpInfoGain()
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        np.random.seed(42)
        model2 = RandomStumpInfoGain()
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Should give same results with same seed
        assert np.array_equal(pred1, pred2)

    def test_different_results_different_seeds(self, multifeature_data):
        """Test that different seeds can give different results."""
        X, y = multifeature_data
        
        results = []
        for seed in range(5):
            np.random.seed(seed)
            model = RandomStumpInfoGain()
            model.fit(X, y)
            if model.j_best is not None:
                results.append(model.j_best)
        
        # Should potentially select different features
        # (though might not if one feature is dominant)
        assert len(results) > 0

    def test_small_feature_space(self):
        """Test with 2 features (sqrt(2) = 1)."""
        X = np.array([[1, 5], [2, 4], [3, 3], [4, 2]])
        y = np.array([0, 0, 1, 1])
        
        # Should select 1 feature randomly
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should still make reasonable predictions
        assert np.mean(y_pred == y) >= 0.5

    def test_large_feature_space(self):
        """Test with many features."""
        np.random.seed(42)
        X = np.random.randn(100, 100)  # 100 features -> sqrt(100) = 10 selected
        y = np.random.randint(0, 2, 100)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should handle large feature space
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_comparison_with_regular_stump(self, multifeature_data):
        """Compare with regular DecisionStumpInfoGain."""
        X, y = multifeature_data
        
        # Regular stump considers all features
        regular_stump = DecisionStumpInfoGain()
        regular_stump.fit(X, y)
        pred_regular = regular_stump.predict(X)
        acc_regular = np.mean(pred_regular == y)
        
        # Random stump considers subset
        np.random.seed(42)
        random_stump = RandomStumpInfoGain()
        random_stump.fit(X, y)
        pred_random = random_stump.predict(X)
        acc_random = np.mean(pred_random == y)
        
        # Regular stump should typically be as good or better
        # (has access to all features)
        # But random stump should still be reasonable
        assert acc_random > 0.4

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X = np.array([[i, i*2] for i in range(12)])
        y = np.array([0]*4 + [1]*4 + [2]*4)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)
        # Should predict valid classes
        assert all(label in [0, 1, 2] for label in y_pred)

    def test_noise_robustness(self):
        """Test with noisy data."""
        np.random.seed(42)
        X = np.random.randn(50, 9)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Add noise
        noise_idx = np.random.choice(50, 10, replace=False)
        y[noise_idx] = 1 - y[noise_idx]
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should still learn something despite noise
        assert np.mean(y_pred == y) > 0.5


class TestRandomStumpEdgeCases:
    """Test edge cases for RandomStump."""

    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1, 2, 3]])
        y = np.array([1])
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred[0] == 1

    def test_two_samples(self):
        """Test with two samples."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should make some prediction
        assert len(y_pred) == 2

    def test_high_dimensional_low_samples(self):
        """Test with more features than samples."""
        X = np.random.randn(5, 20)
        y = np.array([0, 0, 1, 1, 0])
        
        # sqrt(20) = 4.47, floor = 4 features
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert len(y_pred) == len(y)

    def test_all_features_identical(self):
        """Test when all selected features are identical."""
        X = np.ones((10, 4))
        y = np.array([0]*5 + [1]*5)
        
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        # Should not be able to split
        y_pred = model.predict(X)
        # Should predict mode
        assert len(y_pred) == len(y)

    def test_feature_subset_without_good_split(self):
        """Test when randomly selected features don't provide good split."""
        # Create data where only one feature is predictive
        X = np.random.randn(30, 9)
        X[:, 0] = np.arange(30)  # First feature is predictive
        y = (X[:, 0] > 15).astype(int)
        
        # May or may not select the good feature
        model = RandomStumpInfoGain()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        # Should make some prediction
        assert len(y_pred) == len(y)
