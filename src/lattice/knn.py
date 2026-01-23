"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

from .utils import euclidean_dist_squared


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        """
        Simply stores the training data.
        X: shape (n_train, d)
        y: shape (n_train,)
        """
        self.X = X
        self.y = y

    def predict(self, X_hat):
        """
        Predicts labels for X_hat using KNN.

        X_hat: shape (n_test, d)
        Returns: shape (n_test,)
        """

        distances = euclidean_dist_squared(self.X, X_hat)
        nn_indices = np.argsort(distances, axis=0)[: self.k, :]

        nn_labels = self.y[nn_indices]

        y_hat = np.empty(X_hat.shape[0], dtype=int)

        for j in range(X_hat.shape[0]):
            counts = np.bincount(nn_labels[:, j])
            y_hat[j] = np.argmax(counts)

        return y_hat
