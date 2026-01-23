import numpy as np


class LeastSquares:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        return X @ self.w


class LeastSquaresBias:
    def __init__(self, X=None, y=None):
        self.w = None
        self.b = None
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # I will add a column of ones to the end of X
        # to account for the bias term.
        n, d = X.shape
        X_b = np.hstack([X, np.ones((n, 1))])

        # Here I solve for the weights and bias together.
        w_b = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        self.w = w_b[:-1]
        self.b = w_b[-1]

    def predict(self, X):
        return X @ self.w + self.b
