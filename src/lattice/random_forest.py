import numpy as np

from src.lattice.random_tree import RandomTree


class RandomForest:
    def __init__(self, num_trees, max_depth):
        self.num_trees: int = num_trees
        self.max_depth: int = max_depth
        self.trees: list[RandomTree] = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            model = RandomTree(max_depth=self.max_depth)
            model.fit(X, y)
            self.trees.append(model)

    def predict(self, X_pred):
        if not self.trees:
            raise ValueError("RandomForest has not been fit yet.")

        predictions = np.array([tree.predict(X_pred) for tree in self.trees])

        _, n_samples = predictions.shape
        y_hat = np.empty(n_samples, dtype=int)

        for i in range(n_samples):
            counts = np.bincount(predictions[:, i])
            y_hat[i] = np.argmax(counts)

        return y_hat
