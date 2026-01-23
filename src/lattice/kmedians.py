import numpy as np

from .kmeans import KMeans


def l1_distances(X1, X2):
    # Implementation is based on the euclidean_dist_squared function
    # in utils.py.
    return np.sum(np.abs(X1[:, None, :] - X2[None, :, :]), axis=2)


class KMedians(KMeans):
    # We can reuse most of the code structure from KMeans, rather than copy-pasting,
    # by just overriding these few methods.

    def get_assignments(self, X):
        # This implementation is based on the implementation
        # for the get_assignments() method in KMeans.
        D1 = l1_distances(X, self.w)
        return np.argmin(D1, axis=1)

    def update_means(self, X, y):
        # This implementation is based on the implementation
        # for the update_means() method in KMeans.
        for k_i in range(self.k):
            matching = y == k_i
            if matching.any():
                self.w[k_i] = np.median(X[matching], axis=0)

    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)

        return np.sum(np.abs(X - w[y]))
