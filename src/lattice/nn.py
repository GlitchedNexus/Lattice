import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.params = self.init_params(layer_dims)

    def train(self, X, Y, layer_dims, epochs, lr):
        params = self.init_params(layer_dims)
        cost_history = []

        for i in range(epochs):
            Y_hat, caches = self.forward_prop(X, params)
            cost = self.cost_function(Y_hat, Y)
            cost_history.append(cost)
            grads = self.backprop(Y_hat, Y, caches)

            params = self.update_parameters(params, grads, lr)

        return params, cost_history

    def predict(self, X):
        pass

    def init_params(self, layer_dims):
        # np.random.seed(3)
        params = {}
        L = len(layer_dims)

        for l in range(1, L):
            params["W" + str(l)] = (
                np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
            )
            params["b" + str(l)] = np.zeros((layer_dims[l], 1))

        return params

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(np.dot(-1, Z)))
        cache = Z

        return A, cache

    def forward_prop(self, X, params):
        A = X
        caches = []
        L = len(params) // 2
        for l in range(1, L + 1):
            A_prev = A

            Z = np.dot(params["W" + str(l)], A_prev) + params["b" + str(l)]

            linear_cache = (A_prev, params["W" + str(l)], params["b" + str(l)])

            A, activation_cache = self.sigmoid(Z)

            cache = (linear_cache, activation_cache)
            caches.append(cache)

        return A, caches

    def one_layer_backward(self, dA, cache):
        linear_cache, activation_cache = cache

        Z = activation_cache
        dZ = dA * self.sigmoid(Z) * (1 - self.sigmoid(Z))

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def backprop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L - 1)], grads["db" + str(L - 1)] = (
            self.one_layer_backward(dAL, current_cache)
        )

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.one_layer_backward(
                grads["dA" + str(l + 1)], current_cache
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)] - learning_rate * grads["W" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)] - learning_rate * grads["b" + str(l + 1)]
            )

        return parameters
