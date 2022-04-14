import numpy as np


class Adaline_1:

    def __init__(self, n_iter=100, learning_rate=0.01, random_state=42, shuffle=True):
        self.n_iter = n_iter
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.shuffle = shuffle

    def net_input(self, X: np.ndarray):
        return np.dot(X, self.w)

    def random_weights(self, X: np.ndarray):
        rand = np.random.RandomState(self.random_state)
        self.w = rand.normal(loc=0.0, scale=self.learning_rate, size=X.shape[1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        bias = np.ones([X.shape[0],1])
        np.append(X, bias, axis=1)
        self.random_weights(X)
        for i in range(self.n_iter):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = target - output
                self.w += self.learning_rate * xi.dot(error)

    def predict(self, X: np.ndarray):
        return np.where(self.net_input(X) >= 0.0, 1.0, -1.0)

    def score(self, X: np.ndarray, y: np.ndarray):
        counter = 0
        for x, target in zip(X, y):
            p = self.predict(x)
            if p == target:
                counter += 1

        return counter / y.shape[0]
