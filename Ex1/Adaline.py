import numpy as np


class Adaline:

    def __init__(self, n_iter=100, learning_rate=0.01, random_state=42):
        self.n_iter = n_iter
        self.random_state = random_state
        self.learning_rate = learning_rate

    def net_input(self, X: np.ndarray):
        """
        sum(Xi*Wi)+Bi
        (Bi is the last col in X)
        :param X:
        :return:
        """
        # return np.sum((X/100)*self.w)
        return np.dot(X/100, self.w)

    def random_weights(self, X: np.ndarray):
        """
        generate random weights for each feature.
        weight[-1]: bias
        weight[:-1]: weight per feature
        :param X:
        :return:
        """
        rand = np.random.RandomState(self.random_state)
        self.w = rand.normal(loc=0.0, scale=self.learning_rate, size=X.shape[1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        train the model - find the best weights to predict.
        in each iteration,
        fix weight according to each sample error
        :param X:
        :param y:
        :return:
        """
        bias = np.ones([X.shape[0], 1])  # add bias column
        np.append(X, bias, axis=1)
        self.random_weights(X)
        for i in range(self.n_iter):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = (target - output)
                self.w += self.learning_rate * (xi/100).dot(error)


    def predict(self, X: np.ndarray):
        """
        predict X_test, using the new weights.
        if output >= threshold --> 1
        else: -1
        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1.0, -1.0)

    def score(self, X: np.ndarray, y: np.ndarray):
        """
        present of true prediction
        :param X:
        :param y:
        :return:
        """
        counter = 0
        for x, target in zip(X, y):
            p = self.predict(x)
            if p == target:
                counter += 1

        return counter / y.shape[0]
