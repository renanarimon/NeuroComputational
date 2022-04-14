import numpy as np


class Adaline:
    def __init__(self, n_iter=100, random_state=42, learning_rate=0.01, shuffle=True, threshold=0.0):
        self._n_iter = n_iter
        self._random_state = random_state
        self._learning_rate = learning_rate
        self._shuffle = shuffle
        self.threshold = threshold
        self._weight = []

    def randomWeight(self, X: np.ndarray):
        """
        generate random weights for each feature (=X.shape[1] = cols)
        weight[0]: bias
        weight[1:]: weight per feature
        :param X:
        :return:
        """
        rs = np.random.RandomState(self._random_state)
        self._weight = rs.normal(loc=0.0, scale=self._learning_rate, size=X.shape[1] + 1)

    def weightedSum(self, X: np.ndarray) -> np.ndarray:
        """
        sum(Xi*Wi)+Bi
        :param X:
        :return:
        """
        wSum = np.dot(X, self._weight[1:]) # + self._weight[0]
        return wSum

    def activationSigmoid(self, wSum: np.ndarray) -> np.ndarray:
        """
        sigmoid function
        :return:
        """
        return wSum
        # try:
        #     return 1 / (1 + np.exp(-wSum))
        # except Exception as e:
        #     e.__str__()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        train the model - find the best weights to predict.
        error: target - output
        ReWeight:
            * Wi_new = Wi_old + <Xi, error>*learning_rate
            * Bias_i_new = Bias_i_old + sum(error)*learning_rate
        :param X:
        :param y:
        :return:
        """
        self.randomWeight(X)
        for i in range(self._n_iter):
            error = y - self.activationSigmoid(self.weightedSum(X))
            self._weight[1:] += X.T.dot(error) * self._learning_rate
            self._weight[0] += error.sum() * self._learning_rate

    def predict(self, X) -> np.ndarray:
        """
        predict X_test, using the new weights.
        if output >= threshold --> 1
        else: 0
        :param X:
        :return:
        """
        return np.where(self.activationSigmoid(self.weightedSum(X)) >= self.threshold, 1, -1)

    def score(self, X, y):
        """
        count how many right predictions, return the average
        :param X:
        :param y:
        :return:
        """
        countFalse = 0
        for xi, yi in zip(X, y):
            p = self.predict(xi)
            # print(p)
            if p != yi:
                countFalse += 1
        return (len(X) - countFalse) / len(X)


