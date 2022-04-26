import random
import numpy as np

from Adaline import Adaline


def createDataSet(num: int) -> (np.ndarray, np.ndarray):
    dataSet = np.ndarray((num, 3))
    for i in range(num):
        m = random.randint(-10000, 10000)
        n = random.randint(-10000, 10000)
        dataSet[i, 0] = m / 1000
        dataSet[i, 1] = n / 1000
        dataSet[i, 2] = 1 if n / 100 > 1 else -1
    return dataSet[:, 0:2], dataSet[:, 2]


if __name__ == '__main__':
    X_train, y_train = createDataSet(1000)
    X_test, y_test = createDataSet(1000)
    a = Adaline(10, 0.01)
    a.fit(X_train, y_train)
    score = a.score(X_test, y_test)
    print(score)
