import random
import numpy as np

from Adaline import Adaline
from Ada import Ada


def createDataSet_A(num: int) -> (np.ndarray, np.ndarray):
    dataSet = np.ndarray((num, 3))
    for i in range(num):
        m = random.randint(-10000, 10000)
        n = random.randint(-10000, 10000)
        dataSet[i, 0] = m / 100
        dataSet[i, 1] = n / 100
        dataSet[i, 2] = 1 if n / 100 > 1 else -1
    return dataSet[:, 0:2], dataSet[:, 2]


def createDataSet_B(num: int) -> (np.ndarray, np.ndarray):
    dataSet = np.zeros((num, 3))
    for i in range(num):
        m = random.randint(-10000, 10000)
        n = random.randint(-10000, 10000)
        x = m/ 100
        y = n/100
        dataSet[i, 0] = x
        dataSet[i, 1] = y
        dataSet[i, 2] = 1 if 4 <= ((x ** 2) + (y ** 2)) <= 9 else -1

    return dataSet[:, 0:2], dataSet[:, 2]


if __name__ == '__main__':
    X_train, y_train = createDataSet_A(1000)
    X_test, y_test = createDataSet_A(1000)
    a = Adaline(15, 0.1)
    a.fit(X_train, y_train)
    score = a.score(X_test, y_test)
    print("score A: ",score)

    X_train1, y_train1 = createDataSet_B(1000)
    X_test1, y_test1 = createDataSet_B(1000)
    a1 = Adaline(15, 0.01)
    a1.fit(X_train1, y_train1)
    score1 = a1.score(X_test1, y_test1)
    print("score B: ", score1)

