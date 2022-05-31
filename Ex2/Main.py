import random
from SOM import Kohonen
import matplotlib.pyplot as plt
import numpy as np


def createData(size: int, part=1):
    if part == 1:
        rand = np.random.RandomState(0)
        data = rand.randint(0, 1000, (size, 2)) / 1000

    elif part == 2:
        rand = np.random.RandomState(0)
        data1 = rand.randint(0, 500, (int(size * 0.8), 2)) / 1000
        data2 = rand.randint(501, 1000, (int(size * 0.2), 2)) / 1000
        data = np.vstack((data1, data2))
        rand.shuffle(data)

    else:
        data = np.ndarray((size, 2))
        s = 0
        while s < size:
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            if 2 <= ((x ** 2) + (y ** 2)) <= 4:
                data[s, 0] = x
                data[s, 1] = y
                s += 1

    return data


if __name__ == '__main__':
    data = createData(1000, 2)
    model = Kohonen(data, 100)
    model.train_SOM()
