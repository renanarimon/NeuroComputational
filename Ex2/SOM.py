# Return the (g,h) index of the BMU in the grid
import numpy as np
import matplotlib.pyplot as plt
from mpmath import rand


class Kohonen:
    def __init__(self, data: np.ndarray, net_size: int):
        """
        SOM: self organization map - weight of each neuron
        :param data: train_data
        :param net_size:
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (h, h, 2)).astype(float) / 1000
        self.data = data

    def find_BMU(self, sample):
        """
        find the most close neuron for this sample
        clac the oclid distance from this sample to all neurons,
        pick the neuron that minimize the dist
        :param sample:
        :return:
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    # Update the weights of the SOM cells when given a single training example
    # and the model parameters along with BMU coordinates as a tuple
    def update_weights(self, train_ex, learn_rate, radius_sq,
                       BMU_coord, step=3):
        """
        update weight of BMU and its neighboors
        :param train_ex:
        :param learn_rate:
        :param radius_sq:
        :param BMU_coord:
        :param step:
        :return:
        """
        g, h = BMU_coord
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.SOM[g, h, :] += learn_rate * (train_ex - self.SOM[g, h, :])
            return self.SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, g - step), min(self.SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(self.SOM.shape[1], h + step)):
                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (train_ex - self.SOM[i, j, :])
        return self.SOM

    # Main routine for training an SOM. It requires an initialized SOM grid
    # or a partially trained grid as parameter
    def train_SOM(self, learn_rate=.1, radius_sq=1,
                  lr_decay=.1, radius_decay=.1, epochs=10):
        rand = np.random.RandomState(0)
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(self.data)
            for train_ex in self.data:
                g, h = self.find_BMU(train_ex)
                self.SOM = self.update_weights(train_ex,
                                               learn_rate, radius_sq, (g, h))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        self.plot()
        return self.SOM

    def plot(self):
        xs = []
        ys = []
        for i in range(self.SOM.shape[0]):
            for j in range(self.SOM.shape[1]):
                xs.append(self.SOM[i, j, 0])
                ys.append(self.SOM[i, j, 1])

        plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.scatter([xs], [ys], c='r')
        plt.plot(xs, ys, 'r')

        plt.show()
