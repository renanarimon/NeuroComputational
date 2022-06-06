import numpy as np
import matplotlib.pyplot as plt


class Kohonen:
    def __init__(self, data: np.ndarray, net_size: int):
        """
        SOM: self organization map - weight of each neuron
        :param data: train_data
        :param net_size: num of neurons
        """
        rand = np.random.RandomState(0)
        h = np.sqrt(net_size).astype(int)
        self.SOM = rand.randint(0, 1000, (h, h, 2)).astype(float) / 1000
        self.data = data
        self.net_size = net_size

    def find_BMU(self, sample):
        """
        find the most close neuron for this sample
        clac the oclid distance from this sample to all neurons,
        pick the neuron that minimize the dist
        :param sample: single training example
        :return:
        """
        distSq = (np.square(self.SOM - sample)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def update_weights(self, sample, learn_rate, radius_sq, bmu_idx, step=3):
        """
        update weight of BMU and its neighboors
        :param sample: single training example
        :param learn_rate:
        :param radius_sq:
        :param bmu_idx: the best neuron
        :param step:
        :return:
        """
        x, y = bmu_idx
        # if radius is close to zero then only BMU is changed
        if radius_sq < 1e-3:
            self.SOM[x, y, :] += learn_rate * (sample - self.SOM[x, y, :])
            return self.SOM
        # Change all cells in a small neighborhood of BMU
        for i in range(max(0, x - step), min(self.SOM.shape[0], x + step)):
            for j in range(max(0, y - step), min(self.SOM.shape[1], y + step)):
                dist_sq = np.square(i - x) + np.square(j - y)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.SOM[i, j, :] += learn_rate * dist_func * (sample - self.SOM[i, j, :])
        return self.SOM

    def train_SOM(self, learn_rate=.9, radius_sq=1,
                  lr_decay=.1, radius_decay=.1, epochs=10):
        """
        train SOM model - for each sample:
            1. find BMU
            2. update weights
            3. update learning rate
            4. update radius
        :param lr_decay: Rate of decay of the learn_rate
        :param radius_decay: Rate of decay of the radius
        :param epochs: num of iteration
        :return:
        """
        rand = np.random.RandomState(0)
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(self.data)
            for sample in self.data:
                x, y = self.find_BMU(sample)
                self.SOM = self.update_weights(sample,
                                               learn_rate, radius_sq, (x, y))
            self.plot("curr iter: " + str(epoch) + " , learning rate: " + str(round(learn_rate, 3)))
            # Update learning rate and radius
            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return self.SOM

    def plot(self, title):
        X = self.SOM[:, :, 0]  # The X of each point
        Y = self.SOM[:, :, 1]  # The Y of each point

        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i in range(self.SOM.shape[0]):
            xs = []  # x of each point in axis 1 (cols)
            ys = []  # y of each point in axis 1 (rows)
            xh = []  # x of each point in axis 0 (cols)
            yh = []  # x of each point in axis 1 (rows)
            for j in range(self.SOM.shape[1]):
                xs.append(self.SOM[i, j, 0])
                ys.append(self.SOM[i, j, 1])
                xh.append(self.SOM[j, i, 0])
                yh.append(self.SOM[j, i, 1])

            ax.plot(xs, ys, 'r-', markersize=0, linewidth=0.7)
            ax.plot(xh, yh, 'r-', markersize=0, linewidth=0.7)

        ax.plot(X, Y, color='b', marker='o', linewidth=0, markersize=3)
        ax.scatter(self.data[:, 0], self.data[:, 1], c="c", alpha=0.2)
        plt.title(title)
        plt.show()
