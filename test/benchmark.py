import numpy as np


class Benchmark(object):
    # TODO-diego Add more benchmarks
    def __init__(self, train_length=2000, test_length=2000, delay=1):
        """
        Create a train and test data to predict a Mackey Glass non-linear dynamical system
        with T=17
        :param train_length: length of time-series to train
        :type train_length: int
        :param test_length: length of time-series to test
        :type test_length: int
        :param delay: time steps ahead to predict
        :type delay: int
        """
        self.train_length = train_length
        self.test_length = test_length
        self.delay = delay

        data = np.loadtxt('../data/MackeyGlass_t17.txt')
        data = np.atleast_2d(data).T

        self._train_input = data[:train_length]
        self._train_target = data[delay:train_length + delay]

        self._test_input = data[train_length + delay:train_length + test_length + delay]
        self._test_target = data[train_length + 2 * delay:train_length + test_length + 2 * delay]

    @property
    def train_input(self):
        return self._train_input

    @property
    def train_target(self):
        return self._train_target

    @property
    def test_input(self):
        return self._test_input

    @property
    def test_target(self):
        return self._test_target
