import tensorflow as tf


class Optimizer(object):
    def __init__(self, loss, learning_rate):
        """
        Simple optimizer
        :param loss: function to optimize
        :type loss: graph
        :param learning_rate: learning rate to GD algorithm
        :type learning_rate: float
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._optimize_op = optimizer.minimize(loss)

    @property
    def optimize_op(self):
        return self._optimize_op
