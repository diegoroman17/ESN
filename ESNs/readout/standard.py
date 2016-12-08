import tensorflow as tf


class Readout(object):
    def __init__(self, reservoir_size=50, target_size=1):
        """
        Create a readout layer.
        :param reservoir_size: number of neurons in the reservoir
        :type reservoir_size: int
        :param target_size: dimension of time-series output
        :type target_size: int
        """
        self.reservoir_size = reservoir_size
        self.target_size = target_size

        self._states = tf.placeholder(dtype=tf.float32, shape=[None, reservoir_size], name='states')
        self._targets = tf.placeholder(dtype=tf.float32, shape=[None, target_size], name='targets')

        self.W_out = tf.Variable(tf.random_uniform(
            shape=[reservoir_size, target_size],
            minval=-1, maxval=1,
            dtype=tf.float32), name='Wout')

        self.b = tf.Variable(tf.random_uniform(
            shape=[target_size],
            minval=-1, maxval=1,
            dtype=tf.float32), name='b')

        with tf.variable_scope('readout'):
            self._predictions = self._compute_predictions()
            self._loss = self._compute_loss()

    def _compute_predictions(self):
        """
        Compute linear regression.
        :return: the predictions in function of W, b and states
        :rtype: tensor
        """
        return tf.add(tf.matmul(self._states, self.W_out), self.b, name='predictions')

    def _compute_loss(self):
        """
        Compute loss for Ridge regression, but don't work.
        :param param: ridge param
        :type param: float
        :return: loss
        :rtype: tensor
        """
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.square(self._predictions - self._targets))
        return loss

    @property
    def loss(self):
        return self._loss

    @property
    def states(self):
        return self._states

    @property
    def targets(self):
        return self._targets

    @property
    def predictions(self):
        return self._predictions
