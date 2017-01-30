import tensorflow as tf
import numpy as np


class Reservoir(object):
    def __init__(self, reservoir_size=50, input_size=1,
                 alpha=0.9, rho=0.8):
        """
        Create random leaky integrator reservoir without feedback. The reservoir is dynamical respect to length of
        time-series
        :param reservoir_size: number of neurons in the reservoir
        :type reservoir_size: int
        :param input_size: dimension of time-series input
        :type input_size: int
        :param alpha: leaking rate
        :type alpha: float
        :param rho: spectral radius
        :type rho: float
        """
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.alpha = alpha
        self.rho = rho

        self._inputs = tf.placeholder(tf.float32, shape=[None, input_size], name='inputs')
        self._noise_min = tf.placeholder_with_default([0.], shape=[1,], name='noise_min')
        self._noise_max = tf.placeholder_with_default([0.], shape=[1,], name='noise_max')
        self._noise = tf.random_uniform([1], minval=self.noise_min, maxval=self.noise_max,
                                        dtype=tf.float32, name='noise')

        self.W_input = tf.Variable(tf.random_uniform(shape=[input_size, reservoir_size],
                                                     minval=-1, maxval=1, dtype=tf.float32),
                                   name='W_input')

        self.W = tf.Variable(tf.random_uniform(shape=[reservoir_size, reservoir_size],
                                               minval=-1, maxval=1, dtype=tf.float32),
                             dtype=tf.float32, name='W')

        self.b = tf.Variable(tf.random_uniform(shape=[reservoir_size],
                                               minval=-1, maxval=1, dtype=tf.float32),
                             dtype=tf.float32, name='b')

        with tf.variable_scope('reservoir'):
            self._states = self._compute_states()

    def _reservoir_step(self, previous_state, input_n):
        """
        Reservoir step
        :param previous_state: previous state in the reservoir
        :type previous_state: tensor
        :param input_n: time-series step to calculate the actual state
        :type input_n: tensor
        :return: actual state
        :rtype: tensor
        """
        previous_state = tf.reshape(previous_state, [1, self.reservoir_size])

        input_n = tf.reshape(input_n, [1, self.input_size])

        with tf.variable_scope('reservoir_block'):
            state = tf.add(tf.reshape(
                tf.mul(previous_state, (1-self.alpha)) +
                tf.tanh(tf.matmul(previous_state, tf.mul(self.W, self.rho)) +
                        tf.matmul(input_n, self.W_input) + self.noise),
                [self.reservoir_size]), self.b, name='state')

        return state

    def _compute_states(self):
        """
        Compute reservoir states
        :return: list of reservoir states obtained for each time-serie step
        :rtype: list of tensors
        """

        with tf.variable_scope('states'):
            initial_state = tf.zeros([self.reservoir_size], name='initial_state')
            states = tf.scan(self._reservoir_step, self.inputs,
                             initializer=initial_state,
                             name='states')
        return states

    @property
    def inputs(self):
        return self._inputs

    @property
    def noise(self):
        return self._noise

    @property
    def noise_min(self):
        return self._noise_min

    @property
    def noise_max(self):
        return self._noise_max

    @property
    def states(self):
        return self._states
