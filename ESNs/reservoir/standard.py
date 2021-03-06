import tensorflow as tf
import numpy as np
import scipy.sparse.linalg
import functools


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Reservoir(object):
    def __init__(self, data, initial=None, reservoir_size=50,
                 i_scale=1.5, r_scale=1.5, b_scale=0.2, washout=100, prec=tf.float64):
        """
        Create random reservoir without feedback. The reservoir is dynamical respect to length of
        time-series
        :param reservoir_size: number of neurons in the reservoir
        :type reservoir_size: int
        :param input_size: dimension of time-series input
        :type input_size: int
        :param alpha: spectral radius
        :type alpha: float
        """
        self.data = data
        if initial:
            self.initial = initial
        else:
            self.initial = self._initial_state(reservoir_size, prec)
        self._reservoir_size = reservoir_size
        self._washout = washout
        self._prec = prec

        self._input_size = int(data.get_shape()[1])
        self._input_weight, self._weight, self._bias = self._weights_and_bias(self._input_size,
                                                                              reservoir_size,
                                                                              i_scale, r_scale, b_scale,
                                                                              prec)
        self.compute

    @define_scope
    def compute(self):
        """
        Compute reservoir states
        :return: list of reservoir states obtained for each time-serie step
        :rtype: list of tensors
        """
        initial_state = self.initial
        states = tf.scan(self._reservoir_step, self.data,
                         initializer=initial_state)
        return tf.slice(states, [self._washout, 0], [-1, -1])

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
        previous_state = tf.reshape(previous_state, [1, self._reservoir_size])
        input_n = tf.reshape(input_n, [1, self._input_size])
        bias = tf.reshape(self._bias, [1, self._reservoir_size])

        state = tf.matmul(previous_state, self._weight) + tf.matmul(input_n, self._input_weight) + bias
        state = tf.tanh(state)
        state = tf.reshape(state, [self._reservoir_size])
        return state

    @staticmethod
    def _weights_and_bias(n_inputs, n_states, i_scale, r_scale, b_scale, prec=tf.float64):
        input_weight = i_scale * tf.random_normal(shape=[n_inputs, n_states], dtype=prec)
        bias = b_scale * tf.random_normal(shape=[n_states], dtype=prec)

        def _gen_internal_weights(num_neuron, density):
            weights = scipy.sparse.rand(m=num_neuron, n=num_neuron, density=density, format='coo')
            eigen, _ = scipy.sparse.linalg.eigsh(weights, 1)
            weights /= np.abs(eigen[0])
            return weights.toarray()

        if n_states < 20:
            weight = _gen_internal_weights(n_states, 1)
        else:
            weight = _gen_internal_weights(n_states, 10. / n_states)

        weight = weight * r_scale
        return tf.Variable(input_weight), tf.Variable(weight, dtype=prec), tf.Variable(bias)

    @staticmethod
    def _initial_state(n_states, prec=tf.float64):
        initial_state = tf.zeros([n_states], dtype=prec)
        return initial_state
