import tensorflow as tf
import numpy as np
import scipy.sparse.linalg
import functools
from ESNs.reservoir.standard import Reservoir


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


class ReservoirConceptor(Reservoir):
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
        Reservoir.__init__(self, data, initial=initial, reservoir_size=reservoir_size,
                           i_scale=i_scale, r_scale=r_scale, b_scale=b_scale, washout=washout, prec=prec)
        self.conceptor = self._initial_conceptor(reservoir_size, prec)
        self.weight = self._weight
        self.input_weight = self._input_weight
        self.bias = self._bias
        self.compute

    @define_scope
    def compute(self):
        """
        Compute reservoir states
        :return: list of reservoir states obtained for each time-serie step
        :rtype: list of tensors
        """
        initial_state = self.initial
        states = tf.scan(self._reservoir_step_conceptor, self.data,
                         initializer=initial_state)
        return states

    def _reservoir_step_conceptor(self, previous_state, input_n):
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
        bias = tf.reshape(self.bias, [1, self._reservoir_size])

        state = tf.matmul(previous_state, self.weight) + tf.matmul(input_n, self.input_weight) + bias
        state = tf.tanh(state)
        state = tf.matmul(state, self.conceptor)
        state = tf.reshape(state, [self._reservoir_size])
        return state

    @staticmethod
    def _initial_conceptor(n_states, prec=tf.float64):
        initial_conceptor = np.identity(n_states)
        return tf.Variable(initial_conceptor, dtype=prec)
