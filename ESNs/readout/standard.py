import tensorflow as tf
import numpy as np
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


class Readout(object):
    def __init__(self, data, target, tychonov=0.01, prec=tf.float64):
        """
        Create a readout layer.
        :param reservoir_size: number of neurons in the reservoir
        :type reservoir_size: int
        :param target_size: dimension of time-series output
        :type target_size: int
        """
        self.data = data
        self.target = target
        self._tychonov = tychonov
        self._reservoir_size = int(data.get_shape()[1])
        self._output_size = int(target.get_shape()[1])
        self._output_weights = self._weights(self._reservoir_size, self._output_size, prec)
        self.prediction
        self.error
        self.optimize

    @define_scope
    def prediction(self):
        """
        Compute linear regression.
        :return: the predictions in function of W, b and states
        :rtype: tensor
        """
        return tf.matmul(self.data, self._output_weights)

    @define_scope
    def optimize(self):
        """
        Compute loss for Ridge regression, but don't work.
        :param param: ridge param
        :type param: float
        :return: loss
        :rtype: tensor
        """
        identity = tf.constant(np.identity(self._reservoir_size))
        new_weights = tf.matmul(tf.transpose(self.data), self.data)
        new_weights += self._tychonov * identity
        new_weights = tf.matrix_inverse(new_weights)
        new_weights = tf.matmul(new_weights, tf.transpose(self.data))
        new_weights = tf.matmul(new_weights, self.target)
        return tf.assign(self._output_weights, new_weights)

    @define_scope
    def error(self):
        return self._nrmse(self.prediction, self.target)

    @staticmethod
    def _weights(reservoir_size, output_size, prec):
        output_weights = tf.random_uniform(shape=[reservoir_size, output_size], minval=-1, maxval=1, dtype=prec)
        return tf.Variable(output_weights)

    @staticmethod
    def _nrmse(prediction, target):
        error_signal = prediction - target
        variance = 0.5 * tf.nn.moments(prediction, axes=[0])[1] + tf.nn.moments(target, axes=[0])[1]
        nrmse = tf.pow(error_signal, 2)
        nrmse = tf.reduce_mean(nrmse, 0)
        nrmse = tf.sqrt(nrmse / variance)
        return nrmse
