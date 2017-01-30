import tensorflow as tf
import numpy as np
import functools
from ESNs.readout.standard import Readout


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


class ReadoutConceptor(Readout):
    def __init__(self, data, target, tychonov=0.01, prec=tf.float64):
        """
        Create a readout layer.
        :param reservoir_size: number of neurons in the reservoir
        :type reservoir_size: int
        :param target_size: dimension of time-series output
        :type target_size: int
        """
        Readout.__init__(self, data, target, tychonov=tychonov, prec=prec)
        self.output_weights = self._output_weights
