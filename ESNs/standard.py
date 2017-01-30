import tensorflow as tf
import functools
from ESNs.reservoir.standard import Reservoir
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


class EsnNetwork(object):
    def __init__(self, data, target, reservoir_size=50, washout=100, prec=tf.float64):
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
        self.target = self._apply_washout(target, washout)
        self._reservoir_size = reservoir_size
        self._washout = washout
        self._prec = prec
        self._reservoir = Reservoir(data, reservoir_size=reservoir_size, washout=washout, prec=prec)
        self._readout = Readout(self._reservoir.compute, target=self.target,prec=prec)
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        return self._readout.prediction

    @define_scope
    def optimize(self):
        return self._readout.optimize

    @define_scope
    def error(self):
        return self._readout.error

    @staticmethod
    def _apply_washout(signal, washout):
        return tf.slice(signal, [washout, 0], [-1, -1])
