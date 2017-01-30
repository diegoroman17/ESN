from ESNs.reservoir.standard import Reservoir
from ESNs.readout.standard import Readout
import tensorflow as tf
import numpy as np
import functools
from ESNs.reservoir.conceptor import ReservoirConceptor
from ESNs.readout.conceptor import ReadoutConceptor
from test.benchmark import Benchmark


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

def atanh(x):
    return 0.5 * tf.log(tf.div((1 + x), (1 - x)))


class ConceptorNetwork(object):
    def __init__(self,
                 data,
                 target,
                 patterns,
                 reservoir_size=50,
                 aperture=10.,
                 tychonov_reservoir=.0001,
                 tychonov_readout=.01,
                 washout=100,
                 prec=tf.float64):
        self.data = data
        self.target = self._apply_washout(target, washout)
        self.patterns = patterns
        self._reservoir_size = reservoir_size
        self._tychonov_reservoir = tychonov_reservoir
        self._tychonov_readout = tychonov_readout
        self._washout = washout
        self._prec = prec
        self._reservoir = ReservoirConceptor(data, reservoir_size=reservoir_size, washout=washout, prec=prec)
        self._readout = ReadoutConceptor(self._reservoir.compute, target=self.target, prec=prec)
        self.pack_patterns
        """
        self._collector_states_old = []
        self._collector_patterns = []
        self.prec = prec
        self._collector_states = []
        self._x_start = []
        self._collector_conceptors = []
        self.tychonov_readout = tychonov_readout
        self.tychonov_reservoir = tychonov_reservoir
        self._patterns = tf.placeholder(dtype=prec, shape=[None, input_size], name='patterns')
        self._pattern_states = tf.placeholder(dtype=prec, shape=[None, reservoir_size], name='pattern_states')
        self._delayed_pattern_states = tf.placeholder(dtype=prec, shape=[None, reservoir_size],
                                                      name='delayed_pattern_states')
        self.washout = washout
        self.aperture = aperture
        self.diag_reservoir = tf.constant(tychonov_reservoir * np.identity(reservoir_size),
                                          dtype=prec, shape=[reservoir_size, reservoir_size])
        self.diag_readout = tf.constant(tychonov_readout * np.identity(reservoir_size),
                                        dtype=prec, shape=[reservoir_size, reservoir_size])
        self.identity = tf.constant(np.identity(reservoir_size),
                                        dtype=prec, shape=[reservoir_size, reservoir_size])
        self.reservoir = Reservoir(reservoir_size, input_size, washout=washout, prec=prec)
        self.readout = Readout(reservoir_size, input_size, washout=washout, prec=prec)
        self._conceptor = self._compute_conceptor()
        self._new_reservoir, self.w_targets = self._compute_reservoir()
        self._update_reservoir = tf.assign(self.reservoir.W, self.new_reservoir)
        self._readout_param = self._compute_readout()
        self._update_readout = tf.assign(self.readout.W_out, self.readout_param)
        """

    @define_scope
    def _pack_patterns(self):
        con_pat = [self._apply_washout(pat, self._washout) for pat in self.patterns]
        return tf.concat(0, con_pat)

    @define_scope
    def _compute_states(self, data):
        initial_state = self.initial
        states = tf.scan(self._reservoir_step, data, initializer=initial_state)
        return tf.slice(states, [self._washout, 0], [-1, -1]), \
               tf.slice(states, [self._washout-1, tf.shape(data)[0]-self._washout], [-1, -1])

    @define_scope
    def _pack_states(self):
        collection_states = [self._compute_states(data) for data in self.patterns]
        states, states_old = zip(*collection_states)
        return tf.concat(0, states), tf.concat(0, states_old)

    @define_scope
    def load_patterns(self):
        p = self._pack_patterns
        x, x_tilde = self._pack_states
        w_out = tf.matmul(tf.transpose(x), x)
        w_out += self._tychonov_readout*self._identity(self._reservoir_size, self._prec)
        w_out = tf.matrix_inverse(w_out)
        w_out = tf.matmul(w_out, tf.transpose(x))
        w_out = tf.matmul(w_out, p)


        return w_out



    """
    def _compute_conceptor(self):
        with tf.variable_scope('conceptor'):
            correlation = tf.div(tf.matmul(tf.transpose(self.reservoir.states), self.reservoir.states) , tf.cast(
                tf.shape(self.reservoir.states)[0], self.prec))
            s,u,v = tf.svd(correlation,full_matrices=True)
            s = s * self.identity
            s_new = tf.matmul(s, tf.matrix_inverse(s + (self.aperture**(-2.) * self.identity)))
            conceptor = tf.matmul(tf.matmul(u, s_new),tf.transpose(u),name='C')
        return conceptor

    def _compute_reservoir(self):
        with tf.variable_scope('update'):
            X_old = tf.transpose(self.delayed_pattern_states)
            X = tf.transpose(self.pattern_states)
            w_targets = tf.sub(atanh(X), tf.reshape(self.reservoir.b, (self.reservoir.reservoir_size, 1)))
            one = tf.matmul(X_old, tf.transpose(X_old)) + self.diag_reservoir
            two = tf.matrix_inverse(one)
            three = tf.matmul(two, X_old)
            four = tf.matmul(three, tf.transpose(w_targets))
            new_reservoir = four#tf.transpose(four, name='new_reservoir')
        return new_reservoir, w_targets

    def _compute_readout(self):
        with tf.variable_scope('readout'):
            X = tf.transpose(self.pattern_states)
            y_out = tf.transpose(self.patterns)
            one = tf.matmul(X, tf.transpose(X)) + self.diag_readout
            two = tf.matrix_inverse(one)
            three = tf.matmul(two, X)
            four = tf.matmul(three, tf.transpose(y_out))
            readout_param = four
        return readout_param

    def test(self, sess):
        return sess.run(self.reservoir.conceptor_matrix)

    def load_patterns(self, patterns, sess):
        collector_states_old = []
        collector_patterns = []

        for pat in patterns:
            states, states_old, conceptor = sess.run(
                [self.reservoir.states, self.reservoir.states_old, self._conceptor],
                {self.reservoir.inputs: pat})
            self._collector_states.extend(states)
            self._collector_states_old.extend(states_old)
            self._collector_patterns.extend(pat[self.washout:])
            self._collector_conceptors.append(conceptor)
            self._x_start.append(states[-1])

        _ = sess.run(self.update_readout,
                     {self.patterns: self._collector_patterns,
                      self.pattern_states: self._collector_states})
        # clf = Ridge(alpha=0.01, fit_intercept=False)
        # self._collector_states = np.array(self._collector_states)
        # self._collector_patterns = np.array(self._collector_patterns)
        # clf.fit(self._collector_states, self._collector_patterns)
        # sess.run(tf.assign(self.readout.W_out, clf.coef_.T))
        _ = sess.run(self.update_reservoir,
                     {self.pattern_states: self._collector_states,
                      self.delayed_pattern_states: self._collector_states_old})



    def predictions(self, sess):
        return sess.run(self.readout.predictions,
                     {self.readout.states: self._collector_states})

    def autonomous_run(self, idx_conceptor, steps, sess):
        sess.run(tf.assign(self.reservoir.washout, 1))
        sess.run(tf.assign(self.reservoir.conceptor_matrix, self._collector_conceptors[idx_conceptor]))
        # states_filtered = sess.run(self.reservoir.states,
        #                            {self.reservoir.inputs: np.zeros((steps, self.reservoir.input_size)),
        #                             self.reservoir.initial_states: 0.5*np.random.randn(self.reservoir.reservoir_size)})
        states_filtered = sess.run(self.reservoir.states, {self.reservoir.inputs: np.zeros((steps, self.reservoir.input_size)),
                                                           self.reservoir.initial_states: 0.5*np.random.randn(self.reservoir.reservoir_size)})
        pattern_filtered = sess.run(self.readout.predictions, {self.readout.states: states_filtered})
        return states_filtered, pattern_filtered

    @property
    def pattern_states(self):
        return self._pattern_states

    @property
    def delayed_pattern_states(self):
        return self._delayed_pattern_states

    @property
    def patterns(self):
        return self._patterns

    @property
    def new_reservoir(self):
        return self._new_reservoir

    @property
    def update_reservoir(self):
        return self._update_reservoir

    @property
    def readout_param(self):
        return self._readout_param

    @property
    def update_readout(self):
        return self._update_readout

    @property
    def filtered_states(self):
        return self._filtered_states

    @property
    def conceptor_matrix(self, conceptors):
        return self._collector_conceptors[conceptors]
    """

    @staticmethod
    def _apply_washout(signal, washout):
        return tf.slice(signal, [washout, 0], [-1, -1])

    @staticmethod
    def _identity(size, prec):
        return tf.constant(np.identity(size), dtype=prec)

def main():
    p1 = Benchmark(fun='sine', train_length=5000, delay=0, f=8.3333)
    p2 = Benchmark(fun='sine', train_length=5000, delay=0, f=9.3333)
    reservoir_size = 100
    prec = tf.float64
    sess = tf.Session()
    data = tf.placeholder(prec, [None, p1.train_input.shape[1]])
    target = tf.placeholder(prec, [None, p1.train_target.shape[1]])
    patterns = [tf.placeholder(prec, [None, p1.train_input.shape[1]]),
                tf.placeholder(prec, [None, p2.train_input.shape[1]])]
    net = ConceptorNetwork(data, target, patterns, reservoir_size, prec=prec)
    sess.run(tf.global_variables_initializer())
    con_pat = sess.run(net.pack_patterns, {net.patterns[0]:p1.train_input, net.patterns[1]:p2.train_input})
    print(con_pat)

if __name__ == '__main__':
    main()