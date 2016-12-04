import tensorflow as tf
import numpy as np
from ESNs.standard_esn.reservoir import Reservoir
from ESNs.standard_esn.readout import Readout
from sklearn.linear_model import Ridge
from test.benchmark import Benchmark
import time


def train_and_test(sess, reservoir, readout, benchmark):
    """
    Train and test of Echo State Network
    :param sess: session of Tensorflow
    :type sess: session
    :param reservoir: graph of reservoir
    :type reservoir: object
    :param readout: graph of readout
    :type readout: object
    :param benchmark: benchmark of nonlinear dynamics
    :type benchmark: object
    :return: none
    :rtype: none
    """
    # Train stage
    start_time = time.time()
    sess.run(tf.initialize_all_variables())
    states = sess.run(reservoir.states, {reservoir.inputs: benchmark.train_input,
                                         reservoir.noise_min:[-0.001], reservoir.noise_max:[0.001]})

    regr = Ridge(alpha=0.01)
    regr.fit(states, benchmark.train_target)
    sess.run([tf.assign(readout.W_out, np.transpose(regr.coef_)), tf.assign(readout.b, regr.intercept_)])

    states = sess.run(reservoir.states, {reservoir.inputs: benchmark.train_input})
    loss_train = sess.run(readout.loss, {readout.states: states, readout.targets: benchmark.train_target})
    print("At training stage %.2f s the loss is:" % (time.time() - start_time), loss_train)

    # Test stage
    start_time = time.time()
    states = sess.run(reservoir.states, {reservoir.inputs: benchmark.test_input})
    loss_test = sess.run(readout.loss, {readout.states: states, readout.targets: benchmark.test_target})
    print("At testing stage %.2f s the loss is:" % (time.time() - start_time), loss_test)


def main():
    reservoir_size = 100
    benchmark = Benchmark(delay=5)
    reservoir = Reservoir(reservoir_size=reservoir_size)
    readout = Readout(reservoir_size=reservoir_size)
    sess = tf.Session()
    train_and_test(sess, reservoir, readout, benchmark)
    sess.close()


if __name__ == '__main__':
    main()
