import time

import numpy as np
import tensorflow as tf
from Conceptors.standard import ConceptorNetwork
from test.benchmark import Benchmark
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def train_and_test(sess, conceptor_net, washout):
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
    # Pattern generation
    p_1 = Benchmark(delay=0, f=8.8342522)
    p_2 = Benchmark(delay=0, f=9.8342522)
    #p_3 = Benchmark(delay=0, f=12.83425)
    #p_4 = Benchmark(delay=0, f=14.83425)

    # Train stage
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    #conceptor_net.load_patterns([np.hstack((p_1.train_input, p_2.train_input))], sess)
    conceptor_net.load_patterns([p_1.train_input, p_2.train_input], sess)
    _, p1_predictions = conceptor_net.autonomous_run(0, 600, sess)
    _, p2_predictions = conceptor_net.autonomous_run(1, 600, sess)
    #_, p3_predictions = conceptor_net.autonomous_run(2, 1000, sess)
    #_, p4_predictions = conceptor_net.autonomous_run(3, 1000, sess)
    #p1_predictions = conceptor_net.predictions(sess)
    print(conceptor_net.test(sess))
    print('time:', time.time() - start_time)

    plt.plot(p_1.train_input[washout:washout+20])
    plt.plot(p1_predictions[:20])
    plt.savefig('/home/dcabrera/Dropbox/foo1.pdf')
    plt.clf()
    plt.plot(p_2.train_input[washout:washout+20])
    plt.plot(p2_predictions[:20])
    plt.savefig('/home/dcabrera/Dropbox/foo2.pdf')
    """
    plt.clf()
    plt.plot(p_3.train_input[:50])
    plt.plot(p3_predictions[:50])
    plt.savefig('/home/dcabrera/Dropbox/foo3.pdf')
    plt.clf()
    plt.plot(p_4.train_input[:50])
    plt.plot(p4_predictions[:50])
    plt.savefig('/home/dcabrera/Dropbox/foo4.pdf')
    """


def main():
    reservoir_size = 100
    washout = 500
    conceptor_net = ConceptorNetwork(1, reservoir_size, washout=washout, prec=tf.float64)
    sess = tf.Session()
    train_and_test(sess, conceptor_net, washout)
    sess.close()


if __name__ == '__main__':
    main()
