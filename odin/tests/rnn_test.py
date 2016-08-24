# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import cPickle
import unittest
from six.moves import zip, range

import numpy as np

from odin import backend as K
from odin import nnet as N


class RNNTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def simple_rnn_test(self):
        np.random.seed(12082518)

        x = np.random.rand(128, 8, 32)

        X = K.placeholder(shape=(None, 8, 32))
        X1 = K.placeholder(shape=(None, 8, 32))
        X2 = K.placeholder(shape=(None, 8, 32))
        X3 = K.placeholder(shape=(None, 8, 33))
        f = N.SimpleRecurrent(32, activation=K.relu,
                              state_init=K.init.glorot_uniform)

        y = f(X, mask=K.ones(shape=(128, 8)))
        graph = K.ComputationGraph(y)
        self.assertEqual(len(graph.inputs), 1)
        f1 = K.function([X], y)
        x1 = f1(x)

        # ====== different placeholder ====== #
        y = f(X1)
        f2 = K.function([X1], y)
        x2 = f1(x)
        self.assertEqual(np.sum(x1[0] == x2[0]), np.prod(x1[0].shape))

        # ====== pickle load ====== #
        f = cPickle.loads(cPickle.dumps(f))
        y = f(X2)
        f2 = K.function([X2], y)
        x3 = f2(x)
        self.assertEqual(np.sum(x2[0] == x3[0]), np.prod(x2[0].shape))

        # ====== other input shape ====== #
        y = f(X3)
        f3 = K.function([X3], y)
        error_happen = False
        try:
            x3 = f3(np.random.rand(128, 8, 33))
        except ValueError:
            error_happen = True
        self.assertTrue(error_happen)


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
