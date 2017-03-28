# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range, cPickle

import numpy as np

from odin import backend as K
from odin import nnet as N
from odin.utils import Progbar


class SaveTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_save1(self):
        K.set_training(True)
        X = K.placeholder((None, 1, 28, 28))
        f = N.Dense(128, activation=K.relu)
        y = f(X)
        W, b = [K.get_value(p).sum() for p in K.ComputationGraph(y).parameters]
        num_units = f.num_units
        W_init = f.W_init
        b_init = f.b_init
        activation = f.activation

        f = cPickle.loads(cPickle.dumps(f))
        W1, b1 = [K.get_value(p).sum() for p in f.parameters]
        num_units1 = f.num_units
        W_init1 = f.W_init
        b_init1 = f.b_init
        activation1 = f.activation

        self.assertEqual(W1, W)
        self.assertEqual(b1, b)
        self.assertEqual(num_units1, num_units)
        self.assertEqual(W_init1.__name__, W_init.__name__)
        self.assertEqual(b_init.__name__, b_init1.__name__)
        self.assertEqual(activation1, activation)

    def test_load_save2(self):
        K.set_training(True)
        X = K.placeholder((None, 1, 28, 28))

        f = N.Dense(128, activation=K.relu)
        y = f(X)
        yT = f.T(y)
        f1 = K.function(X, y)
        f2 = K.function(X, yT)

        f = cPickle.loads(cPickle.dumps(f))
        y = f(X)
        yT = f.T(y)
        f3 = K.function(X, y)
        f4 = K.function(X, yT)

        x = np.random.rand(12, 1, 28, 28)

        self.assertEqual(f1(x).sum(), f3(x).sum())
        self.assertEqual(f2(x).sum(), f4(x).sum())

    def test_load_save3(self):
        X = K.placeholder(shape=(None, 28, 28))
        ops = N.Sequence([
            N.Dimshuffle(pattern=(0, 1, 2, 'x')),
            N.Conv(8, (3, 3), strides=(1, 1), pad='same', activation=K.relu),
            K.pool2d,
            N.Flatten(outdim=2),
            N.Dense(64, activation=K.relu),
            N.Dense(10, activation=K.softmax)
        ])
        y = ops(X)
        f1 = K.function(X, y)

        ops_ = cPickle.loads(cPickle.dumps(ops, protocol=cPickle.HIGHEST_PROTOCOL))
        y_ = ops_(X)
        f2 = K.function(X, y_)

        x = np.random.rand(32, 28, 28)
        self.assertEqual(np.sum(f1(x) - f2(x)), 0.)
