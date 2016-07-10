# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
import cPickle
from six.moves import zip, range

import numpy as np

from odin import backend as K
from odin import nnet as N


class NNetTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dense(self):
        x = K.placeholder((None, 10))

        f1 = N.Dense(20)
        f2 = N.Dense(30)

        y = f2(f1(x))
        y = f1.T(f2.T(y))

        f = K.function(x, y)
        x = f(np.random.rand(12, 10))

        self.assertEquals(x.shape, (12, 10))
        self.assertEquals(K.get_shape(y), (None, 10))

    def test_conv2D(self):
        x = K.placeholder((None, 3, 28, 28))
        f1 = N.Conv2D(16, (3, 3), stride=(2, 2), pad='same')
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 3, 28, 28))

        self.assertEquals(z.shape, (12, 16, 14, 14))
        self.assertEquals(K.get_shape(y), (None, 16, 14, 14))

        # ====== transpose convolution ====== #
        y = f1.T(y)
        f = K.function(x, y)
        z = f(np.random.rand(12, 3, 28, 28))
        self.assertEquals(z.shape, (12, 3, 28, 28))
        self.assertEquals(K.get_shape(y), (None, 3, 28, 28))

    def test_conv3D(self):
        x = K.placeholder((None, 3, 28, 28, 28))
        f1 = N.Conv3D(16, (3, 3, 3), stride=1, pad='valid')
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 3, 28, 28, 28))

        self.assertEquals(z.shape, (12, 16, 26, 26, 26))
        self.assertEquals(K.get_shape(y), (None, 16, 26, 26, 26))

        # ====== transpose convolution ====== #
        # currently not support
        # y = f1.T(y)
        # f = K.function(x, y)
        # z = f(np.random.rand(12, 3, 28, 28, 28))
        # self.assertEquals(z.shape, (12, 3, 28, 28, 28))
        # self.assertEquals(K.get_shape(y), (None, 3, 28, 28, 28))

    def test_dilatedConv(self):
        x = K.placeholder((None, 3, 28, 28))
        f1 = N.DilatedConv2D(16, (3, 3), dilation=(2, 2))
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 3, 28, 28))

        self.assertEquals(z.shape, (12, 16, 24, 24))
        self.assertEquals(K.get_shape(y), (None, 16, 24, 24))

    def test_batch_norm(self):
        x = K.placeholder((None, 8, 12), for_training=True)
        y = N.BatchNorm()(x)
        f = K.function(x, y)
        z = f(np.random.rand(25, 8, 12))
        self.assertEquals(z.shape, (25, 8, 12))

        # ====== Not training ====== #
        x = K.placeholder((None, 8, 12), for_training=False)
        y = N.BatchNorm()(x)
        f = K.function(x, y)
        z = f(np.random.rand(25, 8, 12))
        self.assertEquals(z.shape, (25, 8, 12))

    def test_noise(self):
        x = K.placeholder((2, 3), for_training=True)
        f1 = N.Noise(sigma=0.5, noise_dims=0, noise_type='gaussian', seed=25)
        y = f1(x)
        f = K.function(x, y)
        z = f(np.ones((2, 3)))
        self.assertEquals(z.tolist(),
            [[1.692906141281128, 1.508149266242981, -0.018559932708740234],
            [1.692906141281128, 1.508149266242981, -0.018559932708740234]])

        f1 = N.Noise(sigma=0.5, noise_dims=1, noise_type='gaussian', seed=25)
        y = f1(x)
        f = K.function(x, y)
        z = f(np.ones((2, 3)))
        self.assertEquals(z.tolist(),
            [[-0.03506612777709961, -0.03506612777709961, -0.03506612777709961],
             [1.66799795627594, 1.66799795627594, 1.66799795627594]])

    def test_dropout(self):
        x = K.placeholder((4, 6), for_training=True)
        f1 = N.Dropout(level=0.5, noise_dims=0, rescale=True, seed=25)
        y = f1(x)
        f = K.function(x, y)
        z = f(np.ones((4, 6)))
        self.assertEquals(z.tolist(),
                          [[2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                           [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                           [2.0, 2.0, 0.0, 0.0, 2.0, 2.0],
                           [2.0, 2.0, 0.0, 0.0, 2.0, 2.0]])

        f1 = N.Dropout(level=0.5, noise_dims=1, rescale=True, seed=25)
        y = f1(x)
        f = K.function(x, y)
        z = f(np.ones((4, 6)))
        self.assertEquals(z.tolist(),
                          [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                           [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def test_shape(self):
        x = K.variable(np.ones((25, 8, 12)))

        def test_func(func):
            y = func(x); yT = func.T(func(x))

            self.assertEquals(K.eval(y).shape, K.get_shape(y))

            self.assertEquals(K.eval(yT).shape, (25, 8, 12))
            self.assertEquals(K.eval(yT).shape, K.get_shape(yT))

        test_func(N.FlattenLeft(outdim=2))
        test_func(N.FlattenLeft(outdim=1))
        test_func(N.Reshape((25, 4, 2, 6, 2)))
        test_func(N.Dimshuffle((2, 0, 1)))

    def test_seq(self):
        X = K.placeholder((None, 1, 28, 28), for_training=True)

        f = N.Sequence([
            N.Conv2D(8, (3, 3), stride=1, pad='same'),
            N.FlattenLeft(outdim=2),
            N.Noise(sigma=0.3, noise_dims=None, noise_type='gaussian', seed=12),
            N.Dense(128, activation=K.relu),
            N.Dropout(level=0.3, noise_dims=None, seed=8),
            N.Dense(10, activation=K.softmax)
        ])
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

        self.assertEquals(f1(x).shape, (2688, 10))
        self.assertEquals(f3(x).shape, (2688, 10))
        self.assertEqual(f1(x).sum(), f3(x).sum())
        self.assertEquals(K.get_shape(y), (None, 10))

        self.assertEquals(f2(x).shape, (12, 1, 28, 28))
        self.assertEquals(f4(x).shape, (12, 1, 28, 28))
        self.assertEqual(f2(x).sum(), f4(x).sum())
        self.assertEquals(K.get_shape(yT), (None, 1, 28, 28))

    def test_load_save1(self):
        X = K.placeholder((None, 1, 28, 28), for_training=True)
        f = N.Dense(128, activation=K.relu)
        f(X)
        W, b = [K.get_value(p).sum() for p in f.parameters]
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
        self.assertEqual(W_init1, W_init)
        self.assertEqual(b_init, b_init1)
        self.assertEqual(activation1, activation)

    def test_load_save2(self):
        X = K.placeholder((None, 1, 28, 28), for_training=True)

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
            N.Dimshuffle((0, 'x', 1, 2)),
            N.Conv2D(8, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
            K.pool2d,
            N.FlattenRight(outdim=2),
            N.Dense(64, activation=K.relu),
            N.Dense(10, activation=K.softmax)
        ])
        y = ops(X)
        f1 = K.function(X, y)

        ops_ = cPickle.loads(cPickle.dumps(ops, protocol=cPickle.HIGHEST_PROTOCOL))
        y_ = ops_(X)
        f2 = K.function(X, y_)

        x = np.random.rand(32, 28, 28)
        self.assertEqual(np.sum(f1(x) - f2(x)) == 0.)

    def test_rnn(self):
        pass


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
