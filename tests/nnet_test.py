# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range, cPickle

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin import nnet as N
from odin.utils import Progbar


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
        self.assertEquals(y.shape.as_list(), [None, 10])

    def test_pool_depool(self):
        X1 = K.placeholder(shape=(None, 12, 8, 25), name='X1')
        X2 = K.placeholder(shape=(None, 12, 8, 25, 18), name='X2')
        x1 = np.random.rand(13, 12, 8, 25)
        x2 = np.random.rand(13, 12, 8, 25, 18)
        prog = Progbar(target=2 * 2 * 2 * 3, print_report=True)

        def check_shape(s1, s2):
            self.assertEqual(tuple(s1), tuple(s2), msg="%s != %s" % (str(s1), str(s2)))
        for pool_size in (2, 3):
            for strides in (2, 3):
                # strides > window_shape not supported due to inconsistency
                # between CPU and GPU implementations
                if pool_size < strides:
                    prog.add(1)
                    continue
                for pad in ('valid', 'same'):
                    for transpose_mode in ('nn', 'pad_margin', 'repeat'):
                        # ====== print prog ====== #
                        prog['test'] = "Size:%d,Stride:%d,Pad:%s,T:%s" % \
                            (pool_size, strides, pad, transpose_mode)
                        prog.add(1)
                        # ====== check ops 4D ====== #
                        down = N.Pool(pool_size=pool_size, strides=strides,
                                      pad=pad, mode='max', transpose_mode=transpose_mode)
                        up = down.T
                        y1 = down(X1)
                        check_shape(K.eval(y1, {X1: x1}).shape[1:], y1.shape.as_list()[1:])
                        y2 = up(y1)
                        check_shape(K.eval(y2, {X1: x1}).shape, x1.shape)
                        # ====== check ops 5D ====== #
                        down = N.Pool(pool_size=pool_size, strides=strides,
                                      pad=pad, mode='max', transpose_mode=transpose_mode)
                        up = down.T
                        y1 = down(X2)
                        check_shape(K.eval(y1, {X2: x2}).shape[1:], y1.shape[1:])
                        y2 = up(y1)
                        check_shape(K.eval(y2, {X2: x2}).shape, x2.shape)

    def test_conv2D(self):
        x = K.placeholder((None, 28, 28, 3))
        f1 = N.Conv(16, (3, 3), strides=(2, 2), pad='same')
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 28, 28, 3))

        self.assertEquals(z.shape, (12, 14, 14, 16))
        self.assertEquals(y.shape.as_list(), [None, 14, 14, 16])

        # ====== transpose convolution ====== #
        y = f1.T(y)
        f = K.function(x, y)
        z = f(np.random.rand(12, 28, 28, 3))
        self.assertEquals(z.shape, (12, 28, 28, 3))
        self.assertEquals(y.shape.as_list(), [None, 28, 28, 3])

    def test_conv3D(self):
        x = K.placeholder((None, 28, 28, 28, 3))
        f1 = N.Conv(16, (3, 3, 3), strides=1, pad='valid')
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 28, 28, 28, 3))

        self.assertEquals(z.shape, (12, 26, 26, 26, 16))
        self.assertEquals(y.shape.as_list(), [None, 26, 26, 26, 16])

        # ====== transpose convolution ====== #
        # currently not support
        # y = f1.T(y)
        # f = K.function(x, y)
        # z = f(np.random.rand(12, 3, 28, 28, 28))
        # self.assertEquals(z.shape, (12, 3, 28, 28, 28))
        # self.assertEquals(K.get_shape(y), (None, 3, 28, 28, 28))

    def test_dilatedConv(self):
        x = K.placeholder((None, 28, 28, 3))
        f1 = N.Conv(16, (3, 3), dilation=(2, 2))
        y = f1(x)

        f = K.function(x, y)
        z = f(np.random.rand(12, 28, 28, 3))

        self.assertEquals(z.shape, (12, 24, 24, 16))
        self.assertEquals(y.shape.as_list(), [None, 24, 24, 16])

    def test_batch_norm(self):
        K.set_training(True)
        x = K.placeholder((None, 8, 12))
        y = N.BatchNorm()(x)
        f = K.function(x, y)
        z = f(np.random.rand(25, 8, 12))
        self.assertEquals(z.shape, (25, 8, 12))

        # ====== Not training ====== #
        K.set_training(False)
        x = K.placeholder((None, 8, 12))
        y = N.BatchNorm()(x)
        f = K.function(x, y)
        z = f(np.random.rand(25, 8, 12))
        self.assertEquals(z.shape, (25, 8, 12))

    def test_noise(self):
        x = K.placeholder((2, 3))
        f1 = N.Noise(level=0.5, noise_dims=0, noise_type='gaussian')
        y = f1(x)
        f = K.function(x, y, defaults={K.is_training(): True})
        z = f(np.ones((2, 3)))
        z = z.tolist()
        self.assertTrue(all(i == z[0] for i in z))

        f1 = N.Noise(level=0.5, noise_dims=1, noise_type='gaussian')
        y = f1(x)
        f = K.function(x, y, defaults={K.is_training(): True})
        z = f(np.ones((2, 3)))
        z = z.T.tolist()
        self.assertTrue(all(i == z[0] for i in z))

    def test_dropout(self):
        x = K.placeholder((4, 6))
        f1 = N.Dropout(level=0.5, noise_dims=0, rescale=True)
        y = f1(x)
        f = K.function(x, y, defaults={K.is_training(): True})
        z = f(np.ones((4, 6)))
        z = z.tolist()
        self.assertTrue(all(i == z[0] for i in z))

        f1 = N.Dropout(level=0.5, noise_dims=1, rescale=True)
        y = f1(x)
        f = K.function(x, y, defaults={K.is_training(): True})
        z = f(np.ones((4, 6)))
        z = z.T.tolist()
        self.assertTrue(all(i == z[0] for i in z))

    def test_shape(self):
        x = K.variable(np.ones((25, 8, 12)))

        def test_func(func):
            y = func(x); yT = func.T(func(x))
            self.assertEquals(K.eval(y).shape, tuple(y.shape.as_list()))
            self.assertEquals(K.eval(yT).shape, (25, 8, 12))
            self.assertEquals(K.eval(yT).shape, tuple(yT.shape.as_list()))

        test_func(N.Flatten(outdim=2))
        test_func(N.Flatten(outdim=1))
        test_func(N.Reshape((25, 4, 2, 6, 2)))
        test_func(N.Dimshuffle((2, 0, 1)))

    def test_seq(self):
        X = K.placeholder((None, 28, 28, 1))
        f = N.Sequence([
            N.Conv(8, (3, 3), strides=1, pad='same'),
            N.Dimshuffle(pattern=(0, 3, 1, 2)),
            N.Flatten(outdim=2),
            N.Noise(level=0.3, noise_dims=None, noise_type='gaussian'),
            N.Dense(128, activation=tf.nn.relu),
            N.Dropout(level=0.3, noise_dims=None),
            N.Dense(10, activation=tf.nn.softmax)
        ])
        y = f(X)
        yT = f.T(y)
        f1 = K.function(X, y, defaults={K.is_training(): True})
        f2 = K.function(X, yT, defaults={K.is_training(): False})

        f = cPickle.loads(cPickle.dumps(f))
        y = f(X)
        yT = f.T(y)
        f3 = K.function(X, y, defaults={K.is_training(): True})
        f4 = K.function(X, yT, defaults={K.is_training(): False})

        x = np.random.rand(12, 28, 28, 1)

        self.assertEquals(f1(x).shape, (2688, 10))
        self.assertEquals(f3(x).shape, (2688, 10))
        self.assertEqual(np.round(f1(x).sum(), 4),
                         np.round(f3(x).sum(), 4))
        self.assertEquals(y.shape.as_list(), (None, 10))

        self.assertEquals(f2(x).shape, (12, 28, 28, 1))
        self.assertEquals(f4(x).shape, (12, 28, 28, 1))
        self.assertEqual(str(f2(x).sum())[:4], str(f4(x).sum())[:4])
        self.assertEquals(yT.shape.as_list(), (None, 28, 28, 1))

    def test_slice_ops(self):
        X = K.placeholder(shape=(None, 28, 28, 28, 3))
        f = N.Sequence([
            N.Conv(32, 3, pad='same', activation=K.linear),
            N.BatchNorm(activation=tf.nn.relu),
            N.Flatten(outdim=4)[:, 8:12, 18:25, 13:],
        ])
        y = f(X)
        fn = K.function(X, y)
        self.assertTrue(fn(np.random.rand(12, 28, 28, 28, 3)).shape[1:] ==
                        tuple(y.shape.as_list()[1:]))
        self.assertEqual(y.shape.as_list()[1:], [4, 7, 883])

    def test_helper_ops_variables(self):
        X = K.placeholder(shape=(10, 20))
        f = N.Sequence([
            N.Dense(12),
            N.Dense(8),
            N.BatchNorm(),
            N.Dense(25, W_init=tf.zeros(shape=(8, 25)))
        ])
        y = f(X)
        self.assertEqual(y.shape.as_list(), [10, 25])
        self.assertEqual(len(f.variables), 10)
        self.assertEqual(len(f.parameters), 7)
        self.assertEqual(len(f.trainable_variables), 9)

    def test_conv_deconv_transpose(self):
        def feval(X, y):
            f = K.function(X, y)
            shape = (np.random.randint(8, 18),) + tuple(X.shape.as_list()[1:])
            x = np.random.rand(*shape)
            return f(x)
        prog = Progbar(target=2 * 3 * 3 * 2 * 2, print_report=True)
        for X in (K.placeholder(shape=(None, 13, 12, 25)),
                  K.placeholder(shape=(None, 13, 12, 8, 25))):
            for strides in (1, 2, 3):
                for filter_size in (3, 4, 5):
                    for num_filters in (8, 25):
                        for pad in ("same", "valid"):
                            for dilation in (1,):
                                # ====== progress ====== #
                                prog['test'] = "#Dim:%d;Stride:%d;Filter:%d;Channel:%d;Pad:%s" % \
                                    (X.shape.ndims, strides, filter_size, num_filters, pad)
                                prog.add(1)
                                # ====== test Conv ====== #
                                f = N.Conv(num_filters=num_filters, filter_size=filter_size,
                                           pad=pad, strides=strides, activation=tf.nn.relu,
                                           dilation=dilation)
                                fT = f.T
                                y = f(X)
                                self.assertEqual(feval(X, y).shape[1:], tuple(y.shape.as_list()[1:]))
                                yT = fT(y)
                                self.assertEqual(feval(X, yT).shape[1:], tuple(yT.shape.as_list()[1:]))
                                self.assertEqual(X.shape.as_list(), yT.shape.as_list())
                                # ====== test Transpose ====== #
                                f = N.TransposeConv(num_filters=num_filters, filter_size=filter_size,
                                    pad=pad, strides=strides, activation=K.relu,
                                    dilation=dilation)
                                fT = f.T
                                y = f(X)
                                self.assertEqual(feval(X, y).shape[1:], tuple(y.shape.as_list()[1:]))
                                yT = fT(y)
                                self.assertEqual(feval(X, yT).shape[1:], tuple(yT.shape.as_list()[1:]))
                                self.assertEqual(X.shape.as_list(), yT.shape.as_list())

if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
