# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range, cPickle

import numpy as np

from odin import backend as K, nnet as N
from odin.config import CONFIG

import lasagne


def random(*shape):
    return np.random.rand(*shape).astype(CONFIG['floatX']) / 12


def random_bin(*shape):
    return np.random.randint(0, 2, size=shape).astype('int32')


def zeros(*shape):
    return np.zeros(shape).astype(CONFIG['floatX'])


class CompareTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_odin_vs_lasagne(self):
        X1 = K.placeholder(shape=(None, 28, 28))
        X2 = K.placeholder(shape=(None, 784))

        def lasagne_net1():
            "FNN"
            i = lasagne.layers.InputLayer(shape=(None, 784))
            i.input_var = X2

            i = lasagne.layers.DenseLayer(i, num_units=32, W=random(784, 32), b=zeros(32),
                nonlinearity=lasagne.nonlinearities.rectify)
            i = lasagne.layers.DenseLayer(i, num_units=16, W=random(32, 16), b=zeros(16),
                nonlinearity=lasagne.nonlinearities.softmax)
            return X2, lasagne.layers.get_output(i)

        def odin_net1():
            "FNN"
            f = N.Sequence([
                N.Dense(32, W_init=random(784, 32), b_init=zeros(32),
                    activation=K.relu),
                N.Dense(16, W_init=random(32, 16), b_init=zeros(16),
                    activation=K.softmax)
            ])
            return X2, f(X2)

        def lasagne_net2():
            "CNN"
            i = lasagne.layers.InputLayer(shape=(None, 28, 28))
            i.input_var = X1

            i = lasagne.layers.DimshuffleLayer(i, (0, 'x', 1, 2))
            i = lasagne.layers.Conv2DLayer(i, 12, (3, 3), stride=(1, 1), pad='same',
                untie_biases=False,
                W=random(12, 1, 3, 3),
                nonlinearity=lasagne.nonlinearities.rectify)
            i = lasagne.layers.Pool2DLayer(i, pool_size=(2, 2), stride=None, mode='max',
                        ignore_border=True)
            i = lasagne.layers.Conv2DLayer(i, 16, (3, 3), stride=(1, 1), pad='same',
                untie_biases=False,
                W=random(16, 12, 3, 3),
                nonlinearity=lasagne.nonlinearities.sigmoid)
            return X1, lasagne.layers.get_output(i)

        def odin_net2():
            "CNN"
            f = N.Sequence([
                N.Dimshuffle((0, 1, 2, 'x')),
                N.Conv(12, (3, 3), strides=(1, 1), pad='same',
                    untie_biases=False,
                    W_init=random(3, 3, 1, 12),
                    activation=K.relu),
                N.Pool(pool_size=(2, 2), strides=None, mode='max',
                       ignore_border=True),
                N.Conv(16, (3, 3), strides=(1, 1), pad='same',
                    untie_biases=False,
                    W_init=random(3, 3, 12, 16),
                    activation=K.sigmoid),
                N.Dimshuffle((0, 3, 1, 2))
            ])
            return X1, f(X1)

        def lasagne_net3():
            "RNN"
            i = lasagne.layers.InputLayer(shape=(None, 28, 28))
            i.input_var = X1

            W = [random(28, 32), random(32, 32), random(32), random_bin(12, 28)]
            i = lasagne.layers.RecurrentLayer(i, num_units=32,
                W_in_to_hid=W[0],
                W_hid_to_hid=W[1],
                b=W[2],
                nonlinearity=lasagne.nonlinearities.rectify,
                hid_init=zeros(1, 32),
                backwards=False,
                learn_init=False,
                gradient_steps=-1,
                grad_clipping=0,
                unroll_scan=False,
                precompute_input=True,
                mask_input=None,
                only_return_final=False)
            return X1, lasagne.layers.get_output(i)

        def odin_net3():
            "RNN"
            W = [random(28, 32), random(32, 32), random(32), random_bin(12, 28)]
            f = N.Sequence([
                N.Dense(num_units=32, W_init=W[0], b_init=W[2],
                    activation=K.linear),
                N.RNN(num_units=32, activation=K.relu,
                    W_init=W[1])
            ])
            return X1, f(X1, hid_init=zeros(1, 32))

        func_list = [
            (lasagne_net1, odin_net1),
            # (lasagne_net2, odin_net2),
            (lasagne_net3, odin_net3)
        ]
        print()
        for i, j in func_list:
            print('Test:', i.__name__, j.__name__)
            seed = np.random.randint(10e8)
            # ====== call the function ====== #
            np.random.seed(seed)
            i = i()
            np.random.seed(seed)
            j = j()
            # ====== create theano function ====== #
            f1 = K.function(i[0], i[1])
            f2 = K.function(j[0], j[1])
            shape = K.get_shape(i[0])
            # ====== get the output ====== #
            x = np.random.rand(*[12 if s is None else s for s in shape])
            y1 = f1(x)
            y2 = f2(x)
            self.assertEqual(y1.shape, y2.shape)
            self.assertAlmostEqual(np.sum(np.abs(y1 - y2)), 0.)

if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
