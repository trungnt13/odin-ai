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
from odin.config import autoconfig

import lasagne


def random(*shape):
    return np.random.rand(*shape).astype(autoconfig['floatX']) / 12


def zeros(*shape):
    return np.zeros(shape).astype(autoconfig['floatX'])


class CompareTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gru(self):
        # ====== pre-define parameters ====== #
        W_in_to_updategate = random(28, 32)
        W_hid_to_updategate = random(32, 32)
        b_updategate = random(32)

        W_in_to_resetgate = random(28, 32)
        W_hid_to_resetgate = random(32, 32)
        b_resetgate = random(32)

        W_in_to_hidden_update = random(28, 32)
        W_hid_to_hidden_update = random(32, 32)
        b_hidden_update = random(32)

        hid_init = random(1, 32)
        x = random(12, 28, 28)
        # ====== odin ====== #
        X = K.placeholder(shape=(None, 28, 28), name='X')
        f = N.Sequence([
            N.Merge([N.Dense(32, W_init=W_in_to_updategate, b_init=b_updategate, activation=K.linear, name='update'),
                     N.Dense(32, W_init=W_in_to_resetgate, b_init=b_resetgate, activation=K.linear, name='reset'),
                     N.Dense(32, W_init=W_in_to_hidden_update, b_init=b_hidden_update, activation=K.linear, name='hidden')],
                    merge_function=K.concatenate),
            N.GRU(32, activation=K.tanh, gate_activation=K.sigmoid,
                  W_init=[W_hid_to_updategate, W_hid_to_resetgate, W_hid_to_hidden_update],
                  state_init=hid_init)
        ])
        y = f(X)
        f = K.function(X, y)
        out1 = f(x)[0]
        # ====== lasagne ====== #
        l = lasagne.layers.InputLayer(shape=(None, 28, 28))
        l.input_var = X
        l = lasagne.layers.GRULayer(l, num_units=32,
            updategate=lasagne.layers.Gate(W_cell=None,
                                           W_in=W_in_to_updategate,
                                           W_hid=W_hid_to_updategate,
                                           b=b_updategate,
                            nonlinearity=lasagne.nonlinearities.sigmoid),
            resetgate=lasagne.layers.Gate(W_cell=None,
                                          W_in=W_in_to_resetgate,
                                          W_hid=W_hid_to_resetgate,
                                          b=b_resetgate,
                            nonlinearity=lasagne.nonlinearities.sigmoid),
            hidden_update=lasagne.layers.Gate(W_cell=None,
                                              W_in=W_in_to_hidden_update,
                                              W_hid=W_hid_to_hidden_update,
                                              b=b_hidden_update,
                            nonlinearity=lasagne.nonlinearities.tanh),
            hid_init=hid_init,
            precompute_input=True
        )
        y = lasagne.layers.get_output(l)
        f = K.function(X, y)
        out2 = f(x)
        # ====== test ====== #
        self.assertEqual(np.sum(np.abs(out1 - out2)), 0.)

    def test_odin_vs_lasagne(self):
        X1 = K.placeholder(shape=(None, 28, 28))
        X2 = K.placeholder(shape=(None, 784))

        def lasagne_net1():
            i = lasagne.layers.InputLayer(shape=(None, 784))
            i.input_var = X2

            i = lasagne.layers.DenseLayer(i, num_units=32, W=random(784, 32), b=zeros(32),
                nonlinearity=lasagne.nonlinearities.rectify)
            i = lasagne.layers.DenseLayer(i, num_units=16, W=random(32, 16), b=zeros(16),
                nonlinearity=lasagne.nonlinearities.softmax)
            return X2, lasagne.layers.get_output(i)

        def odin_net1():
            f = N.Sequence([
                N.Dense(32, W_init=random(784, 32), b_init=zeros(32),
                    activation=K.relu),
                N.Dense(16, W_init=random(32, 16), b_init=zeros(16),
                    activation=K.softmax)
            ])
            return X2, f(X2)

        def lasagne_net2():
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
            f = N.Sequence([
                N.Dimshuffle((0, 'x', 1, 2)),
                N.Conv2D(12, (3, 3), stride=(1, 1), pad='same',
                    untie_biases=False,
                    W_init=random(12, 1, 3, 3),
                    activation=K.relu),
                N.Pool2D(pool_size=(2, 2), strides=None, mode='max',
                    ignore_border=True),
                N.Conv2D(16, (3, 3), stride=(1, 1), pad='same',
                    untie_biases=False,
                    W_init=random(16, 12, 3, 3),
                    activation=K.sigmoid)
            ])
            return X1, f(X1)

        def lasagne_net3():
            i = lasagne.layers.InputLayer(shape=(None, 28, 28))
            i.input_var = X1

            W = [random(28, 32), random(32, 32), random(32)]
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
            W = [random(28, 32), random(32, 32), random(32)]
            f = N.Sequence([
                N.Dense(num_units=32, W_init=W[0], b_init=W[2],
                    activation=K.linear),
                N.SimpleRecurrent(num_units=32, activation=K.relu,
                    W_init=W[1], state_init=zeros(1, 32))
            ])
            return X1, f(X1)[0]

        lasagne_list = [lasagne_net1, lasagne_net2, lasagne_net3]
        odin_list = [odin_net1, odin_net2, odin_net3]

        for i, j in zip(lasagne_list, odin_list):
            name = str(i) + str(j)
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
            self.assertEqual(y1.shape, y2.shape, msg=name)
            self.assertEqual(np.sum(np.abs(y1 - y2)), 0, msg=name)

if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
