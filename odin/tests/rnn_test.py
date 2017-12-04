# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
import timeit
from six.moves import zip, range, cPickle

import numpy as np

from odin import backend as K
from odin import nnet as N
from odin.config import get_ngpu, get_floatX, get_backend

import lasagne


np.random.seed(12082518)


def random(*shape):
    return np.random.rand(*shape).astype(get_floatX()) / 12


def random_bin(*shape):
    return np.random.randint(0, 2, size=shape).astype('int32')


def zeros(*shape):
    return np.zeros(shape).astype(get_floatX())


class RNNTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_lstm(self):
        W_in_to_ingate = random(28, 32) / 12
        W_hid_to_ingate = random(32, 32) / 12
        b_ingate = random(32) / 12

        W_in_to_forgetgate = random(28, 32) / 12
        W_hid_to_forgetgate = random(32, 32) / 12
        b_forgetgate = random(32) / 12

        W_in_to_cell = random(28, 32) / 12
        W_hid_to_cell = random(32, 32) / 12
        b_cell = random(32) / 12

        W_in_to_outgate = random(28, 32) / 12
        W_hid_to_outgate = random(32, 32) / 12
        b_outgate = random(32) / 12

        W_cell_to_ingate = random(32) / 12
        W_cell_to_forgetgate = random(32) / 12
        W_cell_to_outgate = random(32) / 12

        cell_init = random(1, 32) / 12
        hid_init = random(1, 32) / 12
        # ====== pre-define parameters ====== #
        x = random(12, 28, 28)
        x_mask = np.random.randint(0, 2, size=(12, 28))
        # x_mask = np.ones(shape=(12, 28))
        # ====== odin ====== #
        X = K.placeholder(shape=(None, 28, 28), name='X')
        mask = K.placeholder(shape=(None, 28), name='mask', dtype='int32')

        f = N.Sequence([
            N.Merge([N.Dense(32, W_init=W_in_to_ingate, b_init=b_ingate, activation=K.linear),
                     N.Dense(32, W_init=W_in_to_forgetgate, b_init=b_forgetgate, activation=K.linear),
                     N.Dense(32, W_init=W_in_to_cell, b_init=b_cell, activation=K.linear),
                     N.Dense(32, W_init=W_in_to_outgate, b_init=b_outgate, activation=K.linear)
                    ], merge_function=K.concatenate),
            N.LSTM(32, activation=K.tanh, gate_activation=K.sigmoid,
                  W_hid_init=[W_hid_to_ingate, W_hid_to_forgetgate, W_hid_to_cell, W_hid_to_outgate],
                  W_peepholes=[W_cell_to_ingate, W_cell_to_forgetgate, W_cell_to_outgate],
                  input_mode='skip',
                  name='lstm')
        ])
        y = f(X, h0=hid_init, c0=cell_init, mask=mask)
        f = K.function([X, mask], y)
        out1 = f(x, x_mask)
        # ====== lasagne ====== #
        if get_backend() == 'tensorflow':
            self.assertTrue(repr(np.sum(out1))[:4] == repr(43.652363)[:4])
            return
        l = lasagne.layers.InputLayer(shape=(None, 28, 28))
        l.input_var = X
        l_mask = lasagne.layers.InputLayer(shape=(None, 28))
        l_mask.input_var = mask
        l = lasagne.layers.LSTMLayer(l, num_units=32,
            ingate=lasagne.layers.Gate(nonlinearity=lasagne.nonlinearities.sigmoid,
                         W_in=W_in_to_ingate,
                         W_hid=W_hid_to_ingate,
                         W_cell=W_cell_to_ingate,
                         b=b_ingate),
            forgetgate=lasagne.layers.Gate(nonlinearity=lasagne.nonlinearities.sigmoid,
                         W_in=W_in_to_forgetgate,
                         W_hid=W_hid_to_forgetgate,
                         W_cell=W_cell_to_forgetgate,
                         b=b_forgetgate),
            cell=lasagne.layers.Gate(nonlinearity=lasagne.nonlinearities.tanh,
                         W_in=W_in_to_cell,
                         W_hid=W_hid_to_cell,
                         W_cell=None,
                         b=b_cell),
            outgate=lasagne.layers.Gate(nonlinearity=lasagne.nonlinearities.sigmoid,
                         W_in=W_in_to_outgate,
                         W_hid=W_hid_to_outgate,
                         W_cell=W_cell_to_outgate,
                         b=b_outgate),
            nonlinearity=lasagne.nonlinearities.tanh,
            cell_init=cell_init,
            hid_init=hid_init,
            mask_input=l_mask,
            precompute_input=True,
            backwards=False
        )
        y = lasagne.layers.get_output(l)
        f = K.function([X, mask], y)
        out2 = f(x, x_mask)
        # ====== test ====== #
        self.assertAlmostEqual(np.sum(np.abs(out1 - out2)), 0.)

    def test_gru(self):
        # ====== pre-define parameters ====== #
        W_in_to_updategate = random(28, 32)
        W_hid_to_updategate = random(32, 32)
        b_updategate = random(32)
        #
        W_in_to_resetgate = random(28, 32)
        W_hid_to_resetgate = random(32, 32)
        b_resetgate = random(32)
        #
        W_in_to_hidden_update = random(28, 32)
        W_hid_to_hidden_update = random(32, 32)
        b_hidden_update = random(32)
        #
        hid_init = random(1, 32)
        x = random(12, 28, 28)
        x_mask = np.random.randint(0, 2, size=(12, 28))
        # ====== odin ====== #
        X = K.placeholder(shape=(None, 28, 28), name='X')
        mask = K.placeholder(shape=(None, 28), name='mask', dtype='int32')

        f = N.Sequence([
            N.Merge([N.Dense(32, W_init=W_in_to_updategate, b_init=b_updategate, activation=K.linear, name='update'),
                     N.Dense(32, W_init=W_in_to_resetgate, b_init=b_resetgate, activation=K.linear, name='reset'),
                     N.Dense(32, W_init=W_in_to_hidden_update, b_init=b_hidden_update, activation=K.linear, name='hidden')],
                    merge_function=K.concatenate),
            N.GRU(32, activation=K.tanh, gate_activation=K.sigmoid,
                  W_hid_init=[W_hid_to_updategate, W_hid_to_resetgate, W_hid_to_hidden_update],
                  input_mode='skip')
        ])
        y = f(X, h0=hid_init, mask=mask)
        f = K.function([X, mask], y)
        out1 = f(x, x_mask)
        # ====== lasagne ====== #
        if get_backend() == 'tensorflow':
            self.assertTrue(repr(np.sum(out1))[:8] == repr(2490.0596)[:8])
            return
        l = lasagne.layers.InputLayer(shape=(None, 28, 28))
        l.input_var = X
        l_mask = lasagne.layers.InputLayer(shape=(None, 28))
        l_mask.input_var = mask
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
            mask_input=l_mask,
            precompute_input=True
        )
        y = lasagne.layers.get_output(l)
        f = K.function([X, mask], y)
        out2 = f(x, x_mask)
        # ====== test ====== #
        self.assertAlmostEqual(np.sum(np.abs(out1 - out2)), 0.)

    def test_cudnn_rnn_backend(self):
        if get_ngpu() == 0:
            return
        print()
        np.random.seed(1208)
        batch_size = 25
        hidden_size = 12
        X_linear = K.placeholder(shape=(None, 8, 32), name='X_linear')
        X_skip = K.placeholder(shape=(None, 8, 12), name='X_skip')
        for direction_mode in ['bidirectional', 'unidirectional']:
            for nb_layers in [1, 2, 3]:
                for rnn_mode in ['gru', 'lstm', 'rnn_tanh']:
                    for input_mode in ['linear', 'skip']:
                        if input_mode == 'linear':
                            X = X_linear
                            x = np.random.rand(batch_size, 8, 32)
                        else:
                            X = X_skip
                            x = np.random.rand(batch_size, 8, 12)
                        start = timeit.default_timer()
                        y = K.rnn_dnn(X, hidden_size=hidden_size,
                                      rnn_mode=rnn_mode,
                                      input_mode=input_mode,
                                      num_layers=nb_layers,
                                      direction_mode=direction_mode)
                        # perform function
                        f = K.function(X, y)
                        output = f(x)
                        benchmark = timeit.default_timer() - start
                        self.assertEqual([list(i.shape) for i in output],
                                         [[batch_size if j is None else j
                                           for j in K.get_shape(i)]
                                          for i in y])
                        print("*PASSED* [Layers]%s [Mode]%-8s [Input]%-6s [Direction]%s [Benchmark]%.4f" %
                            (nb_layers, rnn_mode, input_mode, direction_mode, benchmark))
                        # [np.array(i).sum() for i in output]

    def test_cudnn_rnn_nnet(self):
        if get_ngpu() == 0:
            return
        print()
        np.random.seed(1208)
        batch_size = 6
        hidden_size = 4
        X_linear = K.placeholder(shape=(None, 3, 8), name='X_linear')
        X_skip = K.placeholder(shape=(None, 3, hidden_size), name='X_skip')
        for direction_mode in ['bidirectional', 'unidirectional']:
            is_bidirectional = direction_mode == 'bidirectional'
            for nb_layers in [2]:
                real_layers = nb_layers * 2 if is_bidirectional else nb_layers
                for rnn_mode in ['gru', 'lstm', 'rnn_relu', 'rnn_tanh']:
                    for init_state, init_state_name in zip([
                        None, # None init
                        K.rand.uniform, # function init
                        K.variable(np.random.rand(real_layers, 1, hidden_size)), # variable
                        K.variable(np.random.rand(real_layers, batch_size, hidden_size)), # variable
                        K.zeros(shape=(real_layers, 1, hidden_size)),
                        K.ones(shape=(real_layers, batch_size, hidden_size))
                    ], ['None', 'Function', 'Var1', 'VarB', 'Tensor1', 'TensorB']):
                        for input_mode in ['linear', 'skip']:
                            if input_mode == 'linear':
                                X = X_linear
                                x = np.random.rand(batch_size, 3, 8)
                            else:
                                X = X_skip
                                x = np.random.rand(batch_size, 3, hidden_size)
                            start = timeit.default_timer()
                            f = N.CudnnRNN(num_units=hidden_size, rnn_mode=rnn_mode,
                                           input_mode=input_mode, num_layers=nb_layers,
                                           direction_mode=direction_mode,
                                           params_split=False,
                                           return_states=True)
                            # perform function
                            y = f(X, h0=init_state, c0=init_state)
                            f = K.function(X, y)
                            output = f(x)
                            benchmark = timeit.default_timer() - start
                            self.assertTrue([list(i.shape) for i in output] ==
                                            [[batch_size if j is None else j
                                              for j in K.get_shape(i)]
                                             for i in y])
                            print("*PASSED* [Layers]%s [Mode]%-8s [Input]%-6s [Direction]%-12s [State]%s [Benchmark]%.4f" %
                                (nb_layers, rnn_mode, input_mode, direction_mode, init_state_name, benchmark))

    def test_simple_rnn(self):
        np.random.seed(12082518)
        x = np.random.rand(128, 8, 32)
        #
        X = K.placeholder(shape=(None, 8, 32))
        X1 = K.placeholder(shape=(None, 8, 32))
        X2 = K.placeholder(shape=(None, 8, 32))
        X3 = K.placeholder(shape=(None, 8, 33))
        f = N.RNN(32, activation=K.relu, input_mode='skip')
        #
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
        error_happen = False
        try:
            y = f(X3)
            f3 = K.function([X3], y)
            x3 = f3(np.random.rand(128, 8, 33))
        except (ValueError, Exception):
            error_happen = True
        self.assertTrue(error_happen)


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
