# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import cPickle
import unittest
import timeit
from six.moves import zip, range

import numpy as np

from odin import backend as K
from odin import nnet as N


class RNNTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cudnn_rnn_backend(self):
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
                        K.init.uniform, # function init
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
                            f = N.CudnnRNN(hidden_size=hidden_size, rnn_mode=rnn_mode,
                                           input_mode=input_mode, num_layers=nb_layers,
                                           direction_mode=direction_mode,
                                           initial_states=init_state,
                                           params_split=False)
                            # perform function
                            y = f(X)
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
        f = N.SimpleRecurrent(32, activation=K.relu)
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
        y = f(X3)
        f3 = K.function([X3], y)
        error_happen = False
        try:
            x3 = f3(np.random.rand(128, 8, 33))
        except (ValueError, Exception):
            error_happen = True
        self.assertTrue(error_happen)


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
