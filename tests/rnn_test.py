# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
import timeit
from six.moves import zip, range, cPickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.ops import init_ops

from odin.utils import uuid, run_script
from odin import backend as K, nnet as N

np.random.seed(1234)


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

  def test_cudnn_rnn(self):
    if get_ngpu() == 0:
      return
    print()
    batch_size = 2
    time_steps = 5
    input_dim = 12
    hidden_dim = 8
    X = K.variable(value=np.random.rand(batch_size, time_steps, input_dim),
                   dtype='float32', name='X')
    for rnn_mode in ('lstm', 'rnn_relu', 'gru'):
      for num_layers in [1, 2]:
        for W_init in [init_ops.glorot_uniform_initializer(seed=1234),
                       init_ops.random_normal_initializer(seed=1234)]:
          for b_init in [0, 1]:
            for bidirectional in (True, False):
              for skip_input in (False,):
                print('RNNmode:%s' % rnn_mode,
                      "#Layers:%d" % num_layers,
                      'Bidirectional:%s' % bidirectional,
                      'SkipInput:%s' % skip_input)
                weights, biases = K.init_rnn(
                    input_dim=input_dim, hidden_dim=hidden_dim, num_gates=rnn_mode,
                    num_layers=num_layers, W_init=W_init, b_init=b_init,
                    skip_input=skip_input, cudnn_vector=False,
                    is_bidirectional=bidirectional, name=None)
                # ====== check number of params ====== #
                params1 = K.params_to_cudnn(weights, biases)
                n = params1.shape[0].value
                nb_params = cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
                    rnn_mode=rnn_mode, num_layers=num_layers,
                    num_units=hidden_dim, input_size=input_dim,
                    input_mode='skip_input' if skip_input else 'linear_input',
                    direction='bidirectional' if bidirectional else 'unidirectional')
                nb_params = K.eval(nb_params)
                assert n == nb_params
                # ====== check cannonical shape match ====== #
                kwargs = {'num_layers': num_layers,
                          'num_units': hidden_dim,
                          'input_mode': 'skip_input' if skip_input else 'linear_input',
                          'direction': 'bidirectional' if bidirectional else 'unidirectional'}
                if rnn_mode == 'lstm':
                  rnn = cudnn_rnn.CudnnLSTM(**kwargs)
                elif rnn_mode == 'gru':
                  rnn = cudnn_rnn.CudnnGRU(**kwargs)
                if rnn_mode == 'rnn_relu':
                  rnn = cudnn_rnn.CudnnRNNRelu(**kwargs)
                if rnn_mode == 'rnn_tanh':
                  rnn = cudnn_rnn.CudnnRNNTanh(**kwargs)
                rnn.build(input_shape=(None, None, input_dim))
                assert len(weights) == len(rnn.canonical_weight_shapes)
                assert len(biases) == len(rnn.canonical_bias_shapes)
                for w, s in zip(weights, rnn.canonical_weight_shapes):
                  assert tuple(w.shape.as_list()) == s
                # ====== check params conversion ====== #
                K.initialize_all_variables()
                params2 = cudnn_rnn_ops.cudnn_rnn_canonical_to_opaque_params(
                    rnn_mode=rnn_mode, num_layers=num_layers,
                    num_units=hidden_dim, input_size=input_dim,
                    input_mode='skip_input' if skip_input else 'linear_input',
                    direction='bidirectional' if bidirectional else 'unidirectional',
                    weights=weights, biases=biases)
                assert np.all(K.eval(params1) == K.eval(params2))
                # ====== odin cudnn implementation ====== #
                name = 'TEST' + uuid(length=25)
                outputs = K.cudnn_rnn(X=X, num_units=hidden_dim, rnn_mode=rnn_mode,
                            num_layers=num_layers, parameters=None,
                            skip_input=skip_input, is_bidirectional=bidirectional,
                            dropout=0.1, name=name)
                K.initialize_all_variables()
                s0 = K.eval(outputs[0]).sum()
                s1 = K.eval(outputs[1]).sum()
                all_variables = K.get_all_variables(scope=name)
                new_weights = [i for i in all_variables
                               if K.role.has_roles(i, roles=K.role.Weight)]
                new_biases = [i for i in all_variables
                              if K.role.has_roles(i, roles=K.role.Bias)]
                new_weights, new_biases = K.sort_cudnn_params(
                    new_weights, new_biases, rnn_mode=rnn_mode)
                assert len(weights) == len(weights)
                assert len(biases) == len(biases)
                for i, j in zip(weights + biases, new_weights + new_biases):
                  assert i.name.split('/')[-1] == j.name.split('/')[-1]
                # ====== CudnnRNN wrapper ====== #
                rnn = N.CudnnRNN(num_units=hidden_dim,
                    W_init=new_weights, b_init=new_biases,
                    rnn_mode=rnn_mode, num_layers=num_layers,
                    skip_input=skip_input, is_bidirectional=bidirectional,
                    return_states=True, dropout=0.)
                outputs = rnn(X)
                K.initialize_all_variables()
                y0 = K.eval(outputs[0]).sum()
                y1 = K.eval(outputs[1]).sum()
                assert y0 == s0
                assert y1 == s1

  def test_save_cudnn_rnn(self):
    np.random.seed(1234)
    X = K.variable(np.random.rand(25, 12, 8))
    num_layers = 2
    num_gates = 'lstm'
    skip_input = False
    is_bidirectional = False
    path = '/tmp/rnn'
    weights, biases = K.init_rnn(input_dim=8, hidden_dim=18,
                                 b_init=init_ops.random_normal_initializer(),
                                 num_layers=num_layers, num_gates=num_gates,
                                 skip_input=skip_input,
                                 is_bidirectional=is_bidirectional)
    rnn = N.CudnnRNN(num_units=18,
                     W_init=weights, b_init=biases,
                     rnn_mode=num_gates, num_layers=num_layers,
                     skip_input=skip_input, is_bidirectional=is_bidirectional,
                     return_states=False,
                     dropout=0., name="CudnnRNNTest")
    y = rnn(X)
    K.initialize_all_variables()
    y = K.eval(y)
    N.serialize(nnops=rnn, path=path, binary_output=True, override=True)
    test_script = r"""
    from __future__ import print_function, division, absolute_import
    import os
    os.environ['ODIN'] = 'gpu,float32,seed=1234'
    import pickle
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.ops import init_ops
    from odin import backend as K, nnet as N
    np.random.seed(1234)
    X = K.variable(np.random.rand(25, 12, 8))
    rnn = N.deserialize("%s", force_restore_vars=True)
    y = rnn(X)
    K.initialize_all_variables()
    y = K.eval(y)
    print(len(rnn.variables),
          sum(np.sum(K.eval(i)) for i in rnn.variables
                    if K.role.has_roles(i, K.role.Weight)),
          sum(np.sum(K.eval(i)) for i in rnn.variables
              if K.role.has_roles(i, K.role.Bias)),
          y.sum(),
          (y**2).sum())
    """ % path
    outputs = run_script(test_script)[1]
    num_variables, w, b, s1, s2 = outputs.split(' ')
    assert int(num_variables) == len(rnn.variables)
    assert np.allclose(float(w),
                       sum(np.sum(K.eval(i)) for i in rnn.variables
                           if K.role.has_roles(i, K.role.Weight)))
    assert np.allclose(float(b),
                       sum(np.sum(K.eval(i)) for i in rnn.variables
                           if K.role.has_roles(i, K.role.Bias)))
    assert np.allclose(float(s1), y.sum())
    assert np.allclose(float(s2), (y**2).sum())

  def test_simple_rnn(self):
    np.random.seed(1234)
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
