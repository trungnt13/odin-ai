from __future__ import division, absolute_import, print_function

import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass
from itertools import chain

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import init_ops

from odin import backend as K
from odin.autoconfig import randint
from odin.nnet.base import NNOp
from odin.nnet.normalization import BatchNorm
from odin.backend.role import (InitialState, Weight, Bias, Parameter,
                               has_roles, BatchNormShiftParameter,
                               BatchNormScaleParameter,
                               BatchNormPopulationMean,
                               BatchNormPopulationInvStd)
from odin.utils import (as_tuple, is_string, is_number, wprint)

# ===========================================================================
# Helper
# ===========================================================================
def _init_input2hidden(ops, rnn_mode, input_mode, W_init, input_dims, hidden_dims):
  # N represent the number of gates
  if 'rnn' in rnn_mode:
    N = 1
    msg = '(W_hid)'
  elif rnn_mode == 'gru':
    N = 3
    msg = '(W_input_to_updategate, W_input_to_resetgate, W_input_to_hiddenupdate)'
  elif rnn_mode == 'lstm':
    N = 4
    msg = '(W_input_to_inputgate, W_input_to_forgetgate, W_input_to_hidden, W_input_to_outputgate)'
  # ====== check input ====== #
  if input_mode != 'skip':
    ops.get_variable_nnop(initializer=W_init, shape=(input_dims, hidden_dims * N),
                     name='W_in', roles=Weight)
    if input_mode == 'norm':
      ops.get_variable_nnop(initializer=init_ops.constant_initializer(0.), shape=(hidden_dims * N,),
                            name='beta', roles=BatchNormShiftParameter)
      ops.get_variable_nnop(initializer=init_ops.constant_initializer(1.), shape=(hidden_dims * N,),
                            name='gamma', roles=BatchNormScaleParameter)
      ops.get_variable_nnop(initializer=init_ops.constant_initializer(0.), shape=(hidden_dims * N,),
                            name='mean', roles=BatchNormPopulationMean)
      ops.get_variable_nnop(initializer=init_ops.constant_initializer(1.), shape=(hidden_dims * N,),
                            name='inv_std', roles=BatchNormPopulationInvStd)
  # skip input mode
  elif input_dims != hidden_dims and \
  input_dims != hidden_dims * N: # 3 gates + 1 hid_update
    raise Exception('Skip input mode, input trailing_dimension=%d '
                    '(the final dim) must equal to the number of hidden '
                    'units (tied input connection), or %d-th the number '
                    'of hidden units = %d, which include: ' + msg %
                    (input_dims, N, hidden_dims * N))


def _check_cudnn_hidden_init(s0, shape, nnops, name):
  nb_layers, batch_size, hidden_size = shape
  # ====== init s0 ====== #
  if s0 is None and hasattr(nnops, name):
    s0 = getattr(nnops, name)
  elif s0 is not None:
    if hasattr(s0, '__call__') or K.is_variable(s0) or \
    isinstance(s0, np.ndarray):
      _ = (nb_layers, 1, hidden_size) \
          if hasattr(s0, '__call__') or isinstance(s0, np.ndarray) \
          else s0.shape
      s0 = nnops.config.create_params(
          s0, shape=_, name=name, roles=InitialState)
    # ====== check s0 shape ====== #
    init_shape = s0.shape
    if s0.shape.ndims == 2:
      if s0.shape[-1].value != hidden_size:
        raise ValueError('init state has %d dimension, but the hidden_size=%d' %
                        (init_shape[-1], hidden_size))
    elif init_shape[::2] != (nb_layers, hidden_size):
      raise ValueError('Require init states of size: %s, but '
                       'given state of size: %s' % (shape, init_shape))
    # ====== return the right shape ====== #
    setattr(nnops, name, s0)
  return s0


# ===========================================================================
# Dynamic RNN
# ===========================================================================
def _infer_variable_role(variables):
  for v in variables:
    name = v.name.split(':')[0].split("/")[-1]
    if 'kernel' in name:
      K.role.add_roles(v, K.role.Weight)
    elif 'bias' in name:
      K.role.add_roles(v, K.role.Bias)
    elif '_w' in name:
      K.role.add_roles(v, K.role.Weight)
    elif '_v' in name:
      K.role.add_roles(v, K.role.Weight)
    else:
      raise ValueError("Unknown role for variable with name: " + name)

def get_cell_info(cell):
  from tensorflow.contrib.rnn.python.ops import rnn_cell
  found_cell = None
  for key, val in inspect.getmembers(K.rnn_cell) + inspect.getmembers(rnn_cell):
    if inspect.isclass(val) and issubclass(val, tf.contrib.rnn.RNNCell):
      if (isinstance(cell, str) and (key == cell or str(val) == cell)) \
      or cell == val:
        found_cell = val; break
  # ====== get cell init info ====== #
  sign = inspect.signature(found_cell.__init__)
  args = []
  kwargs = {}
  for n, p in sign.parameters.items():
    if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                  inspect.Parameter.VAR_KEYWORD) and\
       n != 'self':
      continue
    args.append(n)
    if p.default != inspect.Parameter.empty:
      kwargs[n] = p.default
  return found_cell, args, kwargs

class RNN(NNOp):
  """
  Parameter
  ---------
  attention: None or `tf.contrib.seq2seq.AttentionMechanism`
      two basic attentions: LuongAttention, BahdanauAttention
      or `tf.contrib.rnn.AttentionCellWrapper`
  """

  def __init__(self, cell_type, kwargs={},
               attention=None, attention_kwargs={},
               num_layers=1, bidirectional=False, dynamic=True,
               return_states=False, name=None):
    super(RNN, self).__init__(name=name)
    self._is_initialized_variables = False
    self.cell_type, _args, _kwargs = get_cell_info(cell_type)
    if self.cell_type is None:
      raise RuntimeError("Cannot find any RNNCell with given description: %s"
          % str(cell_type))
    _kwargs.update(kwargs)
    if len(_kwargs) != len(_args):
      raise RuntimeError("Missing following arguments: %s for the RNNCell "
          "of type: %s" % (list(set(_args) - set(_kwargs.keys())),
                           self.cell_type.__name__))
    self.cell_kwargs = _kwargs
    self.num_layers = int(num_layers)
    self.bidirectional = bidirectional
    self.dynamic = dynamic
    self.return_states = return_states
    # ====== attention ====== #
    if attention is True:
      attention = tf.contrib.rnn.AttentionCellWrapper
    if attention is not None and attention is not False and \
    (not issubclass(attention, tf.contrib.seq2seq.AttentionMechanism) and
     attention is not tf.contrib.rnn.AttentionCellWrapper and
     attention is not K.rnn_cell.AttentionCell):
      raise ValueError("`attention` argument must be `None` or instance "
          "of `tensorflow.contrib.seq2seq.AttentionMechanism`.")
    self.attention = attention
    self.attention_kwargs = dict(attention_kwargs)
    # tf.contrib.seq2seq.embedding_attention_seq2seq

  # ==================== Helper ==================== #
  def __cell_creator(self):
    if self.num_layers > 1:
      c = tf.contrib.rnn.MultiRNNCell([self.cell_type(**self.cell_kwargs)
                                       for _ in range(self.num_layers)])
    else:
      c = self.cell_type(**self.cell_kwargs)
    return c

  def __attention_creator(self, cell, X, memory):
    kwargs = self.attention_kwargs
    if self.attention is tf.contrib.rnn.AttentionCellWrapper or \
    self.attention is K.rnn_cell.AttentionCell:
      # attn_length: the size of an attention window
      # attn_size: the size of an attention vector. Equal to cell.output_size by default.
      # attn_vec_size: the number of convolutional features calculated
      #     on attention state and a size of the hidden layer built from
      #     base cell state. Equal attn_size to by default.
      attn_length = kwargs.get('attn_length', cell.output_size)
      attn_size = kwargs.get('attn_size', None)
      attn_vec_size = kwargs.get('attn_vec_size', None)
      cell_with_attention = self.attention(cell,
          attn_length=attn_length, attn_size=attn_size,
          attn_vec_size=attn_vec_size)
    else: # seq2seq attention
      num_units = kwargs.get('num_units', X.shape[-1].value)
      attention_mechanism = self.attention(
          num_units=num_units, memory=memory, **kwargs)
      cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(
          cell=cell, attention_mechanism=attention_mechanism,
          attention_layer_size=attention_mechanism._num_units)
    return cell_with_attention

  # ==================== Cells ==================== #
  @property
  def cell(self):
    # ====== first time create Cell ====== #
    if not hasattr(self, '_cell'):
      self._cell = self.__cell_creator()
    # ====== return Cell ====== #
    self._cell._reuse = False if len(self.variables) == 0 else True
    return self._cell

  @property
  def cell_bw(self):
    if not self.bidirectional:
      raise RuntimeError("`cell_bw` is not supported with `bidirectional=False`")
    # ====== first time create Cell ====== #
    if not hasattr(self, '_cell_bw'):
      self._cell_bw = self.__cell_creator()
    # ====== return Cell ====== #
    self._cell_bw._reuse = False if len(self.variables) == 0 else True
    return self._cell_bw

  def _apply(self, X, state=None, memory=None):
    # time_major: The shape format of the `inputs` and `outputs` Tensors.
    #   If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
    #   If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
    # ====== create attention if necessary ====== #
    cell = self.cell
    if self.bidirectional:
      cell_bw = self.cell_bw
    # create attention cell
    if self.attention:
      if not hasattr(self, "_cell_with_attention"):
        self._cell_with_attention = self.__attention_creator(
            cell, X=X, memory=memory)
        cell = self._cell_with_attention
      # bidirectional attention
      if self.bidirectional:
        if not hasattr(self, "_cell_with_attention_bw"):
          self._cell_with_attention_bw = self.__attention_creator(
              cell_bw, X=X, memory=memory)
        cell_bw = self._cell_with_attention_bw
    # ====== calling rnn_warpper ====== #
    ## Bidirectional
    if self.bidirectional:
      rnn_func = rnn.bidirectional_dynamic_rnn if self.dynamic \
          else rnn.static_bidirectional_rnn
      state_fw, state_bw = None, None
      if isinstance(state, (tuple, list)):
        state_fw = state[0]
        if len(state) > 1:
          state_bw = state[1]
      else:
        state_fw = state
      outputs = rnn_func(cell_fw=cell, cell_bw=cell_bw, inputs=X,
                         initial_state_fw=state_fw,
                         initial_state_bw=state_bw,
                         dtype=X.dtype.base_dtype)
    ## Unidirectional
    else:
      rnn_func = rnn.dynamic_rnn if self.dynamic else rnn.static_rnn
      outputs = rnn_func(cell, inputs=X, initial_state=state,
                         dtype=X.dtype.base_dtype)
    # ====== initialize cell ====== #
    if not self._is_initialized_variables:
      # initialize only once, everytime you call this, the values of
      # variables changed
      K.eval(tf.variables_initializer(self.variables))
      self._is_initialized_variables = True
      _infer_variable_role(self.variables)
    # ====== return ====== #
    if self.bidirectional: # concat outputs
      outputs = (tf.concat(outputs[0], axis=-1), outputs[1])
    if not self.return_states:
      return outputs[0]
    return outputs

  def _transpose(self):
    raise NotImplementedError


class LSTM(RNN):

  def __init__(self, num_units, use_peepholes=False, cell_clip=None,
               num_proj=None, proj_clip=None, forget_bias=1.0,
               activation=None, num_layers=1, bidirectional=False,
               attention=None, attention_kwargs={},
               dynamic=True, return_states=False, name=None):
    super(LSTM, self).__init__(cell_type=tf.contrib.rnn.LSTMCell,
        kwargs={'num_units': num_units, 'use_peepholes': use_peepholes,
                'cell_clip': cell_clip, 'num_proj': num_proj,
                'proj_clip': proj_clip, 'forget_bias': forget_bias,
                'activation': activation},
        num_layers=num_layers, bidirectional=bidirectional,
        attention=attention, attention_kwargs=attention_kwargs,
        dynamic=dynamic, return_states=return_states,
        name=name)


class GRU(RNN):

  def __init__(self, num_units, activation=None,
               attention=None, attention_kwargs={},
               num_layers=1, bidirectional=False, dynamic=True,
               return_states=False, name=None):
    super(GRU, self).__init__(cell_type=tf.contrib.rnn.GRUCell,
        kwargs={'num_units': num_units, 'activation': activation},
        num_layers=num_layers, bidirectional=bidirectional,
        attention=attention, attention_kwargs=attention_kwargs,
        dynamic=dynamic, return_states=return_states,
        name=name)


# ===========================================================================
# DNN
# ===========================================================================
class CudnnRNN(NNOp):

  """CuDNN v5 RNN implementation.

  Parameters
  ----------
  num_units : int
      the number of units within the RNN model.
  W_init:
      initial description for weights
  b_init:
      initial description for bias
  rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
      See cudnn documentation for ``cudnnRNNMode_t``.
  num_layers : int
      the number of layers for the RNN model.
  is_bidirectional : bool (default: False)
      unidirectional: The network operates recurrently from the
                      first input to the last.
      bidirectional: The network operates from first to last then from last
                     to first and concatenates the results at each layer.
  is_training : {None, boolean}
      if None, is_training is conditioned on `odin.backend.is_training()`
  return_states: boolean (defaults: False)
      if True, this Ops returns the [output, hidden_staes, cell_states (lstm)]
      otherwise only return the output
  dropout: float (0.0-1.0)
      whether to enable dropout. With it is 0, dropout is disabled.

  Arguments
  ---------
  X : Tensor
    (batch_size, time_dim, feature_dim)
  h0 : {None, number, Tensor} (default: None)
    if None, all-zeros initial states are used
    (num_layers * num_direction, batch_size, num_units)
  c0 : {None, number, Tensor} (default: None)
    if None, all-zeros initial cell memory are used
    (num_layers * num_direction, batch_size, num_units)
  training : bool (default: None)
    if None, use O.D.I.N training flag,
    otherwise, use given value

  Returns
  -------
  [output, hidden_states, cell_states] for lstm
  [output, hidden_states] for gru and rnn

  output_shape: (batch_size, timesteps,  num_units)
  hidden_shape: (num_layers, batch_size, num_units)
  cell_shape: (num_layers, batch_size,   num_units)

  Note
  ----
  `__call__(training=True)` if you want to get gradient from this NNOp
  """

  def __init__(self, num_units,
          W_init=init_ops.glorot_uniform_initializer(seed=randint()),
          b_init=init_ops.constant_initializer(0.),
          rnn_mode='lstm', num_layers=1,
          skip_input=False, is_bidirectional=False,
          return_states=False, dropout=0., **kwargs):
    super(CudnnRNN, self).__init__(**kwargs)
    # ====== defaults recurrent control ====== #
    self.num_units = int(num_units)
    self.num_layers = int(num_layers)
    self.rnn_mode = str(rnn_mode)
    self.skip_input = bool(skip_input)
    self.is_bidirectional = bool(is_bidirectional)
    self.return_states = bool(return_states)
    self.dropout = dropout

    self.W_init = W_init
    self.b_init = b_init
    if skip_input:
      wprint("`skip_input` is not supported in Tensorflow.")

  # ==================== abstract methods ==================== #
  @property
  def state_shape(self):
    return [self.num_layers * 2 if self.is_bidirectional else 1, None, self.num_units]

  def _transpose(self):
    # flip the input and hidden
    raise NotImplementedError

  def _initialize(self):
    input_shape = self.input_shape_map['X']
    weights, biases = K.init_rnn(
        input_dim=int(input_shape[-1]), hidden_dim=int(self.num_units),
        W_init=self.W_init, b_init=self.b_init,
        num_layers=self.num_layers, num_gates=self.rnn_mode,
        skip_input=self.skip_input,
        is_bidirectional=self.is_bidirectional,
        cudnn_vector=False)
    self._weights_name = [w.name for w in weights]
    self._biases_name = [b.name for b in biases]
    for i in weights + biases:
      self.get_variable_nnop(name=i.name.split('/')[-1].split(':')[0],
                             shape=i.shape.as_list(), initializer=i)

  def _apply(self, X, h0=None, c0=None, training=None):
    if not hasattr(self, '_opaque_params'):
      weights = [K.get_all_variables(full_name=w)[0] for w in self._weights_name]
      biases = [K.get_all_variables(full_name=b)[0] for b in self._biases_name]
      self._opaque_params = K.params_to_cudnn(weights, biases)
    params = self._opaque_params
    outputs = K.cudnn_rnn(X=X, h0=h0, c0=c0,
                          num_units=self.num_units, rnn_mode=self.rnn_mode,
                          num_layers=self.num_layers, parameters=params,
                          skip_input=self.skip_input,
                          is_bidirectional=self.is_bidirectional,
                          dropout=self.dropout,
                          is_training=training)
    if not self.return_states:
      return outputs[0]
    return outputs
