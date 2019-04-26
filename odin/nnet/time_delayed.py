# -*- coding: utf-8 -*-
# This module contain the implementation of Time-delayed neural network
# based on variate source, using both Dense (classic) and Convolution (modern)
# for modeling delayed inputs.
#
# The inputs could be 3-D or 4-D tensor
#  [n_samples, n_timestep, n_features]
# or
#  [n_samples, n_timestep, n_features, n_channels]
#
# References
# ----------
# [1] Waibel, A., Hanazawa, T., Hinton, G., Shikano, K., & Lang, K. J. (1989).
# Phoneme recognition using time-delay neural networks. IEEE transactions on
# acoustics, speech, and signal processing, 37(3), 328-339.
# [2] Peddinti, V., Povey, D., & Khudanpur, S. (2015). A time delay neural
# network architecture for efficient modeling of long temporal contexts.
# In INTERSPEECH (pp. 3214-3218).
# [3] Yoon Kim, Yacine Jernite, David Sontag, and Alexander M. Rush. 2016.
# Character-aware neural language models, AAAI'16
from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import backend as K
from odin.autoconfig import randint
from odin.nnet.base import NNOp
from odin.utils import as_tuple, is_string
from odin.backend.role import Weight, Bias, ConvKernel

_allow_time_pool = ('min', 'max', 'avg', 'sum', 'stat', 'none')

# ===========================================================================
# Time-Delayed Dense
# ===========================================================================
class TimeDelayedDense(NNOp):
  """ TimeDelayedDense

  Parameters
  ----------
  n_new_features : {None, integer, list of integer}
      the amount of weighted sum (Dense layer) will be performed for each
      time slice of `n_time_context`

      if None, or empty list, no projection is performed.
      Otherwise, a shallow Dense network or deep Dense network can be
      specified by a list of integer representing number of hidden unit.

  n_time_context : int
      the length of time window context, i.e. kernel size in
      time dimension for convolution operator

  time_pool : {None, 'max', 'avg', 'stat'}
      pooling in time dimension for each time slice (i.e. different from
      `TimeDelayedConv`, when the pooling is performed after computed
      all time sliced)

      for 'stat' pooling, mean and standard deviation is calculated along
      time-dimension, then output the concatenation of the two.

      if None, concatenate all the output of time slice into a vector,
      which return a 3-D tensor with shape
      [n_sample, n_timestep - n_time_context + 1, n_new_features[-1] * n_time_context]

  """

  def __init__(self, n_new_features, n_time_context,
               time_pool='max', backward=False,
               W_init=init_ops.glorot_uniform_initializer(seed=randint()),
               b_init=init_ops.constant_initializer(0),
               activation=K.linear, **kwargs):
    super(TimeDelayedDense, self).__init__(**kwargs)
    if n_new_features is None:
      self.n_new_features = []
    else:
      self.n_new_features = as_tuple(n_new_features, t=int)
    self.n_time_context = int(n_time_context)
    self.n_layers = len(self.n_new_features)
    # ====== initialization ====== #
    self.W_init = W_init
    self.b_init = b_init
    # ====== activation ====== #
    if activation is None:
      activation = K.linear
    if not isinstance(activation, (tuple, list)):
      activation = (activation,)
    activation = [K.linear if i is None else i
                  for i in activation]
    self.activation = as_tuple(activation, N=self.n_layers)
    # ====== time axis manipulation ====== #
    time_pool = str(time_pool).lower()
    assert time_pool in _allow_time_pool, \
    "Only support: %s; but given: '%s'" % (str(_allow_time_pool), str(time_pool))
    self.time_pool = time_pool
    self.backward = bool(backward)

  def _initialize(self):
    time_dim = self.input_shape[1]
    feat_dim = self.input_shape[2]
    layers = [feat_dim] + list(self.n_new_features)
    for i, (l1, l2) in enumerate(zip(layers, layers[1:])):
      # weights
      self.get_variable_nnop(initializer=self.W_init,
          shape=(l1, l2),
          name='W%d' % i, roles=Weight)
      # biases
      if self.b_init is not None:
        self.get_variable_nnop(initializer=self.b_init,
            shape=(l2,), name='b%d' % i, roles=Bias)

  def _apply(self, X):
    if X.shape.ndims >= 4:
      X = tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
    assert X.shape.ndims == 3, \
    "TimeDelayedConv require 3-D or 4-D input, but given: %s" % str(X)
    # [n_sample, n_timestep, n_features]
    time_dim = X.shape[1].value
    feat_dim = X.shape[2].value
    new_time_dim = time_dim - self.n_time_context + 1
    new_feat_dim = (self.n_new_features[-1]
                    if len(self.n_new_features) > 0 else
                    feat_dim)
    # ====== traverse backward along time axis ====== #
    if self.backward:
      X = tf.reverse(X, 1)
    # ====== prepare VALID padding ====== #
    context_indices = tf.range(0, new_time_dim, delta=1,
                               dtype=tf.int32, name="ContextIndices")
    if self.time_pool in ('concat', 'none'):
      init_dim = new_feat_dim * self.n_time_context
    elif self.time_pool in ('max', 'avg', 'min', 'sum'):
      init_dim = new_feat_dim
    elif self.time_pool == 'stat':
      init_dim = new_feat_dim * 2
    initializer = tf.zeros(
        shape=(tf.shape(X)[0], init_dim),
        dtype=self.get('W0').dtype if len(self.n_new_features) > 0 else X.dtype)

    # ====== step function ====== #
    def _time_step(o, ids):
      ctx = X[:, ids:ids + self.n_time_context, :]
      ctx.set_shape((ctx.shape[0], self.n_time_context, ctx.shape[2]))
      ctx = tf.reshape(ctx, shape=(-1, ctx.shape[2]))
      # applying deep dense network
      for l in range(self.n_layers):
        ctx = K.dot(ctx, self.get('W%d' % l))
        if self.b_init is not None:
          ctx = ctx + self.get('b%d' % l)
        ctx = self.activation[l](ctx)
      ctx = tf.reshape(ctx, shape=(-1, self.n_time_context, new_feat_dim))
      # applying pooling
      if self.time_pool in ('concat', 'none'):
        ctx = tf.reshape(ctx,
                         shape=(tf.shape(ctx)[0], self.n_time_context * new_feat_dim))
      elif self.time_pool == 'max':
        ctx = tf.reduce_max(ctx, axis=1)
      elif self.time_pool == 'min':
        ctx = tf.reduce_min(ctx, axis=1)
      elif self.time_pool == 'sum':
        ctx = tf.reduce_sum(ctx, axis=1)
      elif self.time_pool == 'avg':
        ctx = tf.reduce_mean(ctx, axis=1)
      elif self.time_pool == 'stat':
        mean, var = tf.nn.moments(ctx, axes=1)
        ctx = tf.concat([mean, tf.sqrt(var)], -1)
      return ctx
    # ====== apply convolution ====== #
    output = K.scan_tensors(fn=_time_step,
                            sequences=context_indices,
                            mask=None,
                            initializer=initializer, axis=0,
                            parallel_iterations=12)
    output = tf.transpose(output, perm=(1, 0, 2))
    # [n_sample, n_timestep - n_time_context + 1, n_new_features]
    return output

# ===========================================================================
# Time-Delayed Convolution
# ===========================================================================
class TimeDelayedConv(NNOp):
  """ Time-delayed convolutional neural network
  Input is 3-D tensor `[n_sample, n_timestep, n_features]`
  Output is 2-D time-pooled tensor `[n_sample, n_new_features]`

  From the paper, it is suggested to create multiple `TimeDelayedConv`
  with variate number of feature map and length of context windows,
  then concatenate the outputs for `Dense` layers

  For example:
   - feature_maps = [50, 100, 150, 200, 200, 200, 200]
   - kernels = [1, 2, 3, 4, 5, 6, 7]

  Parameters
  ----------
  n_new_features : int
      The number of learn-able convolutional filters this layer has.

  n_time_context : int
      the length of time window context, i.e. kernel size in
      time dimension for convolution operator

  time_pool : {None, 'max', 'avg', 'stat'}
      pooling in time dimension after convolution operator
      for 'stat' pooling, mean and standard deviation is calculated along
      time-dimension, then output the concatenation of the two.
      if None, no pooling is performed, the output is returned in
      shape [n_samples, n_reduced_timestep, n_new_features]

  """

  def __init__(self, n_new_features, n_time_context,
               time_pool='max', backward=False,
               W_init=init_ops.glorot_uniform_initializer(seed=randint()),
               b_init=init_ops.constant_initializer(0),
               activation=K.linear, **kwargs):
    super(TimeDelayedConv, self).__init__(**kwargs)
    self.n_new_features = int(n_new_features)
    self.n_time_context = int(n_time_context)
    self.W_init = W_init
    self.b_init = b_init
    self.activation = activation
    # ====== time axis manipulation ====== #
    time_pool = str(time_pool).lower()
    assert time_pool in _allow_time_pool, \
    "Only support: %s; but given: '%s'" % (str(_allow_time_pool), str(time_pool))
    self.time_pool = time_pool
    self.backward = bool(backward)

  def _initialize(self):
    time_dim = self.input_shape[1]
    feat_dim = self.input_shape[2]
    channel_dim = 1 if len(self.input_shape) == 3 else self.input_shape[-1]
    # weights
    self.get_variable_nnop(initializer=self.W_init,
        shape=(self.n_time_context, feat_dim, channel_dim, self.n_new_features),
        name='W', roles=ConvKernel)
    if self.b_init is not None:
      self.get_variable_nnop(initializer=self.b_init,
          shape=(self.n_new_features,), name='b', roles=Bias)

  def _apply(self, X):
    if X.shape.ndims == 3:
      X = tf.expand_dims(X, axis=-1)
    assert X.shape.ndims == 4, \
    "TimeDelayedConv require 3-D or 4-D input, but given: '%s'" % str(X)
    # [n_sample, n_timestep, n_features, 1]
    # ====== traverse backward along time axis ====== #
    if self.backward:
      X = tf.reverse(X, 1)
    # ====== apply convolution ====== #
    conved = tf.nn.convolution(input=X, filter=self.get('W'),
        padding="VALID",
        strides=[1, 1],
        data_format="NHWC")
    # [n_sample, n_timestep - n_time_context + 1, 1, n_new_features]
    # ====== apply bias ====== #
    if 'b' in self.variable_info:
      conved += K.dimshuffle(self.get('b'), ('x', 'x', 'x', 0))
    # ====== activation ====== #
    conved = self.activation(conved)
    # ====== applying pooling ====== #
    if self.time_pool == 'none':
      pool = tf.squeeze(conved, 2)
      # [n_sample, n_timestep - n_time_context + 1, n_new_features]
    elif self.time_pool == 'stat':
      mean, var = tf.nn.moments(conved, axes=1, keep_dims=True)
      std = tf.sqrt(var)
      pool = tf.concat([mean, std], -1)
      pool = tf.squeeze(pool, axis=[1, 2])
      # [n_sample, n_new_features * 2]
    elif self.time_pool in ('avg', 'max'):
      fn_pool = tf.nn.max_pool if self.time_pool == 'max' else tf.nn.avg_pool
      pool = fn_pool(conved,
                     ksize=[1, tf.shape(conved)[1], 1, 1],
                     strides=[1, 1, 1, 1],
                     padding='VALID',
                     data_format="NHWC")
      pool = tf.squeeze(pool, axis=[1, 2])
      # [n_sample, n_new_features]
    elif self.time_pool in ('sum'):
      pool = tf.reduce_mean(conved, axis=1)
      pool = tf.squeeze(pool, axis=1)
    # ====== return 2D output ====== #
    return pool
