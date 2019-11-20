from __future__ import absolute_import, division, print_function

import warnings
from abc import ABCMeta, abstractmethod
from enum import IntFlag
from enum import auto as enum_auto

import numpy as np
import tensorflow as tf
import torch
from six import add_metaclass, string_types
from tensorflow import nest
from tensorflow.python import keras

from odin import backend as bk
from odin.networks.attention_mechanism import *
from odin.networks.attention_mechanism import _GROUPS
from odin.utils import as_tuple


@add_metaclass(ABCMeta)
class Attention(keras.Model):
  r"""

  Args:
    input_dim : Integer. Number of input features.
    causal : Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past, suggested in
      (Mishra N., et al., 2018).
    residual : Boolean. If `True`, add residual connection between input `query`
      and the attended output.

    dropout : Float. Dropout probability of the attention scores.
    temporal_dropout : Boolean. If `True`, using the same dropout mask along
      temporal axis (i.e. the 1-st dimension)

    num_heads : Integer. Number of attention heads.
    heads_depth : Integer. The feed-forward network depth of each head
    heads_bias : Boolean or List of Boolean. Specify `use_bias` for each layer
      within a head.
    heads_regularization : Float. Using L2-normalize to encourage diversity among
      attention heads (Kim Y., et al. 2017). If `0.`, turn off normalization
    heads_activation : Activation for each layer within an attention head.

    scale_initializer : String or Number.
      'vaswani' - Suggested by (Vaswani et al. 2017) for large values
      of `input_dim`, the dot products grow large in magnitude, pushing the
      softmax function into regions where it has extremely small gradients.
    scale_tied : Boolean. If `True`, use single scale value for all input
      dimensions, otherwise, create separate scale parameters for each
      dimension.
    scale_trainable : Boolean. If `True`, all scale parameters are trainable.

  Call Arguments:
    query: Query (or target sequence) `Tensor` of shape `[batch_size, Tq, dim]`.
    value: Value (or source sequence) `Tensor` of shape `[batch_size, Tv, dim]`.
    key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
      given, will use `value` for both `key` and `value`, which is the
      most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.

  Output shape:
    Attention outputs of shape `[batch_size, Tq, dim]`.
    Attention scores of shape `[batch_size, Tq, Tv]` or `[batch_size, Tq]` in
    case of location-base attention.
  """

  def __init__(self,
               input_dim,
               causal=False,
               residual=True,
               dropout=0,
               temporal_dropout=False,
               num_heads=0,
               heads_depth=1,
               heads_bias=True,
               heads_regularization=0.,
               heads_activation='linear',
               scale_initializer='vaswani',
               scale_tied=True,
               scale_trainable=False,
               n_mcmc=1,
               temperature=0.5,
               temperature_trainable=False,
               name=None):
    super(Attention, self).__init__(name=name)
    self.input_dim = input_dim
    self.causal = bool(causal)
    self.residual = bool(residual)
    # ====== for dropout ====== #
    self.dropout = dropout
    self.temporal_dropout = bool(temporal_dropout)
    # ====== for hard attention ====== #
    self.n_mcmc = int(n_mcmc)
    self.temperature_trainable = temperature_trainable
    self.temperature = bk.variable(initial_value=temperature,
                                   trainable=temperature_trainable,
                                   dtype='float32',
                                   framework=self)
    # ====== multi-head ====== #
    self.num_heads = int(num_heads)
    self.heads_regularization = heads_regularization
    self.heads_depth = int(heads_depth)
    self.heads_bias = as_tuple(heads_bias, N=self.heads_depth, t=bool)
    self.heads_activation = as_tuple(heads_activation, N=self.heads_depth)
    # ====== initialize scale ====== #
    self.scale_initializer = scale_initializer
    self.scale_tied = scale_tied
    self.scale_trainable = scale_trainable
    if not scale_tied and input_dim is None:
      raise ValueError("If scale_tied=False, the input_dim must be provided.")
    scale = 1
    if scale_initializer is not None:
      if isinstance(scale_initializer, string_types):
        scale_initializer = scale_initializer.lower().strip()
        if scale_initializer == 'vaswani':
          assert input_dim is not None, \
            "input_dim must be provided if scale_initializer='vaswani'"
          scale_initializer = 1 / input_dim**0.5
      scale = bk.parse_initializer(scale_initializer, self)
      if scale_tied:
        scale = bk.variable(initial_value=scale(()),
                            trainable=scale_trainable,
                            framework=self)
      else:
        scale = bk.variable(initial_value=scale(nest.flatten(input_dim)),
                            trainable=scale_trainable,
                            framework=self)
    self.scale = scale
    # ====== init parameters and layers ====== #
    with bk.framework_(self):
      self.query_heads = create_attention_heads(
          input_dim,
          num_heads=self.num_heads,
          depth=self.heads_depth,
          use_bias=self.heads_bias,
          activation=self.heads_activation)
      self.key_heads = create_attention_heads(input_dim,
                                              num_heads=self.num_heads,
                                              depth=self.heads_depth,
                                              use_bias=self.heads_bias,
                                              activation=self.heads_activation)
      self.value_heads = create_attention_heads(
          input_dim,
          num_heads=self.num_heads,
          depth=self.heads_depth,
          use_bias=self.heads_bias,
          activation=self.heads_activation)
      # init default object
      self._mechanism = Inter | PosGlobal | AlignSoft | ScoreLocation
      # query projection for location-based scoring method
      self.location_proj = None
      # target projection use in Local Predictive attention
      self.target_proj = None
      #
      self._local_init()
      self.set_methods()

  def set_methods(self, scoring=None, alignment=None):
    if alignment is not None:
      assert alignment in _GROUPS[2], \
        "alignment must be one of the following: %s" % str(_GROUPS[2])
      self._mechanism |= alignment
    if scoring is not None:
      assert scoring in _GROUPS[3], \
        "scoring must be one of the following: %s" % str(_GROUPS[3])
      self._mechanism |= scoring
    self._mechanism.validate()
    return self

  def _local_init(self):
    pass

  @property
  def mechanism(self):
    return self._mechanism

  def call(self, inputs, mask=None, training=None, return_attention=False):
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    if len(inputs) == 2:
      inputs = (inputs[0], None, inputs[1])
    with bk.framework_(self):
      q, k, v, q_mask, v_mask = self._mechanism.prepare(*inputs, mask=mask)
      assert self.input_dim == q.shape[-1], \
        "Given input_dim=%d but query shape=%s" % (self.input_dim, q.shape)
      q, k, v = [
          x if x is None else f(x) for x, f in zip(
              [q, k, v], [self.query_heads, self.key_heads, self.value_heads])
      ]
      scores = self._mechanism.score(query=q,
                                     key=k,
                                     scale=self.scale,
                                     window_width=None,
                                     q_proj=self.location_proj,
                                     target_proj=self.target_proj)
      out, att = self._mechanism.align(scores=scores,
                                       value=v,
                                       query=q,
                                       q_mask=q_mask,
                                       v_mask=v_mask,
                                       causal=self.causal,
                                       residual=self.residual,
                                       dropout=self.dropout,
                                       temporal_dropout=self.temporal_dropout,
                                       n_mcmc=self.n_mcmc,
                                       temperature=self.temperature,
                                       training=training)
      ### attention heads regularization
      if self.heads_regularization > 0:
        P = self._mechanism.normalize(scores)
        self.add_loss(self.heads_regularization * P)
      ### return attention distribution or not
      if return_attention:
        return out, att
      return out


# ===========================================================================
# Attention classes
# ===========================================================================
class SelfAttention(Attention):
  r""" Self(Intra)-sequence attention using global positioning and
  locative scoring method by default

  Example:

  ```python
  att = net.SelfAttention(dim, **kw)
  y, a = att(query, mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignHard)
  y, a = att(query, mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignRelax)
  y, a = att(query, mask=(q_mask, v_mask), return_attention=True)
  ```
  """

  def _local_init(self):
    self._mechanism |= Intra
    # self attention only support global positioning
    self._mechanism |= PosGlobal
    self._mechanism |= ScoreLocation
    # use for location-base attention (when key and value are not provided)
    self.location_proj = bk.nn.Dense(1, activation='linear', use_bias=True)


class GlobalAttention(Attention):
  r""" Inter-sequence global attention using dot-product scoring method
  by default

  Example:

  ```python
  att = net.GlobalAttention(dim, **kw)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignHard)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignRelax)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  ```
  """

  def _local_init(self):
    self._mechanism |= Inter
    self._mechanism |= PosGlobal
    self._mechanism |= ScoreDotProd


class LocalPredictiveAttention(Attention):
  r""" Inter-sequence local predictive attention using dot-product scoring
  method by default

  Example:

  ```python
  att = net.LocalPredictiveAttention(dim, **kw)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignHard)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  att.set_methods(alignment=net.attention_mechanism.AlignRelax)
  y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
  ```
  """

  def _local_init(self):
    self._mechanism |= Inter
    self._mechanism |= PosLocalP
    self._mechanism |= ScoreDotProd
    # use for location-base attention (when key and value are not provided)
    self.target_proj = bk.nn.Dense(1, activation='linear', use_bias=True)
