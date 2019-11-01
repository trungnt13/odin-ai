# A comprehensive implementation of Attention Mechanism for Neural Networks
# Supporting:
#  * Multi-head attention
#  * Self-attention mechanism
#  * Using more odin.backend function to make it easier transfer between
#     tensorflow and pytorch
#
# References:
#   Bahdanau, D., et al., 2014. Neural Machine Translation by Jointly Learning
#     to Align and Translate. arXiv:1409.0473 [cs, stat].
#   Graves, A., et al., 2014. Neural Turing Machines.
#     arXiv:1410.5401 [cs].
#   Xu, K., et al., 2015. Show, Attend and Tell: Neural Image Caption Generation
#     with Visual Attention. arXiv:1502.03044 [cs].
#   Luong, M.T., et al., 2015. Effective Approaches to Attention-based Neural
#     Machine Translation. arXiv:1508.04025 [cs].
#   Cheng, J., et al., 2016. Long Short-Term Memory-Networks for Machine Reading.
#     arXiv:1601.06733 [cs].
#   Kim, Y., et al., 2017. Structured Attention Networks.
#     arXiv:1702.00887 [cs].
#   Vaswani, A., et al., 2017. Attention Is All You Need.
#     arXiv:1706.03762 [cs].
#   Mishra, N., et al., 2018. A Simple Neural Attentive Meta-Learner.
#     arXiv:1707.03141 [cs, stat].
#   Park, K., 2019. github.com/Kyubyong/transformer.
#   Alexander H. Liu, 2019. github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch.
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import torch
from six import add_metaclass, string_types
from tensorflow import nest
from tensorflow.python import keras

from odin import backend as bk
from odin.utils import as_tuple


# ===========================================================================
# Helper function
# ===========================================================================
def _split_and_concat(x, num_heads):
  return bk.concatenate(bk.split(x, num_heads, axis=-1), axis=0)


def _create_heads(input_dim, num_heads, heads_bias, heads_activation):
  return bk.nn.Sequential([
      bk.nn.Dense(input_dim * num_heads,
                  use_bias=use_bias,
                  activation=activation)
      for use_bias, activation in zip(heads_bias, heads_activation)
  ])


# ===========================================================================
# Base and helper classes
# ===========================================================================
class PositionalEncoder(keras.layers.Layer):
  r""" Positional encoding follow the approach in (Vaswani, 2017)
  For even dimension in the embedding:
    `PE(pos,2i) = sin(pos/10000^(2i/dmodel))`
  and for odd position:
    `PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))`

  """

  def __init__(self,
               output_dim,
               max_len=10000,
               trainable=False,
               mask_zero=False):
    super().__init__()
    self.output_dim = output_dim
    self.mask_zero = bool(mask_zero)
    self.trainable = bool(trainable)
    self.supports_masking = mask_zero
    self.max_len = max_len

    # Applying the cosine to even columns and sin to odds.
    # if zero-masked, dont use the 0 position
    # (i - i % 2) create a sequence of (0,0,1,1,2,2,...) which is needed
    # for two running sequence of sin and cos in odd and even position
    position_encoding = np.array([[
        pos / np.power(10000, (i - i % 2) / output_dim)
        for i in range(output_dim)
    ] if pos != 0 or not mask_zero else [0.] * output_dim
                                  for pos in range(max_len)])
    # [max_len, output_dim]
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # dim 2i
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # dim 2i+1
    if not trainable:
      self.position_encoding = bk.array(position_encoding,
                                        dtype='float32',
                                        framework=self)
    else:
      self.position_encoding = bk.variable(initial_value=position_encoding,
                                           dtype='float32',
                                           trainable=True,
                                           framework=self)

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return bk.not_equal(inputs, 0)

  def call(self, sequence, mask=None, training=None):
    with bk.framework_(self):
      # [batch_size, time_dim]
      positions = bk.tile(bk.expand_dims(bk.arange(sequence.shape[1]), 0),
                          [sequence.shape[0], 1])
      dtype = bk.dtype_universal(positions.dtype)
      if dtype not in ('int32', 'int64'):
        positions = bk.cast(positions, dtype='int32')
      pe = bk.embedding(indices=positions, weight=self.position_encoding)
      return pe

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_dim': self.output_dim,
        'trainable': self.trainable,
        'mask_zero': self.mask_zero,
        'max_len': self.max_len
    })
    return config


@add_metaclass(ABCMeta)
class BaseAttention(keras.Model):

  @abstractmethod
  def calculate_scores(self, query, key=None):
    raise NotImplementedError

  @abstractmethod
  def call(self, query, value=None, key=None, mask=None, training=None):
    raise NotImplementedError


# ===========================================================================
# Attention classes
# ===========================================================================
class SoftAttention(BaseAttention):
  r""" Original implementation from Tensorflow:
  `tensorflow/python/keras/layers/dense_attention.py`
  Copyright 2019 The TensorFlow Authors. All Rights Reserved.

  The meaning of `query`, `value` and `key` depend on the application. In the
  case of text similarity, for example, `query` is the sequence embeddings of
  the first piece of text and `value` is the sequence embeddings of the second
  piece of text. Hence, the attention determines alignment between `query` and
  `value`, `key` is usually the same tensor as value.
  A mapping from `query` to `key` will be learned during the attention.


  Args:
    input_dim : Integer. Number of input features.
    causal : Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past, suggested in
      (Mishra N., et al., 2018).
    residual : Boolean. If `True`, add residual connection between input `query`
      and the attended output.
    return_attention : Boolean. Set to `True` for returning the attention
      scores.

    dropout : Float. Dropout probability of the attention scores.
    temporal_dropout : Boolean. If `True`, using the same dropout mask along
      temporal axis (i.e. the 1-st dimension)

    num_heads : Integer. Number of attention heads.
    heads_depth : Integer. The feed-forward network depth of each head
    heads_bias : Boolean or List of Boolean. Specify `use_bias` for each layer
      within a head.
    heads_norm : Float. Using L2-normalize to encourage diversity among
      attention heads (Kim Y., et al. 2017). If `0.`, turn off normalization
    heads_activation : Activation for each layer within an attention head.
    heads_output_mode : {'cat', 'mean', ''}
      'cat' - concatenate multiple-heads into single output
      'mean' - calculate the mean of all atttention heads.
      '' - do nothing return the output with all heads

    scale_initializer : String or Number.
      'vaswani' - Suggested by (Vaswani, et al. 2017) for large values
      of `input_dim`, the dot products grow large in magnitude, pushing the
      softmax function into regions where it has extremely small gradients.
    scale_tied : Boolean. If `True`, use single scale value for all input
      dimensions, otherwise, create separate scale parameters for each
      dimension.
    scale_trainable : Boolean. If `True`, all scale parameters are trainable.

    attention_type : {'mul', 'add'}.
      'mul' - Scale-dot-product or multiplicative attention style a.k.a.
      Luong or Vaswani-style attention.
      'add' - addictive attention style a.k.a. Bahdanau-style attention.
      'loc' - location-base attention, the attention scores only based on
        the `query` itself (Luong M.T., et al., 2015).
      The model will automatically switch to location-base attention if
      `key` and `value` are not provided

  Call Arguments:
    query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
    value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
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
               input_dim=None,
               causal=False,
               residual=True,
               return_attention=False,
               dropout=0,
               temporal_dropout=False,
               num_heads=0,
               heads_depth=1,
               heads_bias=True,
               heads_norm=0.,
               heads_activation='linear',
               heads_output_mode='cat',
               scale_initializer='vaswani',
               scale_tied=True,
               scale_trainable=False,
               attention_type='mul',
               name=None):
    super().__init__(name=name)
    self.input_dim = input_dim
    self.causal = causal
    self.residual = residual
    self.return_attention = bool(return_attention)
    self.supports_masking = True
    # ====== for dropout ====== #
    self.dropout = dropout
    self.temporal_dropout = bool(temporal_dropout)
    # ====== multi-head ====== #
    self.num_heads = int(num_heads)
    self.heads_output_mode = str(heads_output_mode).lower().strip()
    self.heads_norm = heads_norm
    self.heads_depth = int(heads_depth)
    self.heads_bias = as_tuple(heads_bias, N=self.heads_depth, t=bool)
    self.heads_activation = as_tuple(heads_activation, N=self.heads_depth)
    # create a deep feedforward network for the heads:
    with bk.framework_(self):
      if self.num_heads > 0:
        if input_dim is None:
          raise ValueError("If num_heads > 0, the input_dim must be provided.")
        self.query_heads = _create_heads(input_dim, num_heads, self.heads_bias,
                                         self.heads_activation)
        self.key_heads = _create_heads(input_dim, self.num_heads,
                                       self.heads_bias, self.heads_activation)
        self.value_heads = _create_heads(input_dim, num_heads, self.heads_bias,
                                         self.heads_activation)
      else:  # No heads, just identity function
        self.query_heads = bk.nn.Identity()
        self.key_heads = bk.nn.Identity()
        self.value_heads = bk.nn.Identity()
      # use for location-base attention (when key and value are not provided)
      self.self_projection = bk.nn.Dense(1, activation='linear', use_bias=True)
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
    self.attention_scale = scale
    self.attention_type = str(attention_type).strip().lower()

  def calculate_scores(self, query, key=None):
    """Calculates attention scores (a.k.a logits values).

    Args:
      query: Query tensor of shape `[batch_size * num_heads, Tq, dim]`.
      key: Key tensor of shape `[batch_size * num_heads, Tv, dim]`.

    Returns:
      Tensor of shape `[batch_size * num_heads, Tq, Tv]`.
    """
    if self.attention_type == 'loc' or key is None:
      # [batch_size * num_heads, Tq, dim]
      if not self.scale_tied:
        raise RuntimeError(
            "Self-attention mode, when only query provided doesn't support "
            "untied scale")
      return self.attention_scale * self.self_projection(query)
    elif self.attention_type == 'mul':
      # this is a trick to make attention_scale broadcastable when
      # scale_tied=False
      return bk.matmul(self.attention_scale * query, bk.swapaxes(key, 1, 2))
    elif self.attention_type == 'add':
      # [batch_size * num_heads, Tq, 1, dim]
      q = bk.expand_dims(query, axis=2)
      # [batch_size * num_heads, 1, Tv, dim]
      k = bk.expand_dims(key, axis=1)
      # [batch_size * num_heads, Tq, Tv]
      return bk.reduce_sum(self.attention_scale * bk.tanh(q + k), axis=-1)
    else:
      raise NotImplementedError("No support for attention_type='%s'" %
                                self.attention_type)

  def calculate_scores_norm(self, scores):
    """ With the attention scores is A `[batch_size * num_heads, Tq, Tv]`
    `P = ||A^T*A - I||_2^2`
    """
    # it is easier to assume there is always 1-head at least
    num_heads = max(self.num_heads, 1)

    with bk.framework_(self):
      Tq, Tv = scores.shape[1:]
      # [batch_size, num_heads, Tq * Tv]
      scoresT = bk.reshape(scores, shape=(-1, num_heads, Tq * Tv))
      # [batch_size, Tq * Tv, num_heads]
      scores = bk.swapaxes(scoresT, 1, 2)
      # [batch_size, num_heads, num_heads]
      A = bk.matmul(scoresT, scores)

      I = bk.eye(num_heads, dtype=A.dtype)
      I = bk.expand_dims(I, axis=0)
      # [batch_size, num_heads, num_heads]
      I = bk.tile(I, reps=A.shape[0], axis=0)

      P = bk.norm(A - I, p="fro")**2
    return P

  def _apply_scores(self, scores, value, is_loc_attention, scores_mask=None):
    """Applies attention scores to the given value tensor.

    To use this method in your attention layer, follow the steps:

    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

    Args:
      scores: Scores float tensor of shape `[batch_size * num_heads, Tq, Tv]`.
      value: Value tensor of shape `[batch_size * num_heads, Tv, dim]`.
      scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
        `[batch_size, Tq, Tv]`. If given, scores at positions where
        `scores_mask==False` do not contribute to the result. It must contain
        at least one `True` value in each line along the last dimension.

    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.

    """
    num_heads = max(self.num_heads, 1)
    if scores_mask is not None:
      padding_mask = bk.logical_not(scores_mask)
      if num_heads > 1 and padding_mask.shape[0] != 1:
        padding_mask = bk.tile(padding_mask, reps=num_heads, axis=0)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * bk.cast(padding_mask, dtype=scores.dtype)
    # if the last dimension is 1, no point for applying softmax, hence,
    # softmax to the second last dimension
    attention_distribution = bk.softmax(
        scores, axis=-2 if scores.shape[-1] == 1 else -1)
    if is_loc_attention:  # location-based attention
      return attention_distribution * value, attention_distribution
    else:  # multi-head attention
      return bk.matmul(attention_distribution, value), attention_distribution

  def call(self, query, value=None, key=None, mask=None, training=None):
    # in case value is None, enable location-base mode, only query is given
    if key is None:
      key = value
    is_loc_attention = False
    if value is None and key is None:
      is_loc_attention = True
    if value is None and key is not None:
      raise RuntimeError("value is None but key is not None, in case of "
                         "multi-heads attention, value must be provided and "
                         "key is optional.")

    num_heads = max(self.num_heads, 1)
    if not is_loc_attention:
      assert query.shape[-1] == value.shape[-1] == key.shape[-1], \
        "Query, key and value must has the same feature dimension."

    # we want to keep the original value before projection of each head
    Q, V, K = query, value, key

    with bk.framework_(self):
      # [batch_size * num_heads, Tq, dim]
      query = _split_and_concat(self.query_heads(bk.array(query)), num_heads)
      if not is_loc_attention:
        # [batch_size * num_heads, Tv, dim]
        key = _split_and_concat(self.key_heads(bk.array(key)), num_heads)
        # [batch_size * num_heads, Tv, dim]
        value = _split_and_concat(self.value_heads(bk.array(value)), num_heads)

      # The attention scores [batch_size * num_heads, Tq, Tv]
      scores = self.calculate_scores(query=query, key=key)
      # dropout the attention scores
      if self.dropout > 0:
        scores = bk.dropout(scores,
                            p=self.dropout,
                            axis=1 if self.temporal_dropout else None,
                            training=training)
      # ====== multi-head regularization ====== #
      if self.num_heads > 0 and self.heads_norm > 0:
        self.add_loss(self.heads_norm * self.calculate_scores_norm(scores))

      # ====== prepare the mask ====== #
      if is_loc_attention:  # only 1 mask is need
        if isinstance(mask, (tuple, list)):
          q_mask = mask[0]
        else:
          q_mask = mask
        v_mask = None
      else:
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        if v_mask is not None:
          if v_mask.shape[1] != value.shape[1]:
            raise RuntimeError(
                "Value mask has time dimension %d, but value has time dimension %d"
                % (v_mask.shape[1], value.shape[1]))
          # Mask of shape [batch_size, 1, Tv].
          v_mask = bk.expand_dims(v_mask, axis=-2)

      if self.causal:
        # Creates a lower triangular mask, so position i cannot attend to
        # positions j>i. This prevents the flow of information from the future
        # into the past.
        scores_shape = scores.shape
        # causal_mask_shape = [1, Tq, Tv].
        causal_mask_shape = bk.concatenate(
            [bk.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)
        causal_mask = bk.tril_mask(causal_mask_shape)
      else:
        causal_mask = None
      scores_mask = bk.logical_and(v_mask, causal_mask)

      # ====== applying the attention ====== #
      result, scores_distribution = self._apply_scores(
          scores=scores,
          value=query if value is None else value,
          is_loc_attention=is_loc_attention,
          scores_mask=scores_mask)

      # ====== applying the mask ====== #
      if q_mask is not None:
        if q_mask.shape[1] != query.shape[1]:
          raise RuntimeError(
              "Query mask has time dimension %d, but query has time dimension %d"
              % (q_mask.shape[1], query.shape[1]))
        # Mask of shape [batch_size, Tq, 1].
        q_mask = bk.expand_dims(q_mask, axis=-1)
        if num_heads > 1:
          q_mask = bk.tile(q_mask, reps=num_heads, axis=0)
        result *= bk.cast(q_mask, dtype=result.dtype)

      # ====== final aggregation ====== #
      result = bk.reshape(result, shape=(-1, num_heads, [1], [2]))
      if self.heads_output_mode == 'mean':
        # [batch_size, Tq, dim]
        result = bk.reduce_sum(result, axis=1) / num_heads
      elif self.heads_output_mode in ('concat', 'cat', 'concatenate'):
        # [batch_size, Tq, dim * num_heads]
        result = bk.flatten(bk.swapaxes(result, 1, 2), outdim=3)
      elif self.num_heads == 0:
        # [batch_size, Tq, dim]
        result = bk.squeeze(result, axis=1)
      # ====== residual connection ====== #
      if self.residual:
        result += Q

    if self.return_attention:
      return result, scores_distribution
    return result

  def compute_mask(self, query, value=None, key=None, mask=None):
    with bk.framework_(self):
      if mask:
        # Self-attention
        if value is None and key is None:
          q_mask = mask[0] if isinstance(mask, (tuple, list)) else mask
        # Multi-heads attention
        else:
          q_mask = mask[0]
          if q_mask is None:
            return None
        return bk.array(q_mask)
      return None

  def get_config(self):
    config = {'causal': self.causal}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
