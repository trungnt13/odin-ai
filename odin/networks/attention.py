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

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import torch
from tensorflow import nest
from tensorflow.python import keras

from odin import backend as bk
from odin.utils import as_tuple


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


class BaseAttention(keras.Model):
  pass


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

  Args:
    causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past.
    return_score: Boolean. Set to `True` for returning the attention scores.

  Call Arguments:
    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
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

  """

  def __init__(self,
               input_dim=None,
               causal=False,
               return_scores=False,
               units=64,
               num_heads=0,
               heads_bias=True,
               heads_norm=0.,
               heads_activation='relu',
               scale_initializer='one',
               scale_tied=True,
               attention_type='mul',
               attention_activation='softmax',
               **kwargs):
    super().__init__(**kwargs)
    self.causal = causal
    self.return_scores = bool(return_scores)
    self.supports_masking = True
    # ====== multi-head ====== #
    self.units = int(units)
    self.heads_norm = heads_norm
    self.heads_bias = bool(heads_bias)
    self.num_heads = int(num_heads)
    # create a deep feedforward network for the heads:
    if self.num_heads > 0:
      with bk.framework_(self):
        self.query_heads = bk.nn.Parallel()
        self.key_heads = bk.nn.Parallel()
        self.value_heads = bk.nn.Parallel()
    # ====== initialize scale ====== #
    if not scale_tied and input_dim is None:
      raise ValueError("If scale_tied=False, the input_dim must be given")
    scale = 1
    if scale_initializer is not None:
      scale = bk.parse_initializer(scale_initializer, self)
      if scale_tied:
        scale = bk.variable(initial_value=scale(()),
                            trainable=True,
                            framework=self)
      else:
        scale = bk.variable(initial_value=scale(nest.flatten(input_dim)),
                            trainable=True,
                            framework=self)
    self.attention_scale = scale
    self.attention_type = str(attention_type).strip().lower()
    self.attention_activation = bk.parse_activation(attention_activation, self)

  def calculate_scores(self, query, key):
    """Calculates attention scores (a.k.a logits values).

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.

    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    if self.attention_type == 'mul':
      # this is a trick to make attention_scale broadcastable when
      # scale_tied=False
      return bk.matmul(self.attention_scale * query, bk.swapaxes(key, 1, 2))
    elif self.attention_type == 'add':
      # [batch_size, Tq, 1, dim]
      q = bk.expand_dims(query, axis=2)
      # [batch_size, 1, Tv, dim]
      k = bk.expand_dims(key, axis=1)
      return bk.reduce_sum(self.attention_scale * bk.tanh(q + k), axis=-1)
    else:
      raise NotImplementedError("No support for attention_type='%s'" %
                                self.attention_type)

  def calculate_scores_norm(self, scores):
    """ With the attention scores is A `[batch_size, Tq, Tv, num_heads]`
    `P = ||A^T*A - I||_2^2`
    """
    with bk.framework_(self):
      batch_size, Tq, Tv, num_heads = scores.shape
      # [batch_size, TqTv, num_heads]
      scores = bk.reshape(scores, shape=(batch_size, Tq * Tv, num_heads))
      # [batch_size, num_heads, TqTv]
      scoresT = bk.swapaxes(scores, 1, 2)
      I = bk.eye(self.head)
      I = bk.expand_dims(I, axis=0)
      # [batch_size, num_heads, num_heads]
      I = bk.tile(I, reps=scores.shape[0], axis=0)
      P = bk.norm(bk.matmul(scoresT, scores) - I, p="fro")**2
    return P

  def _apply_scores(self, scores, value, scores_mask=None):
    """Applies attention scores to the given value tensor.

    To use this method in your attention layer, follow the steps:

    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

    Args:
      scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
      value: Value tensor of shape `[batch_size, Tv, dim]`.
      scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
        `[batch_size, Tq, Tv]`. If given, scores at positions where
        `scores_mask==False` do not contribute to the result. It must contain
        at least one `True` value in each line along the last dimension.

    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.

    """
    if scores_mask is not None:
      padding_mask = bk.logical_not(scores_mask)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * bk.cast(padding_mask, dtype=scores.dtype)
    # [batch_size, Tq, Tv]
    attention_distribution = self.attention_activation(scores)
    return bk.matmul(attention_distribution, value)

  def call(self, query, value=None, key=None, mask=None, training=None):
    if value is None:
      value = query
    if key is None:
      key = value

    with bk.framework_(self):
      query = bk.array(query)
      key = bk.array(key)
      value = bk.array(value)
      scores = self.calculate_scores(query=query, key=key)

      # ====== multi-head regularization ====== #
      if self.num_heads > 0 and self.heads_norm > 0:
        self.add_loss(self.heads_norm * self.calculate_scores_norm(scores))

      # ====== prepare the mask ====== #
      q_mask = mask[0] if mask else None
      v_mask = mask[1] if mask else None
      if v_mask is not None:
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
      result = self._apply_scores(scores=scores,
                                  value=value,
                                  scores_mask=scores_mask)

      # ====== applying the mask ====== #
      if q_mask is not None:
        # Mask of shape [batch_size, Tq, 1].
        q_mask = bk.expand_dims(q_mask, axis=-1)
        result *= bk.cast(q_mask, dtype=result.dtype)
    return result

  def compute_mask(self, inputs, mask=None):
    with bk.framework_(self):
      if mask:
        q_mask = mask[0]
        if q_mask is None:
          return None
        return bk.array(q_mask)
      return None

  def get_config(self):
    config = {'causal': self.causal}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


# ===========================================================================
# Soft and Hard attention
# ===========================================================================
class HardAttention(SoftAttention):
  pass
