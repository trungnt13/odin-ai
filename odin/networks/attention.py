from __future__ import absolute_import, division, print_function

import tensorflow as tf
import torch
from tensorflow import nest
from tensorflow.python import keras

from odin import backend as bk


# ===========================================================================
# Base classe
# ===========================================================================
class BaseAttention(keras.layers.Layer):
  pass


# ===========================================================================
# Attention classes
# ===========================================================================
class SoftAttention(BaseAttention):
  """ Original implementation from Tensorflow:
  `tensorflow/python/keras/layers/dense_attention.py`
  Copyright 2019 The TensorFlow Authors. All Rights Reserved.

  Base Attention class, modified for supporting:
   - Multi-head attention
   - Self-attention mechanism
   - Using more odin.backend function to make it easier transfer between
     tensorflow and pytorch

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

  References:
    [1] Graves, A., et al., 2014. Neural Turing Machines. arXiv:1410.5401 [cs].
    [2] Xu, K., et al., 2015. Show, Attend and Tell: Neural Image Caption
        Generation with Visual Attention. arXiv:1502.03044 [cs].
    [3] Luong, M.T., et al., 2015. Effective Approaches to Attention-based
        Neural Machine Translation. arXiv:1508.04025 [cs].
    [4] Kim, Y., Denton, C., Hoang, L., Rush, A.M., 2017. Structured Attention
        Networks. arXiv:1702.00887 [cs].
    [5] Vaswani, A., et al., 2017. Attention Is All You Need.
        arXiv:1706.03762 [cs].

  """

  def __init__(self,
               input_dim=None,
               causal=False,
               return_score=False,
               multihead_norm=0.,
               scale_initializer='one',
               scale_tied=True,
               attention_type='mul',
               attention_activation='softmax',
               **kwargs):
    super(SoftAttention, self).__init__(**kwargs)
    self.causal = causal
    self.return_score = bool(return_score)
    self.multihead_norm = multihead_norm
    self.supports_masking = True
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
      # TODO: this might be wrong
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
    base_config = super(BaseAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# ===========================================================================
# Soft and Hard attention
# ===========================================================================
class HardAttention(SoftAttention):
  pass
