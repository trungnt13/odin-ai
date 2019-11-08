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
#   Wang, Y., et al., 2019. Transformer-based Acoustic Modeling for Hybrid
#     Speech Recognition. arXiv:1910.09799 [cs, eess].
#   Park, K., 2019. github.com/Kyubyong/transformer
#   Alexander H. Liu, 2019. github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch
#   Macar O.U., 2019. https://github.com/uzaymacar/attention-mechanisms
from __future__ import absolute_import, division, print_function

import warnings
from enum import IntFlag
from enum import auto as enum_auto
from functools import partial

from odin import backend as bk
from odin import bay


class AttentionMechanism(IntFlag):
  r""" The taxomony of all attention
  To use this method in your attention layer, follow the steps:
  * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
    `[batch_size, Tv]` to calculate the attention `scores`.
  * Pass `scores` and `value` tensors to this method. The method applies
    `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
    returns `matmul(attention_distribution, value).
  * Apply `query_mask` and return the result.

  All family of attention mechanism is summarized into follow
  a hierarchical structure, the order is as follow:

  The input space of the attention mechansim:
    - `Intra` (a.k.a. self-attention):
    - `Inter`:

  The attending positions within the input space:
    - `PosGlobal`: global attention
    - `PosLocalM`: local monotonic positioning
    - `PosLocalP`: local predictive positioning

  The alignment of the position:
    - `AlignSoft`:
    - `AlignRelax`: using gumble softmax for "relaxed" hard attention
    - `AlignHard`:

  The score function in which the attention logits are calculated:
    - `ScoreLocative`:
    - `ScoreAdditive`:
    - `ScoreDotProd`:
    - `ScoreCosine`:
    - `ScoreGeneral`:

  Since many studies try to group attention algorithm into categories, we take
  a more flexbile approach that allow a random path passing through each stage
  to create the final algorithm, e.g.
    - `Intra` to `PosGlobal` to `AlignSoft` to `ScoreLocative`
    - `Inter` to `PosGlobal` to `AlignHard` to `ScoreConcat`
  and so on.

  # TODO:
  * Down sampled multihead attention
  * Sparse attention
  """
  # ====== input space ====== #
  Intra = enum_auto()  # a.k.a. self-attention
  Inter = enum_auto()  # a.k.a. inter-attention
  # ====== attending positions ====== #
  PosGlobal = enum_auto()
  PosLocalM = enum_auto()  # local monotonic
  PosLocalP = enum_auto()  # local predictive
  # ====== alignment function ====== #
  AlignSoft = enum_auto()
  AlignHard = enum_auto()
  AlignRelax = enum_auto()
  # ====== alignment score function ====== #
  ScoreLocative = enum_auto()
  ScoreAdditive = enum_auto()
  ScoreDotProd = enum_auto()
  ScoreCosine = enum_auto()
  ScoreGeneral = enum_auto()

  def __or__(self, other):
    # delete the duplicated bit, then setting the new bit
    att = super().__or__(other)
    for group in _GROUPS:
      if other in group:
        for g in group:
          if g == other:
            continue
          att = att & ~g
        break
    return att

  def __str__(self):
    text = super().__str__()
    text = text.replace(self.__class__.__name__ + '.', '')
    return text

  @property
  def is_self_attention(self):
    r""" self-attention is intra-attention, in contrast to inter-attention
    which determines the alignment between two different sequences. """
    self.validate()
    if Intra in self:
      return True
    return False

  @property
  def is_soft_attention(self):
    return AlignSoft in self

  def validate(self):

    def count_and_check(groups):
      duplication = [g for g in groups if g in self]
      c = len(duplication)
      if c == 0:
        raise ValueError(
            "The created mechanism must contain one of the following: %s" %
            ', '.join([str(g) for g in groups]))
      elif c > 1:
        raise ValueError(
            "The created mechanism contain duplicated methods of the same stage: %s"
            % ', '.join([str(g) for g in duplication]))

    for g in _GROUPS:
      count_and_check(g)
    return self

  def check_inputs(self, query, key=None, value=None, mask=None):
    # by default, if key is not provide, using value
    if key is None:
      key = value
    query = bk.array(query, ignore_none=True)
    key = bk.array(key, ignore_none=True)
    value = bk.array(value, ignore_none=True)
    # ====== check if intra-attention ====== #
    if self.is_self_attention:
      if (key is not None or value is not None):
        warnings.warn(
            "Self-attention (intra-attention) need only query, "
            "ignore provided key and value",
            category=UserWarning)
      if key is not None:
        key = query
      if value is not None:
        value = query
    elif value is None and key is not None:
      raise RuntimeError("value is None but key is not None, value must be "
                         "provided and key is optional.")
    # ====== masks ====== #
    if self.is_self_attention:  # only 1 mask is need
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
    # ====== return ====== #
    return query, key, value, \
      bk.array(q_mask, ignore_none=True), bk.array(v_mask, ignore_none=True)

  def normalize_scores(self, scores, num_heads=1):
    r""" Normalize attention scores using "fro"-norm that encouraging diversity
    among attention heads math::`P = ||A^T*A - I||_2^2` (Kim et al. 2017)

    Arguments:
      scores: Tensor with shape `[batch_size * num_heads, Tq, Tv]`
    """
    # it is easier to assume there is always 1-head at least
    num_heads = max(num_heads, 1)
    Tq, Tv = scores.shape[1:]
    # [batch_size, num_heads, Tq * Tv]
    scoresT = bk.reshape(scores, shape=(-1, num_heads, Tq * Tv))
    # [batch_size, Tq * Tv, num_heads]
    scores = bk.swapaxes(scoresT, 1, 2)
    # [batch_size, num_heads, num_heads]
    A = bk.matmul(scoresT, scores)
    # [batch_size, num_heads, num_heads]
    I = bk.eye(num_heads, dtype=A.dtype)
    I = bk.expand_dims(I, axis=0)
    I = bk.tile(I, reps=A.shape[0], axis=0)
    # normalized
    P = bk.norm(A - I, p="fro")**2
    return P

  def score(self,
            query,
            key=None,
            scale=1,
            window_width=None,
            q_proj=None,
            v_proj=None):
    r"""
    Arguments:
      query: Query (or target sequence) tensor of shape
        `[batch_size * num_heads, Tq, dim]`.
      key: Key (or source sequence) tensor of shape
        `[batch_size * num_heads, Tv, dim]`.
      window_width : `None`, `Integer` or `Float` ([0, 1]). The total number of
        frames for a single window in local attention (i.e. `left + 1 + right`)
        Can be given as a fixed number of frames (`int`), or percentage of
        the sequence length (`float`). If `None`, use `Tq`
      q_proj : `Dense`, instance of dense or fully connected layer
        - for `ScoreLocative`, the number of hidden unit is `1`
        - for `ScoreGeneral`, the number of hidden unit is `dim`
      v_proj : `Dense`, instance of dense of full connected layer

    Returns:
      Tensor of shape `[batch_size * num_heads, Tq, Tv]`, or
       `[batch_size * num_heads, Tq, 1]` if `key` is not provided
    """
    Tq = query.shape[1]
    Tv = Tq if key is None else key.shape[1]
    # scale shape is `[]` or `[dim]`
    scale = bk.array(scale, dtype=query.dtype)
    ### Check the window width
    if window_width is None:
      window_width = Tq
    elif window_width < 1:
      window_width = window_width * Tv
    window_width = int(window_width)
    ### Locative attention
    if AttentionMechanism.ScoreLocative in self:
      if PosLocalM in self or PosLocalP in self:
        raise NotImplementedError(
            "ScoreLocative only support Global attention.")
      # [batch_size * num_heads, Tq, dim]
      scores = bk.reduce_mean(scale) * q_proj(query)
      assert scores.shape[-1] == 1, \
        " q_proj must have only 1 hidden unit, but given %d" % scores.shape[-1]
      return scores
    ### Other score mode need key tensor
    if key is None:
      raise ValueError("key must be provided for attention type: %s" %
                       str(self))
    ### Attention position (local or global)
    if PosLocalM in self:
      key = key[:, -window_width:]
    elif PosLocalP in self:
      pt = bk.sigmoid(v_proj(bk.reshape(query, ([0], -1))))
      # `[batch_size * num_heads, 1]`
      pt = Tv * pt
      # `[batch_size * num_heads, Tv]`
      # Eq (10) (Luong et al. 2015)
      gauss_est = bk.exp(-bk.square(bk.arange(Tv, dtype=pt.dtype) - pt) /
                         (2 * bk.square(window_width / 2)))
      # `[batch_size * num_heads, 1, Tv]`
      gauss_est = bk.expand_dims(gauss_est, axis=1)
    ### Additive or concat method
    if AttentionMechanism.ScoreAdditive in self:
      # [batch_size * num_heads, Tq, 1, dim]
      q = bk.expand_dims(query, axis=2)
      # [batch_size * num_heads, 1, Tv, dim]
      k = bk.expand_dims(key, axis=1)
      # [batch_size * num_heads, Tq, Tv]
      scores = bk.reduce_sum(scale * bk.tanh(q + k), axis=-1)
    ### Dot product or multiplicative scoring
    elif AttentionMechanism.ScoreDotProd in self:
      # this is a trick to make attention_scale broadcastable when
      # scale_tied=False
      scores = bk.matmul(scale * query, bk.swapaxes(key, 1, 2))
    ### cosine scoring
    elif AttentionMechanism.ScoreCosine in self:
      # [batch_size * num_heads, Tq, 1, dim]
      q = bk.expand_dims(query, axis=2)
      # [batch_size * num_heads, 1, Tv, dim]
      k = bk.expand_dims(key, axis=1)
      # [batch_size * num_heads, Tq, Tv, dim]
      scores = (q * k) / (bk.norm(q, p=2) * bk.norm(k, p=2))
      scores = bk.reduce_sum(scale * scores, axis=-1, keepdims=False)
    ### general method with only project on the query
    elif AttentionMechanism.ScoreGeneral in self:
      query = q_proj(query)
      assert query.shape[-1] == key.shape[-1], \
        " q_proj must have %d hidden units, but given %d units" % \
          (key.shape[-1], query.shape[-1])
      scores = bk.matmul(scale * query, bk.swapaxes(key, 1, 2))
    else:
      raise NotImplementedError("No support for attention_type='%s'" %
                                str(self))
    ### applying the local-predictive attention
    if PosLocalP in self:
      scores = scores * gauss_est
    return scores

  def align(self,
            scores,
            value,
            q_mask=None,
            v_mask=None,
            causal=False,
            dropout=0,
            temporal_dropout=False,
            n_mcmc=1,
            temperature=0.5,
            training=None):
    r"""Applies attention scores to the given value tensor.

    Args:
      scores: Attention Scores float tensor of shape
        `[batch_size * num_heads, Tq, Tv]`.
      value: Value tensor of shape `[batch_size * num_heads, Tv, dim]`.
      q_mask: A boolean query mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      v_mask: A boolean value mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.
      n_mcmc (`Integer`) : number of mcmc samples for estimating the gradient
        of hard attention
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of RelaxedOneHotCategorical distributions. The temperature
        should be positive.

    Returns:
      attended sequence: Tensor of shape `[batch_size * num_heads, Tq, dim]`,
        or, in case of hard attention, tensor of shape
        `[n_mcmc, batch_size, Tq, dim]`.
      attention distribution : Tensor or one-hot OneHotCategorical distribution
        of shape `[n_mcmc, batch_size * num_heads, Tq]` for self-attention, and
        `[n_mcmc, batch_size * num_heads, Tq, Tv]` for inter-attention.
    """
    Tq = scores.shape[1]
    Tv = scores.shape[2]
    # ====== Causal mask ====== #
    if causal:
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
    if v_mask is not None:
      # LocalM applied
      if PosLocalM in self:
        v_mask = v_mask[:, -Tv:]
      # Mask of shape [batch_size, 1, Tv].
      v_mask = bk.expand_dims(v_mask, axis=-2)
      v_mask = bk.cast(v_mask, 'bool')
    scores_mask = bk.logical_and(v_mask, causal_mask)
    ### applying the scores mask
    if scores_mask is not None:
      padding_mask = bk.logical_not(scores_mask)
      num_heads = bk.cast(scores.shape[0] / padding_mask.shape[0], 'int32')
      if num_heads > 1 and padding_mask.shape[0] != 1:
        padding_mask = bk.tile(padding_mask, reps=num_heads, axis=0)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * bk.cast(padding_mask, dtype=scores.dtype)
    # ====== convert attention score to distribution ====== #
    # if the last dimension is 1, no point for applying softmax, hence,
    # softmax to the second last dimension
    ### soft attention
    if AlignSoft in self:
      attention_distribution = bk.softmax(
          scores, axis=-2 if scores.shape[-1] == 1 else -1)
    ### relaxed hard attention
    elif AlignRelax in self:
      attention_distribution = bay.distributions.RelaxedOneHotCategorical(
          temperature=temperature,
          logits=bk.squeeze(scores, axis=-1)
          if scores.shape[-1] == 1 else scores)
      fsample = partial(bay.Distribution.sample, sample_shape=n_mcmc)
      attention_distribution = bay.coercible_tensor(
          attention_distribution, convert_to_tensor_fn=fsample)
    ### hard attention
    elif AlignHard in self:
      attention_distribution = bay.distributions.OneHotCategorical(
          logits=bk.squeeze(scores, axis=-1)
          if scores.shape[-1] == 1 else scores,
          dtype=value.dtype)
      fsample = partial(bay.Distribution.sample, sample_shape=n_mcmc)
      attention_distribution = bay.coercible_tensor(
          attention_distribution, convert_to_tensor_fn=fsample)
    # ======  dropout the attention scores ====== #
    if dropout > 0:
      attention_distribution = bk.dropout(attention_distribution,
                                          p=dropout,
                                          axis=1 if temporal_dropout else None,
                                          training=training)
    # ====== applying the attention ====== #
    if self.is_self_attention and ScoreLocative in self:
      if attention_distribution.shape[-1] != 1:
        attention_distribution = bk.expand_dims(attention_distribution, axis=-1)
      result = attention_distribution * value
    else:
      if PosLocalM in self:
        value = value[:, -Tv:]
      result = bk.matmul(attention_distribution, value)
    # ====== applying the Query mask ====== #
    if q_mask is not None:
      assert q_mask.shape[1] == Tq,\
        "Query mask has time dimension %d, but query has time dimension %d" \
          % (q_mask.shape[1], Tq)
      # Mask of shape [batch_size, Tq, 1].
      q_mask = bk.expand_dims(q_mask, axis=-1)
      num_heads = bk.cast(scores.shape[0] / q_mask.shape[0], 'int32')
      if num_heads > 1 and q_mask.shape[0] != 1:
        q_mask = bk.tile(q_mask, reps=num_heads, axis=0)
      result *= bk.cast(q_mask, dtype=result.dtype)
    # ====== return ====== #
    return result, attention_distribution

  def __call__(self, query, key=None, value=None):
    if residual:
      result += bk.tile(Q, reps=num_heads, axis=0) if num_heads > 1 else Q


# shortcut to make it easier
Intra = AttentionMechanism.Intra
Inter = AttentionMechanism.Inter
PosGlobal = AttentionMechanism.PosGlobal
PosLocalM = AttentionMechanism.PosLocalM
PosLocalP = AttentionMechanism.PosLocalP
AlignSoft = AttentionMechanism.AlignSoft
AlignRelax = AttentionMechanism.AlignRelax
AlignHard = AttentionMechanism.AlignHard
ScoreLocative = AttentionMechanism.ScoreLocative
ScoreAdditive = AttentionMechanism.ScoreAdditive
ScoreDotProd = AttentionMechanism.ScoreDotProd
ScoreCosine = AttentionMechanism.ScoreCosine
ScoreGeneral = AttentionMechanism.ScoreGeneral

_GROUPS = [
    (Intra, Inter), \
    (PosGlobal, PosLocalM, PosLocalP), \
    (AlignSoft, AlignHard, AlignRelax), \
    (ScoreLocative, ScoreAdditive, ScoreDotProd, ScoreCosine, ScoreGeneral)
]
