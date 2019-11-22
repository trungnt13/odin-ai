# A comprehensive implementation of Attention Mechanism for Neural Networks
# Supporting:
#  * Multi-head attention
#  * Self-attention mechanism
#  * Using more odin.backend function to make it easier transfer between
#     tensorflow and pytorch
#
# Some suggestion for designing attention-based model:
# * Attention First, Feedforward Later (the sandwich design) is suggested in
#   (Press et al. 2019), more attention layers at the bottom, and more
#   feed-forward layer at the end.
# * Balancing the number of self-attention and feedforward sublayers appears
#   to be a desirable property
# * Use scale `1/sqrt(dim)` for dot-product attention (Vaswani et al. 2017).
# *
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
#   Press, O., Smith, N.A., Levy, O., n.d. 2019. Improving Transformer Models by
#     Reordering their Sublayers 8.

from __future__ import absolute_import, division, print_function

import warnings
from enum import IntFlag
from enum import auto as enum_auto
from functools import partial

from odin import backend as bk
from odin import bay
from odin.utils import as_tuple


def _split_and_concat(x, num_heads):
  return bk.stack(bk.split(x, num_heads, axis=-1), axis=0)


def _get_num_heads(query):
  r""" return the number of attention heads.
  return 0 if no multi-heads attention applied """
  ndim = query.ndim if hasattr(query, 'ndim') else query.shape.ndims
  if ndim == 3:
    num_heads = 0
  else:  # multi-heads attention
    num_heads = query.shape[0]
  return num_heads


def create_attention_heads(input_dim,
                           num_heads=2,
                           depth=1,
                           use_bias=True,
                           activation='relu'):
  r""" Create multi-heads attention projection
  `[batch_size, Tq, dim]` to `[num_heads, batch_size, Tq, dim]`
  """
  num_heads = int(num_heads)
  depth = int(depth)
  if num_heads > 1 and depth > 0:
    use_bias = as_tuple(use_bias, N=depth, t=bool)
    activation = as_tuple(activation, N=depth)
    layers = [
        bk.nn.Dense(input_dim * num_heads, use_bias=bias, activation=activ)
        for bias, activ in zip(use_bias, activation)
    ]
    layers.append(bk.nn.Lambda(partial(_split_and_concat, num_heads=num_heads)))
    return bk.nn.Sequential(layers)
  else:
    return bk.nn.Identity()


class AttentionMechanism(IntFlag):
  r""" The taxomony of all attention
  The meaning of `query`, `value` and `key` depend on the application. In the
  case of text similarity, for example, `query` is the sequence embeddings of
  the first piece of text and `value` is the sequence embeddings of the second
  piece of text. Hence, the attention determines alignment between `query` and
  `value`, `key` is usually the same tensor as value.
  A mapping from `query` to `key` will be learned during the attention.

  To use this method in your attention layer, follow the steps:
    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

  The following method call order is recommended:
    * `validate`: make the no duplicated steps stored in the `AttentionMechanism`
    * `prepare`: prepare the query, key, value and masks according to the given
      mechanism.
    * `score`: get the attention scores given the query and the key
    * `normalize` (optional): normalize the multi-heads attention scores.
    * `align`: create attention distribution, use this distribution to align
      the query and the value

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
    - `ScoreLocation`:
    - `ScoreAdditive`:
    - `ScoreDotProd`:
    - `ScoreCosine`:
    - `ScoreGeneral`:

  Since many studies try to group attention algorithm into categories, we take
  a more flexbile approach that allow a random path passing through each stage
  to create the final algorithm, e.g.
    - `Intra` to `PosGlobal` to `AlignSoft` to `ScoreLocation`
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
  ScoreLocation = enum_auto()
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

  @property
  def is_hard_attention(self):
    return AlignSoft not in self

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

  def prepare(self, query, key=None, value=None, mask=None):
    r""" Preparing input for attention model

    Returns:
      query: Query (or target sequence) tensor of shape `[batch_size, Tq, dim]`.
      key: Key (or source sequence) tensor of shape `[batch_size, Tv, dim]`.
      value: Value (or source sequence) tensor of shape `[batch_size, Tv, dim]`.
      mask: list of the following
        * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
            If given, the output will be zero at the positions where
            `mask==False`.
        * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
            If given, will apply the mask such that values at positions where
            `mask==False` do not contribute to the result.
    """
    # by default, if key is not provide, using value
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
    ### inter-attention
    else:
      if key is None:
        key = value
      if value is None:  # value must always provided
        raise RuntimeError("value must be given of inter-sequences attention.")
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

  def normalize(self, scores):
    r""" Normalize attention scores using "fro"-norm that encouraging diversity
    among attention heads math::`P = ||A^T*A - I||_2^2` (Kim et al. 2017)

    Arguments:
      scores: Tensor with shape `[batch_size * num_heads, Tq, Tv]`
    """
    # it is easier to assume there is always 1-head at least
    num_heads = _get_num_heads(scores)
    if num_heads == 0:
      return bk.cast(0., scores.dtype)
    # [batch_size, num_heads, Tq * Tv]
    scoresT = bk.swapaxes(bk.reshape(scores, shape=([0], [1], -1)), 0, 1)
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
            target_proj=None):
    r"""
    Arguments:
      query: Query (or target sequence) tensor of shape
        `[batch_size, Tq, dim]` or `[num_heads, batch_size, Tq, dim]` in case
        of multi-heads attention.
      key: Key (or source sequence) tensor of shape
        `[batch_size, Tv, dim]` or `[num_heads, batch_size, Tv, dim]` in case
        of multi-heads attention.
      scale: single `Scalar` or `Tensor` of shape `[dim]` for scaling
        the attention scores, suggested `1/sqrt(dim)` in (Vaswani et al. 2017).
      window_width : `None`, `Integer` or `Float` ([0, 1]). The total number of
        frames for a single window in local attention (i.e. `left + 1 + right`)
        Can be given as a fixed number of frames (`int`), or percentage of
        the sequence length (`float`). If `None`, use `Tq`
      q_proj : `Dense`, instance of dense or fully connected layer
        - for `ScoreLocation`, the number of hidden unit is `1`
        - for `ScoreGeneral`, the number of hidden unit is `dim`
      target_proj : `Dense`, for predictive local attention, applying
        a fully connected network on target sequence (i.e. the query) to
        predict the position on source sequence (i.e. the key).
        The layer must has output dimension equal to 1 and return logit value.

    Returns:
      Tensor of shape `[num_heads, batch_size, Tq, Tv]`, or
       `[num_heads, batch_size, Tq, 1]` if `ScoreLocation`
    """
    ### Check if multi-head attention is used
    num_heads = _get_num_heads(query)
    if num_heads > 0:
      query = bk.reshape(query, [-1] + [i for i in query.shape[2:]])
      if key is not None:
        key = bk.reshape(key, [-1] + [i for i in key.shape[2:]])
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
    if AttentionMechanism.ScoreLocation in self:
      if PosLocalM in self or PosLocalP in self:
        raise NotImplementedError(
            "ScoreLocation only support Global attention, but given: %s" %
            str(self))
      # [batch_size * num_heads, Tq, dim]
      scores = bk.reduce_mean(scale) * q_proj(query)
      assert scores.shape[-1] == 1, \
        " q_proj must have only 1 hidden unit, but given %d" % scores.shape[-1]
    ### Other score mode need the key tensor
    else:
      if key is None:
        raise ValueError("key must be provided for attention type: %s" %
                         str(self))
      ### Attention position (local or global)
      if PosLocalM in self:
        key = key[:, -window_width:]
      elif PosLocalP in self:
        pt = bk.sigmoid(target_proj(bk.reshape(query, ([0], -1))))
        assert pt.shape[-1] == 1, \
          "target_proj must project the query [., Tq * dim] to [., 1], i.e. " + \
            "predicting the attention position on source sequence using " + \
              "knowledge from target sequence."
        pt = Tv * pt  # `[batch_size * num_heads, 1]`
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
    ### get back the multi-heads shape
    if num_heads > 0:
      scores = bk.reshape(scores,
                          shape=[num_heads, -1] + [i for i in scores.shape[1:]])
    return scores

  def align(self,
            scores,
            value,
            query=None,
            q_mask=None,
            v_mask=None,
            causal=False,
            residual=False,
            dropout=0,
            temporal_dropout=False,
            n_mcmc=1,
            temperature=0.5,
            training=None):
    r"""Applies attention scores to the given value tensor.

    Arguments:
      scores: Attention Scores float tensor of shape
        `[num_heads, batch_size, Tq, Tv]`.
      value: Value (or source sequence) tensor of shape
        `[num_heads, batch_size, Tv, dim]`.
      query: Query (or target sequence) tensor of shape
        `[num_heads, batch_size, Tq, dim]`.
      q_mask: A boolean query mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      v_mask: A boolean value mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.
      dropout : Float. Dropout probability of the attention scores.
      temporal_dropout : Boolean. If `True`, using the same dropout mask along
        temporal axis (i.e. the 1-st dimension)
      n_mcmc (`Integer`) : number of mcmc samples for estimating the gradient
        of hard attention
      temperature: An 0-D `Tensor`, representing the temperature
        of a set of RelaxedOneHotCategorical distributions. The temperature
        should be positive.

    Returns:
      attended sequence: Tensor of shape
        * `[n_mcmc, num_heads, batch_size, Tq, dim]` for (hard + multi-heads)
        * `[n_mcmc, batch_size, Tq, dim]` for (hard + no-head)
        * `[num_heads, batch_size, Tq, dim]` for (soft + multi-heads)
        * `[batch_size, Tq, dim]` for (soft + no-head)
      attention distribution : for soft attention, return Tensor of shape
        * `[num_heads, batch_size, Tq]` for self-attention
        * `[num_heads, batch_size, Tq, Tv]` for inter-attention.
        for hard attention, return one-hot categorical distribution of shape
        * `[n_mcmc, num_heads, batch_size, Tq]` for self-attention
        * `[n_mcmc, num_heads, batch_size, Tq, Tv]` for inter-attention.
        if multi-heads attention wasn't used, omit the `[num_heads]`.
    """
    num_heads = _get_num_heads(scores)
    if num_heads == 0:
      Tq = scores.shape[1]
      Tv = scores.shape[2]
    else:
      Tq = scores.shape[2]
      Tv = scores.shape[3]
    if value is None:
      if query is None:
        raise ValueError("both query and value are None, "
                         "at least one of them must be given")
      value = query
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
      if num_heads > 0:
        v_mask = bk.expand_dims(v_mask, axis=0)
    scores_mask = bk.logical_and(v_mask, causal_mask)
    ### applying the scores mask
    if scores_mask is not None:
      padding_mask = bk.logical_not(scores_mask)
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
    attention = bk.dropout(attention_distribution,
                           p_drop=dropout,
                           axis=1 if temporal_dropout else None,
                           training=training and dropout > 0)
    # ====== applying the attention ====== #
    if self.is_self_attention and ScoreLocation in self:
      result = bk.expand_dims(bk.array(attention), axis=-1) * value  \
          if attention.shape[-1] != 1 else \
            attention * value
    else:
      if PosLocalM in self:
        value = value[:, -Tv:] if num_heads == 0 else value[:, :, -Tv:]
      result = bk.matmul(attention, value)
    # ====== applying the Query mask ====== #
    if q_mask is not None:
      assert q_mask.shape[1] == Tq,\
        "Query mask has time dimension %d, but query has time dimension %d" \
          % (q_mask.shape[1], Tq)
      # Mask of shape [batch_size, Tq, 1].
      q_mask = bk.expand_dims(q_mask, axis=-1)
      result *= bk.cast(q_mask, dtype=result.dtype)
    # ====== residual connection ====== #
    if residual:
      if query is None:
        raise ValueError("query must be given for residual connection")
      result += query
    # ====== return ====== #
    return result, attention_distribution

  def compute_mask(self, mask=None):
    if mask:
      q_mask = mask[0] if isinstance(mask, (tuple, list)) else mask
      return bk.array(q_mask)


# shortcut to make it easier
Intra = AttentionMechanism.Intra
Inter = AttentionMechanism.Inter
PosGlobal = AttentionMechanism.PosGlobal
PosLocalM = AttentionMechanism.PosLocalM
PosLocalP = AttentionMechanism.PosLocalP
AlignSoft = AttentionMechanism.AlignSoft
AlignRelax = AttentionMechanism.AlignRelax
AlignHard = AttentionMechanism.AlignHard
ScoreLocation = AttentionMechanism.ScoreLocation
ScoreAdditive = AttentionMechanism.ScoreAdditive
ScoreDotProd = AttentionMechanism.ScoreDotProd
ScoreCosine = AttentionMechanism.ScoreCosine
ScoreGeneral = AttentionMechanism.ScoreGeneral

_GROUPS = [
    (Intra, Inter), \
    (PosGlobal, PosLocalM, PosLocalP), \
    (AlignSoft, AlignHard, AlignRelax), \
    (ScoreLocation, ScoreAdditive, ScoreDotProd, ScoreCosine, ScoreGeneral)
]
