# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright (c) 2017 - 2018 scVAE authors
# Copyright (c) 2018 - 2019 SISUA authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TODO: NotImplementedError: <class 'odin.bay.distributions.zero_inflated.ZeroInflated'>
# does not support batch slicing; must implement _params_event_ndims.
# ======================================================================== #
"""The ZeroInflated distribution class."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import (Bernoulli, Independent,
                                                         distribution)
from tensorflow_probability.python.internal import (reparameterization,
                                                    tensor_util)
from tensorflow_probability.python.util.seed_stream import SeedStream

__all__ = ['ZeroInflated']


def _make_broadcastable(*tensors):
  # TODO: still hard for broadcastable here
  shape = None
  max_rank = 0
  for t in tensors:
    if len(t.shape) > max_rank:
      max_rank = len(t.shape)
      shape = tf.shape(t)
  return [
      tf.broadcast_to(t, shape) if len(t.shape) < max_rank else t
      for t in tensors
  ]


class ZeroInflated(distribution.Distribution):
  r"""Zero-inflated distribution.

  The `zero-inflated` object implements batched zero-inflated distributions.
  The zero-inflated model is defined by a zero-inflation rate
  and a python list of `Distribution` objects.

  Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
  `entropy_lower_bound`.
  """

  def __init__(self,
               count_distribution,
               inflated_distribution=None,
               logits=None,
               probs=None,
               eps=1e-8,
               validate_args=False,
               allow_nan_stats=True,
               name="ZeroInflated"):
    r"""Initialize a zero-inflated distribution.

    A `ZeroInflated` is defined by a zero-inflation rate (`inflated_distribution`,
    representing the probabilities of excess zeros) and a `Distribution` object
    having matching dtype, batch shape, event shape, and continuity
    properties (the dist).

    Arguments:
      count_distribution : A `tfp.distributions.Distribution` instance.
        The instance must have `batch_shape` matching the zero-inflation
        distribution.
      inflated_distribution: `tfp.distributions.Bernoulli`-like instance.
        Manages the probability of excess zeros, the zero-inflated rate.
        Must have either scalar `batch_shape` or `batch_shape` matching
        `count_distribution.batch_shape`.
      logits: An N-D `Tensor` representing the log-odds of a excess zeros
        A zero-inflation rate, where the probability of excess zeros is
        sigmoid(logits).
        Only one of `logits` or `probs` should be passed in.
      probs: An N-D `Tensor` representing the probability of a zero event.
        Each entry in the `Tensor` parameterizes an independent
        ZeroInflated distribution.
        Only one of `logits` or `probs` should be passed in.
      validate_args: Python `bool`, default `False`. If `True`, raise a runtime
        error if batch or event ranks are inconsistent between pi and any of
        the distributions. This is only checked if the ranks cannot be
        determined statically at graph construction time.
      allow_nan_stats: Boolean, default `True`. If `False`, raise an
       exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).

    References:
      Liu, L. & Blei, D.M.. (2017). Zero-Inflated Exponential Family Embeddings.
        Proceedings of the 34th International Conference on Machine Learning,
        in PMLR 70:2140-2148

    """
    parameters = dict(locals())
    with tf.compat.v1.name_scope(name) as name:
      # main count distribution
      if not isinstance(count_distribution, distribution.Distribution):
        raise TypeError("count_distribution must be a Distribution instance"
                        " but saw: %s" % count_distribution)
      # Zero inflation distribution
      if inflated_distribution is None:
        inflated_distribution = Bernoulli(logits=logits,
                                          probs=probs,
                                          dtype=tf.int32,
                                          validate_args=validate_args,
                                          allow_nan_stats=allow_nan_stats,
                                          name="ZeroInflatedRate")
      elif not isinstance(inflated_distribution, distribution.Distribution):
        raise TypeError("inflated_distribution must be a Distribution instance"
                        " but saw: %s" % inflated_distribution)
      # Matching the event shape
      inflated_ndim = len(inflated_distribution.event_shape)
      count_ndim = len(count_distribution.event_shape)
      if inflated_ndim < count_ndim:
        inflated_distribution = Independent(inflated_distribution,
                                            count_ndim - inflated_ndim)
      self._count_distribution = count_distribution
      self._inflated_distribution = inflated_distribution
      #
      if self._count_distribution.batch_shape.ndims is None:
        raise ValueError(
            "Expected to know rank(batch_shape) from count_distribution")
      if self._inflated_distribution.batch_shape.ndims is None:
        raise ValueError(
            "Expected to know rank(batch_shape) from inflated_distribution")
      self._eps = tensor_util.convert_nonref_to_tensor(
          eps, dtype_hint=count_distribution.dtype, name='eps')
    # We let the zero-inflated distribution access _graph_parents since its arguably
    # more like a baseclass.
    super(ZeroInflated, self).__init__(
        dtype=self._count_distribution.dtype,
        reparameterization_type=self._count_distribution.
        reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        name=name,
    )

  def __getitem__(self, slices):
    return self.copy(
        count_distribution=self.count_distribution.__getitem__(slices),
        inflated_distribution=self.inflated_distribution.__getitem__(slices))

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    if isinstance(self._inflated_distribution, Independent):
      return self._inflated_distribution.distribution.logits_parameter()
    return self._inflated_distribution.logits_parameter()

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    if isinstance(self._inflated_distribution, Independent):
      return self._inflated_distribution.distribution.probs_parameter()
    return self._inflated_distribution.probs_parameter()

  @property
  def count_distribution(self):
    return self._count_distribution

  @property
  def inflated_distribution(self):
    return self._inflated_distribution

  def _batch_shape_tensor(self):
    return self._count_distribution._batch_shape_tensor()

  def _batch_shape(self):
    return self._count_distribution._batch_shape()

  def _event_shape_tensor(self):
    return self._count_distribution._event_shape_tensor()

  def _event_shape(self):
    return self._count_distribution._event_shape()

  def _mean(self):
    # These should all be the same shape by virtue of matching
    # batch_shape and event_shape.
    probs, d_mean = _make_broadcastable(self.probs,
                                        self._count_distribution.mean())
    return (1 - probs) * d_mean

  def _variance(self):
    r"""
    (1 - pi) * (d.var + d.mean^2) - [(1 - pi) * d.mean]^2

    Note: mean(ZeroInflated) = (1 - pi) * d.mean

    where:
     - pi is zero-inflated rate
     - d is count distribution
    """
    # These should all be the same shape by virtue of matching
    # batch_shape and event_shape.
    d = self._count_distribution
    probs, d_mean, d_variance = _make_broadcastable(self.probs, d.mean(),
                                                    d.variance())
    return ((1 - probs) * (d_variance + tf.square(d_mean)) -
            tf.math.square(self._mean()))

  def _log_prob(self, x):
    # this version use logits and log_prob which is more numerical stable
    x = tf.convert_to_tensor(x, dtype=self.dtype)
    d = self._count_distribution
    eps = self._eps
    pi = self.logits
    llk = d.log_prob(x)
    # prepare broadcast
    llk, pi = _make_broadcastable(llk, pi)
    #
    t1 = llk - pi
    t2 = tf.nn.softplus(-pi)
    y_0 = tf.nn.softplus(t1) - t2
    y_1 = t1 - t2
    return tf.where(x > eps, y_1, y_0)

  # def _log_prob(self, x):
  #   x = tf.convert_to_tensor(x, name="x")
  #   d = self._count_distribution
  #   logits = self.logits
  #   pi = self.probs
  #   eps = self._eps
  #   # count distribution llk
  #   log_prob = d.log_prob(x)
  #   prob = tf.math.exp(log_prob)
  #   # make pi and anything come out of count_distribution
  #   # broadcast-able
  #   pi, prob, log_prob = _make_broadcastable(pi, prob, log_prob)
  #   # This equation is validated
  #   # Equation (13) reference: u_{ij} = 1 - pi_{ij}
  #   y_0 = tf.math.log(pi + (1. - pi) * prob + eps)
  #   # y_0 = tf.nn.softplus(log_prob - logits) + tf.nn.softplus(-logits)
  #   y_1 = tf.math.log(1. - pi + eps) + log_prob
  #   # note: sometimes pi can get to 1 and y_1 -> -inf
  #   return tf.where(x > eps, y_1, y_0)

  def _sample_n(self, n, seed):
    seed = SeedStream(seed, salt="ZeroInflated")
    mask = self.inflated_distribution.sample(n, seed())
    samples = self.count_distribution.sample(n, seed())
    tf.assert_equal(
        tf.rank(samples) >= tf.rank(mask),
        True,
        message=f"Cannot broadcast zero inflated mask of shape {mask.shape} "
        f"to sample shape {samples.shape}")
    samples, mask = _make_broadcastable(samples, mask)
    # mask = 1 => new_sample = 0
    # mask = 0 => new_sample = sample
    return samples * tf.cast(1 - mask, samples.dtype)

  # ******************** shortcut for denoising ******************** #
  def denoised_mean(self):
    return self.count_distribution.mean()

  def denoised_variance(self):
    return self.count_distribution.variance()
