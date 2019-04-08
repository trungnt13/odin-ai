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
# ======================================================================== #
"""The ZeroInflated distribution class."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow_probability.python.distributions import (distribution,
  Bernoulli, Independent)
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    'ZeroInflated'
]

def _broadcast_rate(probs, *others):
  # make the shape broadcast-able
  others = list(others)
  others_ndims = [o.shape.ndims for o in others]
  assert len(set(others_ndims)) == 1
  others_ndims = others_ndims[0]

  probs_ndims = probs.shape.ndims

  if others_ndims < probs_ndims:
    for i in range(probs_ndims - others_ndims):
      others = [tf.expand_dims(o, -1)
                for o in others]
  elif others_ndims > probs_ndims:
    for i in range(others_ndims - probs_ndims):
      probs = tf.expand_dims(probs, -1)
  return [probs] + others

class ZeroInflated(distribution.Distribution):
  """zero-inflated distribution.

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
               validate_args=False,
               allow_nan_stats=True,
               name="ZeroInflated"):
    """Initialize a zero-inflated distribution.

    A `ZeroInflated` is defined by a zero-inflation rate (`inflated_distribution`,
    representing the probabilities of excess zeros) and a `Distribution` object
    having matching dtype, batch shape, event shape, and continuity
    properties (the dist).

    Parameters
    ----------
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

    References
    ----------
    Liu, L. & Blei, D.M.. (2017). Zero-Inflated Exponential Family Embeddings.
    Proceedings of the 34th International Conference on Machine Learning,
    in PMLR 70:2140-2148

    """
    parameters = dict(locals())
    self._runtime_assertions = []

    with tf.compat.v1.name_scope(name) as name:
      if not isinstance(count_distribution, distribution.Distribution):
        raise TypeError("count_distribution must be a Distribution instance"
                        " but saw: %s" % count_distribution)
      self._count_distribution = count_distribution

      if inflated_distribution is None:
        inflated_distribution = Bernoulli(
            logits=logits,
            probs=probs,
            dtype=tf.int32,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name="ZeroInflatedRate")
      elif not isinstance(inflated_distribution, distribution.Distribution):
        raise TypeError("inflated_distribution must be a Distribution instance"
                        " but saw: %s" % inflated_distribution)
      self._inflated_distribution = inflated_distribution

      if self._count_distribution.batch_shape.ndims is None:
        raise ValueError(
            "Expected to know rank(batch_shape) from count_disttribution")
      if self._inflated_distribution.batch_shape.ndims is None:
        raise ValueError(
            "Expected to know rank(batch_shape) from inflated_distribution")

      # create Independent Bernoulli distribution that the batch_shape
      # of count_distribution matching batch_shape of inflated_distribution
      inflated_batch_ndims = self._inflated_distribution.batch_shape.ndims
      count_batch_ndims = self._count_distribution.batch_shape.ndims
      if count_batch_ndims < inflated_batch_ndims:
        self._inflated_distribution = Independent(
            self._inflated_distribution,
            reinterpreted_batch_ndims=inflated_batch_ndims - count_batch_ndims,
            name="ZeroInflatedRate")
      elif count_batch_ndims > inflated_batch_ndims:
        raise ValueError("count_distribution has %d-D batch_shape, which smaller"
          "than %d-D batch_shape of inflated_distribution" %
          (count_batch_ndims, inflated_batch_ndims))

      # Ensure that all batch and event ndims are consistent.
      if validate_args:
        self._runtime_assertions.append(
            tf.assert_equal(
                self._count_distribution.batch_shape_tensor(),
                self._inflated_distribution.batch_shape_tensor(),
                message=("dist batch shape must match logits|probs batch shape"))
        )

    # We let the zero-inflated distribution access _graph_parents since its arguably
    # more like a baseclass.
    reparameterization_type = [
        self._count_distribution.reparameterization_type,
        self._inflated_distribution.reparameterization_type]
    if any(i == reparameterization.NOT_REPARAMETERIZED
           for i in reparameterization_type):
      reparameterization_type = reparameterization.NOT_REPARAMETERIZED
    else:
      reparameterization_type = reparameterization.FULLY_REPARAMETERIZED

    super(ZeroInflated, self).__init__(
        dtype=self._count_distribution.dtype,
        reparameterization_type=reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=self._count_distribution._graph_parents +
        self._inflated_distribution._graph_parents,
        name=name)

  @property
  def logits(self):
    """Log-odds of a `1` outcome (vs `0`)."""
    if isinstance(self._inflated_distribution, Independent):
      return self._inflated_distribution.distribution.logits
    return self._inflated_distribution.logits

  @property
  def probs(self):
    """Probability of a `1` outcome (vs `0`)."""
    if isinstance(self._inflated_distribution, Independent):
      return self._inflated_distribution.distribution.probs
    return self._inflated_distribution.probs

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
    with tf.compat.v1.control_dependencies(self._runtime_assertions):
      # These should all be the same shape by virtue of matching
      # batch_shape and event_shape.
      probs, d_mean = _broadcast_rate(self.probs, self._count_distribution.mean())
      return (1 - probs) * d_mean

  def _variance(self):
    """
    (1 - pi) * (d.var + d.mean^2) - [(1 - pi) * d.mean]^2

    Note: mean(ZeroInflated) = (1 - pi) * d.mean
    where:
     - pi is zero-inflated rate
     - d is count distribution
    """
    with tf.compat.v1.control_dependencies(self._runtime_assertions):
      # These should all be the same shape by virtue of matching
      # batch_shape and event_shape.
      d = self._count_distribution

      probs, d_mean, d_variance = _broadcast_rate(
          self.probs, d.mean(), d.variance())
      return (1 - probs) * \
      (d_variance + tf.square(d_mean)) - \
      tf.square(self._mean())

  def _log_prob(self, x):
    with tf.compat.v1.control_dependencies(self._runtime_assertions):
      x = tf.convert_to_tensor(x, name="x")
      d = self._count_distribution
      pi = self.probs

      d_prob = d.prob(x)
      d_log_prob = d.log_prob(x)

      # make pi and anything come out of count_distribution
      # broadcast-able
      pi, d_prob, d_log_prob = _broadcast_rate(pi, d_prob, d_log_prob)

      # This equation is validated
      # Equation (13) reference: u_{ij} = 1 - pi_{ij}
      y_0 = tf.log(pi + (1 - pi) * d_prob)
      y_1 = tf.log(1 - pi) + d_log_prob
      return tf.where(x == 0, y_0, y_1)

  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  def _sample_n(self, n, seed):
    with tf.compat.v1.control_dependencies(self._runtime_assertions):
      seed = seed_stream.SeedStream(seed, salt="ZeroInflated")
      mask = self.inflated_distribution.sample(n, seed())
      samples = self.count_distribution.sample(n, seed())
      mask, samples = _broadcast_rate(mask, samples)
      # mask = 1 => new_sample = 0
      # mask = 0 => new_sample = sample
      return samples * tf.cast(1 - mask, samples.dtype)

  # ******************** shortcut for denoising ******************** #
  def denoised_mean(self):
    return self.count_distribution.mean()

  def denoised_variance(self):
    return self.count_distribution.variance()
