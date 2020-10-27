from typing import List

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (Distribution,
                                                         kullback_leibler)

__all__ = ['CombinedDistribution']


class CombinedDistribution(Distribution):
  r""" Convert a list of homogeneous distributions into a single distribution
  by concatenating their output along event shape

  If the `event_shape` mismatch, it is flattened.
  """

  def __init__(self, distributions, validate_args=False, name=None):
    parameters = dict(locals())
    batch_shape = [d.batch_shape for d in distributions]
    for shape in batch_shape:
      tf.assert_equal(
          shape, batch_shape,
          "All distributions must have the same batch_shape but given: %s" %
          str(batch_shape))
    self._distributions = distributions
    # check if need flatten
    flatten = False
    event_shape = [d.event_shape for d in distributions]
    s0 = event_shape[0]
    for s in event_shape[1:]:
      if len(s) != len(s0) or not all(i == j for i, j in zip(s[:-1], s0[:-1])):
        flatten = True
    self._flatten = flatten
    super(CombinedDistribution, self).__init__(
        dtype=self._distributions[0].dtype,
        reparameterization_type=self._distributions[0].reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=self._distributions[0].allow_nan_stats,
        parameters=parameters,
        name=name)

  @property
  def distributions(self) -> List[Distribution]:
    return self._distributions

  def _batch_shape_tensor(self):
    return self.distributions[0].batch_shape_tensor()

  def _batch_shape(self):
    return self.distributions[0].batch_shape

  def _event_shape_tensor(self):
    shape = self.distributions[0].event_shape_tensor()
    if self._flatten:
      shape = tf.reduce_prod(shape)
      for d in self.distributions[1:]:
        shape += tf.reduce_prod(d.event_shape_tensor())
    else:
      shape = tf.concat(
          [
              shape[:-1],
              tf.reduce_sum(
                  [d.event_shape_tensor()[-1] for d in self.distributions])
          ],
          axis=0,
      )
    return shape

  def _event_shape(self):
    if self._flatten:
      return (tf.reduce_sum(
          [tf.reduce_prod(d.event_shape) for d in self.distributions]),)
    else:
      shape = self.distributions[0].event_shape
      return tf.concat(
          [shape[:-1], (sum(d.event_shape[-1] for d in self.distributions),)],
          axis=0)

  def __getitem__(self, slices):
    return self.copy(
        distributions=[d.__getitem__(slices) for d in self.distributions])

  ######## Helpers
  def _concat_events(self, tensors):
    if self._flatten:
      tensors = tf.concat(
          [
              tf.reshape(
                  t, tf.concat([t.shape[:-len(d.event_shape)], (-1,)], axis=0))
              for t, d in zip(tensors, self._distributions)
          ],
          axis=-1,
      )
    return tf.concat(tensors, axis=-1)

  def _split_evidences(self, x):
    x = tf.nest.flatten(x)
    if len(x) >= len(self.distributions):
      return x
    elif len(x) == 1:
      x = x[0]
      if self._flatten:
        size = np.cumsum(
            [0] + [int(np.prod(d.event_shape)) for d in self.distributions])
        x = [x[..., s:e] for s, e in zip(size, size[1:])]
        x = [
            tf.reshape(i, tf.concat([i.shape[:-1], d.event_shape], axis=0))
            for i, d in zip(x, self.distributions)
        ]
      else:
        size = np.cumsum([0] + [d.event_shape[-1] for d in self.distributions])
        x = [x[..., s:e] for s, e in zip(size, size[1:])]
      return x
    raise RuntimeError("Given %s distributions but only %s evidences" %
                       (str(self.distributions), str([i.shape for i in x])))

  ######## Methods from Distribution
  def _log_prob(self, x, **kwargs):
    return sum(
        d._log_prob(x, **kwargs)
        for x, d in zip(self._split_evidences(x), self.distributions))

  def _log_cdf(self, x, **kwargs):
    return sum(
        d._log_cdf(x, **kwargs)
        for x, d in zip(self._split_evidences(x), self.distributions))

  ######## Statistics
  def _sample_n(self, n, seed, **kwargs):
    return self._concat_events(
        [d.sample(n, seed, **kwargs) for d in self._distributions])

  def _entropy(self, **kwargs):
    return self._concat_events(
        [d._entropy(**kwargs) for d in self.distributions])

  def _mean(self, **kwargs):
    return self._concat_events([d._mean(**kwargs) for d in self.distributions])

  def _variance(self, **kwargs):
    return self._concat_events(
        [d._variance(**kwargs) for d in self.distributions])

  def _stddev(self, **kwargs):
    return self._concat_events(
        [d._stddev(**kwargs) for d in self.distributions])

  def _mode(self, **kwargs):
    return self._concat_events([d._mode(**kwargs) for d in self.distributions])


@kullback_leibler.RegisterKL(CombinedDistribution, CombinedDistribution)
def _kl_independent(a: CombinedDistribution,
                    b: CombinedDistribution,
                    name='kl_combined'):
  r"""Batched KL divergence `KL(a || b)` for CombinedDistribution distributions.

  Just the summation of all distributions KL
  """
  kl = 0.
  for d1, d2 in zip(a.distributions, b.distributions):
    kl += kullback_leibler.kl_divergence(d1, d2, name=name)
  return kl
