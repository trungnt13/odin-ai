from typing import List, Optional

from odin.backend.types_helpers import Axis
import numpy as np
import tensorflow as tf
from odin.utils import as_tuple
from typeguard import typechecked
from tensorflow_probability.python.distributions import (Distribution,
                                                         kullback_leibler,
                                                         Blockwise)

__all__ = [
    'Batchwise',
]


# ===========================================================================
# Main
# ===========================================================================
class Batchwise(Distribution):
  """Concatenate distributions of the same type along specific axis in
  batch dimensions.

  See also
  --------
  `odin.bay.helpers.concat_distribution` :
      create a new distributions of the same type with concatenated parameters
  """

  def __init__(self,
               distributions: List[Distribution],
               axis: Axis = 0,
               validate_args: bool = False,
               name: Optional[str] = None):
    parameters = dict(locals())
    distributions = as_tuple(distributions)
    # validate distribution types
    dist_types = set([type(d) for d in distributions])
    assert len(dist_types) == 1, \
      f'Only concatenate homogeneous type of distribution but given: {dist_types}'
    # validate shape information
    shape_info = [(d.batch_shape, d.event_shape) for d in distributions]
    batch_ref = shape_info[0][0]
    event_ref = shape_info[0][1]
    for batch, event in shape_info:
      tf.assert_equal(
          batch.ndims, batch_ref.ndims,
          f"Rank of batch shapes mismatch {batch.ndims} != {batch_ref.ndims} ")
      tf.assert_equal(event, event_ref,
                      f"Event shapes mismatch {event} != {event_ref} ")
    self._distributions = distributions
    self._batch_ndims = batch_ref.ndims
    self._axis = int(axis) % self._batch_ndims
    super(Batchwise, self).__init__(
        dtype=self._distributions[0].dtype,
        reparameterization_type=self._distributions[0].reparameterization_type,
        validate_args=validate_args,
        allow_nan_stats=self._distributions[0].allow_nan_stats,
        parameters=parameters,
        name=name)

  @property
  def distributions(self) -> List[Distribution]:
    return self._distributions

  @property
  def axis(self) -> Axis:
    return self._axis

  def _batch_shape_tensor(self):
    shapes = [d.batch_shape_tensor() for d in self.distributions]
    newshapes = []
    for i in range(self._batch_ndims):
      if i != self.axis:
        s = shapes[0][i]
      else:
        s = tf.reduce_sum([sj[i] for sj in shapes])
      newshapes.append(s)
    return tf.concat(newshapes, axis=0)

  def _batch_shape(self):
    shapes = [d.batch_shape for d in self.distributions]
    newshapes = []
    for i in range(self._batch_ndims):
      if i != self.axis:
        s = shapes[0][i]
      else:
        s = sum(sj[i] for sj in shapes)
      newshapes.append(s)
    return tf.TensorShape(newshapes)

  def _event_shape_tensor(self):
    return self.distributions[0].event_shape_tensor()

  def _event_shape(self):
    return self.distributions[0].event_shape

  def __getitem__(self, slices):
    raise NotImplementedError()

  ######## Methods from Distribution
  def _log_prob(self, x, **kwargs):
    llk = []
    start = 0
    axis = self.axis
    for di in self.distributions:
      n = di.batch_shape[axis]
      xi = tf.gather(x, tf.range(start, start + n), axis=self.axis)
      llk.append(di._log_prob(xi, **kwargs))
      start += n
    return tf.concat(llk, axis=axis)

  def _log_cdf(self, x, **kwargs):
    lcd = []
    start = 0
    axis = self.axis
    for di in self.distributions:
      n = di.batch_shape[axis]
      xi = tf.gather(x, tf.range(start, start + n), axis=self.axis)
      lcd.append(di._log_cdf(xi, **kwargs))
      start += n
    return tf.concat(lcd, axis=axis)

  ######## Statistics
  def _sample_n(self, n, seed, **kwargs):
    samples = [d.sample(n, seed, **kwargs) for d in self.distributions]
    return tf.concat(samples, axis=self.axis + 1)

  def _entropy(self, **kwargs):
    return tf.concat([d.entropy(**kwargs) for d in self.distributions],
                     axis=self.axis)

  def _mean(self, **kwargs):
    return tf.concat([d.mean(**kwargs) for d in self.distributions],
                     axis=self.axis)

  def _variance(self, **kwargs):
    return tf.concat([d.variance(**kwargs) for d in self.distributions],
                     axis=self.axis)

  def _stddev(self, **kwargs):
    return tf.concat([d.stddev(**kwargs) for d in self.distributions],
                     axis=self.axis)

  def _mode(self, **kwargs):
    return tf.concat([d.mode(**kwargs) for d in self.distributions],
                     axis=self.axis)

  def _quantile(self, value, **kwargs):
    raise NotImplementedError('quantile is not implemented: {}'.format(
        type(self).__name__))

  def _covariance(self, **kwargs):
    raise NotImplementedError('covariance is not implemented: {}'.format(
        type(self).__name__))

  def __str__(self):
    s = super().__str__()
    s = s.replace(type(self).__name__, type(self.distributions[0]).__name__)
    return s

  def __repr__(self):
    s = super().__repr__()
    s = s.replace(type(self).__name__, type(self.distributions[0]).__name__)
    return s


@kullback_leibler.RegisterKL(Batchwise, Batchwise)
def _kl_independent(a: Batchwise, b: Batchwise, name='kl_batch_concatenation'):
  r"""Batched KL divergence `KL(a || b)` for concatenated distributions.

  Just the summation of all distributions KL
  """
  KLs = []
  for d1, d2 in zip(a.distributions, b.distributions):
    KLs.append(kullback_leibler.kl_divergence(d1, d2, name=name))
  return tf.concat(KLs, axis=a.axis)
