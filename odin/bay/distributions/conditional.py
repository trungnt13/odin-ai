import collections

import tensorflow as tf
from tensorflow_probability.python.distributions import \
    distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import (assert_util, prefer_static,
                                                    tensor_util,
                                                    tensorshape_util)

__all__ = ['ConditionalTensor']


class ConditionalTensor(distribution_lib.Distribution):
  r""" Create a conditional distribution by concatenate a Tensor to the the
  component distribution.

  This distribution is useful for creating a conditional variational
  autoencoder, e.g. concatenating the labels to the latent space
  """

  def __init__(self,
               distribution,
               conditional_tensor,
               validate_args=False,
               name=None):
    parameters = dict(locals())
    with tf.name_scope(name or
                       ('ConditionalTensor' + distribution.name)) as name:
      self._distribution = distribution
      # this is hideous but it work
      if hasattr(self._distribution, 'KL_divergence'):
        self.KL_divergence = self._distribution.KL_divergence
      self._conditional_tensor = tf.convert_to_tensor(conditional_tensor,
                                                      dtype_hint=tf.float32,
                                                      name="conditional_tensor")
      super(ConditionalTensor, self).__init__(
          dtype=self._distribution.dtype,
          reparameterization_type=self._distribution.reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self) -> distribution_lib.Distribution:
    return self._distribution

  @property
  def conditional_tensor(self):
    return self._conditional_tensor

  def __getitem__(self, slices):
    # Because slicing is parameterization-dependent, we only implement slicing
    # for instances of ConditionalTensor, not subclasses thereof.
    if type(self) is not ConditionalTensor:
      return super(ConditionalTensor, self).__getitem__(slices)

    slices = (tuple(slices) if isinstance(slices, collections.Sequence) else
              (slices,))
    if Ellipsis not in slices:
      slices = slices + (Ellipsis,)
    slices = slices + (slice(None),) * len(self.event_shape)
    return self.copy(distribution=self.distribution[slices],
                     conditional_tensor=self.conditional_tensor[slices])

  def _batch_shape_tensor(self):
    return self.distribution.batch_shape_tensor()

  def _batch_shape(self):
    return self.distribution.batch_shape

  def _event_shape_tensor(self):
    # If both `distribution.batch_shape` and `distribution.tensor_shape` are
    # known statically, then Distribution won't call this method.  But this
    # method may be called wheh only one of them is statically known.
    event_shape = self.distribution.event_shape
    if not tensorshape_util.is_fully_defined(event_shape):
      event_shape = self.distribution.event_shape_tensor()
    tensor_shape = self.conditional_tensor.shape
    if not tensorshape_util.is_fully_defined(tensor_shape):
      tensor_shape = tf.shape(self.conditional_tensor)
    return event_shape[:-1] + (event_shape[-1] + tensor_shape[-1],)

  def _event_shape(self):
    event_shape = self.distribution.event_shape
    tensor_shape = self.conditional_tensor.shape
    # just concatenate the last dimension
    return event_shape[:-1] + (event_shape[-1] + tensor_shape[-1],)

  def _sample_n(self, n, seed, **kwargs):
    s = self.distribution.sample(sample_shape=n, seed=seed, **kwargs)
    t = tf.repeat(tf.expand_dims(self.conditional_tensor, axis=0), n, axis=0)
    return tf.concat([s, t], axis=-1)

  def _log_prob(self, x, **kwargs):
    return self.distribution.log_prob(x, **kwargs)

  def _log_cdf(self, x, **kwargs):
    return self.distribution.log_cdf(x, **kwargs)

  def _entropy(self, **kwargs):
    return self.distribution.entropy(**kwargs)

  def _mean(self, **kwargs):
    return tf.concat(
        [self.distribution.mean(**kwargs), self.conditional_tensor], axis=-1)

  def _variance(self, **kwargs):
    return tf.concat(
        [self.distribution.variance(**kwargs), self.conditional_tensor],
        axis=-1)

  def _stddev(self, **kwargs):
    return tf.concat(
        [self.distribution.stddev(**kwargs), self.conditional_tensor], axis=-1)

  def _mode(self, **kwargs):
    return tf.concat(
        [self.distribution.mode(**kwargs), self.conditional_tensor], axis=-1)

  def _default_event_space_bijector(self):
    return self.distribution._experimental_default_event_space_bijector()  # pylint: disable=protected-access


@kullback_leibler.RegisterKL(ConditionalTensor, ConditionalTensor)
def _kl_independent(a, b, name='kl_conditionaltensor'):
  r"""Batched KL divergence `KL(a || b)` for ConditionalTensor distributions.

  This will just ignore the concatenated tensor and return `kl_divergence`
  of the original distribution.

  Arguments:
    a: Instance of `ConditionalTensor`.
    b: Instance of `ConditionalTensor`.
    name: (optional) name to use for created ops. Default 'kl_independent'.

  Returns:
    Batchwise `KL(a || b)`.

  Raises:
    ValueError: If the event space for `a` and `b`, or their underlying
      distributions don't match.
  """
  return kullback_leibler.kl_divergence(a.distribution,
                                        b.distribution,
                                        name=name)
