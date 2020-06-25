import types

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers.distribution_layer import _event_size

__all__ = ['DeterministicLayer', 'VectorDeterministicLayer']


class DeterministicLayer(tfl.DistributionLambda):
  r"""
  ```none
  pmf(x; loc) = 1, if x == loc, else 0
  cdf(x; loc) = 1, if x >= loc, else 0
  ```
  """

  def __init__(self,
               event_shape=(),
               log_prob=None,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(DeterministicLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, log_prob, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          log_prob=None,
          validate_args=False,
          name='DeterministicLayer'):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat(
        [tf.shape(input=params)[:-1], event_shape],
        axis=0,
    )
    dist = tfd.Deterministic(loc=tf.reshape(params, output_shape),
                             validate_args=validate_args,
                             name=name)
    # override the log-prob function
    if log_prob is not None and callable(log_prob):
      dist.log_prob = types.MethodType(log_prob, dist)
    return dist

  @staticmethod
  def params_size(event_shape, name='DeterministicLayer_params_size'):
    r""" The number of `params` needed to create a single distribution. """
    return _event_size(event_shape, name)


class VectorDeterministicLayer(tfl.DistributionLambda):
  r"""
  ```none
  pmf(x; loc)
    = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
    = 0, otherwise.
  ```
  """

  def __init__(self,
               event_shape=(),
               log_prob=None,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(VectorDeterministicLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, log_prob, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          log_prob=None,
          validate_args=False,
          name='VectorDeterministicLayer'):
    r""" Create the distribution instance from a `params` vector. """
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat(
        [tf.shape(input=params)[:-1], event_shape],
        axis=0,
    )
    dist = tfd.VectorDeterministic(loc=tf.reshape(params, output_shape),
                                   validate_args=validate_args,
                                   name=name)
    if log_prob is not None and callable(log_prob):
      dist.log_prob = types.MethodType(log_prob, dist)
    return dist

  @staticmethod
  def params_size(event_shape, name='VectorDeterministicLayer_params_size'):
    r""" The number of `params` needed to create a single distribution. """
    return _event_size(event_shape, name)
