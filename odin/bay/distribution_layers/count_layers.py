from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import _event_size

from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'PoissonLayer', 'NegativeBinomialDispLayer', 'NegativeBinomialLayer',
    'ZINegativeBinomialDispLayer', 'ZINegativeBinomialLayer', 'ZIPoissonLayer'
]

PoissonLayer = tfl.IndependentPoisson


class NegativeBinomialLayer(DistributionLambda):
  """An independent NegativeBinomial Keras layer.

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.
  given_log_count : boolean
    is the input representing log count values or the count itself
  dispersion : {'full', 'share', 'single'}
    'full' creates a dispersion value for each individual data point,
    'share' creates a single vector of dispersion for all examples, and
    'single' uses a single value as dispersion for all data points.
    Note: the dispersion in this case is the probability of success.
  convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
    instance and returns a `tf.Tensor`-like object.
    Default value: `tfd.Distribution.sample`.
  validate_args: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  """

  def __init__(self,
               event_shape=(),
               given_log_count=True,
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    dispersion = str(dispersion).lower()
    assert dispersion in ('full', 'single', 'share'), \
      "Only support three different dispersion value: 'full', 'single' and " + \
        "'share', but given: %s" % dispersion
    super(NegativeBinomialLayer,
          self).__init__(lambda t: type(self).new(
              t, event_shape, given_log_count, dispersion, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_count=True,
          dispersion='full',
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'NegativeBinomialLayer',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32),
                                               tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ],
                               axis=0)
      ndims = output_shape.shape[0]

      total_count_params, logits_params = tf.split(params, 2, axis=-1)
      if dispersion == 'single':
        logits_params = tf.reduce_mean(logits_params)
      elif dispersion == 'share':
        logits_params = tf.reduce_mean(logits_params,
                                       axis=tf.range(0,
                                                     ndims - 1,
                                                     dtype='int32'),
                                       keepdims=True)
      if given_log_count:
        total_count_params = tf.exp(total_count_params, name='total_count')
      return tfd.Independent(
          tfd.NegativeBinomial(total_count=tf.reshape(total_count_params,
                                                      output_shape),
                               logits=tf.reshape(logits_params, output_shape)
                               if dispersion == 'full' else logits_params,
                               validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'NegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape,
                             name=name or 'NegativeBinomial_params_size')


class NegativeBinomialDispLayer(DistributionLambda):
  """An alternative parameterization of the NegativeBinomial Keras layer.

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.
  given_log_mean : `bool`
    is the input representing log mean values or the count mean itself
  given_log_mean : `bool`
    is the input representing log mean values or the count mean itself
  dispersion : {'full', 'share', 'single'}
    'full' creates a dispersion value for each individual data point,
    'share' creates a single vector of dispersion for all examples, and
    'single' uses a single value as dispersion for all data points.
  convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
    instance and returns a `tf.Tensor`-like object.
    Default value: `tfd.Distribution.sample`.
  validate_args: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  """

  def __init__(self,
               event_shape=(),
               given_log_mean=True,
               given_log_disp=True,
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    dispersion = str(dispersion).lower()
    assert dispersion in ('full', 'single', 'share'), \
      "Only support three different dispersion value: 'full', 'single' and " + \
        "'share', but given: %s" % dispersion
    super(NegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, given_log_mean, given_log_disp,
                                 dispersion, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_mean=True,
          given_log_disp=True,
          dispersion='full',
          validate_args=False,
          name=None):
    """ Create the distribution instance from a `params` vector. """
    with tf.compat.v1.name_scope(name, 'NegativeBinomialDispLayer',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32),
                                               tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ],
                               axis=0)
      loc_params, disp_params = tf.split(params, 2, axis=-1)
      if dispersion == 'single':
        disp_params = tf.reduce_mean(disp_params)
      elif dispersion == 'share':
        disp_params = tf.reduce_mean(disp_params,
                                     axis=tf.range(0,
                                                   output_shape.shape[0] - 1,
                                                   dtype='int32'),
                                     keepdims=True)
      if given_log_mean:
        loc_params = tf.exp(loc_params, name='loc')
      if given_log_disp:
        disp_params = tf.exp(disp_params, name='disp')
      return tfd.Independent(
          NegativeBinomialDisp(loc=tf.reshape(loc_params, output_shape),
                               disp=tf.reshape(disp_params, output_shape)
                               if dispersion == 'full' else disp_params,
                               validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'NegativeBinomialDisp_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape,
                             name=name or 'NegativeBinomialDisp_params_size')


# ===========================================================================
# Zero inflated
# ===========================================================================
class ZIPoissonLayer(DistributionLambda):
  """A Independent zero-inflated Poisson keras layer
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZIPoissonLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'ZIPoissonLayer', [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32),
                                               tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ],
                               axis=0)
      (log_rate_params, logits_params) = tf.split(params, 2, axis=-1)
      zip = ZeroInflated(count_distribution=tfd.Poisson(
          log_rate=tf.reshape(log_rate_params, output_shape),
          validate_args=validate_args),
                         logits=tf.reshape(logits_params, output_shape),
                         validate_args=validate_args)
      return tfd.Independent(
          zip,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name,
                                 'ZeroInflatedNegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(
          event_shape, name=name or 'ZeroInflatedNegativeBinomial_params_size')


class ZINegativeBinomialLayer(DistributionLambda):
  """A Independent zero-inflated negative binomial keras layer

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.
  given_log_count : boolean
    is the input representing log count values or the count itself
  dispersion : {'full', 'share', 'single'}
    'full' creates a dispersion value for each individual data point,
    'share' creates a single vector of dispersion for all examples, and
    'single' uses a single value as dispersion for all data points.
  convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
    instance and returns a `tf.Tensor`-like object.
    Default value: `tfd.Distribution.sample`.
  validate_args: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  """

  def __init__(self,
               event_shape=(),
               given_log_count=True,
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZINegativeBinomialLayer,
          self).__init__(lambda t: type(self).new(
              t, event_shape, given_log_count, dispersion, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_count=True,
          dispersion='full',
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'ZINegativeBinomialLayer',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32),
                                               tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ],
                               axis=0)
      ndims = output_shape.shape[0]
      (total_count_params, logits_params, rate_params) = tf.split(params,
                                                                  3,
                                                                  axis=-1)
      if dispersion == 'single':
        logits_params = tf.reduce_mean(logits_params)
      elif dispersion == 'share':
        logits_params = tf.reduce_mean(logits_params,
                                       axis=tf.range(0,
                                                     ndims - 1,
                                                     dtype='int32'),
                                       keepdims=True)
      if given_log_count:
        total_count_params = tf.exp(total_count_params, name='total_count')
      nb = tfd.NegativeBinomial(total_count=tf.reshape(total_count_params,
                                                       output_shape),
                                logits=tf.reshape(logits_params, output_shape)
                                if dispersion == 'full' else logits_params,
                                validate_args=validate_args)
      zinb = ZeroInflated(count_distribution=nb,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
      return tfd.Independent(
          zinb,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name,
                                 'ZeroInflatedNegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 3 * _event_size(
          event_shape, name=name or 'ZeroInflatedNegativeBinomial_params_size')


class ZINegativeBinomialDispLayer(DistributionLambda):
  """A Independent zero-inflated negative binomial (alternative
  parameterization) keras layer

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.
  given_log_mean : boolean
    is the input representing log count values or the count itself
  given_log_disp : boolean
    is the input representing log dispersion values
  dispersion : {'full', 'share', 'single'}
    'full' creates a dispersion value for each individual data point,
    'share' creates a single vector of dispersion for all examples, and
    'single' uses a single value as dispersion for all data points.
  convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
    instance and returns a `tf.Tensor`-like object.
    Default value: `tfd.Distribution.sample`.
  validate_args: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  """

  def __init__(self,
               event_shape=(),
               given_log_mean=True,
               given_log_disp=True,
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZINegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, given_log_mean, given_log_disp,
                                 dispersion, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_mean=True,
          given_log_disp=True,
          dispersion='full',
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'ZINegativeBinomialDispLayer',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32),
                                               tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ],
                               axis=0)
      # splitting the parameters
      (loc_params, disp_params, rate_params) = tf.split(params, 3, axis=-1)
      if dispersion == 'single':
        disp_params = tf.reduce_mean(disp_params)
      elif dispersion == 'share':
        disp_params = tf.reduce_mean(disp_params,
                                     axis=tf.range(0,
                                                   output_shape.shape[0] - 1,
                                                   dtype='int32'),
                                     keepdims=True)
      # as count value, do exp if necessary
      if given_log_mean:
        loc_params = tf.exp(loc_params, name='loc')
      if given_log_disp:
        disp_params = tf.exp(disp_params, name='disp')
      # create the distribution
      nb = NegativeBinomialDisp(loc=tf.reshape(loc_params, output_shape),
                                disp=tf.reshape(disp_params, output_shape)
                                if dispersion == 'full' else disp_params,
                                validate_args=validate_args)
      zinb = ZeroInflated(count_distribution=nb,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
      return tfd.Independent(
          zinb,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'ZINegativeBinomialDisp_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 3 * _event_size(event_shape,
                             name=name or 'ZINegativeBinomialDisp_params_size')
