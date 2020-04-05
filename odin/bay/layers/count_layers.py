from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import _event_size

from odin.backend import parse_activation
from odin.backend.maths import softplus1
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'PoissonLayer', 'NegativeBinomialDispLayer', 'NegativeBinomialLayer',
    'ZINegativeBinomialDispLayer', 'ZINegativeBinomialLayer', 'ZIPoissonLayer'
]

PoissonLayer = tfl.IndependentPoisson


class NegativeBinomialLayer(DistributionLambda):
  r"""An independent NegativeBinomial Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    count_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures
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
               count_activation='exp',
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    dispersion = str(dispersion).lower()
    assert dispersion in ('full', 'single', 'share'), \
      "Only support three different dispersion value: 'full', 'single' and " + \
        "'share', but given: %s" % dispersion
    super(NegativeBinomialLayer, self).__init__(
        lambda t: type(self).new(t, event_shape,
                                 parse_activation(count_activation, self),
                                 dispersion, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.exp,
          dispersion='full',
          validate_args=False,
          name="NegativeBinomialLayer"):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
        event_shape,
    ],
                             axis=0)
    ndims = output_shape.shape[0]
    #
    total_count_params, logits_params = tf.split(params, 2, axis=-1)
    if dispersion == 'single':
      logits_params = tf.reduce_mean(logits_params)
    elif dispersion == 'share':
      logits_params = tf.reduce_mean(logits_params,
                                     axis=tf.range(0, ndims - 1, dtype='int32'),
                                     keepdims=True)
    total_count_params = count_activation(total_count_params)
    return tfd.Independent(tfd.NegativeBinomial(
        total_count=tf.reshape(total_count_params, output_shape),
        logits=tf.reshape(logits_params, output_shape)
        if dispersion == 'full' else logits_params,
        validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(event_shape=(), name="NegativeBinomial_params_size"):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape, name=name)


class NegativeBinomialDispLayer(DistributionLambda):
  r"""An alternative parameterization of the NegativeBinomial Keras layer.

  The order of input parameters are: mean, dispersion

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    mean_activation : activation for the non-negative mean
    disp_activation : activation for the non-negative dispersion
    dispersion : {'full', 'share', 'single'}
      'full' creates a dispersion value for each individual data point,
      'share' creates a single dispersion vector of `event_shape` for all examples,
      and 'single' uses a single value as dispersion for all data points.
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
               mean_activation='softplus',
               disp_activation='softplus1',
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    dispersion = str(dispersion).lower()
    self.dispersion = dispersion
    assert dispersion in ('full', 'single', 'share'), \
      "Only support three different dispersion value: 'full', 'single' and " + \
        "'share', but given: %s" % dispersion
    super(NegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(mean_activation, self),
            parse_activation(disp_activation, self), dispersion, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          mean_activation=tf.nn.softplus,
          disp_activation=softplus1,
          dispersion='full',
          validate_args=False,
          name="NegativeBinomialDispLayer"):
    """ Create the distribution instance from a `params` vector. """
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
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
    loc_params = mean_activation(loc_params)
    disp_params = disp_activation(disp_params)
    return tfd.Independent(NegativeBinomialDisp(
        loc=tf.reshape(loc_params, output_shape),
        disp=tf.reshape(disp_params, output_shape)
        if dispersion == 'full' else disp_params,
        validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(event_shape=(), name="NegativeBinomialDisp_params_size"):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape, name=name)


# ===========================================================================
# Zero inflated
# ===========================================================================
class ZIPoissonLayer(DistributionLambda):
  r"""A Independent zero-inflated Poisson keras layer """

  def __init__(self,
               event_shape=(),
               activation='linear',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(ZIPoissonLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(activation, self), validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          activation=tf.identity,
          validate_args=False,
          name="ZIPoissonLayer"):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
        event_shape,
    ],
                             axis=0)
    (log_rate_params, logits_params) = tf.split(params, 2, axis=-1)
    return tfd.Independent(ZeroInflated(count_distribution=tfd.Poisson(
        log_rate=activation(tf.reshape(log_rate_params, output_shape)),
        validate_args=validate_args),
                                        logits=tf.reshape(
                                            logits_params, output_shape),
                                        validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(event_shape=(), name="ZeroInflatedPoisson_params_size"):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape, name=name)


class ZINegativeBinomialLayer(DistributionLambda):
  r"""A Independent zero-inflated negative binomial keras layer

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    count_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures
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
               count_activation='exp',
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(ZINegativeBinomialLayer, self).__init__(
        lambda t: type(self).new(t, event_shape,
                                 parse_activation(count_activation, self),
                                 dispersion, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.exp,
          dispersion='full',
          validate_args=False,
          name="ZINegativeBinomialLayer"):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
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
                                     axis=tf.range(0, ndims - 1, dtype='int32'),
                                     keepdims=True)
    total_count_params = count_activation(total_count_params)
    nb = tfd.NegativeBinomial(total_count=tf.reshape(total_count_params,
                                                     output_shape),
                              logits=tf.reshape(logits_params, output_shape)
                              if dispersion == 'full' else logits_params,
                              validate_args=validate_args)
    zinb = ZeroInflated(count_distribution=nb,
                        logits=tf.reshape(rate_params, output_shape),
                        validate_args=validate_args)
    return tfd.Independent(zinb,
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(
      event_shape=(), name="ZeroInflatedNegativeBinomial_params_size"):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 3 * _event_size(event_shape, name=name)


class ZINegativeBinomialDispLayer(DistributionLambda):
  r"""A Independent zero-inflated negative binomial (alternative
  parameterization) keras layer.

  The order of input parameters are: mean, dispersion, dropout rate

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    mean_activation : activation for the non-negative mean
    disp_activation : activation for the non-negative dispersion
    dispersion : {'full', 'share', 'single'}
      'full' creates a dispersion value for each individual data point,
      'share' creates a single dispersion vector of `event_shape` for all examples,
      and 'single' uses a single value as dispersion for all data points.
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
               mean_activation='softplus',
               disp_activation='softplus1',
               dispersion='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    self.dispersion = dispersion
    super(ZINegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(mean_activation, self),
            parse_activation(disp_activation, self), dispersion, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          mean_activation=tf.nn.softplus,
          disp_activation=softplus1,
          dispersion='full',
          validate_args=False,
          name="ZINegativeBinomialDispLayer"):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-tf.size(event_shape)],
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
    loc_params = mean_activation(loc_params)
    disp_params = disp_activation(disp_params)
    # create the distribution
    nb = NegativeBinomialDisp(loc=tf.reshape(loc_params, output_shape),
                              disp=tf.reshape(disp_params, output_shape)
                              if dispersion == 'full' else disp_params,
                              validate_args=validate_args)
    zinb = ZeroInflated(count_distribution=nb,
                        logits=tf.reshape(rate_params, output_shape),
                        validate_args=validate_args)
    return tfd.Independent(zinb,
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(event_shape=(), name="ZINegativeBinomialDisp_params_size"):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 3 * _event_size(event_shape, name=name)
