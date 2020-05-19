from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
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
    'PoissonLayer',
    'NegativeBinomialDispLayer',
    'NegativeBinomialLayer',
    'ZINegativeBinomialDispLayer',
    'ZINegativeBinomialLayer',
    'ZIPoissonLayer',
    'MultinomialLayer',
    'DirichletMultinomialLayer',
    'BinomialLayer',
]

PoissonLayer = tfl.IndependentPoisson


# ===========================================================================
# Negative binomial
# ===========================================================================
def _dispersion(disp, event_shape, is_logits, name, n_components=1):
  dispersion = str(disp).lower().strip()
  assert dispersion in ('full', 'single', 'share'), \
    "Only support three different dispersion value: 'full', 'single' and " + \
      "'share', but given: %s" % dispersion
  disp = None
  if n_components > 1:
    shape_single = (n_components, 1)
    shape_share = tf.concat(
        [[n_components], tf.nest.flatten(event_shape)], axis=0)
  else:
    shape_single = (1,)
    shape_share = tf.nest.flatten(event_shape)
  ######## logits values
  if is_logits:
    if dispersion == 'single':
      disp = tf.Variable(tf.zeros(shape_single),
                         trainable=True,
                         dtype=keras.backend.floatx(),
                         name=f"{name}_logits")
    elif dispersion == 'share':
      disp = tf.Variable(tf.zeros(shape_share),
                         trainable=True,
                         dtype=keras.backend.floatx(),
                         name=f"{name}_logits")
  ######## raw dispersion values
  else:
    if dispersion == 'single':
      disp = tf.Variable(tf.random.normal(shape_single),
                         trainable=True,
                         dtype=keras.backend.floatx(),
                         name=f"{name}_raw")
    elif dispersion == 'share':
      disp = tf.Variable(tf.random.normal(shape_share),
                         trainable=True,
                         dtype=keras.backend.floatx(),
                         name=f"{name}_raw")
  return disp


class NegativeBinomialLayer(DistributionLambda):
  r"""An independent NegativeBinomial Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    count_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures
    dispersion : {'full', 'share', 'single'}
      - 'full' creates a dispersion value for each individual data point,
      - 'share' creates a single vector of dispersion for all examples, and
      - 'single' uses a single value as dispersion for all data points.
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
    disp = _dispersion(dispersion,
                       event_shape,
                       is_logits=True,
                       name="dispersion")
    super(NegativeBinomialLayer, self).__init__(
        lambda t: type(self).new(
            t,
            event_shape,
            count_activation=parse_activation(count_activation, self),
            validate_args=validate_args,
            disp=disp,
        ), convert_to_tensor_fn, **kwargs)
    self.disp = disp

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.exp,
          validate_args=False,
          name="NegativeBinomialLayer",
          disp=None):
    r"""Create the distribution instance from a `params` vector."""
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
    if disp is None:
      total_count, logits = tf.split(params, 2, axis=-1)
      logits = tf.reshape(logits, output_shape)
    else:
      total_count = params
      logits = disp
    total_count = tf.reshape(total_count, output_shape)
    total_count = count_activation(total_count)
    return tfd.Independent(
        tfd.NegativeBinomial(total_count=total_count,
                             logits=logits,
                             validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(),
                  dispersion='full',
                  name="NegativeBinomialLayer_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    if dispersion == 'full':
      return 2 * _event_size(event_shape, name=name)
    return _event_size(event_shape, name=name)


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
    disp = _dispersion(dispersion,
                       event_shape,
                       is_logits=False,
                       name="dispersion")
    super(NegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(
            t,
            event_shape,
            mean_activation=parse_activation(mean_activation, self),
            disp_activation=parse_activation(disp_activation, self),
            validate_args=validate_args,
            disp=disp,
        ), convert_to_tensor_fn, **kwargs)
    self.disp = disp

  @staticmethod
  def new(params,
          event_shape=(),
          mean_activation=tf.nn.softplus,
          disp_activation=softplus1,
          validate_args=False,
          name="NegativeBinomialDispLayer",
          disp=None):
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
    if disp is None:
      loc, disp = tf.split(params, 2, axis=-1)
      disp = tf.reshape(disp, output_shape)
    else:
      loc = params
    loc = tf.reshape(loc, output_shape)
    loc = mean_activation(loc)
    disp = disp_activation(disp)
    return tfd.Independent(
        NegativeBinomialDisp(loc=loc, disp=disp, validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(),
                  dispersion='full',
                  name="NegativeBinomialDispLayer_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    if dispersion == 'full':
      return 2 * _event_size(event_shape, name=name)
    return _event_size(event_shape, name=name)


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
    (log_rate_params, logits_params) = tf.split(params, 2, axis=-1)
    return tfd.Independent(
        ZeroInflated(count_distribution=tfd.Poisson(
            log_rate=activation(tf.reshape(log_rate_params, output_shape)),
            validate_args=validate_args),
                     logits=tf.reshape(logits_params, output_shape),
                     validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name="ZeroInflatedPoisson_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)


class ZINegativeBinomialLayer(DistributionLambda):
  r"""A Independent zero-inflated negative binomial keras layer

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    count_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures
    dispersion, inflation : {'full', 'share', 'single'}
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
               inflation='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    disp = _dispersion(dispersion,
                       event_shape,
                       is_logits=True,
                       name="dispersion")
    rate = _dispersion(inflation, event_shape, is_logits=True, name="inflation")
    super(ZINegativeBinomialLayer, self).__init__(
        lambda t: type(self).new(
            t,
            event_shape,
            count_activation=parse_activation(count_activation, self),
            validate_args=validate_args,
            disp=disp,
            rate=rate,
        ), convert_to_tensor_fn, **kwargs)
    self.disp = disp
    self.rate = rate

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.exp,
          validate_args=False,
          name="ZINegativeBinomialLayer",
          disp=None,
          rate=None):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat((tf.shape(input=params)[:-1], event_shape), axis=0)
    if disp is None:  # full dispersion
      if rate is None:
        total_count, logits, rate = tf.split(params, 3, axis=-1)
        rate = tf.reshape(rate, output_shape)
      else:
        total_count, logits = tf.split(params, 2, axis=-1)
      logits = tf.reshape(logits, output_shape)
    else:  # share dispersion
      if rate is None:
        total_count, rate = tf.split(params, 2, axis=-1)
        rate = tf.reshape(rate, output_shape)
      else:
        total_count = params
      logits = disp
    total_count = tf.reshape(total_count, output_shape)
    total_count = count_activation(total_count)
    nb = tfd.NegativeBinomial(total_count=total_count,
                              logits=logits,
                              validate_args=validate_args)
    zinb = ZeroInflated(count_distribution=nb,
                        logits=rate,
                        validate_args=validate_args)
    return tfd.Independent(zinb,
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           name=name)

  @staticmethod
  def params_size(event_shape=(),
                  dispersion='full',
                  inflation='full',
                  name="ZeroInflatedNegativeBinomial_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    size = _event_size(event_shape, name=name)
    total = 3 * size
    if dispersion != 'full':
      total -= size
    if inflation != 'full':
      total -= size
    return total


class ZINegativeBinomialDispLayer(DistributionLambda):
  r"""A Independent zero-inflated negative binomial (alternative
  parameterization) keras layer.

  The order of input parameters are: mean, dispersion, dropout rate

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    mean_activation : activation for the non-negative mean
    disp_activation : activation for the non-negative dispersion
    dispersion, inflation : {'full', 'share', 'single'}
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
               inflation='full',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    disp = _dispersion(dispersion,
                       event_shape,
                       is_logits=True,
                       name="dispersion")
    rate = _dispersion(inflation, event_shape, is_logits=True, name="inflation")
    super(ZINegativeBinomialDispLayer, self).__init__(
        lambda t: type(self).new(
            t,
            event_shape,
            mean_activation=parse_activation(mean_activation, self),
            disp_activation=parse_activation(disp_activation, self),
            disp=disp,
            rate=rate,
            validate_args=validate_args,
        ), convert_to_tensor_fn, **kwargs)
    self.disp = disp
    self.rate = rate

  @staticmethod
  def new(params,
          event_shape=(),
          mean_activation=tf.nn.softplus,
          disp_activation=softplus1,
          validate_args=False,
          name="ZINegativeBinomialDispLayer",
          disp=None,
          rate=None):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat((tf.shape(input=params)[:-1], event_shape), axis=0)
    ### splitting the parameters
    if disp is None:  # full dispersion
      if rate is None:
        loc, disp, rate = tf.split(params, 3, axis=-1)
        rate = tf.reshape(rate, output_shape)
      else:
        loc, disp = tf.split(params, 2, axis=-1)
      disp = tf.reshape(disp, output_shape)
    else:  # share dispersion
      if rate is None:
        loc, rate = tf.split(params, 2, axis=-1)
        rate = tf.reshape(rate, output_shape)
      else:
        loc = params
    # as count value, do exp if necessary
    loc = tf.reshape(loc, output_shape)
    loc = mean_activation(loc)
    disp = disp_activation(disp)
    # create the distribution
    nb = NegativeBinomialDisp(loc=loc, disp=disp, validate_args=validate_args)
    zinb = ZeroInflated(count_distribution=nb,
                        logits=rate,
                        validate_args=validate_args)
    return tfd.Independent(zinb,
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           name=name)

  @staticmethod
  def params_size(event_shape=(),
                  dispersion='full',
                  inflation='full',
                  name="ZINegativeBinomialDisp_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    size = _event_size(event_shape, name=name)
    total = 3 * size
    if dispersion != 'full':
      total -= size
    if inflation != 'full':
      total -= size
    return total


# ===========================================================================
# Binomial Multinomial layer
# ===========================================================================
class MultinomialLayer(tfl.DistributionLambda):

  def __init__(self,
               event_shape=(),
               count_activation='softplus',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False):
    super().__init__(
        lambda t: MultinomialLayer.new(t, event_shape, count_activation,
                                       validate_args), convert_to_tensor_fn)

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.nn.softplus,
          validate_args=False,
          name='MultinomialLayer'):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    count_activation = parse_activation(count_activation, 'tf')
    total_count = count_activation(params[..., 0])
    logits = params[..., 1:]
    return tfd.Multinomial(total_count=total_count,
                           logits=logits,
                           validate_args=validate_args,
                           name=name)

  @staticmethod
  def params_size(event_shape=(), name='MultinomialLayer_params_size'):
    r"""The number of `params` needed to create a single distribution."""
    return _event_size(event_shape, name=name) + 1.


class DirichletMultinomialLayer(tfl.DistributionLambda):
  r""" Dirichlet-Multinomial compound distribution.

  K=2 equal to Beta-Binomial distribution
  """

  def __init__(self,
               event_shape=(),
               count_activation='softplus',
               alpha_activation='softplus',
               clip_for_stable=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False):
    super().__init__(
        lambda t: DirichletMultinomialLayer.
        new(t, event_shape, count_activation, alpha_activation, clip_for_stable,
            validate_args), convert_to_tensor_fn)

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.nn.softplus,
          alpha_activation=tf.nn.softplus,
          clip_for_stable=True,
          validate_args=False,
          name='DirichletMultinomialLayer'):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    count_activation = parse_activation(count_activation, 'tf')
    alpha_activation = parse_activation(alpha_activation, 'tf')
    total_count = count_activation(params[..., 0])
    concentration = alpha_activation(params[..., 1:])
    if clip_for_stable:
      concentration = tf.clip_by_value(concentration, 1e-3, 1e3)
    return tfd.DirichletMultinomial(total_count=total_count,
                                    concentration=concentration,
                                    validate_args=validate_args,
                                    name=name)

  @staticmethod
  def params_size(event_shape=(), name='DirichletMultinomialLayer_params_size'):
    r"""The number of `params` needed to create a single distribution."""
    return _event_size(event_shape, name=name) + 1.


class BinomialLayer(tfl.DistributionLambda):
  r""" Binomial distribution, each entry is a flipping of the coin K times (
    parameterized by `total_count` """

  def __init__(self,
               event_shape=(),
               count_activation='softplus',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False):
    super().__init__(
        lambda t: BinomialLayer.new(t, event_shape, count_activation,
                                    validate_args), convert_to_tensor_fn)

  @staticmethod
  def new(params,
          event_shape=(),
          count_activation=tf.nn.softplus,
          validate_args=False,
          name='BinomialLayer'):
    r"""Create the distribution instance from a `params` vector."""
    count_activation = parse_activation(count_activation, 'tf')
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat((tf.shape(params)[:-1], event_shape), axis=0)
    total_count, logits = tf.split(params, 2, axis=-1)
    total_count = tf.reshape(total_count, output_shape)
    logits = tf.reshape(logits, output_shape)
    return tfd.Independent(
        tfd.Binomial(total_count=count_activation(total_count),
                     logits=logits,
                     validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name='BinomialLayer_params_size'):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)
