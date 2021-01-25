from __future__ import absolute_import, division, print_function

import types

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from odin.backend import parse_activation
from odin.backend.maths import softplus1
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated
from six import string_types
from tensorflow.python.keras.utils import tf_utils as keras_tf_utils
# By importing `distributions` as `tfd`, docstrings will show
# `tfd.Distribution`. We import `bijectors` the same way, for consistency.
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers.distribution_layer import _event_size
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc

__all__ = [
    'DistributionLambda',
    'MultivariateNormalLayer',
    'GammaLayer',
    'BetaLayer',
    'DirichletLayer',
    'GaussianLayer',
    'NormalLayer',
    'LogNormalLayer',
    'LogisticLayer',
    'update_convert_to_tensor_fn',
]

DistributionLambda = tfl.DistributionLambda
LogisticLayer = tfl.IndependentLogistic


# ===========================================================================
# Helper
# ===========================================================================
def update_convert_to_tensor_fn(dist, fn):
  assert isinstance(dist, dtc._TensorCoercible), \
  "dist must be output from tfd.DistributionLambda"
  assert callable(fn), "fn must be callable"
  if isinstance(fn, property):
    fn = fn.fget
  dist._concrete_value = None
  dist._convert_to_tensor_fn = fn
  return dist


def _preprocess_eventshape(params, event_shape, n_dims=1):
  if isinstance(event_shape, string_types):
    if event_shape.lower().strip() == 'auto':
      event_shape = params.shape[-n_dims:]
    else:
      raise ValueError("Not support for event_shape='%s'" % event_shape)
  return event_shape


# ===========================================================================
# Simple distribution
# ===========================================================================
class BetaLayer(DistributionLambda):
  r"""An independent Beta Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    concentration_activation : activation function for `concentration1`,
      must return only positive values.
    beta_activation : activation function for `concentration0`,
      must return only positive values.
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
               concentration_activation='softplus1',
               beta_activation='softplus1',
               clip_for_stable=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(BetaLayer, self).__init__(
        lambda t: type(self).
        new(t, event_shape, concentration_activation, beta_activation,
            clip_for_stable, validate_args), convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          concentration_activation=softplus1,
          beta_activation=softplus1,
          clip_for_stable=True,
          validate_args=False,
          name="BetaLayer"):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    concentration_activation = parse_activation(concentration_activation, 'tf')
    beta_activation = parse_activation(beta_activation, 'tf')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat((tf.shape(input=params)[:-1], event_shape), axis=0)
    # alpha, beta
    concentration1, concentration0 = tf.split(params, 2, axis=-1)
    #
    concentration1 = tf.reshape(concentration1, output_shape)
    concentration1 = concentration_activation(concentration1)
    #
    concentration0 = tf.reshape(concentration0, output_shape)
    concentration0 = beta_activation(concentration0)
    if clip_for_stable:
      concentration0 = tf.clip_by_value(concentration0, 1e-3, 1e3)
      concentration1 = tf.clip_by_value(concentration1, 1e-3, 1e3)
    return tfd.Independent(
        tfd.Beta(concentration1=concentration1,
                 concentration0=concentration0,
                 validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name='BetaLayer_params_size'):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)


class DirichletLayer(DistributionLambda):
  r"""   The Dirichlet distribution is defined over the
  [`(k-1)`-simplex](https://en.wikipedia.org/wiki/Simplex) using a positive,
  length-`k` vector `concentration` (`k > 1`). The Dirichlet is identically the
  Beta distribution when `k = 2`.

  Arguments:
    concentration_activation: activation function return positive floating-point `Tensor`
      indicating mean number of class occurrences; aka "alpha"
    clip_for_stable : bool (default: True)
      clipping the concentration into range [1e-3, 1e3] for stability
  """

  def __init__(self,
               event_shape=(),
               concentration_activation='softplus1',
               concentration_clip=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(DirichletLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, concentration_activation,
                                 concentration_clip, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          concentration_activation=softplus1,
          concentration_clip=True,
          validate_args=False,
          name="DirichletLayer"):
    r"""Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    # Clips the Dirichlet parameters to the numerically stable KL region
    concentration_activation = parse_activation(concentration_activation, 'tf')
    params = concentration_activation(params)
    if concentration_clip:
      params = tf.clip_by_value(params, 1e-3, 1e3)
    return tfd.Dirichlet(concentration=params,
                         validate_args=validate_args,
                         name=name)

  @staticmethod
  def params_size(event_shape=(), name='DirichletLayer_params_size'):
    r"""The number of `params` needed to create a single distribution."""
    return _event_size(event_shape, name=name)


class GaussianLayer(DistributionLambda):
  r"""An independent normal Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    scale_activation : activation function for scale parameters, default:
      `softplus1(x) = softplus(x) + softplus_inverse(1.0)`
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object.
      Default value: `tfd.Distribution.sample`.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs. Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
  """

  def __init__(self,
               event_shape=(),
               loc_activation='linear',
               scale_activation='softplus1',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(GaussianLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(loc_activation, self),
            parse_activation(scale_activation, self), validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape,
          loc_activation,
          scale_activation,
          validate_args,
          name="GaussianLayer"):
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
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    loc_params = tf.reshape(loc_activation(loc_params), output_shape)
    scale_params = tf.reshape(scale_activation(scale_params), output_shape)
    return tfd.Independent(
        tfd.Normal(loc=loc_params,
                   scale=scale_params,
                   validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name="GaussianLayer_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)


class LogNormalLayer(DistributionLambda):
  r"""An independent LogNormal Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    scale_activation : activation function for scale parameters, default:
      `softplus1(x) = softplus(x) + softplus_inverse(1.0)`
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
               loc_activation='linear',
               scale_activation='softplus1',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(LogNormalLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(loc_activation, self),
            parse_activation(scale_activation, self), validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape,
          loc_activation,
          scale_activation,
          validate_args=False,
          name="LogNormalLayer"):
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
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    loc_params = tf.reshape(loc_activation(loc_params), output_shape)
    scale_params = tf.reshape(scale_activation(scale_params), output_shape)
    return tfd.Independent(
        tfd.LogNormal(loc=loc_params,
                      scale=scale_params,
                      validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name="LogNormal_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)


class GammaLayer(DistributionLambda):
  r"""An independent Gamma Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object.
      Default value: `tfd.Distribution.sample`.
    concentration_activation : activation function return positive values.
    rate_activation : activation function return positive values.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               concentration_activation='softplus1',
               rate_activation='softplus1',
               validate_args=False,
               **kwargs):
    super(GammaLayer, self).__init__(
        lambda t: type(self).new(
            t, event_shape, parse_activation(concentration_activation, self),
            parse_activation(rate_activation, self), validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape,
          concentration_activation=softplus1,
          rate_activation=softplus1,
          validate_args=False,
          name="GammaLayer"):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(
        tf.convert_to_tensor(value=event_shape,
                             name='event_shape',
                             dtype=tf.int32),
        tensor_name='event_shape',
    )
    output_shape = tf.concat((tf.shape(input=params)[:-1], event_shape), axis=0)
    concentration, rate = tf.split(params, 2, axis=-1)
    concentration = tf.reshape(concentration, output_shape)
    concentration = concentration_activation(concentration)
    rate = tf.reshape(rate, output_shape)
    rate = rate_activation(rate)
    return tfd.Independent(
        tfd.Gamma(concentration=concentration,
                  rate=rate,
                  validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        name=name,
    )

  @staticmethod
  def params_size(event_shape=(), name="GammaLayer_params_size"):
    r"""The number of `params` needed to create a single distribution."""
    return 2 * _event_size(event_shape, name=name)


# ===========================================================================
# Multivariate distribution
# ===========================================================================
class MultivariateNormalLayer(DistributionLambda):
  r"""A `d`-variate Multivariate Normal distribution Keras layer:

  Different covariance mode:
   - tril (lower triangle): `d + d * (d + 1) // 2` params.
   - diag (diagonal) : `d + d` params.
   - full (full) : `d + d * d` params.

  Arguments:
    event_size: Scalar `int` representing the size of single draw from this
      distribution.
    covariance : {'diag', 'tril', 'full'}
    loc_activation : activation function for loc (a.k.a mean), default:
      'identity'
    scale_activation : activation function for scale, default:
      `softplus1(x) = softplus(x) + softplus_inverse(1.0)`
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object. For examples, see
      `class` docstring.
      Default value: `tfd.Distribution.sample`.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.
  """

  def __init__(self,
               event_shape,
               covariance='diag',
               loc_activation=None,
               scale_activation=None,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(MultivariateNormalLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, covariance, loc_activation,
                                 scale_activation, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape,
          covariance,
          loc_activation=None,
          scale_activation=None,
          validate_args=False,
          name=None):
    r"""Create the distribution instance from a `params` vector."""
    covariance = str(covariance).lower().strip()
    event_size = tf.reduce_prod(event_shape)
    assert covariance in ('full', 'tril', 'diag'), \
      f"No support for given covariance: '{covariance}'"
    if name is None:
      name = f"MultivariateNormal{covariance.capitalize()}"
    # parameters
    params = tf.convert_to_tensor(value=params, name='params')
    loc = params[..., :event_size]
    if loc_activation is not None:
      loc = loc_activation(loc)
    scale = params[..., event_size:]
    ### the distribution
    if covariance == 'tril':
      scale_tril = tfb.FillScaleTriL(
          diag_bijector=scale_activation,
          diag_shift=np.array(1e-5, params.dtype.as_numpy_dtype()),
          validate_args=validate_args,
      )
      return tfd.MultivariateNormalTriL(loc=loc,
                                        scale_tril=scale_tril(scale),
                                        validate_args=validate_args,
                                        name=name)
    elif covariance == 'diag':
      if scale_activation is None:
        scale = tf.nn.softplus(scale)
      return tfd.MultivariateNormalDiag(loc=loc,
                                        scale_diag=scale,
                                        validate_args=validate_args,
                                        name=name)
    elif covariance == 'full':
      raise NotImplementedError(
          'MVN full covariance is deprecated, '
          'use `scale_tril=tf.linalg.cholesky(covariance_matrix)` instead')

  @staticmethod
  def params_size(event_size, covariance='diag', name=None):
    """The number of `params` needed to create a single distribution."""
    covariance = str(covariance).lower().strip()
    assert covariance in ('full', 'tril', 'diag'), \
    "No support for given covariance: '%s'" % covariance
    if covariance == 'tril':
      return event_size + event_size * (event_size + 1) // 2
    elif covariance == 'diag':
      return event_size + event_size
    elif covariance == 'full':
      raise NotImplementedError(
          'MVN full covariance is deprecated, '
          'use `scale_tril=tf.linalg.cholesky(covariance_matrix)` instead')


# ===========================================================================
# Shortcut
# ===========================================================================
NormalLayer = GaussianLayer
