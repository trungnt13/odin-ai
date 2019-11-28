from __future__ import absolute_import, division, print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
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

from odin.backend import parse_activation
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'DistributionLambda', 'MultivariateNormalLayer', 'DeterministicLayer',
    'VectorDeterministicLayer', 'GammaLayer', 'BetaLayer', 'DirichletLayer',
    'GaussianLayer', 'NormalLayer', 'LogNormalLayer', 'LogisticLayer',
    'update_convert_to_tensor_fn'
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
class DeterministicLayer(DistributionLambda):
  """
  ```none
  pmf(x; loc) = 1, if x == loc, else 0
  cdf(x; loc) = 1, if x >= loc, else 0
  ```
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(DeterministicLayer,
          self).__init__(lambda t: type(self).new(t, validate_args),
                         convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    return tfd.Deterministic(loc=params, validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """ The number of `params` needed to create a single distribution. """
    return event_size


class VectorDeterministicLayer(DistributionLambda):
  """
  ```none
  pmf(x; loc)
    = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
    = 0, otherwise.
  ```
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(VectorDeterministicLayer,
          self).__init__(lambda t: type(self).new(t, validate_args),
                         convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    return tfd.VectorDeterministic(loc=params, validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """ The number of `params` needed to create a single distribution. """
    return event_size


class DirichletLayer(DistributionLambda):
  r"""
  Arguments:
    pre_softplus : bool (default: False)
      applying softplus activation on the parameters before parameterizing
    clip_for_stable : bool (default: True)
      clipping the concentration into range [1e-3, 1e3] for stability
  """

  def __init__(self,
               event_shape='auto',
               pre_softplus=False,
               clip_for_stable=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(DirichletLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, pre_softplus, clip_for_stable,
                                 validate_args), convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape='auto',
          pre_softplus=False,
          clip_for_stable=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    event_shape = _preprocess_eventshape(params, event_shape)
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        event_shape,
    ],
                             axis=0)
    # Clips the Dirichlet parameters to the numerically stable KL region
    if pre_softplus:
      params = tf.nn.softplus(params)
    if clip_for_stable:
      params = tf.clip_by_value(params, 1e-3, 1e3)
    return tfd.Independent(tfd.Dirichlet(concentration=tf.reshape(
        params, output_shape),
                                         validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return _event_size(event_shape, name=name or 'DirichletLayer_params_size')


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
               scale_activation=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(GaussianLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, scale_activation, validate_args
                                ), convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, event_shape, scale_activation, validate_args, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        event_shape,
    ],
                             axis=0)
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    scale_params = parse_activation(scale_activation, 'tf')(scale_params)
    return tfd.Independent(tfd.Normal(loc=tf.reshape(loc_params, output_shape),
                                      scale=tf.reshape(scale_params,
                                                       output_shape),
                                      validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape,
                           name=name or 'GaussianLayer_params_size')


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
               scale_activation='softplus1',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(LogNormalLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, scale_activation, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          scale_activation='softplus1',
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        event_shape,
    ],
                             axis=0)
    loc_params, scale_params = tf.split(params, 2, axis=-1)
    scale_params = parse_activation(scale_activation, 'tf')(scale_params)
    return tfd.Independent(tfd.LogNormal(loc=tf.reshape(loc_params,
                                                        output_shape),
                                         scale=tf.reshape(
                                             scale_params, output_shape),
                                         validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf1.name_scope(name, 'LogNormal_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'LogNormal_params_size')


class GammaLayer(DistributionLambda):
  r"""An independent Gamma Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
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
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(GammaLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        event_shape,
    ],
                             axis=0)
    concentration_params, rate_params = tf.split(params, 2, axis=-1)
    return tfd.Independent(tfd.Gamma(concentration=tf.reshape(
        concentration_params, output_shape),
                                     rate=tf.reshape(rate_params, output_shape),
                                     validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape, name=name or 'Gamma_params_size')


class BetaLayer(DistributionLambda):
  r"""An independent Beta Keras layer.

  Arguments:
    event_shape: integer vector `Tensor` representing the shape of single
      draw from this distribution.
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
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(BetaLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        event_shape,
    ],
                             axis=0)
    concentration1_params, concentration0_params = tf.split(params, 2, axis=-1)
    return tfd.Independent(tfd.Beta(
        concentration1=tf.reshape(concentration1_params, output_shape),
        concentration0=tf.reshape(concentration0_params, output_shape),
        validate_args=validate_args),
                           reinterpreted_batch_ndims=tf.size(input=event_shape),
                           validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    event_shape = tf.convert_to_tensor(value=event_shape,
                                       name='event_shape',
                                       dtype=tf.int32)
    return 2 * _event_size(event_shape, name=name or 'Beta_params_size')


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
    covariance_type : {'diag', 'tril', 'full'}
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
               event_size,
               covariance_type='diag',
               scale_activation='softplus1',
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               **kwargs):
    super(MultivariateNormalLayer, self).__init__(
        lambda t: type(self).new(t, event_size, covariance_type,
                                 scale_activation, validate_args),
        convert_to_tensor_fn, **kwargs)

  @staticmethod
  def new(params,
          event_size,
          covariance_type,
          scale_activation,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    covariance_type = str(covariance_type).lower().strip()
    assert covariance_type in ('full', 'tril', 'diag'), \
    "No support for given covariance_type: '%s'" % covariance_type
    scale_fn = parse_activation(scale_activation, 'tf')
    params = tf.convert_to_tensor(value=params, name='params')
    if covariance_type == 'tril':
      scale_tril = tfb.ScaleTriL(diag_shift=np.array(
          1e-5, params.dtype.as_numpy_dtype()),
                                 validate_args=validate_args)
      return tfd.MultivariateNormalTriL(loc=params[..., :event_size],
                                        scale_tril=scale_tril(
                                            scale_fn(params[..., event_size:])),
                                        validate_args=validate_args)
    elif covariance_type == 'diag':
      return tfd.MultivariateNormalDiag(loc=params[..., :event_size],
                                        scale_diag=scale_fn(
                                            params[..., event_size:]),
                                        validate_args=validate_args)
    elif covariance_type == 'full':
      return tfd.MultivariateNormalFullCovariance(
          loc=params[..., :event_size],
          covariance_matrix=tf.reshape(scale_fn(params[..., event_size:]),
                                       (event_size, event_size)),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, covariance_type='diag', name=None):
    """The number of `params` needed to create a single distribution."""
    covariance_type = str(covariance_type).lower().strip()
    assert covariance_type in ('full', 'tril', 'diag'), \
    "No support for given covariance_type: '%s'" % covariance_type
    if covariance_type == 'tril':
      return event_size + event_size * (event_size + 1) // 2
    elif covariance_type == 'diag':
      return event_size + event_size
    elif covariance_type == 'full':
      return event_size + event_size * event_size


# ===========================================================================
# Shortcut
# ===========================================================================
NormalLayer = GaussianLayer
