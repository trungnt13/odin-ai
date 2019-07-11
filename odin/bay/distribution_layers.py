from __future__ import absolute_import, division, print_function

from six import string_types

# Dependency imports
import numpy as np
import tensorflow as tf

# By importing `distributions` as `tfd`, docstrings will show
# `tfd.Distribution`. We import `bijectors` the same way, for consistency.
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import layers as tfl
from tensorflow_probability.python.layers.distribution_layer import _event_size
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.layers.internal import distribution_tensor_coercible as dtc
from tensorflow.python.keras.utils import tf_utils as keras_tf_utils

__all__ = [
    'DistributionLambda',
    'MultivariateNormal',
    'Bernoulli',
    'OneHotCategorical',
    'Poisson',
    'NegativeBinomial',
    'Gamma',
    'Dirichlet',
    'Normal',
    'LogNormal',
    'Logistic',
    'ZeroInflatedPoisson',
    'ZeroInflatedNegativeBinomial',
    'update_convert_to_tensor_fn'
]

DistributionLambda = tfl.DistributionLambda
Bernoulli = tfl.IndependentBernoulli
Poisson = tfl.IndependentPoisson
Logistic = tfl.IndependentLogistic

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
class OneHotCategorical(DistributionLambda):
  """ A `d`-variate OneHotCategorical Keras layer from `d` params.

  Parameters
  ----------
  convert_to_tensor_fn: callable
    that takes a `tfd.Distribution` instance and returns a
    `tf.Tensor`-like object. For examples, see `class` docstring.
    Default value: `tfd.Distribution.sample`.

  sample_dtype: `dtype`
    Type of samples produced by this distribution.
    Default value: `None` (i.e., previous layer's `dtype`).

  validate_args: `bool` (default `False`)
    When `True` distribution parameters are checked for validity
    despite possibly degrading runtime performance.
    When `False` invalid inputs may silently render incorrect outputs.
    Default value: `False`.

  **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  Note
  ----
  If input as probability values is given, it will be clipped by value
  [1e-8, 1 - 1e-8]

  """

  def __init__(self,
               convert_to_tensor_fn=tfd.Distribution.sample,
               probs_input=False,
               sample_dtype=None,
               activity_regularizer=None,
               validate_args=False,
               **kwargs):
    super(OneHotCategorical, self).__init__(
        lambda t: OneHotCategorical.new(t, probs_input, sample_dtype, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, probs_input=False, dtype=None, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'OneHotCategorical', [params]):
      return tfd.OneHotCategorical(
          logits=params if not probs_input else None,
          probs=tf.clip_by_value(params, 1e-8, 1 - 1e-8) if probs_input else None,
          dtype=dtype or params.dtype.base_dtype,
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size

class Dirichlet(DistributionLambda):

  """
  Parameters
  ----------
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
               activity_regularizer=None,
               validate_args=False,
               **kwargs):
    super(Dirichlet, self).__init__(
        lambda t: type(self).new(
          t, event_shape, pre_softplus, clip_for_stable, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape='auto',
          pre_softplus=False, clip_for_stable=True,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    event_shape = _preprocess_eventshape(params, event_shape)
    with tf.compat.v1.name_scope(name, 'Dirichlet',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      # Clips the Dirichlet parameters to the numerically stable KL region
      if pre_softplus:
        params = tf.nn.softplus(params)
      if clip_for_stable:
        params = tf.clip_by_value(params, 1e-3, 1e3)
      return tfd.Independent(
          tfd.Dirichlet(
              concentration=tf.reshape(params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Dirichlet_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return _event_size(event_shape, name=name or 'Dirichlet_params_size')

class Normal(DistributionLambda):
  """An independent normal Keras layer.

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.

  softplus_scale : bool
    if True, `scale = softplus(params) + softplus_inverse(1.0)`

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
               softplus_scale=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               activity_regularizer=None,
               validate_args=False,
               **kwargs):
    super(Normal, self).__init__(
        lambda t: type(self).new(t, event_shape, softplus_scale, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), softplus_scale=True,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'Normal',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      if softplus_scale:
        scale_params = tf.math.softplus(scale_params) + tfd.softplus_inverse(1.0)
      return tfd.Independent(
          tfd.Normal(
              loc=tf.reshape(loc_params, output_shape),
              scale=tf.reshape(scale_params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Normal_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 2 * _event_size(
          event_shape, name=name or 'Normal_params_size')

class LogNormal(DistributionLambda):
  """An independent LogNormal Keras layer.

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.

  softplus_scale : bool
    if True, `scale = softplus(params) + softplus_inverse(1.0)`

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
               softplus_scale=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(LogNormal, self).__init__(
        lambda t: type(self).new(t, event_shape, softplus_scale, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), softplus_scale=True,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'LogNormal',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      if softplus_scale:
        scale_params = tf.math.softplus(scale_params) + tfd.softplus_inverse(1.0)
      return tfd.Independent(
          tfd.LogNormal(
              loc=tf.reshape(loc_params, output_shape),
              scale=tf.reshape(scale_params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'LogNormal_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'LogNormal_params_size')

class Gamma(DistributionLambda):
  """An independent Gamma Keras layer.

  Parameters
  ----------
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
               activity_regularizer=None,
               **kwargs):
    super(Gamma, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'Gamma',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      concentration_params, rate_params = tf.split(params, 2, axis=-1)
      return tfd.Independent(
          tfd.Gamma(
              concentration=tf.reshape(concentration_params, output_shape),
              rate=tf.reshape(rate_params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Gamma_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'Gamma_params_size')

class NegativeBinomial(DistributionLambda):
  """An independent NegativeBinomial Keras layer.

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.

  given_log_count : boolean
    is the input representing log count values or the count itself

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
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(NegativeBinomial, self).__init__(
        lambda t: type(self).new(t, event_shape, given_log_count, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), given_log_count=True,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'NegativeBinomial',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      total_count_params, logits_params = tf.split(params, 2, axis=-1)
      if given_log_count:
        total_count_params = tf.exp(total_count_params, name='total_count')
      return tfd.Independent(
          tfd.NegativeBinomial(
              total_count=tf.reshape(total_count_params, output_shape),
              logits=tf.reshape(logits_params, output_shape),
              validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'NegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 2 * _event_size(event_shape,
                             name=name or 'NegativeBinomial_params_size')

# ===========================================================================
# Multivariate distribution
# ===========================================================================
class MultivariateNormal(DistributionLambda):
  """A `d`-variate Multivariate Normal distribution Keras layer:

  Different covariance mode:
   - tril (lower triangle): `d + d * (d + 1) // 2` params.
   - diag (diagonal) : `d + d` params.
   - full (full) : `d + d * d` params.

  Typical choices for `convert_to_tensor_fn` include:

  - `tfd.Distribution.sample`
  - `tfd.Distribution.mean`
  - `tfd.Distribution.mode`
  - `lambda s: s.mean() + 0.1 * s.stddev()`

    Parameters
    ----------
    event_size: Scalar `int` representing the size of single draw from this
      distribution.

    covariance_type : {'diag', 'tril', 'full'}

    softplus_scale : bool
      if True, `scale = softplus(params) + softplus_inverse(1.0)`

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
               softplus_scale=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(MultivariateNormal, self).__init__(
        lambda t: type(self).new(t, event_size, covariance_type, softplus_scale, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_size, covariance_type, softplus_scale,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    covariance_type = str(covariance_type).lower().strip()
    assert covariance_type in ('full', 'tril', 'diag'), \
    "No support for given covariance_type: '%s'" % covariance_type

    scale_fn = lambda x: tf.math.softplus(x) + tfd.softplus_inverse(1.0) \
    if bool(softplus_scale) else x

    with tf.compat.v1.name_scope(name, 'MultivariateNormal',
                                 [params, event_size]):
      params = tf.convert_to_tensor(value=params, name='params')

      if covariance_type == 'tril':
        scale_tril = tfb.ScaleTriL(
            diag_shift=np.array(1e-5, params.dtype.as_numpy_dtype()),
            validate_args=validate_args)
        return tfd.MultivariateNormalTriL(
            loc=params[..., :event_size],
            scale_tril=scale_tril(scale_fn(params[..., event_size:])),
            validate_args=validate_args)

      elif covariance_type == 'diag':
        return tfd.MultivariateNormalDiag(
            loc=params[..., :event_size],
            scale_diag=scale_fn(params[..., event_size:]))

      elif covariance_type == 'full':
        return tfd.MultivariateNormalFullCovariance(
            loc=params[..., :event_size],
            covariance_matrix=tf.reshape(scale_fn(params[..., event_size:]),
                                         (event_size, event_size)))

  @staticmethod
  def params_size(event_size, covariance_type, name=None):
    """The number of `params` needed to create a single distribution."""
    covariance_type = str(covariance_type).lower().strip()
    assert covariance_type in ('full', 'tril', 'diag'), \
    "No support for given covariance_type: '%s'" % covariance_type

    with tf.compat.v1.name_scope(name, 'MultivariateNormal_params_size',
                                 [event_size]):
      if covariance_type == 'tril':
        return event_size + event_size * (event_size + 1) // 2
      elif covariance_type == 'diag':
        return event_size + event_size
      elif covariance_type == 'full':
        return event_size + event_size * event_size

# ===========================================================================
# Complex distributions
# ===========================================================================
class ZeroInflatedPoisson(DistributionLambda):
  """A Independent zero-inflated Poisson keras layer
  """

  def __init__(self,
               event_shape=(),
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZeroInflatedPoisson, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    from odin.bay.distributions import ZeroInflated

    with tf.compat.v1.name_scope(name, 'ZeroInflatedPoisson',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      (log_rate_params, logits_params) = tf.split(params, 2, axis=-1)
      zip = ZeroInflated(
          count_distribution=tfd.Poisson(
              log_rate=tf.reshape(log_rate_params, output_shape),
              validate_args=validate_args),
          logits=tf.reshape(logits_params, output_shape),
          validate_args=validate_args)
      return tfd.Independent(zip,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), tied_inflation_rate=False,
                  name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name,
                                 'ZeroInflatedNegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 2 * _event_size(event_shape,
                  name=name or 'ZeroInflatedNegativeBinomial_params_size')

class ZeroInflatedNegativeBinomial(DistributionLambda):
  """A Independent zero-inflated negative binomial keras layer

  Parameters
  ----------
  event_shape: integer vector `Tensor` representing the shape of single
    draw from this distribution.

  given_log_count : boolean
    is the input representing log count values or the count itself

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
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZeroInflatedNegativeBinomial, self).__init__(
        lambda t: type(self).new(t, event_shape, given_log_count, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), given_log_count=True,
          validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    from odin.bay.distributions import ZeroInflated

    with tf.compat.v1.name_scope(name, 'ZeroInflatedNegativeBinomial',
                                 [params, event_shape]):
      params = tf.convert_to_tensor(value=params, name='params')
      event_shape = dist_util.expand_to_vector(
          tf.convert_to_tensor(
              value=event_shape, name='event_shape', dtype=tf.int32),
          tensor_name='event_shape')
      output_shape = tf.concat([
          tf.shape(input=params)[:-1],
          event_shape,
      ], axis=0)
      (total_count_params, logits_params,
       rate_params) = tf.split(params, 3, axis=-1)
      if given_log_count:
        total_count_params = tf.exp(total_count_params, name='total_count')
      nb = tfd.NegativeBinomial(
          total_count=tf.reshape(total_count_params, output_shape),
          logits=tf.reshape(logits_params, output_shape),
          validate_args=validate_args)
      zinb = ZeroInflated(count_distribution=nb,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
      return tfd.Independent(zinb,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), tied_inflation_rate=False,
                  name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name,
                                 'ZeroInflatedNegativeBinomial_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(
          value=event_shape, name='event_shape', dtype=tf.int32)
      return 3 * _event_size(event_shape,
                  name=name or 'ZeroInflatedNegativeBinomial_params_size')
