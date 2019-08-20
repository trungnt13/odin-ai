from __future__ import absolute_import, division, print_function

# Dependency imports
import numpy as np
import tensorflow as tf
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

from odin.bay.distribution_util_layers import *
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'DistributionLambda', 'MultivariateNormalLayer', 'BernoulliLayer',
    'DeterministicLayer', 'OneHotCategoricalLayer', 'GammaLayer',
    'DirichletLayer', 'GaussianLayer', 'NormalLayer', 'LogNormalLayer',
    'LogisticLayer', 'PoissonLayer', 'NegativeBinomialLayer',
    'NegativeBinomialDispLayer', 'ZIPoissonLayer', 'ZINegativeBinomialLayer',
    'ZIBernoulliLayer', 'update_convert_to_tensor_fn'
]

DistributionLambda = tfl.DistributionLambda
BernoulliLayer = tfl.IndependentBernoulli
PoissonLayer = tfl.IndependentPoisson
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
  """Scalar `Deterministic` distribution on the real line.

  The scalar `Deterministic` distribution is parameterized by a [batch] point
  `loc` on the real line.  The distribution is supported at this point only,
  and corresponds to a random variable that is constant, equal to `loc`.

  See [Degenerate rv](https://en.wikipedia.org/wiki/Degenerate_distribution).

  #### Mathematical Details

  The probability mass function (pmf) and cumulative distribution function (cdf)
  are

  ```none
  pmf(x; loc) = 1, if x == loc, else 0
  cdf(x; loc) = 1, if x >= loc, else 0
  ```

  For vectorized version, the probability mass function (pmf) is

  ```none
  pmf(x; loc)
    = 1, if All[Abs(x - loc) <= atol + rtol * Abs(loc)],
    = 0, otherwise.
  ```

  Parameters
  ----------
  vectorized : `bool` (default=False)
    The `VectorDeterministic` distribution is parameterized by a [batch] point
    `loc in R^k`.  The distribution is supported at this point only,
    and corresponds to a random variable that is constant, equal to `loc`.

  """

  def __init__(self,
               vectorized=False,
               convert_to_tensor_fn=tfd.Distribution.sample,
               activity_regularizer=None,
               validate_args=False,
               **kwargs):
    super(DeterministicLayer,
          self).__init__(lambda t: type(self).new(t, vectorized, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params, vectorized, validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'DeterministicLayer', [params]):
      if vectorized:
        return tfd.VectorDeterministic(loc=params, validate_args=validate_args)
      else:
        return tfd.Deterministic(loc=params, validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """ The number of `params` needed to create a single distribution. """
    return event_size


class OneHotCategoricalLayer(DistributionLambda):
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
    super(OneHotCategoricalLayer, self).__init__(
        lambda t: type(self).new(t, probs_input, sample_dtype, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, probs_input=False, dtype=None, validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'OneHotCategoricalLayer', [params]):
      return tfd.OneHotCategorical(
          logits=params if not probs_input else None,
          probs=tf.clip_by_value(params, 1e-8, 1 -
                                 1e-8) if probs_input else None,
          dtype=dtype or params.dtype.base_dtype,
          validate_args=validate_args)

  @staticmethod
  def params_size(event_size, name=None):
    """The number of `params` needed to create a single distribution."""
    return event_size


class DirichletLayer(DistributionLambda):
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
    super(DirichletLayer,
          self).__init__(lambda t: type(self).new(
              t, event_shape, pre_softplus, clip_for_stable, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_shape='auto',
          pre_softplus=False,
          clip_for_stable=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    event_shape = _preprocess_eventshape(params, event_shape)
    with tf.compat.v1.name_scope(name, 'DirichletLayer', [params, event_shape]):
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
      return tfd.Independent(
          tfd.Dirichlet(concentration=tf.reshape(params, output_shape),
                        validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Dirichlet_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return _event_size(event_shape, name=name or 'Dirichlet_params_size')


class GaussianLayer(DistributionLambda):
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
    super(GaussianLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, softplus_scale, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          softplus_scale=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'NormalLayer', [params, event_shape]):
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
      if softplus_scale:
        scale_params = tf.math.softplus(scale_params) + tfd.softplus_inverse(
            1.0)
      return tfd.Independent(
          tfd.Normal(loc=tf.reshape(loc_params, output_shape),
                     scale=tf.reshape(scale_params, output_shape),
                     validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Normal_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'Normal_params_size')


class LogNormalLayer(DistributionLambda):
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
    super(LogNormalLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, softplus_scale, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          softplus_scale=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'LogNormalLayer', [params, event_shape]):
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
      if softplus_scale:
        scale_params = tf.math.softplus(scale_params) + tfd.softplus_inverse(
            1.0)
      return tfd.Independent(
          tfd.LogNormal(loc=tf.reshape(loc_params, output_shape),
                        scale=tf.reshape(scale_params, output_shape),
                        validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'LogNormal_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'LogNormal_params_size')


class GammaLayer(DistributionLambda):
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
    super(GammaLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params, event_shape=(), validate_args=False, name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'GammaLayer', [params, event_shape]):
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
      return tfd.Independent(
          tfd.Gamma(concentration=tf.reshape(concentration_params,
                                             output_shape),
                    rate=tf.reshape(rate_params, output_shape),
                    validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'Gamma_params_size', [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape, name=name or 'Gamma_params_size')


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
      ndims = output_shape.shape[0]

      loc_params, disp_params = tf.split(params, 2, axis=-1)
      if dispersion == 'single':
        disp_params = tf.reduce_mean(disp_params)
      elif dispersion == 'share':
        disp_params = tf.reduce_mean(disp_params,
                                     axis=tf.range(0, ndims - 1, dtype='int32'),
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
# Multivariate distribution
# ===========================================================================
class MultivariateNormalLayer(DistributionLambda):
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
    super(MultivariateNormalLayer,
          self).__init__(lambda t: type(self).new(
              t, event_size, covariance_type, softplus_scale, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_size,
          covariance_type,
          softplus_scale,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    covariance_type = str(covariance_type).lower().strip()
    assert covariance_type in ('full', 'tril', 'diag'), \
    "No support for given covariance_type: '%s'" % covariance_type

    scale_fn = lambda x: tf.math.softplus(x) + tfd.softplus_inverse(1.0) \
    if bool(softplus_scale) else x

    with tf.compat.v1.name_scope(name, 'MultivariateNormalLayer',
                                 [params, event_size]):
      params = tf.convert_to_tensor(value=params, name='params')

      if covariance_type == 'tril':
        scale_tril = tfb.ScaleTriL(diag_shift=np.array(
            1e-5, params.dtype.as_numpy_dtype()),
                                   validate_args=validate_args)
        return tfd.MultivariateNormalTriL(
            loc=params[..., :event_size],
            scale_tril=scale_tril(scale_fn(params[..., event_size:])),
            validate_args=validate_args)

      elif covariance_type == 'diag':
        return tfd.MultivariateNormalDiag(loc=params[..., :event_size],
                                          scale_diag=scale_fn(
                                              params[..., event_size:]))

      elif covariance_type == 'full':
        return tfd.MultivariateNormalFullCovariance(
            loc=params[..., :event_size],
            covariance_matrix=tf.reshape(scale_fn(params[..., event_size:]),
                                         (event_size, event_size)))

  @staticmethod
  def params_size(event_size, covariance_type='diag', name=None):
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
  def params_size(event_shape=(), tied_inflation_rate=False, name=None):
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
    super(ZINegativeBinomialLayer,
          self).__init__(lambda t: type(self).new(
              t, event_shape, given_log_count, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_count=True,
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
      (total_count_params, logits_params, rate_params) = tf.split(params,
                                                                  3,
                                                                  axis=-1)
      if given_log_count:
        total_count_params = tf.exp(total_count_params, name='total_count')
      nb = tfd.NegativeBinomial(total_count=tf.reshape(total_count_params,
                                                       output_shape),
                                logits=tf.reshape(logits_params, output_shape),
                                validate_args=validate_args)
      zinb = ZeroInflated(count_distribution=nb,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
      return tfd.Independent(
          zinb,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), tied_inflation_rate=False, name=None):
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
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZINegativeBinomialDispLayer,
          self).__init__(lambda t: type(self).new(
              t, event_shape, given_log_mean, given_log_disp, validate_args),
                         convert_to_tensor_fn,
                         activity_regularizer=activity_regularizer,
                         **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_log_mean=True,
          given_log_disp=True,
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
      (loc_params, disp_params, rate_params) = tf.split(params, 3, axis=-1)
      if given_log_mean:
        loc_params = tf.exp(loc_params, name='loc')
      if given_log_disp:
        disp_params = tf.exp(disp_params, name='disp')
      nb = NegativeBinomialDisp(loc=tf.reshape(loc_params, output_shape),
                                disp=tf.reshape(disp_params, output_shape),
                                validate_args=validate_args)
      zinb = ZeroInflated(count_distribution=nb,
                          logits=tf.reshape(rate_params, output_shape),
                          validate_args=validate_args)
      return tfd.Independent(
          zinb,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), tied_inflation_rate=False, name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'ZINegativeBinomialDisp_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 3 * _event_size(event_shape,
                             name=name or 'ZINegativeBinomialDisp_params_size')


class ZIBernoulliLayer(DistributionLambda):
  """A Independent zero-inflated bernoulli keras layer

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
               given_logits=True,
               convert_to_tensor_fn=tfd.Distribution.sample,
               validate_args=False,
               activity_regularizer=None,
               **kwargs):
    super(ZIBernoulliLayer, self).__init__(
        lambda t: type(self).new(t, event_shape, given_logits, validate_args),
        convert_to_tensor_fn,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @staticmethod
  def new(params,
          event_shape=(),
          given_logits=True,
          validate_args=False,
          name=None):
    """Create the distribution instance from a `params` vector."""
    with tf.compat.v1.name_scope(name, 'ZIBernoulliLayer',
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
      (bernoulli_params, rate_params) = tf.split(params, 2, axis=-1)
      bernoulli_params = tf.reshape(bernoulli_params, output_shape)
      bern = tfd.Bernoulli(logits=bernoulli_params if given_logits else None,
                           probs=bernoulli_params if not given_logits else None,
                           validate_args=validate_args)
      zibern = ZeroInflated(count_distribution=bern,
                            logits=tf.reshape(rate_params, output_shape),
                            validate_args=validate_args)
      return tfd.Independent(
          zibern,
          reinterpreted_batch_ndims=tf.size(input=event_shape),
          validate_args=validate_args)

  @staticmethod
  def params_size(event_shape=(), tied_inflation_rate=False, name=None):
    """The number of `params` needed to create a single distribution."""
    with tf.compat.v1.name_scope(name, 'ZeroInflatedBernoulli_params_size',
                                 [event_shape]):
      event_shape = tf.convert_to_tensor(value=event_shape,
                                         name='event_shape',
                                         dtype=tf.int32)
      return 2 * _event_size(event_shape,
                             name=name or 'ZeroInflatedBernoulli_params_size')


# ===========================================================================
# Shortcut
# ===========================================================================
NormalLayer = GaussianLayer
