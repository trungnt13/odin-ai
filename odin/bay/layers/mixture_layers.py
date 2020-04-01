from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers import (
    CategoricalMixtureOfOneHotCategorical, MixtureLogistic, MixtureSameFamily)

from odin.backend import parse_activation
from odin.backend.maths import softplus1
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated

__all__ = [
    'MixtureLogisticLayer', 'MixtureSameFamilyLayer', 'MixtureGaussianLayer',
    'CategoricalMixtureOfOneHotCategorical', 'MixtureNegativeBinomialLayer'
]
MixtureLogisticLayer = MixtureLogistic
MixtureSameFamilyLayer = MixtureSameFamily


class MixtureGaussianLayer(tfp.layers.DistributionLambda):
  r""" Initialize the mixture of gaussian distributions layer.

  Arguments:
    n_components: Number of component distributions in the mixture
      distribution.
    loc_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures in default parameterization, or
      `mean` in alternative approach.
    scale_activation: activation function for the success rate (default
      parameterization), or the non-negative dispersion (alternative approach).
    covariance_type : {'tril', 'diag' (default), 'spherical'/'none'}
        String describing the type of covariance parameters to use.
        Must be one of:
        'tril' - each component has its own general covariance matrix
        'diag' - each component has its own diagonal covariance matrix
        'spherical'/'none' - each component has its own single variance
    convert_to_tensor_fn: Python `callable` that takes a `tfd.Distribution`
      instance and returns a `tf.Tensor`-like object.
      Default value: `tfd.Distribution.sample`.
    validate_args: Python `bool`, default `False`. When `True` distribution
      parameters are checked for validity despite possibly degrading runtime
      performance. When `False` invalid inputs may silently render incorrect
      outputs.
      Default value: `False`.
    **kwargs: Additional keyword arguments passed to `tf.keras.Layer`.

  Attributes:
    mixture_distribution: `tfp.distributions.Categorical`-like instance.
      Manages the probability of selecting components. The number of
      categories must match the rightmost batch dimension of the
      `components_distribution`. Must have either scalar `batch_shape` or
      `batch_shape` matching `components_distribution.batch_shape[:-1]`.
    components_distribution: `tfp.distributions.Distribution`-like instance.
      Right-most batch dimension indexes components, i.e.
      `[batch_dim, component_dim, ...]`

  References:
    Bishop, C.M. (1994). Mixture density networks.
  """

  def __init__(self,
               event_shape=(),
               n_components=2,
               covariance='none',
               loc_activation='linear',
               scale_activation='softplus1',
               convert_to_tensor_fn=tfp.distributions.Distribution.sample,
               validate_args=False,
               **kwargs):
    super().__init__(
        lambda params: MixtureGaussianLayer.new(
            params, event_shape, n_components, covariance,
            parse_activation(loc_activation, self),
            parse_activation(scale_activation, self), validate_args),
        convert_to_tensor_fn, **kwargs)
    self.event_shape = event_shape
    self.n_components = n_components
    self.covariance = str(covariance).strip().lower()

  @staticmethod
  def new(params,
          event_shape=(),
          n_components=2,
          covariance='none',
          loc_activation=tf.identity,
          scale_activation='softplus1',
          validate_args=False,
          name=None):
    r""" Create the distribution instance from a `params` vector. """
    params = tf.convert_to_tensor(value=params, name='params')
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    components_size = MixtureGaussianLayer.components_size(
        event_shape, covariance)
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        [n_components],
        event_shape,
    ],
                             axis=0)
    ### Create the mixture
    mixture = tfp.distributions.Categorical(logits=params[..., :n_components],
                                            name="MixtureWeights")
    ### Create the components
    params = tf.reshape(
        params[..., n_components:],
        tf.concat(
            [tf.shape(input=params)[:-1], [n_components, components_size]],
            axis=0))
    # ====== initialize the components ====== #
    if covariance == 'none':
      def_name = 'IndependentGaussian'
      loc, scale = tf.split(params, 2, axis=-1)
      loc = tf.reshape(loc_activation(loc), output_shape)
      scale = tf.reshape(scale_activation(scale), output_shape)
      components = tfp.distributions.Independent(
          tfp.distributions.Normal(loc=loc,
                                   scale=scale,
                                   validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape))
    # Diagonal
    elif covariance == 'diag':
      def_name = 'MultivariateGaussianDiag'
      loc, scale = tf.split(params, 2, axis=-1)
      loc = loc_activation(loc)
      scale = scale_activation(scale)
      components = tfp.distributions.MultivariateNormalDiag(loc=loc,
                                                            scale_diag=scale)
    # lower-triangle
    elif covariance in ('full', 'tril'):
      def_name = 'MultivariateGaussianTriL'
      event_size = tf.reduce_prod(event_shape)
      loc = loc_activation(params[..., :event_size])
      scale = scale_activation(params[..., event_size:])
      scale_tril = tfp.bijectors.FillScaleTriL(
          diag_shift=np.array(1e-5, params.dtype.as_numpy_dtype()))
      components = tfp.distributions.MultivariateNormalTriL(
          loc=loc, scale_tril=scale_tril(scale))
    # error
    else:
      raise NotImplementedError("No support for covariance: '%s'" % covariance)
    ### the mixture distribution
    return tfp.distributions.MixtureSameFamily(
        mixture_distribution=mixture,
        components_distribution=components,
        name="Mixture%s" % def_name if name is None else str(name))

  @staticmethod
  def components_size(event_shape, covariance):
    event_size = tf.convert_to_tensor(value=tf.reduce_prod(event_shape),
                                      name='params_size',
                                      dtype_hint=tf.int32)
    event_size = dist_util.prefer_static_value(event_size)
    covariance = covariance.lower()
    if covariance == 'none':
      return event_size + event_size
    elif covariance == 'diag':  # only the diagonal
      return event_size + event_size
    elif covariance in ('full', 'tril'):  # lower triangle
      return event_size + event_size * (event_size + 1) // 2
    return NotImplementedError("No support for covariance: '%s'" % covariance)

  @staticmethod
  def params_size(event_shape, n_components=2, covariance='diag'):
    r"""Number of `params` needed to create a `MixtureNegativeBinomialLayer`
    distribution.

    Returns:
     params_size: The number of parameters needed to create the mixture
       distribution.
    """
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    n_components = dist_util.prefer_static_value(n_components)
    component_params_size = MixtureGaussianLayer.components_size(
        event_shape, covariance)
    return n_components + n_components * component_params_size


class MixtureNegativeBinomialLayer(tfp.layers.DistributionLambda):
  r"""Initialize the mixture of NegativeBinomial distributions layer.

  Arguments:
    n_components: Number of component distributions in the mixture
      distribution.
    mean_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures in default parameterization, or
      `mean` in alternative approach.
    disp_activation: activation function for the success rate (default
      parameterization), or the non-negative dispersion (alternative approach).
    alternative: `bool`, using default parameterization of
      `total_count` and `probs_success`, or the alternative with `mean` and
      `dispersion`. Default: `False`
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

  Attributes:
    mixture_distribution: `tfp.distributions.Categorical`-like instance.
      Manages the probability of selecting components. The number of
      categories must match the rightmost batch dimension of the
      `components_distribution`. Must have either scalar `batch_shape` or
      `batch_shape` matching `components_distribution.batch_shape[:-1]`.
    components_distribution: `tfp.distributions.Distribution`-like instance.
      Right-most batch dimension indexes components, i.e.
      `[batch_dim, component_dim, ...]`

  References:
    Bishop, C.M. (1994). Mixture density networks.
    Liu, L.-P., Blei, D.M.. Zero-Inflated Exponential Family Embeddings.
  """

  def __init__(self,
               event_shape=(),
               n_components=2,
               mean_activation='softplus1',
               disp_activation=None,
               dispersion='full',
               alternative=False,
               zero_inflated=False,
               convert_to_tensor_fn=tfp.distributions.Distribution.sample,
               validate_args=False,
               **kwargs):
    if disp_activation is None:
      disp_activation = 'softplus1' if alternative else 'linear'
    super().__init__(
        lambda params: MixtureNegativeBinomialLayer.new(
            params, event_shape, n_components,
            parse_activation(mean_activation, self),
            parse_activation(disp_activation, self), dispersion, alternative,
            zero_inflated, validate_args), convert_to_tensor_fn, **kwargs)
    self.event_shape = event_shape
    self.n_components = n_components
    self.zero_inflated = zero_inflated

  @staticmethod
  def new(
      params,
      event_shape=(),
      n_components=2,
      mean_activation=softplus1,
      disp_activation=tf.identity,
      dispersion='full',
      alternative=False,
      zero_inflated=False,
      validate_args=False,
  ):
    """Create the distribution instance from a `params` vector."""
    params = tf.convert_to_tensor(value=params, name='params')
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        [n_components],
        event_shape,
    ],
                             axis=0)
    mixture = tfp.distributions.Categorical(logits=params[..., :n_components])
    if zero_inflated:
      mean, disp, rate = tf.split(params[..., n_components:], 3, axis=-1)
      rate = tf.reshape(rate, output_shape)
    else:
      mean, disp = tf.split(params[..., n_components:], 2, axis=-1)
      rate = None
    mean = tf.reshape(mean, output_shape)
    disp = tf.reshape(disp, output_shape)

    if dispersion == 'single':
      disp = tf.reduce_mean(disp)
    elif dispersion == 'share':
      disp = tf.reduce_mean(disp,
                            axis=tf.range(0,
                                          output_shape.shape[0] - 1,
                                          dtype='int32'),
                            keepdims=True)
    mean = mean_activation(mean)
    disp = disp_activation(disp)

    if alternative:
      NBtype = NegativeBinomialDisp
      name = 'NegBinDisp'
    else:
      NBtype = tfp.distributions.NegativeBinomial
      name = 'NegBin'
    components = tfp.distributions.Independent(
        NBtype(mean, disp, validate_args=validate_args),
        reinterpreted_batch_ndims=tf.size(input=event_shape),
        validate_args=validate_args)
    if zero_inflated:
      name = 'ZI' + name
      components = ZeroInflated(count_distribution=components,
                                logits=rate,
                                validate_args=False)
    return tfp.distributions.MixtureSameFamily(mixture,
                                               components,
                                               validate_args=False,
                                               name='Mixture%s' % name)

  def params_size(self):
    r"""Number of `params` needed to create a `MixtureNegativeBinomialLayer`
    distribution.

    Returns:
     params_size: The number of parameters needed to create the mixture
       distribution.
    """
    n_components = tf.convert_to_tensor(value=self.n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    params_size = tf.convert_to_tensor(value=tf.reduce_prod(self.event_shape) *
                                       (3 if self.zero_inflated else 2),
                                       name='params_size')
    n_components = dist_util.prefer_static_value(n_components)
    params_size = dist_util.prefer_static_value(params_size)
    return n_components + n_components * params_size
