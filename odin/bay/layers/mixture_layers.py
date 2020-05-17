from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import keras
from tensorflow_probability.python.internal import \
    distribution_util as dist_util
from tensorflow_probability.python.layers import (
    CategoricalMixtureOfOneHotCategorical, MixtureLogistic, MixtureSameFamily)

from odin.backend import parse_activation
from odin.backend.maths import softplus1
from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated
from odin.bay.layers.count_layers import _dispersion

__all__ = [
    'MixtureLogisticLayer', 'MixtureSameFamilyLayer', 'MixtureGaussianLayer',
    'CategoricalMixtureOfOneHotCategorical', 'MixtureNegativeBinomialLayer'
]
MixtureLogisticLayer = MixtureLogistic
MixtureSameFamilyLayer = MixtureSameFamily


def _to_loc_scale(params_split,
                  params,
                  loc,
                  scale,
                  loc_shape=None,
                  scale_shape=None):
  if loc is None and scale is None:
    loc, scale = params_split()
    if loc_shape is not None:
      loc = tf.reshape(loc, loc_shape)
    if scale_shape is not None:
      scale = tf.reshape(scale, scale_shape)
  elif not (loc is None or scale is None):  # provided both
    pass
  elif loc is not None:  # provided loc
    scale = params
    if scale_shape is not None:
      scale = tf.reshape(scale, scale_shape)
  elif scale is not None:  # provided scale
    loc = params
    if loc_shape is not None:
      loc = tf.reshape(loc, loc_shape)
  return loc, scale


class MixtureGaussianLayer(tfp.layers.DistributionLambda):
  r""" Initialize the mixture of gaussian distributions layer.

  Arguments:
    n_components: Number of component distributions in the mixture
      distribution.
    covariance : {'tril', 'diag' (default), 'spherical'/'none'}
        String describing the type of covariance parameters to use.
        Must be one of:
        'tril' - each component has its own general covariance matrix
        'diag' - each component has its own diagonal covariance matrix
        'none' - each component has its own single variance
        'spherical' - all components use the same single variance
    tie_mixtures : a Boolean. If True, force all samples in minibatch use the
      same Categorical distribution (mixture), by taking the mean of the logits
    tie_components : a Boolean. If True, force all samples in the minibatch
      has the same parameters for components distribution
    loc_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures in default parameterization, or
      `mean` in alternative approach.
    scale_activation: activation function for the success rate (default
      parameterization), or the non-negative dispersion (alternative approach).
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
               tie_mixtures=False,
               tie_loc=False,
               tie_scale=False,
               loc_activation='linear',
               scale_activation='softplus1',
               convert_to_tensor_fn=tfp.distributions.Distribution.sample,
               validate_args=False,
               **kwargs):
    event_size = tf.convert_to_tensor(value=tf.reduce_prod(event_shape),
                                      name='event_size',
                                      dtype_hint=tf.int32)
    event_size = dist_util.prefer_static_value(event_size)
    if covariance != 'none':  # diag and tril is multivariate Gaussian
      event_shape = event_size
    if not tie_mixtures:
      if tie_loc and tie_scale:
        raise ValueError(
            "Mixture distribution has no support for tie_mixtures=False "
            "and both loc and scale are tied")
    logits, loc, scale = None, None, None
    if tie_mixtures:
      logits = tf.Variable([0.] * n_components,
                           trainable=True,
                           dtype=keras.backend.floatx(),
                           name="mixture_logits")
    if tie_loc:
      if covariance == 'none':
        shape = tf.concat(
            [[n_components], tf.nest.flatten(event_shape)], axis=0)
      else:
        shape = (n_components, event_size)
      loc = tf.Variable(
          tf.random.normal(shape),
          trainable=True,
          dtype=keras.backend.floatx(),
          name="components_loc",
      )
    if tie_scale:
      if covariance == 'none':
        shape = tf.concat(
            [[n_components], tf.nest.flatten(event_shape)], axis=0)
      elif covariance == 'diag':
        shape = (n_components, event_size)
      else:
        shape = (n_components, event_size * (event_size + 1) // 2)
      scale = tf.Variable(
          tf.random.normal(shape),
          trainable=True,
          dtype=keras.backend.floatx(),
          name="components_scale",
      )
    super().__init__(
        lambda params: MixtureGaussianLayer.new(
            params,
            event_shape,
            n_components=n_components,
            covariance=covariance,
            loc_activation=parse_activation(loc_activation, self),
            scale_activation=parse_activation(scale_activation, self),
            validate_args=validate_args,
            logits=logits,
            loc=loc,
            scale=scale), convert_to_tensor_fn, **kwargs)
    self.logits = logits
    self.loc = loc
    self.scale = scale
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
          logits=None,
          loc=None,
          scale=None,
          name=None):
    r""" Create the distribution instance from a `params` vector. """
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    event_size = tf.reduce_prod(event_shape)
    components_size = MixtureGaussianLayer.components_size(
        event_shape,
        covariance=covariance,
        tie_loc=loc is not None,
        tie_scale=scale is not None)
    ### shapes
    params_shape = tf.shape(input=params)[:-1]
    output_shape = tf.concat(
        (params_shape, [n_components],
         event_shape if covariance == 'none' else [event_size]),
        axis=0)
    ### Create the mixture
    if logits is None:
      logits = params[..., :n_components]
      params = params[..., n_components:]
    mixture = tfp.distributions.Categorical(logits=logits,
                                            name="MixtureWeights")
    ## loc-scale params
    shape = tf.concat(
        [tf.shape(input=params)[:-1], [n_components, components_size]], axis=0)
    params = tf.cond(tf.greater(components_size, 0),
                     true_fn=lambda: tf.reshape(params, shape),
                     false_fn=lambda: params)
    # ====== initialize the components ====== #
    if covariance == 'none':
      def_name = 'IndependentGaussian'
      loc, scale = _to_loc_scale(lambda: tf.split(params, 2, axis=-1),
                                 params,
                                 loc,
                                 scale,
                                 loc_shape=output_shape,
                                 scale_shape=output_shape)
      loc = loc_activation(loc)
      scale = scale_activation(scale)
      components = tfp.distributions.Independent(
          tfp.distributions.Normal(loc=loc,
                                   scale=scale,
                                   validate_args=validate_args),
          reinterpreted_batch_ndims=tf.size(input=event_shape))
    # Diagonal
    elif covariance == 'diag':
      def_name = 'MultivariateGaussianDiag'
      loc, scale = _to_loc_scale(lambda: tf.split(params, 2, axis=-1),
                                 params,
                                 loc,
                                 scale,
                                 loc_shape=output_shape,
                                 scale_shape=output_shape)
      loc = loc_activation(loc)
      scale = scale_activation(scale)
      components = tfp.distributions.MultivariateNormalDiag(loc=loc,
                                                            scale_diag=scale)
    # lower-triangle
    elif covariance in ('full', 'tril'):
      def_name = 'MultivariateGaussianTriL'
      scale_shape = tf.concat(
          (params_shape, [n_components], [event_size * (event_size + 1) // 2]),
          axis=0)
      loc, scale = _to_loc_scale(
          lambda: (params[..., :event_size], params[..., event_size:]),
          params,
          loc,
          scale,
          loc_shape=output_shape,
          scale_shape=scale_shape)
      loc = loc_activation(loc)
      scale = scale_activation(scale)
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
  def components_size(event_shape, covariance, tie_loc, tie_scale):
    event_size = tf.convert_to_tensor(value=tf.reduce_prod(event_shape),
                                      name='params_size',
                                      dtype_hint=tf.int32)
    event_size = dist_util.prefer_static_value(event_size)
    covariance = covariance.lower()
    loc_size = event_size
    if covariance == 'none':  # loc + scale
      scale_size = event_size
    elif covariance == 'diag':  # only the diagonal
      scale_size = event_size
    elif covariance in ('full', 'tril'):  # lower triangle
      scale_size = event_size * (event_size + 1) // 2
    else:
      raise NotImplementedError("No support for covariance: '%s'" % covariance)
    if tie_loc and tie_scale:
      return 0
    if not (tie_loc or tie_scale):
      return loc_size + scale_size
    if tie_loc:
      return scale_size
    if tie_scale:
      return loc_size

  @staticmethod
  def params_size(event_shape,
                  n_components=2,
                  covariance='diag',
                  tie_mixtures=False,
                  tie_loc=False,
                  tie_scale=False):
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
        event_shape,
        covariance=covariance,
        tie_loc=tie_loc,
        tie_scale=tie_scale)
    total = n_components + n_components * component_params_size
    if tie_mixtures:
      total -= n_components
    return total


# ===========================================================================
# Mixture mass layer
# ===========================================================================
class MixtureNegativeBinomialLayer(tfp.layers.DistributionLambda):
  r"""Initialize the mixture of NegativeBinomial distributions layer.

  Arguments:
    n_components: Number of component distributions in the mixture
      distribution.
    dispersion : {'full', 'share', 'single'}
      - 'full' creates a dispersion value for each individual data point,
      - 'share' creates a single dispersion vector of `event_shape` for
        all examples,
      - and 'single' uses a single value as dispersion for all data points.
    tie_mixtures : a Boolean. If True, force all samples in minibatch use the
      same Categorical distribution (mixture), by taking the mean of the logits
    mean_activation: activation function return non-negative floating-point,
      i.e. the `total_count` of failures in default parameterization, or
      `mean` in alternative approach.
    disp_activation: activation function for the success rate (default
      parameterization), or the non-negative dispersion (alternative approach).
    alternative: `bool`, using default parameterization of
      `total_count` and `probs_success`, or the alternative with `mean` and
      `dispersion`. Default: `False`
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
               tie_mixtures=False,
               tie_mean=False,
               dispersion='full',
               inflation='full',
               mean_activation='softplus1',
               disp_activation=None,
               alternative=False,
               zero_inflated=False,
               convert_to_tensor_fn=tfp.distributions.Distribution.sample,
               validate_args=False,
               **kwargs):
    if not tie_mixtures:
      if tie_mean and dispersion != 'full':
        raise ValueError(
            "Mixture distribution has no support for tie_mixtures=False "
            "and both mean and dispersion are tied")
    if zero_inflated:
      if inflation == 'full' and tie_mean and dispersion != 'full':
        raise ValueError("ZeroInflated distribution has no support for "
                         "batch-wise inflation rate but tied mean and "
                         "dispersion (this is broadcasting issue).")
    logits, mean, disp, rate = None, None, None, None
    shape = tf.concat([[n_components], tf.nest.flatten(event_shape)], axis=0)
    if tie_mixtures:
      logits = tf.Variable([0.] * n_components,
                           trainable=True,
                           dtype=keras.backend.floatx(),
                           name="mixture_logits")
    if tie_mean:
      mean = tf.Variable(tf.random.normal(shape),
                         trainable=True,
                         dtype=keras.backend.floatx(),
                         name="components_mean")
    disp = _dispersion(dispersion,
                       event_shape,
                       is_logits=not alternative,
                       name='dispersion',
                       n_components=n_components)
    rate = _dispersion(inflation,
                       event_shape,
                       is_logits=True,
                       name='inflation',
                       n_components=n_components)
    if disp_activation is None:
      disp_activation = 'softplus1' if alternative else 'linear'
    super().__init__(
        lambda params: MixtureNegativeBinomialLayer.new(
            params,
            event_shape,
            n_components=n_components,
            mean_activation=parse_activation(mean_activation, self),
            disp_activation=parse_activation(disp_activation, self),
            alternative=alternative,
            zero_inflated=zero_inflated,
            validate_args=validate_args,
            logits=logits,
            mean=mean,
            disp=disp,
            rate=rate), convert_to_tensor_fn, **kwargs)
    self.logits = logits
    self.mean = mean
    self.disp = disp
    self.rate = rate
    self.event_shape = event_shape
    self.n_components = n_components
    self.zero_inflated = zero_inflated

  @staticmethod
  def new(params,
          event_shape=(),
          n_components=2,
          mean_activation=softplus1,
          disp_activation=tf.identity,
          alternative=False,
          zero_inflated=False,
          validate_args=False,
          logits=None,
          mean=None,
          disp=None,
          rate=None):
    r""" Create the distribution instance from a `params` vector. """
    n_components = tf.convert_to_tensor(value=n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    event_size = tf.convert_to_tensor(
        tf.reduce_prod(event_shape),
        dtype_hint=tf.int32,
        name='event_size',
    )
    ### prepare params
    params = tf.convert_to_tensor(value=params, name='params')
    event_shape = dist_util.expand_to_vector(tf.convert_to_tensor(
        value=event_shape, name='event_shape', dtype=tf.int32),
                                             tensor_name='event_shape')
    output_shape = tf.concat([
        tf.shape(input=params)[:-1],
        [n_components],
        event_shape,
    ],
                             axis=0)
    ### Create the mixture
    if logits is None:
      logits = params[..., :n_components]
      params = params[..., n_components:]
    mixture = tfp.distributions.Categorical(logits=logits,
                                            name="MixtureWeights")
    ### zero_inflated
    if zero_inflated:
      if mean is None:
        mean = params[..., :n_components * event_size]
        mean = tf.reshape(mean, output_shape)
        params = params[..., n_components * event_size:]
      disp, rate = _to_loc_scale(lambda: tf.split(params, 2, axis=-1),
                                 params,
                                 loc=disp,
                                 scale=rate,
                                 loc_shape=output_shape,
                                 scale_shape=output_shape)
    else:  # negative binomial
      rate = None
      mean, disp = _to_loc_scale(lambda: tf.split(params, 2, axis=-1),
                                 params,
                                 loc=mean,
                                 scale=disp,
                                 loc_shape=output_shape,
                                 scale_shape=output_shape)
    ### applying activation
    mean = mean_activation(mean)
    disp = disp_activation(disp)
    ### alternative parameterization
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
    ### zero-inflated
    if zero_inflated:
      name = 'ZI' + name
      components = ZeroInflated(count_distribution=components,
                                logits=rate,
                                validate_args=False)
    return tfp.distributions.MixtureSameFamily(mixture,
                                               components,
                                               validate_args=False,
                                               name='Mixture%s' % name)

  @staticmethod
  def params_size(event_shape,
                  n_components=2,
                  zero_inflated=False,
                  dispersion='full',
                  inflation='full',
                  tie_mixtures=False,
                  tie_mean=False):
    r"""Number of `params` needed to create a `MixtureNegativeBinomialLayer`
    distribution.

    Returns:
     params_size: The number of parameters needed to create the mixture
       distribution.
    """
    n_components = tf.convert_to_tensor(
        n_components,
        dtype_hint=tf.int32,
        name='n_components',
    )
    event_size = tf.convert_to_tensor(
        tf.reduce_prod(event_shape),
        dtype_hint=tf.int32,
        name='event_size',
    )
    n_components = dist_util.prefer_static_value(n_components)
    event_size = dist_util.prefer_static_value(event_size)
    # single components
    params_size = (3 if zero_inflated else 2) * event_size
    if tie_mean:
      params_size -= event_size
    if dispersion != 'full':
      params_size -= event_size
    if zero_inflated and inflation != 'full':
      params_size -= event_size
    # total number of all components
    total = n_components + n_components * params_size
    if tie_mixtures:
      total -= n_components
    return total
