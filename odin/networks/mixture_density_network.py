from __future__ import absolute_import, division, print_function

from functools import partial
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture
from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dense
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda, _get_convert_to_tensor_fn, _serialize,
    _serialize_function)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc
from tensorflow_probability.python.layers.internal import \
    tensor_tuple as tensor_tuple

from odin import backend as bk
from odin.bay.helpers import KLdivergence, coercible_tensor, kl_divergence

__all__ = ['MixtureDensityNetwork']

_COV_TYPES = ('none', 'diag', 'full')


class MixtureDensityNetwork(Dense):
  r"""A mixture of Gaussian Keras layer.

  Arguments:
    units : `int`
      number of output features for each component.
    n_components : `int` (default=`2`)
      The number of mixture components.
    covariance_type : {'none', 'diag', 'full'}
      String describing the type of covariance parameters to use.
      Must be one of:
        'none' - each component is a collection of univariate gaussian distributions,
        'diag' - each component has its own diagonal covariance matrix,
        'full' - full covariance with parameterized lower triangle
                 covariance matrix,
    dropout : a Scalar. Dropout the activation before parameterizing the
      distributions

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
               units,
               n_components=8,
               covariance_type='none',
               convert_to_tensor_fn=tfd.Distribution.sample,
               softplus_scale=True,
               activation='linear',
               use_bias=True,
               dropout=0.0,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    covariance_type = str(covariance_type).lower()
    assert covariance_type in _COV_TYPES, \
    "No support for covariance_type: '%s', the support value are: %s" % \
      (covariance_type, ', '.join(_COV_TYPES))
    self._covariance_type = covariance_type
    self._n_components = int(n_components)
    self._is_sampling = True \
      if convert_to_tensor_fn == tfd.Distribution.sample else False
    self._convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)
    self._softplus_scale = bool(softplus_scale)
    self._dropout = dropout
    # We'll need to keep track of who's calling who since the functional
    # API has a different way of injecting `_keras_history` than the
    # `keras.Sequential` way.
    self._enter_dunder_call = False
    # store the distribution from last call
    self._last_distribution = None
    # ====== calculating the number of parameters ====== #
    if covariance_type == 'none':
      component_params_size = units + units
    elif covariance_type == 'diag':  # only the diagonal
      component_params_size = units + units
    elif covariance_type == 'full':  # lower triangle
      component_params_size = units + units * (units + 1) // 2
    else:
      raise NotImplementedError
    self._component_params_size = component_params_size
    params_size = self.n_components + self.n_components * component_params_size
    self._event_size = units
    super(MixtureDensityNetwork,
          self).__init__(units=params_size,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
    # ====== create prior ====== #
    self._prior = None

  def set_prior(self, loc=0., log_scale=np.log(np.expm1(1)), mixture_logits=1.):
    r""" Set the prior for mixture density network

    loc : Scalar or Tensor with shape `[n_components, event_size]`
    log_scale : Scalar or Tensor with shape
      `[n_components, event_size]` for 'none' and 'diag' component, and
      `[n_components, event_size*(event_size +1)//2]` for 'full' component.
    mixture_logits : Scalar or Tensor with shape `[n_components]`
    """
    event_size = self.event_size
    if self.covariance_type == 'diag':
      scale_shape = [self.n_components, event_size]
      fn = lambda l, s: tfd.MultivariateNormalDiag(loc=l,
                                                   scale_diag=tf.nn.softplus(s))
    elif self.covariance_type == 'none':
      scale_shape = [self.n_components, event_size]
      fn = lambda l, s: tfd.Independent(
          tfd.Normal(loc=l, scale=tf.nn.softplus(s)), 1)
    elif self.covariance_type == 'full':
      scale_shape = [self.n_components, event_size * (event_size + 1) // 2]
      fn = lambda l, s: tfd.MultivariateNormalTriL(
          loc=l, scale_tril=tfb.ScaleTriL(diag_shift=1e-5)(tf.nn.softplus(s)))
    #
    if isinstance(log_scale, Number) or tf.rank(log_scale) == 0:
      loc = tf.fill([self.n_components, self.event_size], loc)
    #
    if isinstance(log_scale, Number) or tf.rank(log_scale) == 0:
      log_scale = tf.fill(scale_shape, log_scale)
    #
    if mixture_logits is None:
      mixture_logits = 1.
    if isinstance(mixture_logits, Number) or tf.rank(mixture_logits) == 0:
      mixture_logits = tf.fill([self.n_components], mixture_logits)
    #
    loc = tf.cast(loc, self.dtype)
    log_scale = tf.cast(log_scale, self.dtype)
    mixture_logits = tf.cast(mixture_logits, self.dtype)
    self._prior = tfd.MixtureSameFamily(
        components_distribution=fn(loc, log_scale),
        mixture_distribution=tfd.Categorical(logits=mixture_logits),
        name="prior")
    return self

  @property
  def event_size(self):
    return self._event_size

  @property
  def covariance_type(self):
    return self._covariance_type

  @property
  def n_components(self):
    return self._n_components

  @property
  def component_params_size(self):
    return self._component_params_size

  @property
  def prior(self):
    return self._prior

  @prior.setter
  def prior(self, p):
    assert isinstance(p, (tfd.Distribution, type(None)))
    self._prior = p

  @property
  def posterior(self):
    r""" Return the last parametrized distribution, i.e. the result from the
    last `call` """
    return self._last_distribution

  def __call__(self, inputs, *args, **kwargs):
    self._enter_dunder_call = True
    distribution, _ = super(MixtureDensityNetwork,
                            self).__call__(inputs, *args, **kwargs)
    self._last_distribution = distribution
    self._enter_dunder_call = False
    return distribution

  def call(self, inputs, training=None, n_mcmc=1, projection=True, prior=None):
    params = super().call(inputs) if projection else inputs
    if self._dropout > 0:
      params = bk.dropout(params, p_drop=self._dropout, training=training)
    n_components = tf.convert_to_tensor(value=self.n_components,
                                        name='n_components',
                                        dtype_hint=tf.int32)
    # ====== mixture weights ====== #
    mixture_coefficients = params[..., :n_components]
    mixture_dist = tfd.Categorical(logits=mixture_coefficients,
                                   name="MixtureWeights")
    # ====== initialize the components ====== #
    params = tf.reshape(
        params[..., n_components:],
        tf.concat([tf.shape(input=params)[:-1], [n_components, -1]], axis=0))
    if bool(self._softplus_scale):
      scale_fn = lambda x: tf.math.softplus(x) + tfd.softplus_inverse(1.0)
    else:
      scale_fn = lambda x: x

    if self.covariance_type == 'none':
      cov = 'IndependentNormal'
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      components_dist = tfd.Independent(tfd.Normal(
          loc=loc_params, scale=scale_fn(scale_params)),
                                        reinterpreted_batch_ndims=1)
    #
    elif self.covariance_type == 'diag':
      cov = 'MultivariateNormalDiag'
      loc_params, scale_params = tf.split(params, 2, axis=-1)
      components_dist = tfd.MultivariateNormalDiag(
          loc=loc_params, scale_diag=scale_fn(scale_params))
    #
    elif self.covariance_type == 'full':
      cov = 'MultivariateNormalTriL'
      loc_params = params[..., :self.event_size]
      scale_params = scale_fn(params[..., self.event_size:])
      scale_tril = tfb.ScaleTriL(
          diag_shift=np.array(1e-5, params.dtype.as_numpy_dtype()))
      components_dist = tfd.MultivariateNormalTriL(
          loc=loc_params, scale_tril=scale_tril(scale_params))
    else:
      raise NotImplementedError
    # ====== finally the mixture ====== #
    d = tfd.MixtureSameFamily(mixture_distribution=mixture_dist,
                              components_distribution=components_dist,
                              name="Mixture%s" % cov)
    # Wraps the distribution to return both dist and concrete value."""
    convert_to_tensor_fn = self._convert_to_tensor_fn
    if self._is_sampling is True:
      convert_to_tensor_fn = partial(convert_to_tensor_fn,
                                     sample_shape=[n_mcmc])
    distribution, value = coercible_tensor(
        d, convert_to_tensor_fn=convert_to_tensor_fn, return_value=True)
    if self._enter_dunder_call:
      # Its critical to return both distribution and concretization
      # so Keras can inject `_keras_history` to both. This is what enables
      # either to be used as an input to another Keras `Model`.
      return distribution, value
    # injecting KL object
    distribution.KL_divergence = KLdivergence(
        distribution,
        prior=self.prior if prior is None else prior,
        n_mcmc=n_mcmc)
    return distribution

  def kl_divergence(self, prior=None, analytic=False, n_mcmc=1, reverse=True):
    r""" KL(q||p) where `p` is the posterior distribution returned from last
    call

    Arguments:
      prior : instance of `tensorflow_probability.Distribution`
        prior distribution of the latent
      analytic : `bool` (default=`False`). Using closed form solution for
        calculating divergence, otherwise, sampling with MCMC
      reverse : `bool`. If `True`, calculate `KL(q||p)` else `KL(p||q)`
      n_mcmc : `int` (default=`1`)
        number of MCMC sample if `analytic=False`

    Return:
      kullback_divergence : Tensor [n_mcmc, batch_size, ...]
    """
    if n_mcmc is None:
      n_mcmc = 1
    if prior is None:
      prior = self._prior
    assert isinstance(prior, tfd.Distribution), "prior is not given!"
    if self.posterior is None:
      raise RuntimeError(
          "DenseDistribution must be called to create the distribution before "
          "calculating the kl-divergence.")

    kullback_div = kl_divergence(q=self.posterior,
                                 p=prior,
                                 analytic=bool(analytic),
                                 reverse=bool(reverse),
                                 q_sample=int(n_mcmc),
                                 auto_remove_independent=True)
    if analytic:
      kullback_div = tf.expand_dims(kullback_div, axis=0)
      if n_mcmc > 1:
        ndims = kullback_div.shape.ndims
        kullback_div = tf.tile(kullback_div, [n_mcmc] + [1] * (ndims - 1))
    return kullback_div

  def log_prob(self, x):
    r""" Calculating the log probability (i.e. log likelihood) using the last
    distribution returned from call """
    return self.posterior.log_prob(x)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    # the number of output units is equal to event_size, not number of
    # hidden units
    return input_shape[:-1].concatenate(self.event_size)

  def get_config(self):
    """Returns the config of this layer. """
    config = {
        'convert_to_tensor_fn': _serialize(self._convert_to_tensor_fn),
        'covariance_type': self._covariance_type,
        'n_components': self._n_components,
        'softplus_scale': self._softplus_scale,
        'dropout': self._dropout
    }
    base_config = super(MixtureDensityNetwork, self).get_config()
    base_config.update(config)
    return base_config
