from __future__ import absolute_import, division, print_function

import inspect
from functools import partial
from numbers import Number
from typing import Callable, Optional, Text, Type, Union

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow_probability.python.bijectors import ScaleTriL
from tensorflow_probability.python.distributions import (Categorical,
                                                         Distribution,
                                                         Independent,
                                                         MixtureSameFamily,
                                                         MultivariateNormalDiag,
                                                         MultivariateNormalTriL,
                                                         Normal)
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda, _get_convert_to_tensor_fn, _serialize,
    _serialize_function)

from odin import backend as bk
from odin.bay.distribution_alias import parse_distribution
from odin.bay.distribution_layers import VectorDeterministicLayer
from odin.bay.helpers import KLdivergence, kl_divergence
from odin.bay.layers.distribution_util_layers import Moments, Sampling

__all__ = [
    'DenseDeterministic', 'DenseDistribution', 'MixtureDensityNetwork',
    'MixtureMassNetwork'
]


class DenseDistribution(Dense):
  r""" Using `Dense` layer to parameterize the tensorflow_probability
  `Distribution`

  Arguments:
    event_shape : `int`
      number of output units.
    posterior : the posterior distribution, a distribution alias or Distribution
      type can be given for later initialization (Default: 'normal').
    prior : {`None`, `tensorflow_probability.Distribution`}
      prior distribution, used for calculating KL divergence later.
    use_bias : `bool` (default=`True`)
      enable biases for the Dense layers
    posterior_kwargs : `dict`. Keyword arguments for initializing the posterior
      `DistributionLambda`

  Return:
    `tensorflow_probability.Distribution`
  """

  def __init__(self,
               event_shape=(),
               posterior='normal',
               posterior_kwargs={},
               prior=None,
               convert_to_tensor_fn=Distribution.sample,
               dropout=0.0,
               activation='linear',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    assert prior is None or isinstance(prior, Distribution), \
      "prior can be None or instance of tensorflow_probability.Distribution"
    # duplicated event_shape or event_size in posterior_kwargs
    posterior_kwargs = dict(posterior_kwargs)
    if 'event_shape' in posterior_kwargs:
      event_shape = posterior_kwargs.pop('event_shape')
    if 'event_size' in posterior_kwargs:
      event_shape = posterior_kwargs.pop('event_size')
    convert_to_tensor_fn = posterior_kwargs.pop('convert_to_tensor_fn',
                                                Distribution.sample)
    # process the posterior
    # TODO: support give instance of DistributionLambda directly
    if inspect.isclass(posterior) and issubclass(posterior, DistributionLambda):
      post_layer = posterior
    else:
      post_layer, _ = parse_distribution(posterior)
    self._n_mcmc = [1]
    self._posterior_layer = post_layer(
        event_shape,
        convert_to_tensor_fn=partial(Distribution.sample,
                                     sample_shape=self._n_mcmc) \
          if convert_to_tensor_fn == Distribution.sample else \
            convert_to_tensor_fn,
        **posterior_kwargs)
    # create layers
    self._convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)
    self._posterior = posterior
    self._prior = prior
    self._event_shape = event_shape
    self._posterior_kwargs = posterior_kwargs
    self._dropout = dropout
    # set more descriptive name
    name = kwargs.pop('name', None)
    if name is None:
      name = 'dense_%s' % (posterior if isinstance(posterior, string_types) else
                           posterior.__class__.__name__)
    kwargs['name'] = name
    # params_size could be static function or method
    params_size = self._posterior_layer.params_size() \
      if 'self' in inspect.getfullargspec(post_layer.params_size).args else \
        post_layer.params_size(event_shape)
    super(DenseDistribution,
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
    # store the distribution from last call
    self._last_distribution = None

  @property
  def event_shape(self):
    return tf.nest.flatten(self._event_shape)

  @property
  def event_size(self):
    return tf.cast(tf.reduce_prod(self._event_shape), tf.int32)

  @property
  def prior(self):
    return self._prior

  @prior.setter
  def prior(self, p):
    assert isinstance(p, (Distribution, type(None)))
    self._prior = p

  @property
  def posterior_layer(self):
    return self._posterior_layer

  @property
  def posterior(self):
    r""" Return the last parametrized distribution, i.e. the result from the
    last `call` """
    return self._last_distribution

  @tf.function
  def sample(self, sample_shape=(), seed=None):
    r""" Sample from prior distribution """
    if self._prior is None:
      raise RuntimeError("prior hasn't been provided for the %s" %
                         self.__class__.__name__)
    return self.prior.sample(sample_shape=sample_shape, seed=seed)

  def call(self, inputs, training=None, n_mcmc=1, projection=True, prior=None):
    params = super().call(inputs) if projection else inputs
    if self._dropout > 0:
      params = bk.dropout(params, p_drop=self._dropout, training=training)
    # modifying the Lambda to return given number of n_mcmc samples
    self._n_mcmc[0] = n_mcmc
    posterior = self.posterior_layer(params, training=training)
    self._last_distribution = posterior
    # NOTE: all distribution has the method kl_divergence, so we cannot use it
    posterior.KL_divergence = KLdivergence(
        posterior, prior=self.prior if prior is None else prior, n_mcmc=n_mcmc)
    return posterior

  def kl_divergence(self, prior=None, analytic=True, n_mcmc=1, reverse=True):
    r""" KL(q||p) where `p` is the posterior distribution returned from last
    call

    Arguments:
      prior : instance of `tensorflow_probability.Distribution`
        prior distribution of the latent
      analytic : `bool` (default=`True`). Using closed form solution for
        calculating divergence, otherwise, sampling with MCMC
      reverse : `bool`. If `True`, calculate `KL(q||p)` else `KL(p||q)`
      n_mcmc : `int` (default=`1`)
        number of MCMC sample if `analytic=False`

    Return:
      kullback_divergence : Tensor [n_mcmc, batch_size, ...]
    """
    if prior is None:
      prior = self._prior
    assert isinstance(prior, Distribution), "prior is not given!"
    if self.posterior is None:
      raise RuntimeError(
          "DenseDistribution must be called to create the distribution before "
          "calculating the kl-divergence.")

    kullback_div = kl_divergence(q=self.posterior,
                                 p=prior,
                                 analytic=bool(analytic),
                                 reverse=reverse,
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

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return "<Dense shape:%s #params:%d posterior:%s prior:%s dropout:%.2f kw:%s>" % \
      (self.event_shape, self.units, self.posterior_layer.name, str(self.prior),
       self._dropout, str(self._posterior_kwargs))

  def get_config(self):
    config = super().get_config()
    config['convert_to_tensor_fn'] = _serialize(self._convert_to_tensor_fn)
    config['event_shape'] = self._event_shape
    config['posterior'] = self._posterior
    config['prior'] = self._prior
    config['dropout'] = self._dropout
    config['posterior_kwargs'] = self._posterior_kwargs
    return config


# ===========================================================================
# Shortcuts
# ===========================================================================
class MixtureDensityNetwork(DenseDistribution):

  def __init__(self,
               units,
               n_components=8,
               covariance='none',
               loc_activation='linear',
               scale_activation='softplus1',
               convert_to_tensor_fn=Distribution.sample,
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
    self.covariance = covariance
    self.n_components = n_components
    super().__init__(event_shape=units,
                     posterior='mixgaussian',
                     prior=None,
                     posterior_kwargs=dict(n_components=int(n_components),
                                           covariance=str(covariance),
                                           loc_activation=loc_activation,
                                           scale_activation=scale_activation),
                     convert_to_tensor_fn=convert_to_tensor_fn,
                     dropout=dropout,
                     activation='linear',
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     **kwargs)

  def set_prior(self, loc=0., log_scale=np.log(np.expm1(1)), mixture_logits=1.):
    r""" Set the prior for mixture density network

    loc : Scalar or Tensor with shape `[n_components, event_size]`
    log_scale : Scalar or Tensor with shape
      `[n_components, event_size]` for 'none' and 'diag' component, and
      `[n_components, event_size*(event_size +1)//2]` for 'full' component.
    mixture_logits : Scalar or Tensor with shape `[n_components]`
    """
    event_size = self.event_size
    if self.covariance == 'diag':
      scale_shape = [self.n_components, event_size]
      fn = lambda l, s: MultivariateNormalDiag(loc=l,
                                               scale_diag=tf.nn.softplus(s))
    elif self.covariance == 'none':
      scale_shape = [self.n_components, event_size]
      fn = lambda l, s: Independent(Normal(loc=l, scale=tf.math.softplus(s)), 1)
    elif self.covariance == 'full':
      scale_shape = [self.n_components, event_size * (event_size + 1) // 2]
      fn = lambda l, s: MultivariateNormalTriL(
          loc=l, scale_tril=ScaleTriL(diag_shift=1e-5)(tf.math.softplus(s)))
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
    self._prior = MixtureSameFamily(
        components_distribution=fn(loc, log_scale),
        mixture_distribution=Categorical(logits=mixture_logits),
        name="prior")
    return self


class MixtureMassNetwork(DenseDistribution):

  def __init__(self,
               event_shape=(),
               n_components=2,
               mean_activation='softplus1',
               disp_activation=None,
               dispersion='full',
               alternative=False,
               zero_inflated=False,
               convert_to_tensor_fn=Distribution.sample,
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
    self.n_components = n_components
    self.dispersion = dispersion
    self.zero_inflated = zero_inflated
    self.alternative = alternative
    super().__init__(event_shape=event_shape,
                     posterior='mixnb',
                     prior=None,
                     posterior_kwargs=dict(
                         n_components=int(n_components),
                         mean_activation=mean_activation,
                         disp_activation=disp_activation,
                         dispersion=dispersion,
                         alternative=alternative,
                         zero_inflated=zero_inflated,
                     ),
                     convert_to_tensor_fn=convert_to_tensor_fn,
                     dropout=dropout,
                     activation='linear',
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     **kwargs)


class DenseDeterministic(DenseDistribution):
  r""" Similar to `keras.Dense` layer but return a
  `tensorflow_probability.VectorDeterministic` distribution to represent
  the output, hence, making it compatible to the probabilistic framework.
  """

  def __init__(self,
               units,
               dropout=0.0,
               activation='linear',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(event_shape=int(units),
                     posterior='vdeterministic',
                     posterior_kwargs={},
                     prior=None,
                     convert_to_tensor_fn=Distribution.sample,
                     dropout=dropout,
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
