from __future__ import absolute_import, division, print_function

import inspect
from functools import partial
from typing import Callable, Optional, Text, Type, Union

import tensorflow as tf
from six import string_types
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import (
    DistributionLambda, _get_convert_to_tensor_fn, _serialize,
    _serialize_function)

from odin import backend as bk
from odin.bay.distribution_alias import _dist_mapping, parse_distribution
from odin.bay.distribution_layers import VectorDeterministicLayer
from odin.bay.helpers import KLdivergence, kl_divergence
from odin.networks.distribution_util_layers import Moments, Sampling

__all__ = ['DenseDeterministic', 'DenseDistribution']


class DenseDeterministic(Dense):
  """ Similar to `keras.Dense` layer but return a
  `tensorflow_probability.Deterministic` distribution to represent the output,
  hence, make it compatible to probabilistic frameworks
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DenseDeterministic,
          self).__init__(units=units,
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

  def call(self, inputs, n_mcmc=1, **kwargs):
    outputs = super(DenseDeterministic, self).call(inputs)
    distribution = VectorDeterministicLayer(convert_to_tensor_fn=partial(
        Distribution.sample, sample_shape=[n_mcmc]))(outputs)
    distribution.KL_divergence = KLdivergence(distribution, prior=None)
    return distribution


class DenseDistribution(Dense):
  r""" Using `Dense` layer to parameterize the tensorflow_probability
  `Distribution`

  Arguments:
    units : `int`
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
               prior=None,
               posterior_kwargs={},
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
    post_layer, _ = parse_distribution(posterior)
    self._n_mcmc = [1]
    self._posterior_layer = post_layer(
        event_shape,
        convert_to_tensor_fn=partial(Distribution.sample,
                                     sample_shape=self._n_mcmc) if
        convert_to_tensor_fn == Distribution.sample else convert_to_tensor_fn,
        **posterior_kwargs)
    # create layers
    self._convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)
    self._posterior = posterior
    self._prior = prior
    self._event_shape = event_shape
    self._posterior_kwargs = posterior_kwargs
    self._dropout = dropout
    super(DenseDistribution,
          self).__init__(units=post_layer.params_size(event_shape),
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
  def prior(self):
    return self._prior

  @prior.setter
  def prior(self, p):
    assert isinstance(p, (Distribution, type(None)))
    self._prior = p

  @property
  def distribution_layer(self):
    return self._posterior_layer

  @property
  def posterior(self):
    r""" Return the last parametrized distribution, i.e. the result from the
    last `call` """
    return self._last_distribution

  def call(self, inputs, training=None, n_mcmc=1, projection=True, prior=None):
    params = super().call(inputs) if projection else inputs
    if self._dropout > 0:
      params = bk.dropout(params, p_drop=self._dropout, training=training)
    # modifying the Lambda to return given number of n_mcmc samples
    self._n_mcmc[0] = n_mcmc
    posterior = self._posterior_layer(params, training=training)
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

  def get_config(self):
    config = super().get_config()
    config['convert_to_tensor_fn'] = _serialize(self._convert_to_tensor_fn)
    config['event_shape'] = self._event_shape
    config['posterior'] = self._posterior
    config['prior'] = self._prior
    config['dropout'] = self._dropout
    config['posterior_kwargs'] = self._posterior_kwargs
    return config
