from __future__ import absolute_import, division, print_function

import inspect
from typing import Callable, Optional, Type, Union

import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.layers import Dense, Lambda
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers import DistributionLambda

from odin.bay.distribution_layers import VectorDeterministicLayer
from odin.bay.helpers import Statistic, kl_divergence
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

  def call(self, inputs, **kwargs):
    outputs = super(DenseDeterministic, self).call(inputs)
    return VectorDeterministicLayer()(outputs)


class DenseDistribution(Sequential):
  """ using `Dense` layer to parameterize tensorflow_probability `Distribution`

  Parameters
  ----------
  units : `int`
    number of output units.
  posterior : {`DistributionLambda`, `callable`, `type`}
    posterior distribution, the class or a callable can be given for later
    initialization.
  prior : {`None`, `tensorflow_probability.Distribution`}
    prior distribution, used for calculating KL divergence later.
  use_bias : `bool` (default=`True`)

  call_mode : `odin.bay.helpers.Statistic` (default=`Statistic.SAMPLE`)

  name : `str` (default='DenseDistribution')

  Return
  ------
  sample, mean, variance, stddev : [n_samples, batch_size, units]
    depend on the `call_mode`, multiple statistics could be returned
  """

  def __init__(
      self,
      units,
      posterior: Union[DistributionLambda, Type[DistributionLambda], Callable],
      prior: Optional[Distribution] = None,
      activation='linear',
      use_bias=True,
      call_mode: Statistic = Statistic.DIST,
      posterior_kwargs={},
      name="DenseDistribution"):
    assert prior is None or isinstance(prior, Distribution), \
     "prior can be None or instance of tensorflow_probability.Distribution"
    assert isinstance(call_mode, Statistic), \
      "call_mode must be instance of odin.bay.helpers.Statistic"
    units = int(units)
    use_bias = bool(use_bias)
    # process the posterior
    if isinstance(posterior, DistributionLambda):
      layer = posterior
    elif (isinstance(posterior, type) and
          issubclass(posterior, DistributionLambda)) or \
          isinstance(posterior, Callable):
      args = inspect.getfullargspec(posterior).args
      posterior_kwargs = {
          i: j for i, j in posterior_kwargs.items() if i in args
      }
      layer = posterior(units, **posterior_kwargs)
    else:
      raise ValueError("No support for posterior of type: %s" %
                       str(type(layer)))
    # layer must be DistributionLambda
    assert isinstance(layer, DistributionLambda), \
      "The callable must return instance of DistributionLambda, but given: %s" \
        % (str(type(layer)))
    # create layers
    params_size = layer.params_size(units)
    layers = [
        Dense(params_size, activation=activation, use_bias=use_bias), layer
    ]
    super(DenseDistribution, self).__init__(layers=layers, name=name)
    # basics
    self._units = units
    self._params_size = params_size
    self._posterior = posterior
    self._prior = prior
    self._activation = activation
    self._use_bias = bool(use_bias)
    self._call_mode = call_mode
    self._last_distribution = None
    self._posterior_kwargs = posterior_kwargs
    # check class init
    for arg in inspect.getfullargspec(self.__init__).args:
      if arg not in locals():
        raise ValueError("Invalid subclassing %s, the arguments to __init__ "
                         "must contain DenseDistribution arguments." %
                         str(self.__class__))

  def get_config(self):
    config = {
        'name': self.name,
        'configs': {
            'units': self._units,
            'posterior': self._posterior,
            'prior': self._prior,
            'activation': self._activation,
            'use_bias': self._use_bias,
            'call_mode': self._call_mode,
            'posterior_kwargs': self._posterior_kwargs
        }
    }
    if self._build_input_shape:
      config['build_input_shape'] = self._build_input_shape
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'name' in config:
      name = config['name']
      build_input_shape = config.get('build_input_shape')
    else:
      name = None
      build_input_shape = None
    kw = {
        i: config['configs'][i]
        for i in inspect.getfullargspec(cls.__init__).args
        if i in config['configs']
    }
    model = cls(name=name, **kw)
    if not model.inputs and build_input_shape:
      model.build(build_input_shape)
    return model

  @property
  def prior(self):
    return self._prior

  @property
  def distribution_layer(self):
    return self.layers[-1]

  @property
  def posterior(self):
    """ Return the last parametrized distribution, i.e. the result from `call`
    """
    return self._last_distribution

  @property
  def units(self):
    return self._units

  def _apply_distribution(self, x):
    if hasattr(x, '_distribution') and \
      x._distribution == self._last_distribution:
      dist = x._distribution
    else:
      dist = super(DenseDistribution, self).call(x)
    return dist

  def mean(self, x):
    dist = self._apply_distribution(x)
    y = Moments(mean=True, variance=False)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def variance(self, x):
    dist = self._apply_distribution(x)
    y = Moments(mean=False, variance=True)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def stddev(self, x):
    v = self.variance(x)
    y = Lambda(tf.math.sqrt)(v)
    setattr(y, '_distribution', v._distribution)
    return y

  def sample(self, x, n_samples=1):
    if n_samples is None or n_samples <= 0:
      n_samples = 1
    dist = self._apply_distribution(x)
    y = Sampling(n_samples=n_samples)(dist)
    setattr(y, '_distribution', dist)
    self._last_distribution = y._distribution
    return y

  def call(self, x, training=None, n_samples=1, mode=None):
    """
    Parameters
    ----------
    x : {`numpy.ndarray`, `tensorflow.Tensor`}

    training : {`None`, `bool`} (default=`None`)

    n_samples : {`None`, `int`} (default=`1`)

    mode : {`None`, `odin.bay.helpers.Statistic`} (default=`None`)
      decide which of the statistics will be return from the distribution,
      this value will overide the default value of the class
    """
    results = []
    variance = None
    call_mode = self._call_mode if not isinstance(mode, Statistic) else mode
    # special case only need the distribution
    if Statistic.DIST == call_mode:
      dist = super(DenseDistribution, self).call(x)
      self._last_distribution = dist
      return dist
    # convert to tensor modes
    if Statistic.SAMPLE in call_mode:
      results.append(self.sample(x, n_samples=n_samples))
    if Statistic.MEAN in call_mode:
      results.append(self.mean(x))
    if Statistic.VAR in call_mode or Statistic.STDDEV in call_mode:
      variance = self.variance(x)
      if Statistic.VAR in call_mode:
        results.append(variance)
      if Statistic.STDDEV in call_mode:
        y = Lambda(tf.math.sqrt)(variance)
        setattr(y, '_distribution', variance._distribution)
        results.append(y)
    if Statistic.DIST in call_mode:
      assert len(results) > 0
      results.append(results[0]._distribution)
    return results[0] if len(results) == 1 else tuple(results)

  def kl_divergence(self, x=None, prior=None, analytic_kl=True, n_samples=1):
    """
    Parameters
    ---------
    x : `Tensor`
      optional input for parametrizing the distribution, if not given,
      used the last result from `call`

    prior : instance of `tensorflow_probability.Distribution`
      prior distribution of the latent

    analytic_kl : `bool` (default=`True`)
      using closed form solution for calculating divergence,
      otherwise, sampling with MCMC

    n_samples : `int` (default=`1`)
      number of MCMC sample if `analytic_kl=False`

    Return
    ------
    kullback_divergence : Tensor [n_samples, batch_size, ...]
    """
    if n_samples is None:
      n_samples = 1
    if prior is None:
      prior = self._prior
    elif self._prior is None:
      self._prior = prior
    assert isinstance(prior, Distribution), "prior is not given!"

    if x is not None:
      self(x, mode=Statistic.DIST)
    if self.posterior is None:
      raise RuntimeError(
          "DenseDistribution must be called to create the distribution before "
          "calculating the kl-divergence.")

    kullback_div = kl_divergence(q=self.posterior,
                                 p=prior,
                                 use_analytic_kl=bool(analytic_kl),
                                 q_sample=int(n_samples),
                                 auto_remove_independent=True)
    if analytic_kl:
      kullback_div = tf.expand_dims(kullback_div, axis=0)
      if n_samples > 1:
        ndims = kullback_div.shape.ndims
        kullback_div = tf.tile(kullback_div, [n_samples] + [1] * (ndims - 1))
    return kullback_div

  def log_prob(self, x):
    """ Calculating the log probability (i.e. log likelihood) """
    assert self.units == x.shape[-1], \
      "Number of features mismatch, units=%d  input_shape=%s" % \
        (self.units, str(x.shape))
    if self.posterior is None:
      raise RuntimeError(
          "DenseDistribution must be called to create the distribution before "
          "calculating the log-likelihood.")
    return self.posterior.log_prob(x)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return '<DenseDistribution units:%d #params:%g posterior:%s prior:%s>' %\
      (self.units, self._params_size / self.units,
       self.layers[-1].__class__.__name__,
       self.prior.__class__.__name__)
