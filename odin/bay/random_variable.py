from __future__ import absolute_import, division, print_function

import dataclasses
import inspect
import types
from copy import deepcopy
from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from six import string_types
from tensorflow.python import array_ops, keras

from odin.bay import distributions as obd
from odin.bay import layers as obl
from odin.bay.distribution_alias import parse_distribution
from odin.utils.cache_utils import cache_memory

__all__ = ['RandomVariable']


# ===========================================================================
# Helper
# ===========================================================================
def _args_and_defaults(func):
  spec = inspect.getfullargspec(func)
  if spec.defaults is not None:
    defaults = {i: j for i, j in zip(spec.args[::-1], spec.defaults[::-1])}
  else:
    defaults = {}
  if spec.kwonlydefaults is not None:
    defaults.update(spec.kwonlydefaults)
  return spec.args + spec.kwonlyargs, defaults


def _default_prior(event_shape, layer, prior):
  if prior is not None:
    return prior
  ## Normal
  if layer == obl.GaussianLayer:
    prior = obd.Independent(
        obd.Normal(loc=tf.zeros(shape=event_shape),
                   scale=tf.ones(shape=event_shape)), 1)
  ## Multivariate Normal
  elif issubclass(layer, obl.MultivariateNormalLayer):
    cov = layer._partial_kwargs['covariance']
    if cov == 'diag':  # diagonal covariance
      loc = tf.zeros(shape=event_shape)
      if tf.rank(loc) == 0:
        loc = tf.expand_dims(loc, axis=-1)
      prior = obd.MultivariateNormalDiag(loc=loc, scale_identity_multiplier=1.)
    else:  # low-triangle covariance
      bijector = tfp.bijectors.FillScaleTriL(
          diag_bijector=tfp.bijectors.Identity(), diag_shift=1e-5)
      size = tf.reduce_prod(event_shape)
      loc = tf.zeros(shape=[size])
      scale_tril = bijector.forward(tf.ones(shape=[size * (size + 1) // 2]))
      prior = obd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
  ## Log Normal
  elif layer == obl.LogNormalLayer:
    prior = obd.Independent(
        obd.LogNormal(loc=tf.zeros(shape=event_shape),
                      scale=tf.ones(shape=event_shape)), 1)
  return prior


# ===========================================================================
# Main-Method
# ===========================================================================
@dataclasses.dataclass(init=True,
                       repr=True,
                       eq=True,
                       order=False,
                       unsafe_hash=False,
                       frozen=True)
class RandomVariable:
  r""" Description of a random variable for the Bayesian model.

  Arguments:
    event_shape : a tuple of Integer. The shape tuple of distribution
      event shape
    posterior : a String. Alias for a distribution, for examples:
      - 'bern' : `Bernoulli` distribution
      - ('pois', 'poisson'): `Poisson` distribution
      - ('norm', 'gaus') : `IndependentGaussian` distribution
      - 'diag' : diagonal multivariate Gaussian distribution
      - 'tril' : full (or lower triangle) multivariate Gaussian distribution
      - 'lognorm' : LogNormal distribution
      - 'nb' : negative binomial
      - 'nbd' : negative binomial using mean-dispersion parameterization
      - 'zinb' or 'zinbd' : zero-inflated negative binomial
      - 'mdn' : mixture density network (`IndependentNormal` components)
      - 'mixdiag' : mixture of multivariate diagonal normals
      - 'mixtril' : mixture of multivariate full or triL (lower-triangle) normals
      - 'vdeterministic' : vectorized deterministic distribution
      or loss function named in `tensorflow.losses` or `keras.activations`,
      then a VectorDeterministic distribution is created and the `log_prob`
      function is replaced with given loss function, for example:
      - 'binary_crossentropy'
      - 'categorical_crossentropy'
      - 'categorical_hinge'
      - 'cosine_similarity'
      - 'mean_absolute_error'
      - 'mean_squared_error'
    name : a String. Identity of the random variable.
    kwargs : a Dictionary. Keyword arguments for initializing the
      `DistributionLambda` of the posterior.

  Example:
    x = RandomVariable(event_shape=12, posterior='gaus')
    dist = x.create_posterior()
  """
  event_shape: List[int] = ()
  posterior: str = 'gaus'
  prior: str = None
  name: str = 'RandomVariable'
  kwargs: dict = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    assert isinstance(self.posterior, string_types)

  ######## Basic methods
  def keys(self):
    for i in dataclasses.fields(self):
      yield i.name

  def values(self):
    for i in dataclasses.fields(self):
      yield i.default

  def __iter__(self):
    for i in dataclasses.fields(self):
      yield i.name, i.default

  def __len__(self):
    return len(dataclasses.fields(self))

  def __getitem__(self, key):
    return getattr(self, key)

  def copy(self, **kwargs):
    obj = deepcopy(self)
    return dataclasses.replace(obj, **kwargs)

  ######## query distribution type
  @cache_memory
  def _dummy_dist(self, remove_independent=True):
    # deterministic case
    if self.is_deterministic:
      return obd.VectorDeterministic(loc=(0.,))
    # stochastic
    layer, _ = parse_distribution(self.posterior)
    # extra kwargs for params_size
    args, defaults = _args_and_defaults(layer.params_size)
    _, init_defaults = _args_and_defaults(layer.__init__)
    kw = {}
    if len(args) > 1:
      args = args[1:]
      for a in args:
        if a in self.kwargs:
          kw[a] = self.kwargs[a]
        elif a in defaults:
          kw[a] = defaults[a]
        elif a in init_defaults:
          kw[a] = init_defaults[a]
    # get the params_size
    if inspect.getfullargspec(layer.params_size).args[0] == 'event_size':
      size = layer.params_size(1, **kw)
      event_shape = 1
    else:
      size = layer.params_size(1, **kw)
      event_shape = (1,)
    param_shape = (1, size)
    # create a dummy dist
    params = array_ops.empty(shape=param_shape, dtype=tf.float32)
    dist = layer(event_shape)(params)
    # get original distribution
    if remove_independent:
      while isinstance(dist, obd.Independent):
        dist = dist.distribution
    return dist

  @property
  def is_mixture(self):
    dist = self._dummy_dist()
    if isinstance(dist, (obd.Mixture, obd.MixtureSameFamily)):
      return True
    return False

  @property
  def is_zero_inflated(self):
    dist = self._dummy_dist()
    if isinstance(dist, obd.ZeroInflated):
      return True
    return False

  @property
  def is_deterministic(self):
    if 'deterministic' in self.posterior:
      return True
    if self.posterior in dir(tf.losses) or \
      self.posterior in dir(keras.activations):
      return True
    return False

  ######## create posterior distribution
  def create_posterior(self, name=None) -> obl.DistributionLambda:
    r""" Initiate a Distribution for the random variable """
    prior = self.prior
    event_shape = tf.nest.flatten(self.event_shape)
    posterior = self.posterior
    kwargs = dict(self.kwargs)
    llk_fn = None  # custom log_prob
    name = self.name if name is None else str(name)
    # ====== deterministic distribution with loss function from tensorflow ====== #
    if posterior in dir(tf.losses) or posterior in dir(keras.activations):
      distribution_layer = obl.VectorDeterministicLayer
      if posterior in dir(tf.losses):
        activation = kwargs.pop('activation', 'relu')
        fn = tf.losses.get(str(posterior))
      else:  # just activation function, loss default MSE
        activation = keras.activations.get(self.posterior)
        fn = tf.losses.get(kwargs.pop('loss', 'mse'))
      llk_fn = lambda self, y_true: -fn(y_true, self.posterior.mean())
    # ====== probabilistic loss ====== #
    else:
      distribution_layer = parse_distribution(self.posterior)[0]
      activation = 'linear'
    # ====== create distribution layers ====== #
    activation = kwargs.pop('activation', activation)
    if posterior in ('mdn', 'mixdiag', 'mixfull', 'mixtril'):
      kwargs.pop('covariance', None)
      layer = obl.MixtureDensityNetwork(event_shape,
                                        loc_activation=activation,
                                        scale_activation='softplus1',
                                        covariance=dict(
                                            mdn='none',
                                            mixdiag='diag',
                                            mixfull='tril',
                                            mixtril='tril')[posterior],
                                        name=name,
                                        **kwargs)
      layer.set_prior()
    else:
      layer = obl.DenseDistribution(
          event_shape,
          posterior=distribution_layer,
          prior=_default_prior(event_shape, distribution_layer, prior),
          activation=activation,
          posterior_kwargs=kwargs,
          name=name,
      )
    ### custom loss as log_prob
    if llk_fn is not None:
      layer.log_prob = types.MethodType(llk_fn, layer)
    return layer
