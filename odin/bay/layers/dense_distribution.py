from __future__ import absolute_import, division, print_function

import inspect
from numbers import Number
from types import LambdaType
from typing import Any, Callable, Dict, List, Optional, Union, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from six import string_types
from tensorflow import Tensor
from tensorflow.python.keras import Model
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras.initializers.initializers_v2 import Initializer
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.regularizers import Regularizer
from tensorflow.python.training.tracking import base as trackable
from tensorflow_probability.python.bijectors import FillScaleTriL
from tensorflow_probability.python.distributions import (
  Categorical, Distribution, Independent, MixtureSameFamily,
  MultivariateNormalDiag, MultivariateNormalTriL, Normal)
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import \
  DistributionLambda

from odin import backend as bk
from odin.bay.helpers import (KLdivergence, coercible_tensor,
                              is_binary_distribution, is_discrete_distribution,
                              is_mixture_distribution,
                              is_zeroinflated_distribution, kl_divergence)
from odin.networks import NetConf
from odin.utils import as_tuple

__all__ = [
  'DenseDeterministic',
  'DistributionDense',
  'MixtureDensityNetwork',
  'MixtureMassNetwork',
  'DistributionNetwork',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _params_size(layer, event_shape, **kwargs):
  spec = inspect.getfullargspec(layer.params_size)
  args = spec.args + spec.kwonlyargs
  if 'event_size' == args[0]:
    event_shape = tf.reduce_prod(event_shape)
  # extra kwargs from function closure
  kw = {}
  if len(args) > 1:
    fn = layer._make_distribution_fn
    closures = {
      k: v.cell_contents
      for k, v in zip(fn.__code__.co_freevars, fn.__closure__)
    }
    for k in args[1:]:
      if k in closures:
        kw[k] = closures[k]
  kw.update({k: v for k, v in kwargs.items() if k in spec.args})
  return layer.params_size(event_shape, **kw)


def _get_all_args(fn):
  spec = inspect.getfullargspec(fn)
  return spec.args + spec.kwonlyargs


# ===========================================================================
# Main classes
# ===========================================================================
class DistributionDense(Layer):
  """ Using `Dense` layer to parameterize the tensorflow_probability
  `Distribution`

  Parameters
  ----------
  event_shape : List[int]
      distribution event shape, by default ()
  posterior : {str, DistributionLambda, Callable[..., Distribution]}
      Instrution for creating the posterior distribution, could be one of
      the following:
      - string : alias of the distribution, e.g. 'normal', 'mvndiag', etc.
      - DistributionLambda : an instance or type.
      - Callable : a callable that accept a Tensor as inputs and return a Distribution.
  posterior_kwargs : Dict[str, Any], optional
      keywords arguments for initialize the DistributionLambda if a type is
      given as posterior.
  prior : Union[Distribution, Callable[[], Distribution]]
      prior Distribution, or a callable which return a prior.
  autoregressive: bool
      using maksed autoregressive dense network, by default False
  dropout : float, optional
      dropout on the dense layer, by default 0.0
  projection : bool, optional
      enable dense layers for projecting the inputs into parameters for distribution,
      by default True
  flatten_inputs : bool, optional
      flatten to 2D, by default False
  units : Optional[int], optional
      explicitly given total number of distribution parameters, by default None

  Return
  -------
  `tensorflow_probability.Distribution`
  """

  def __init__(
      self,
      event_shape: Union[int, Sequence[int]] = (),
      posterior: Union[str, DistributionLambda,
                       Callable[[Tensor], Distribution]] = 'normal',
      posterior_kwargs: Optional[Dict[str, Any]] = None,
      prior: Optional[Union[Distribution, Callable[[], Distribution]]] = None,
      convert_to_tensor_fn: Callable[
        [Distribution], Tensor] = Distribution.sample,
      activation: Union[str, Callable[[Tensor], Tensor]] = 'linear',
      autoregressive: bool = False,
      use_bias: bool = True,
      kernel_initializer: Union[str, Initializer] = 'glorot_normal',
      bias_initializer: Union[str, Initializer] = 'zeros',
      kernel_regularizer: Union[None, str, Regularizer] = None,
      bias_regularizer: Union[None, str, Regularizer] = None,
      activity_regularizer: Union[None, str, Regularizer] = None,
      kernel_constraint: Union[None, str, Constraint] = None,
      bias_constraint: Union[None, str, Constraint] = None,
      dropout: float = 0.0,
      projection: bool = True,
      flatten_inputs: bool = False,
      units: Optional[int] = None,
      **kwargs,
  ):
    if posterior_kwargs is None:
      posterior_kwargs = {}
    ## store init arguments (this is not intended for serialization but
    # for cloning)
    init_args = dict(locals())
    del init_args['self']
    del init_args['__class__']
    del init_args['kwargs']
    init_args.update(kwargs)
    self._init_args = init_args
    ## check prior type
    assert isinstance(prior, (Distribution, type(None))) or callable(prior), \
      ("prior can only be None or instance of Distribution, DistributionLambda"
       f",  but given: {prior}-{type(prior)}")
    self._projection = bool(projection)
    self.flatten_inputs = bool(flatten_inputs)
    ## duplicated event_shape or event_size in posterior_kwargs
    posterior_kwargs = dict(posterior_kwargs)
    if 'event_shape' in posterior_kwargs:
      event_shape = posterior_kwargs.pop('event_shape')
    if 'event_size' in posterior_kwargs:
      event_shape = posterior_kwargs.pop('event_size')
    convert_to_tensor_fn = posterior_kwargs.pop('convert_to_tensor_fn',
                                                Distribution.sample)
    ## process the posterior
    self._posterior_layer = None
    self._callable_posterior = False
    if isinstance(posterior, DistributionLambda):
      self._posterior_layer = posterior
      self._posterior_class = type(posterior)
    elif inspect.isclass(posterior) and issubclass(posterior,
                                                   DistributionLambda):
      self._posterior_class = posterior
    elif isinstance(posterior, string_types):
      from odin.bay.distribution_alias import parse_distribution
      self._posterior_class, _ = parse_distribution(posterior)
    elif callable(posterior):
      self._callable_posterior = True
      if isinstance(posterior, LambdaType):
        posterior = tf.autograph.experimental.do_not_convert(posterior)
      self._posterior_layer = posterior
      self._posterior_class = type(posterior)
    else:
      raise ValueError('posterior could be: string, DistributionLambda, '
                       f'callable or type; but give: {posterior}')
    self._posterior = posterior
    self._posterior_kwargs = posterior_kwargs
    self._posterior_sample_shape = ()
    ## create layers
    self._convert_to_tensor_fn = convert_to_tensor_fn
    self._prior = prior
    self._event_shape = event_shape
    self._dropout = dropout
    ## set more descriptive name
    name = kwargs.pop('name', None)
    if name is None:
      posterior_name = (posterior if isinstance(posterior, string_types) else
                        posterior.__class__.__name__)
      name = f'dense_{posterior_name}'
    kwargs['name'] = name
    ## params_size could be static function or method
    if not projection:
      self._params_size = 0
    else:
      if not hasattr(self.posterior_layer, 'params_size'):
        if units is None:
          raise ValueError(
            f'posterior layer of type {type(self.posterior_layer)} '
            "doesn't has method params_size, number of parameters "
            'must be provided as `units` argument, but given: None')
        self._params_size = int(units)
      else:
        self._params_size = int(
          _params_size(self.posterior_layer, event_shape,
                       **self._posterior_kwargs))
    super().__init__(**kwargs)
    self.autoregressive = autoregressive
    if autoregressive:
      from odin.bay.layers.autoregressive_layers import AutoregressiveDense
      self._dense = AutoregressiveDense(
        params=self._params_size / self.event_size,
        event_shape=(self.event_size,),
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint)
    else:
      self._dense = Dense(units=self._params_size,
                          activation=activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activity_regularizer=activity_regularizer,
                          kernel_constraint=kernel_constraint,
                          bias_constraint=bias_constraint)
    # store the distribution from last call,
    self._most_recently_built_distribution = None
    # We'll need to keep track of who's calling who since the functional
    # API has a different way of injecting `_keras_history` than the
    # `keras.Sequential` way.
    self._enter_dunder_call = False
    spec = inspect.getfullargspec(self.posterior_layer)
    self._posterior_call_kw = set(spec.args + spec.kwonlyargs)

  def build(self, input_shape) -> 'DistributionDense':
    self._dense.build(input_shape)
    return self

  @property
  def params_size(self) -> int:
    return self._params_size

  @property
  def projection(self) -> bool:
    return self._projection and self.params_size > 0

  @property
  def is_binary(self) -> bool:
    return is_binary_distribution(self.posterior_layer)

  @property
  def is_discrete(self) -> bool:
    return is_discrete_distribution(self.posterior_layer)

  @property
  def is_mixture(self) -> bool:
    return is_mixture_distribution(self.posterior_layer)

  @property
  def is_zero_inflated(self) -> bool:
    return is_zeroinflated_distribution(self.posterior_layer)

  @property
  def event_shape(self) -> List[int]:
    shape = self._event_shape
    if not (tf.is_tensor(shape) or isinstance(shape, tf.TensorShape)):
      shape = tf.nest.flatten(shape)
    return shape

  @property
  def event_size(self) -> int:
    return tf.cast(tf.reduce_prod(self._event_shape), tf.int32)

  @property
  def prior(self) -> Optional[Union[Distribution, Callable[[], Distribution]]]:
    return self._prior

  @prior.setter
  def prior(self,
            p: Optional[Union[Distribution, Callable[[],
                                                     Distribution]]] = None):
    self._prior = p

  def set_prior(self,
                p: Optional[Union[Distribution,
                                  Callable[[], Distribution]]] = None):
    self.prior = p
    return self

  def _sample_fn(self, dist):
    return dist.sample(sample_shape=self._posterior_sample_shape)

  @property
  def convert_to_tensor_fn(self) -> Callable[..., Tensor]:
    if self._convert_to_tensor_fn == Distribution.sample:
      return self._sample_fn
    else:
      return self._convert_to_tensor_fn

  @property
  def posterior_layer(
      self) -> Union[DistributionLambda, Callable[..., Distribution]]:
    if self._callable_posterior:
      ...
    elif not isinstance(self._posterior_layer, DistributionLambda):
      self._posterior_layer = self._posterior_class(
        self._event_shape,
        convert_to_tensor_fn=self.convert_to_tensor_fn,
        **self._posterior_kwargs)
    return self._posterior_layer

  @property
  def posterior(self) -> Distribution:
    r""" Return the most recent parametrized distribution,
    i.e. the result from the last `call` """
    return self._most_recently_built_distribution

  @tf.function
  def sample(self, sample_shape=(), seed=None):
    r""" Sample from prior distribution """
    if self._prior is None:
      raise RuntimeError("prior hasn't been provided for the %s" %
                         self.__class__.__name__)
    return self.prior.sample(sample_shape=sample_shape, seed=seed)

  def __call__(self, inputs, *args, **kwargs):
    if self._callable_posterior:
      self._enter_dunder_call = True
      distribution, _ = super().__call__(inputs, *args, **kwargs)
      self._enter_dunder_call = False
    else:
      distribution = super().__call__(inputs, *args, **kwargs)
    return distribution

  def call(self,
           inputs,
           training=None,
           sample_shape=(),
           projection=None,
           **kwargs):
    ## NOTE: a 2D inputs is important here, but we don't want to flatten
    # automatically
    if self.flatten_inputs:
      inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
    params = inputs
    ## do not use tf.cond here, it infer the wrong shape when
    # trying to build the layer in Graph mode.
    projection = projection if projection is not None else self.projection
    if projection:
      params = self._dense(params)
      if self.autoregressive:
        params = tf.concat(tf.unstack(params, axis=-1), axis=-1)
    ## applying dropout
    if self._dropout > 0:
      params = bk.dropout(params, p_drop=self._dropout, training=training)
    ## create posterior distribution
    self._posterior_sample_shape = sample_shape
    kw = dict()
    if 'training' in self._posterior_call_kw:
      kw['training'] = training
    if 'sample_shape' in self._posterior_call_kw:
      kw['sample_shape'] = sample_shape
    for k, v in kwargs.items():
      if k in self._posterior_call_kw:
        kw[k] = v
    posterior = self.posterior_layer(params, **kw)
    # tensorflow tries to serialize the distribution, which raise exception
    # when saving the graphs, to avoid this, store it as non-tracking list.
    with trackable.no_automatic_dependency_tracking_scope(self):
      self._most_recently_built_distribution = posterior
    ## NOTE: all distribution has the method kl_divergence, so we cannot use it
    posterior.KL_divergence = KLdivergence(
      posterior, prior=self.prior,
      sample_shape=None)  # None mean reuse sampled data here
    ## special case callable (act as DistributionLambda)
    if self._callable_posterior:
      posterior, value = coercible_tensor(posterior,
                                          self.convert_to_tensor_fn,
                                          return_value=True)
      if self._enter_dunder_call:
        # Its critical to return both distribution and concretization
        # so Keras can inject `_keras_history` to both. This is what enables
        # either to be used as an input to another Keras `Model`.
        return posterior, value
    return posterior

  def kl_divergence(self,
                    prior=None,
                    analytic=True,
                    sample_shape=1,
                    reverse=True):
    """ KL(q||p) where `p` is the posterior distribution returned from last
    call

    Parameters
    -----------
    prior : instance of `tensorflow_probability.Distribution`
        prior distribution of the latent
    analytic : `bool` (default=`True`). Using closed form solution for
        calculating divergence, otherwise, sampling with MCMC
    reverse : `bool`.
        If `True`, calculate `KL(q||p)` else `KL(p||q)`
    sample_shape : `int` (default=`1`)
        number of MCMC sample if `analytic=False`

    Returns
    --------
      kullback_divergence : Tensor [sample_shape, batch_size, ...]
    """
    if prior is None:
      prior = self._prior
    assert isinstance(prior, Distribution), "prior is not given!"
    if self.posterior is None:
      raise RuntimeError(
        "DistributionDense must be called to create the distribution before "
        "calculating the kl-divergence.")

    kullback_div = kl_divergence(q=self.posterior,
                                 p=prior,
                                 analytic=bool(analytic),
                                 reverse=reverse,
                                 q_sample=sample_shape)
    if analytic:
      kullback_div = tf.expand_dims(kullback_div, axis=0)
      if isinstance(sample_shape, Number) and sample_shape > 1:
        ndims = kullback_div.shape.ndims
        kullback_div = tf.tile(kullback_div, [sample_shape] + [1] * (ndims - 1))
    return kullback_div

  def log_prob(self, x):
    r""" Calculating the log probability (i.e. log likelihood) using the last
    distribution returned from call """
    return self.posterior.log_prob(x)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    if self.prior is None:
      prior = 'None'
    elif isinstance(self.prior, Distribution):
      prior = (
        f"<{self.prior.__class__.__name__} "
        f"batch:{self.prior.batch_shape} event:{self.prior.event_shape}>")
    else:
      prior = str(self.prior)
    posterior = self._posterior_class.__name__
    if hasattr(self, 'input_shape'):
      inshape = self.input_shape
    else:
      inshape = None
    if hasattr(self, 'output_shape'):
      outshape = self.output_shape
    else:
      outshape = None
    return (
      f"<'{self.name}' autoregr:{self.autoregressive} proj:{self.projection} "
      f"in:{inshape} out:{outshape} event:{self.event_shape} "
      f"#params:{self._params_size} post:{posterior} prior:{prior} "
      f"dropout:{self._dropout:.2f} kw:{self._posterior_kwargs}>")

  def get_config(self) -> dict:
    return dict(self._init_args)


# ===========================================================================
# Shortcuts
# ===========================================================================
class MixtureDensityNetwork(DistributionDense):
  r""" Mixture Density Network

  Mixture of Gaussian parameterized by neural network

  For arguments information: `odin.bay.layers.mixture_layers.MixtureGaussianLayer`
  """

  def __init__(
      self,
      units: int,
      n_components: int = 2,
      covariance: str = 'none',
      tie_mixtures: bool = False,
      tie_loc: bool = False,
      tie_scale: bool = False,
      loc_activation: Union[str, Callable] = 'linear',
      scale_activation: Union[str, Callable] = 'softplus1',
      convert_to_tensor_fn: Callable = Distribution.sample,
      use_bias: bool = True,
      dropout: float = 0.0,
      kernel_initializer: Union[str, Initializer, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Initializer, Callable] = 'zeros',
      kernel_regularizer: Union[str, Regularizer, Callable] = None,
      bias_regularizer: Union[str, Regularizer, Callable] = None,
      activity_regularizer: Union[str, Regularizer, Callable] = None,
      kernel_constraint: Union[str, Constraint, Callable] = None,
      bias_constraint: Union[str, Constraint, Callable] = None,
      **kwargs,
  ):
    self.covariance = covariance
    self.n_components = n_components
    super().__init__(event_shape=units,
                     posterior='mixgaussian',
                     posterior_kwargs=dict(n_components=int(n_components),
                                           covariance=str(covariance),
                                           loc_activation=loc_activation,
                                           scale_activation=scale_activation,
                                           tie_mixtures=bool(tie_mixtures),
                                           tie_loc=bool(tie_loc),
                                           tie_scale=bool(tie_scale)),
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

  def set_prior(self,
                loc=0.,
                log_scale=np.log(np.expm1(1)),
                mixture_logits=None):
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
        loc=l, scale_tril=FillScaleTriL(diag_shift=1e-5)(tf.math.softplus(s)))
    #
    if isinstance(log_scale, Number) or tf.rank(log_scale) == 0:
      loc = tf.fill([self.n_components, self.event_size], loc)
    #
    if isinstance(log_scale, Number) or tf.rank(log_scale) == 0:
      log_scale = tf.fill(scale_shape, log_scale)
    #
    if mixture_logits is None:
      p = 1. / self.n_components
      mixture_logits = np.log(p / (1. - p))
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


class MixtureMassNetwork(DistributionDense):
  r""" Mixture Mass Network

  Mixture of NegativeBinomial parameterized by neural network
  """

  def __init__(
      self,
      event_shape: List[int] = (),
      n_components: int = 2,
      dispersion: str = 'full',
      inflation: str = 'full',
      tie_mixtures: bool = False,
      tie_mean: bool = False,
      mean_activation: Union[str, Callable] = 'softplus1',
      disp_activation: Union[str, Callable] = None,
      alternative: bool = False,
      zero_inflated: bool = False,
      convert_to_tensor_fn: Callable = Distribution.sample,
      dropout: float = 0.0,
      use_bias: bool = True,
      kernel_initializer: Union[str, Initializer, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Initializer, Callable] = 'zeros',
      kernel_regularizer: Union[str, Regularizer, Callable] = None,
      bias_regularizer: Union[str, Regularizer, Callable] = None,
      activity_regularizer: Union[str, Regularizer, Callable] = None,
      kernel_constraint: Union[str, Constraint, Callable] = None,
      bias_constraint: Union[str, Constraint, Callable] = None,
      **kwargs,
  ):
    self.n_components = n_components
    self.dispersion = dispersion
    self.zero_inflated = zero_inflated
    self.alternative = alternative
    super().__init__(event_shape=event_shape,
                     posterior='mixnb',
                     prior=None,
                     posterior_kwargs=dict(n_components=int(n_components),
                                           mean_activation=mean_activation,
                                           disp_activation=disp_activation,
                                           dispersion=dispersion,
                                           inflation=inflation,
                                           alternative=alternative,
                                           zero_inflated=zero_inflated,
                                           tie_mixtures=bool(tie_mixtures),
                                           tie_mean=bool(tie_mean)),
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


class DenseDeterministic(DistributionDense):
  r""" Similar to `keras.Dense` layer but return a
  `tensorflow_probability.VectorDeterministic` distribution to represent
  the output, hence, making it compatible to the probabilistic framework.
  """

  def __init__(
      self,
      units: int,
      dropout: float = 0.0,
      activation: Union[str, Callable] = 'linear',
      use_bias: bool = True,
      kernel_initializer: Union[str, Initializer, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Initializer, Callable] = 'zeros',
      kernel_regularizer: Union[str, Regularizer, Callable] = None,
      bias_regularizer: Union[str, Regularizer, Callable] = None,
      activity_regularizer: Union[str, Regularizer, Callable] = None,
      kernel_constraint: Union[str, Constraint, Callable] = None,
      bias_constraint: Union[str, Constraint, Callable] = None,
      **kwargs,
  ):
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


class DistributionNetwork(Model):
  """A simple sequential network that will output a Distribution
  or multiple Distrubtions

  Parameters
  ----------
  distributions : List[Layer]
      List of output Layers that parameterize the Distrubtions
  network : Union[Layer, NetConf], optional
      a network
  name : str, optional
      by default 'DistributionNetwork'
  """

  def __init__(
      self,
      distributions: List[Layer],
      network: Union[Layer, NetConf] = NetConf([128, 128], flatten_inputs=True),
      name: str = 'DistributionNetwork',
  ):
    super().__init__(name=name)
    ## prepare the preprocessing layers
    if isinstance(network, NetConf):
      network = network.create_network()
    assert isinstance(network, Layer), \
      f'network must be instance of keras.layers.Layer but given {network}'
    self.network = network
    ## prepare the output distribution
    from odin.bay.random_variable import RVconf
    self.distributions = []
    for d in as_tuple(distributions):
      if isinstance(d, RVconf):
        d = d.create_posterior()
      assert isinstance(d, Layer), \
        ('distributions must be a list of Layer that return Distribution '
         f'in call(), but given {d}')
      self.distributions.append(d)
    # others
    self.network_kws = _get_all_args(self.network.call)
    self.distributions_kws = [_get_all_args(d.call) for d in self.distributions]

  def build(self, input_shape) -> 'DistributionNetwork':
    super().build(input_shape)
    return self

  def preprocess(self, inputs, **kwargs):
    hidden = self.network(
      inputs, **{k: v for k, v in kwargs.items() if k in self.network_kws})
    return hidden

  def call(self, inputs, **kwargs):
    hidden = self.preprocess(inputs, **kwargs)
    # applying the distribution transformation
    outputs = []
    for dist, args in zip(self.distributions, self.distributions_kws):
      o = dist(hidden, **{k: v for k, v in kwargs.items() if k in args})
      outputs.append(o)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

  def __str__(self):
    from odin.backend.keras_helpers import layer2text
    shape = (self.network.input_shape
             if hasattr(self.network, 'input_shape') else None)
    s = f'[DistributionNetwork]{self.name}'
    s += f'\n input_shape:{shape}\n '
    s += '\n '.join(layer2text(self.network).split('\n'))
    s += '\n Distribution:'
    for d in self.distributions:
      s += f'\n  {d}'
    return s
