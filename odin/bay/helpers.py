from __future__ import absolute_import, division, print_function

import collections
import inspect
from enum import Flag, auto
from numbers import Number
from typing import Callable, List, Optional, Text, Union, Sequence

import numpy as np
import tensorflow as tf
from odin.bay import distributions as obd
from odin.utils import as_tuple
from six import string_types
from tensorflow import Tensor
from tensorflow.python import keras
from tensorflow.python.ops import array_ops
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.distributions.joint_distribution import JointDistribution
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.layers.distribution_layer import (
    _get_convert_to_tensor_fn, _serialize)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible as dtc
from tensorflow_probability.python.layers.internal import \
    tensor_tuple as tensor_tuple

__all__ = [
    'print_distribution',
    'coercible_tensor',
    'kl_divergence',
    'is_binary_distribution',
    'is_discrete_distribution',
    'is_mixture_distribution',
    'is_zeroinflated_distribution',
    'concat_distributions',
    'batch_slice',
]


# ===========================================================================
# distribution type
# ===========================================================================
def _dist(dist):
  if isinstance(dist, DistributionLambda):
    dist = dist(keras.Input((None,), None))
  # distribution layer
  if isinstance(dist, tfd.Distribution):
    while isinstance(dist, tfd.Independent):
      dist = dist.distribution
    dist = type(dist)
  elif inspect.isclass(dist) and issubclass(dist, DistributionLambda):
    dist = dist()(array_ops.empty(shape=(1, dist.params_size((1,))),
                                  dtype=tf.float32))
  else:
    raise ValueError("No support for distribution of type: %s" % str(dist))
  # remove unnecessary classes
  dist = [
      t for t in type.mro(dist)
      if issubclass(t, tfd.Distribution) and t not in (
          tfd.Independent, tfd.Distribution, tfd.TransformedDistribution,
          dtc._TensorCoercible)
  ]
  return dist


def is_binary_distribution(dist):
  if isinstance(dist, tfd.Distribution):
    s = dist.sample(100).numpy()
    return np.all(np.unique(s.astype('float32')) == [0., 1.])
  for dist in _dist(dist):
    if issubclass(dist, (obd.OneHotCategorical, obd.RelaxedOneHotCategorical,
                         obd.Bernoulli, obd.RelaxedBernoulli)):
      return True
  return False


def is_discrete_distribution(dist):
  if isinstance(dist, tfd.Distribution):
    s = dist.sample(100).numpy()
    return np.all(s.astype('float32') == s.astype('int32'))
  for dist in _dist(dist):
    if issubclass(dist,
                  (obd.Poisson, obd.NegativeBinomial, obd.NegativeBinomialDisp,
                   obd.Categorical, obd.Binomial, obd.Multinomial)):
      return True
  return False


def is_mixture_distribution(dist):
  for dist in _dist(dist):
    if issubclass(dist, (obd.Mixture, obd.MixtureSameFamily)):
      return True
  return False


def is_zeroinflated_distribution(dist):
  for dist in _dist(dist):
    if issubclass(dist, obd.ZeroInflated):
      return True
  return False


# ===========================================================================
# Logging
# ===========================================================================
def _dist2text(dist):
  cls = dist.__class__.__name__
  return (f"{cls} dtype:{dist.dtype.name} "
          f"batch:{dist.batch_shape} event:{dist.event_shape}")


def _extract_desc(dist, name, pad):
  assert isinstance(dist, tfd.Distribution), \
    f"dist must be instance of Distribution but given {type(dist)}"
  text = f"{pad}{(name + ':' if len(name) > 0 else '')}{_dist2text(dist)}\n"
  pad += " "
  text += f"{pad}Initialization:\n"
  for key, val in sorted(dist.parameters.items()):
    if isinstance(val, tfd.Distribution):
      text += _extract_desc(val, key, f"{pad}  ")
    elif tf.is_tensor(val):
      text += f"{pad}  {key}: {val.shape} {val.dtype.name}\n"
    else:
      text += f"{pad}  {key}: {val}\n"
  text += f"{pad}Tensors:\n"
  for key, val in sorted(inspect.getmembers(dist)):
    if (tf.is_tensor(val) or isinstance(val, np.ndarray)) and \
      key not in dist.parameters:
      text += f"{pad}  {key}: {val.shape} {val.dtype.name}\n"
  return text[:-1]


def print_distribution(dist, return_text=False):
  r""" Special function for printing out distribution information """
  assert isinstance(dist, tfd.Distribution)
  text = _extract_desc(dist, '', '')
  if return_text:
    return text[:-1]
  print(text)


# ===========================================================================
# Objectives
# ===========================================================================
def coercible_tensor(d: tfd.Distribution,
                     convert_to_tensor_fn=tfd.Distribution.sample,
                     return_value: bool = False) -> tfd.Distribution:
  r""" make a distribution convertible to Tensor using the
  `convert_to_tensor_fn`

  This code is copied from: `distribution_layers.py` tensorflow_probability
  """
  assert isinstance(d, tfd.Distribution), \
    "dist must be instance of tensorflow_probability.Distribution"
  convert_to_tensor_fn = _get_convert_to_tensor_fn(convert_to_tensor_fn)
  if inspect.isfunction(convert_to_tensor_fn) and \
    convert_to_tensor_fn in list(tfd.Distribution.__dict__.values()):
    convert_to_tensor_fn = getattr(type(d), convert_to_tensor_fn.__name__)
  # Wraps the distribution to return both dist and concrete value."""
  distribution = dtc._TensorCoercible(distribution=d,
                                      convert_to_tensor_fn=convert_to_tensor_fn)
  ### prepare the value
  value = distribution._value()
  value._tfp_distribution = distribution
  distribution.shape = value.shape
  distribution.get_shape = value.get_shape
  ### return
  if return_value:
    return distribution, value
  return distribution


# ===========================================================================
# Objectives
# ===========================================================================
def kl_divergence(
    q: Union[Distribution, Callable[[], Distribution]],
    p: Union[Distribution, Callable[[], Distribution]],
    analytic: bool = False,
    q_sample: Union[int,
                    Callable[[Distribution], Tensor]] = lambda q: q.sample(),
    reduce_axis: Sequence[int] = (),
    reverse: bool = True,
    free_bits: Optional[float] = None,
) -> Tensor:
  """ Calculating `KL(q(x)||p(x))` (if reverse=True) or
  `KL(p(x)||q(x))` (if reverse=False)

  Parameters
  ----------
  q : `tensorflow_probability.Distribution` or `Callable`,
      the approximated posterior distribution
  p : `tensorflow_probability.Distribution` or `Callable`,
      the prior distribution
  analytic : bool (default: False)
      if True, use the close-form solution  for
  q_sample : {callable, Tensor, Number}
      callable for extracting sample from `q(x)` (takes `q` posterior distribution
      as input argument)
  reduce_axis : {None, int, tuple}. Reduce axis when use MCMC to estimate KL
      divergence, default `()` mean keep all original dimensions.
  reverse : `bool`. If `True`, calculating `KL(q||p)` which optimizes `q`
      (or p_model) by greedily filling in the highest modes of data (or, in
      other word, placing low probability to where data does not occur).
      Otherwise, `KL(p||q)` a.k.a maximum likelihood, place high probability
      at anywhere data occur (i.e. averagely fitting the data).
  free_bits : `float` (optional)
      maximum(lambda, KL) as stated in (Kingma et al. 2016)

  Returns
  -------
  A Tensor with the batchwise KL-divergence between `distribution_a`
      and `distribution_b`.  The shape is `[batch_dims]` for analytic KL,
      otherwise, `[sample_shape, batch_dims]`.

  References
  ----------
  Kingma, D.P., et al., 2016. Improved variational inference with inverse
      autoregressive flow, Advances in Neural Information Processing
      Systems. Curran Associates, Inc., pp. 4743–4751.

  Example
  -------
  ```python
  p = bay.distributions.OneHotCategorical(logits=[1, 2, 3])

  w = bk.variable(np.random.rand(2, 3).astype('float32'))
  q = bay.distributions.OneHotCategorical(w)

  opt = tf.optimizers.Adam(learning_rate=0.01,
                           beta_1=0.9,
                           beta_2=0.999,
                           epsilon=1e-07,
                           amsgrad=False)
  for i in range(1000):
    with tf.GradientTape() as tape:
      kl = bay.kl_divergence(q=q, p=p, q_sample=lambda q: q.sample(1000))
      grads = bk.grad(tf.reduce_mean(kl), w, tape=tape)
      opt.apply_gradients(grads_and_vars=[(g, v) for g, v in zip(grads, [w])])
      if i % 10 == 0:
        print("#%3d KL: %.4f" % (i, tf.reduce_mean(kl).numpy()))
  print(q.sample())
  ```
  """
  if callable(p) and not isinstance(p, Distribution):
    p = p()
    assert isinstance(p, Distribution), \
      f"callable must return a Distribution, but returned: {p}"
  if callable(q) and not isinstance(q, Distribution):
    q = q()
    assert isinstance(q, Distribution), \
      f"callable must return a Distribution, but returned: {q}"
  ### add independent if necessary
  if isinstance(q, tfd.Independent) and not isinstance(p, tfd.Independent):
    p = tfd.Independent(
        p,
        reinterpreted_batch_ndims=len(q.event_shape) - len(p.event_shape),
    )
  ### removing Independent
  if not bool(reverse):
    q, p = [q, p][::-1]
  ### analytic KL
  if bool(analytic):
    kl = tfd.kl_divergence(q, p)
  ### non-analytic KL
  else:
    # using MCMC sampling for estimating the KL
    if callable(q_sample):
      z = q_sample(q)
    elif q_sample is None:  # TensorCoercible
      z = tf.convert_to_tensor(q)
    else:
      z = q.sample(q_sample)
    # calculate the output, then perform reduction
    kl = q.log_prob(z) - p.log_prob(z)
  ### free-bits
  if free_bits is not None:
    kl = tf.maximum(kl, tf.constant(free_bits, dtype=kl.dtype))
  kl = tf.reduce_mean(input_tensor=kl, axis=reduce_axis)
  return kl


class KLdivergence:
  r""" This class freezes the arguments of `kl_divergence` so it could be call
  later without the required arguments.

    - Calculating KL(q(x)||p(x)) (if reverse=True) or
    - KL(p(x)||q(x)) (if reverse=False)

  Parameters
  ----------
  posterior : `tensorflow_probability.Distribution`, the approximated
    posterior distribution
  prior : `tensorflow_probability.Distribution`, the prior distribution
  analytic : bool (default: False)
    if True, use the close-form solution  for
  sample_shape : {Tensor, Number}
    number of MCMC samples for MCMC estimation of KL-divergence
  reverse : `bool`. If `True`, calculating `KL(q||p)` which optimizes `q`
    (or p_model) by greedily filling in the highest modes of data (or, in
    other word, placing low probability to where data does not occur).
    Otherwise, `KL(p||q)` a.k.a maximum likelihood, or expectation
    propagation place high probability at anywhere data occur
    (i.e. averagely fitting the data).
  keepdims : a Boolean. If True, expand the dimension to preserve the MCMC
    dimension in case of analytic KL.

  Note
  ----
  this class return 0. if the prior is not given (i.e. prior=None)
  """

  def __init__(self,
               posterior,
               prior=None,
               analytic=False,
               sample_shape=(),
               reverse=True,
               free_bits=None,
               keepdims=False):
    self.posterior = posterior
    self.prior = prior
    self.analytic = bool(analytic)
    self.sample_shape = sample_shape
    self.reverse = bool(reverse)
    self.keepdims = bool(keepdims)
    self.free_bits = free_bits

  def __str__(self):
    if hasattr(self.posterior, 'shape'):
      post_shape = self.posterior.shape
    else:
      post_shape = f"{self.posterior.batch_shape + self.posterior.event_shape}"
    if hasattr(self.prior, 'shape'):
      prior_shape = self.prior.shape
    else:
      prior_shape = f"{self.prior.batch_shape + self.prior.event_shape}"
    return (f"<{self.__class__.__name__} "
            f"post:({self.posterior.__class__.__name__}, {post_shape})"
            f" prior:({self.prior.__class__.__name__}, {prior_shape})"
            f" analytic:{self.analytic} reverse:{self.reverse}"
            f" sample:{self.sample_shape}>")

  def __repr__(self):
    return self.__str__()

  def __call__(self,
               prior=None,
               analytic=None,
               sample_shape=-1,
               reverse=None,
               keepdims=False,
               free_bits=None):
    prior = self.prior if prior is None else prior
    analytic = self.analytic if analytic is None else bool(analytic)
    sample_shape = self.sample_shape if sample_shape == -1 else sample_shape
    reverse = self.reverse if reverse is None else bool(reverse)
    keepdims = self.keepdims if keepdims is None else bool(keepdims)
    free_bits = self.free_bits if free_bits is None else free_bits
    if prior is None:
      return 0.
    div = kl_divergence(q=self.posterior,
                        p=prior,
                        analytic=analytic,
                        reverse=reverse,
                        q_sample=sample_shape,
                        free_bits=free_bits)
    if analytic and keepdims:
      div = tf.expand_dims(div, axis=0)
    return div


# ===========================================================================
# COncatenation of distributions
# ===========================================================================
# must hand define all the parameters here
# NOTE: this list is to be updated, or a smarter solution for automatically
# mining all the parameters
dist_params = {
    # complex
    obd.Independent: ['distribution', 'reinterpreted_batch_ndims'],
    obd.ZeroInflated: ['count_distribution', 'inflated_distribution'],
    obd.MixtureSameFamily: ['mixture_distribution', 'components_distribution'],
    obd.Blockwise: ['distributions'],
    obd.ConditionalTensor: ['distribution', 'conditional_tensor'],
    # Exponential
    obd.Gamma: ['concentration', 'rate'],
    # Gaussians
    obd.Normal: ['loc', 'scale'],
    obd.LogNormal: ['loc', 'scale'],
    obd.MultivariateNormalDiag: ['loc', 'scale'],
    obd.MultivariateNormalTriL: ['loc', 'scale'],
    obd.MultivariateNormalFullCovariance: ['loc', 'scale'],
    # Count
    obd.NegativeBinomialDisp: ['loc', 'disp'],
    obd.NegativeBinomial: ['total_count', 'logits_parameter'],
    obd.Poisson: ['log_rate_parameter'],
    # Binary and probability
    obd.Gumbel: ['loc', 'scale'],
    obd.Bernoulli: ['logits_parameter'],
    obd.Dirichlet: ['concentration'],
    obd.Beta: ['concentration1', 'concentration0'],
    obd.OneHotCategorical: ['logits_parameter'],
    obd.Categorical: ['logits_parameter'],
    # others
    obd.Laplace: ['loc', 'scale'],
    obd.WishartTriL: ['df', 'scale'],
    obd.Uniform: ['low', 'high'],
    obd.Multinomial: ['total_count', 'logits_parameter'],
    obd.Deterministic: ['loc', 'atol', 'rtol'],
    obd.VectorDeterministic: ['loc', 'atol', 'rtol'],
}

for _type, _names in dist_params.items():
  assert isinstance(_names, (tuple, list)) and all(
      isinstance(name, string_types) for name in _names), \
        "Error defining parameters of distributions"
  assert isinstance(_type, type) and issubclass(_type, obd.Distribution),\
        "Error defining parameters of distributions"
  assert all(hasattr(_type, name) for name in _names), \
        "Error defining parameters of distributions"


def _find_axis_for_stack(dists, given_axis):
  """This algorithm find any axis that is different among minibatches as
  an axis for concatenation."""
  # check event shape is consistent
  if given_axis is not None:
    return int(given_axis)
  is_joint_dist = isinstance(dists[0], JointDistribution)
  event_ref = dists[0].event_shape
  batch_ref = dists[0].batch_shape
  if is_joint_dist:
    event_ref = event_ref[0]
    batch_ref = batch_ref[0]
  # check shape matching conditions
  assertions = []
  for d in dists[1:]:
    if is_joint_dist:
      event_shape = tf.reduce_sum(d.event_shape)  # assume concatenation
      ndims = d.batch_shape[0].ndims
    else:
      event_shape = d.event_shape
      ndims = d.batch_shape.ndims
    assertions.append(tf.assert_equal(event_ref, event_shape))
    assertions.append(tf.assert_equal(batch_ref.ndims, ndims))
  # searching for different dimension.
  with tf.control_dependencies(assertions):
    axis = []
    for d in dists:
      shape = d.batch_shape[0] if is_joint_dist else d.batch_shape
      for ax, (i, j) in enumerate(zip(batch_ref, shape)):
        if i != j:
          axis.append(ax)
    # default, just  return the first one
    if len(axis) == 0:
      return 0
    assert len(set(axis)) == 1, \
      "Multiple dimensions are found to be different among the distributions, "\
        "expect only 1 different dimension."
    return axis[0]


def _with_batch_dim(tensor, dist):
  if isinstance(tensor, tf.Tensor) and \
    tensor.shape.ndims > 0 and \
    len(dist.batch_shape) == 0:
    return tf.expand_dims(tensor, axis=0)
  return tensor


def _MVNdiag(dists, axis, kwargs):
  scale = [d.scale for d in dists]
  scale_diag = None
  scale_identity_multiplier = None
  if isinstance(scale[0], tf.linalg.LinearOperatorDiag):
    scale_diag = tf.concat(
        [_with_batch_dim(s.diag, d) for s, d in zip(scale, dists)], axis=axis)
  elif isinstance(scale[0], tf.linalg.LinearOperatorScaledIdentity):
    multiplier = [s.multiplier for s in scale]
    for m in multiplier[1:]:
      tf.assert_equal(m, multiplier[0])
    scale_identity_multiplier = multiplier[0]
  loc = tf.concat([_with_batch_dim(d.loc, d) for d in dists], axis=axis)
  kwargs.update(
      dict(loc=loc,
           scale_diag=scale_diag,
           scale_identity_multiplier=scale_identity_multiplier))
  return obd.MultivariateNormalDiag(**kwargs)


def _MVNtril(dists, axis, kwargs):
  scale = tf.concat([_with_batch_dim(d.scale.to_dense(), d) for d in dists],
                    axis=axis)
  loc = tf.concat([_with_batch_dim(d.loc, d) for d in dists], axis=axis)
  kwargs.update(dict(loc=loc, scale_tril=scale))
  return obd.MultivariateNormalTriL(**kwargs)


def _MVNfull(dists, axis, kwargs):
  scale = [_with_batch_dim(d.scale.to_dense(), d) for d in dists]
  scale = [s @ tf.linalg.matrix_transpose(s) for s in scale]
  scale = tf.concat(scale, axis=axis)
  loc = tf.concat([_with_batch_dim(d.loc, d) for d in dists], axis=axis)
  kwargs.update(dict(loc=loc, covariance_matrix=scale))
  return obd.MultivariateNormalFullCovariance(**kwargs)


def concat_distributions(dists: List[tfd.Distribution],
                         axis: Optional[int] = None,
                         validate_args: bool = False,
                         allow_nan_stats: bool = True,
                         name: Optional[Text] = None) -> tfd.Distribution:
  """This layer create a new `Distribution` by concatenate parameters of
  multiple distributions of the same type along given `axis`

  Note
  ----
  If your distribution is the output from `DistributionLambda`,
      this function will remove all the keras history
  """
  dists = as_tuple(dists)
  if len(dists) == 1:
    return dists[0]
  if len(dists) == 0:
    raise ValueError("No distributions were given")
  axis = _find_axis_for_stack(dists, given_axis=axis)
  # ====== get the proper distribution type ====== #
  dist_type = type(dists[0])
  # _TensorCoercible will messing up with the parameters of the
  # distribution
  if issubclass(dist_type, dtc._TensorCoercible):
    dist_type = type.mro(dist_type)[2]
    assert issubclass(dist_type, tfd.Distribution) and not issubclass(
        dist_type, dtc._TensorCoercible)
  #TODO: issues concatenating JointDistribution, use Batchwise.
  if issubclass(dist_type, JointDistribution):
    from odin.bay.distributions.batchwise import Batchwise
    return Batchwise(dists, axis=axis, validate_args=validate_args, name=name)
  # ====== special cases ====== #
  dist_func = None
  if dist_type == obd.MultivariateNormalDiag:
    dist_func = _MVNdiag
  elif dist_type == obd.MultivariateNormalTriL:
    dist_func = _MVNtril
  elif dist_type == obd.MultivariateNormalFullCovariance:
    dist_func = _MVNfull
  if dist_func is not None:
    kwargs = dict(validate_args=validate_args, allow_nan_stats=allow_nan_stats)
    if name is not None:
      kwargs['name'] = name
    return dist_func(dists, axis, kwargs)

  ### no more distribution, tensor of parameters is return during the
  # recursive operator
  if issubclass(dist_type, (tuple, list)) and all(
      isinstance(i, obd.Distribution) or tf.is_tensor(i)
      for i in tf.nest.flatten(dists)):
    return [
        tf.concat(x, axis=axis)
        if tf.is_tensor(x) else concat_distributions(x, axis=axis)
        for x in zip(*dists)
    ]
  elif issubclass(dist_type, tf.Tensor):
    shapes = [d.shape for d in dists]
    if shapes[0].ndims == 0 or all(i == 1 for i in shapes[0]):
      # make sure all the number is the same (we cannot concatenate numbers)
      for d in dists[1:]:
        tf.assert_equal(d, dists[0])
      return dists[0]
    return tf.concat(dists, axis=axis)
  elif issubclass(dist_type, obd.Distribution):
    pass  # continue with all distribution parameters
  else:
    return dists[0]

  # get all params for concatenate
  if dist_type not in dist_params:
    raise RuntimeError("Unknown distribution of type '%s' for concatenation" %
                       str(dist_type))
  params_name = dist_params[dist_type]

  # start concat the params
  params = {}
  for p in params_name:
    attrs = [getattr(d, p) for d in dists]
    is_method = False
    if inspect.ismethod(attrs[0]):
      attrs = [a() for a in attrs]
      is_method = True
    if is_method and '_parameter' == p[-10:]:
      p = p[:-10]
    params[p] = concat_distributions(attrs, axis=axis)

  # extra arguments
  if name is not None:
    params['name'] = name
  args = inspect.getfullargspec(dist_type.__init__).args
  if 'allow_nan_stats' in args:
    params['allow_nan_stats'] = allow_nan_stats
  if 'validate_args' in args:
    params['validate_args'] = validate_args
  dist = dist_type(**params)

  return dist


# ===========================================================================
# Batch Slice
# ===========================================================================
def batch_slice(dist: tfd.Distribution, indices, name=None) -> tfd.Distribution:
  r""" Apply indexing on batch dimension of the distribution's parameters and
  return a new `Distribution` """
  assert isinstance(dist, tfd.Distribution), \
    "dist must be instance of Distribution, but given: %s" % str(type(dist))
  if name is None:
    name = dist.name
  dist_type = dist.__class__
  kwargs = dict(validate_args=dist._validate_args,
                allow_nan_stats=dist._allow_nan_stats,
                name=name)
  ## compound distribution
  if isinstance(dist, tfd.Independent):
    return dist_type(
        distribution=batch_slice(dist.distribution, indices),
        reinterpreted_batch_ndims=dist._reinterpreted_batch_ndims,
        **kwargs,
    )
  elif isinstance(dist, tfd.MixtureSameFamily):
    return dist_type(
        mixture_distribution=batch_slice(dist._mixture_distribution, indices),
        components_distribution=batch_slice(dist._components_distribution,
                                            indices),
        reparameterize=dist._reparameterize,
        **kwargs,
    )
  elif isinstance(dist, obd.ZeroInflated):
    return dist_type(
        count_distribution=batch_slice(dist.count_distribution, indices),
        inflated_distribution=batch_slice(dist.inflated_distribution, indices),
        **kwargs,
    )
  # this is very ad-hoc solution
  params = dist.parameters.copy()
  for key, val in list(params.items()):
    if isinstance(val, (np.ndarray, tf.Tensor)):
      params[key] = tf.gather(val, indices=indices, axis=0)
  params.update(kwargs)
  return dist_type(**params)
