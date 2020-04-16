from __future__ import absolute_import, division, print_function

import inspect
from typing import List, Optional, Text

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import (assert_util,
                                                    distribution_util,
                                                    prefer_static,
                                                    tensorshape_util)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible

from odin.bay import distributions as obd
from odin.bay.distributions.negative_binomial_disp import NegativeBinomialDisp
from odin.bay.distributions.zero_inflated import ZeroInflated

__all__ = ['concat_distribution', 'slice_distribution']

# ===========================================================================
# Helpers
# ===========================================================================
# must hand define all the parameters here
# NOTE: this list is to be updated, or a smarter solution for automatically
# mining all the parameters
dist_params = {
    # complex
    obd.Independent: ['distribution', 'reinterpreted_batch_ndims'],
    ZeroInflated: ['count_distribution', 'inflated_distribution'],
    obd.MixtureSameFamily: ['mixture_distribution', 'components_distribution'],
    # Exponential
    obd.Gamma: ['concentration', 'rate'],
    # Gaussians
    obd.Normal: ['loc', 'scale'],
    obd.LogNormal: ['loc', 'scale'],
    obd.MultivariateNormalDiag: ['loc', 'scale'],
    obd.MultivariateNormalTriL: ['loc', 'scale'],
    obd.MultivariateNormalFullCovariance: ['loc', 'scale'],
    # Count
    NegativeBinomialDisp: ['loc', 'disp'],
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


# ===========================================================================
# Main code
# ===========================================================================
def _find_axis_for_stack(dists, given_axis):
  # check event shape is consistent
  if given_axis is not None:
    return int(given_axis)

  event_shape = dists[0].event_shape
  batch_shape = dists[0].batch_shape

  assertions = []
  for d in dists[1:]:
    assertions.append(tf.assert_equal(event_shape, d.event_shape))
    assertions.append(tf.assert_equal(batch_shape.ndims, d.batch_shape.ndims))

  with tf.control_dependencies(assertions):
    axis = []
    for d in dists:
      shape = d.batch_shape
      for ax, (i, j) in enumerate(zip(batch_shape, shape)):
        if i != j:
          axis.append(ax)
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


# ===========================================================================
# Special distribution cases for concatenation
# ===========================================================================
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


def concat_distribution(dists: List[tfd.Distribution],
                        axis: Optional[int] = None,
                        validate_args: bool = False,
                        allow_nan_stats: bool = True,
                        name: Optional[Text] = None) -> tfd.Distribution:
  r""" This layer create a new `Distribution` by concatenate parameters of
  multiple distributions of the same type along given `axis`

  Note
  ----
  If your distribution is the output from
  `tensorflow_probability.DistributionLambda`, this function will remove all
  the keras tracking ultilities, for better solution checkout
  `odin.networks.distribution_util_layer.ConcatDistribution`
  """
  if not isinstance(dists, (tuple, list)):
    dists = [dists]
  if len(dists) == 1:
    return dists[0]
  if len(dists) == 0:
    raise ValueError("No distributions were given")
  axis = _find_axis_for_stack(dists, given_axis=axis)

  dist_type = type(dists[0])
  # _TensorCoercible will messing up with the parameters of the
  # distribution
  if issubclass(dist_type, distribution_tensor_coercible._TensorCoercible):
    dist_type = type.mro(dist_type)[2]
    assert issubclass(dist_type, tfd.Distribution) and not issubclass(
        dist_type, distribution_tensor_coercible._TensorCoercible)

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

  # no more distribution, tensor of parameters is return during the
  # recursive operator
  if issubclass(dist_type, tf.Tensor):
    if dists[0].shape.ndims == 0:
      for d in dists[1:]:
        # make sure all the number is the same (we cannot concatenate numbers)
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
    params[p] = concat_distribution(attrs, axis=axis)

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
# Slice the distribution
# ===========================================================================
def slice_distribution(index, dist: tfd.Distribution,
                       name=None) -> tfd.Distribution:
  r""" Apply indexing on distribution parameters and return another
  `Distribution` """
  assert isinstance(dist, tfd.Distribution), \
    "dist must be instance of Distribution, but given: %s" % str(type(dist))
  if name is None:
    name = dist.name
  ## compound distribution
  if isinstance(dist, tfd.Independent):
    return tfd.Independent(
        distribution=slice_distribution(index, dist.distribution),
        reinterpreted_batch_ndims=dist.reinterpreted_batch_ndims,
        name=name)
  elif isinstance(dist, ZeroInflated):
    return ZeroInflated(\
      count_distribution=slice_distribution(index, dist.count_distribution),
      inflated_distribution=slice_distribution(index, dist.inflated_distribution),
      name=name)

  # this is very ad-hoc solution
  params = dist.parameters.copy()
  for key, val in list(params.items()):
    if isinstance(val, (np.ndarray, tf.Tensor)):
      params[key] = tf.gather(val, indices=index, axis=0)
  return dist.__class__(**params)
