from __future__ import absolute_import, division, print_function

import inspect
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import (assert_util, prefer_static,
                                                    tensorshape_util)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible

__all__ = ['stack_distributions']


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


def stack_distributions(dists: List[tfd.Distribution],
                        axis: Optional[int] = None) -> tfd.Distribution:
  """ Automatically and recursively stack multiple distribution of the
  same type and same `event_shape` into single distribution with
  concatenated `batch_shape`

  The stacking is done recursively done for any distribution contained in
  `Independent`, `ZeroInflated` or any complex distribution

  Parameters
  ----------
  dists : List of `tensorflow_probability.Distribution`
  axis : {`int`, `None`}
    the batch dimension for merging the distribution, if `None`,
    by default the first dimension, or the dimension that is different
    among batches.

  Note
  ----
  This function is only stacking the mismatch batch dimension,
  and due to many detail involves broadcasting the shape of the parameters,
  this function only applied in eager mode.
  """
  if not isinstance(dists, (tuple, list)):
    dists = [dists]
  if len(dists) == 1:
    return dists[0]
  if len(dists) == 0:
    raise ValueError("No distributions were given")
  # already assume all the distribution has the same type
  t = type(dists[0])
  try:
    all_params = set(t._params_event_ndims().keys())
  except NotImplementedError:
    all_params = set()

  # _TensorCoercible will messing up with the parameters of the
  # distribution
  if issubclass(t, distribution_tensor_coercible._TensorCoercible):
    t = type.mro(t)[2]
    assert issubclass(t, tfd.Distribution) and not issubclass(
        t, distribution_tensor_coercible._TensorCoercible)

  # params is arguments of the __init__ function
  params = dists[0].parameters
  for key in inspect.getfullargspec(t.__init__).args:
    if key not in params:
      val = getattr(dists[0], key, '__NO_ARGUMENT_FOUND__')
      if val != '__NO_ARGUMENT_FOUND__':
        params[key] = val
  params = {k: getattr(dists[0], k, v) for k, v in params.items()}

  # prefer logits
  if 'probs' in params and 'logits' in params:
    del params['probs']

  axis = _find_axis_for_stack(dists, given_axis=axis)
  new_params = {}
  for key, val in params.items():
    # another nested distribution
    if isinstance(val, tfd.Distribution):
      new_params[key] = stack_distributions([getattr(d, key) for d in dists])
    # Tensor parameters
    elif key in all_params:
      all_x = []
      for d in dists:
        full_shape = tf.concat([d.batch_shape, d.event_shape], axis=0)
        x = getattr(d, key)
        for _ in range(len(full_shape) - len(x.shape)):
          x = tf.expand_dims(x, axis=0)
        repeats = [i / j for i, j in zip(full_shape, x.shape)]
        if any(i > 1 for i in repeats):
          x = tf.tile(x, multiples=repeats)
        all_x.append(x)
      val = tf.concat(all_x, axis=axis)
      new_params[key] = val
    # primitive values
    else:
      if key in new_params:
        tf.assert_equal(val, new_params[key])
      new_params[key] = val

  specs = inspect.getfullargspec(t.__init__)
  for key in list(new_params.keys()):
    if key not in specs.args:
      del new_params[key]

  new_dist = t(**new_params)
  return new_dist
