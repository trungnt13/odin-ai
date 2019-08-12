from __future__ import absolute_import, division, print_function

import inspect
from typing import List

import tensorflow as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import (assert_util, prefer_static,
                                                    tensorshape_util)
from tensorflow_probability.python.layers.internal import \
    distribution_tensor_coercible


def _find_axis_for_stack(dists):
  # check event shape is consistent
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
    assert len(set(axis)) == 1, \
      "Multiple dimensions are found to be different among the distributions, "\
        "expect only 1 different dimension."
    return axis[0]


def stack_distributions(dists: List[tfd.Distribution]) -> tfd.Distribution:
  """ Automatically and recursively stack multiple distribution of the
  same type and same `event_shape` into single distribution with
  concatenated `batch_shape`

  Parameters
  ----------
  dists : List of `tensorflow_probability.Distribution`

  Note
  ----
  This function is only stacking the mismatch batch dimension
  """
  if not isinstance(dists, (tuple, list)):
    dists = [dists]
  t = type(dists[0])

  # _TensorCoercible will messing up with the parameters of the
  # distribution
  if issubclass(t, distribution_tensor_coercible._TensorCoercible):
    t = type.mro(t)[2]
    assert issubclass(t, tfd.Distribution) and not issubclass(
        t, distribution_tensor_coercible._TensorCoercible)

  params = dists[0].parameters
  for key in inspect.getfullargspec(t.__init__).args:
    if key not in params:
      val = getattr(dists[0], key, '__NO_ARGUMENT_FOUND__')
      if val != '__NO_ARGUMENT_FOUND__':
        params[key] = val

  axis = _find_axis_for_stack(dists)
  new_params = {}

  for key, val in params.items():
    # another nested distribution
    if isinstance(val, tfd.Distribution):
      new_params[key] = stack_distributions([getattr(d, key) for d in dists])
    # Tensor
    elif tf.is_tensor(val):
      val = tf.concat([getattr(d, key) for d in dists], axis=axis)
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
