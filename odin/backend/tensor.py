# A collection of functions for processing, manipulating and calculating
# necessary information from tensors
#
# Aim for unifiying the API among numpy, tensorflow and pytorch.
# Since numpy is most popular framework, the function name is strictly
# following numpy name and arguments.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from __future__ import absolute_import, division, print_function

import copy
import inspect
import itertools
import numbers
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np
import scipy as sp
import tensorflow as tf
import torch
from six import string_types
from six.moves import builtins
from tensorflow import nest
from tensorflow.python.ops import init_ops

from odin.utils import as_tuple, is_number, is_same_shape, is_string, uuid

# TODO: add stack for setting framework context
_FRAMEWORK_STACK = ['numpy']


class framework_(object):
  """
  ```python
  with bk.framework_('tensorflow') as fw1:
    print("Context 1:", bk.get_framework(), fw1)
    with bk.framework_('pytorch') as fw2:
      print("Context 2:", bk.get_framework(), fw2)
    print("Context 1:", bk.get_framework())
  print("Default:", bk.get_framework())

  bk.framework_('tensorflow')
  print("Current:", bk.get_framework())
  bk.reset_framework()
  print("Reset:", bk.get_framework())

  # Context 1: tensorflow <module 'tensorflow'>
  # Context 2: pytorch <module 'torch'>
  # Context 1: tensorflow
  # Default: numpy
  # Current: tensorflow
  # Reset: numpy
  ```
  """

  def __init__(self, framework):
    framework = parse_framework(framework)
    _FRAMEWORK_STACK.append(framework)
    self._framework = framework

  def __enter__(self):
    return parse_framework(self._framework)

  def __exit__(self, *args):
    reset_framework()


def reset_framework():
  if len(_FRAMEWORK_STACK) > 1:
    _FRAMEWORK_STACK.pop()


def get_framework():
  return _FRAMEWORK_STACK[-1]


def parse_framework(alias):
  """ Convert a string or object to appropriate framework module: numpy,
  tensorflow or torch """
  if inspect.ismodule(alias):
    if alias in (tf, torch, np):
      return alias
    alias = str(alias)
  elif alias is None:
    return get_framework()
  elif inspect.isclass(alias):
    alias = ''.join([str(i) for i in type.mro(alias)])
  elif not isinstance(alias, string_types):
    alias = type(alias)
    alias = ''.join([str(i) for i in type.mro(alias)])

  alias = alias.strip().lower()
  if any(i in alias for i in ['tf', 'tensorflow']):
    return tf
  if any(i in alias for i in ['torch', 'pytorch', 'pt', 'tr']):
    return torch
  return np


# ===========================================================================
# Helper
# ===========================================================================
def _normalize_axis(axis, ndim):
  if axis is None:
    return None
  if isinstance(axis, (tuple, list)):
    return tuple([a % ndim if a is not None else a for a in axis])
  return axis % ndim


def dtype_universal(dtype,
                    torch_dtype=False,
                    tf_dtype=False,
                    np_dtype=False,
                    framework=None):
  if dtype is None:
    return dtype

  if sum([torch_dtype, tf_dtype, np_dtype]) > 1:
    raise ValueError("Cannot only return dtype for 1 framework a time.")
  if isinstance(dtype, tf.dtypes.DType):
    dtype = dtype.name
  elif isinstance(dtype, torch.dtype):
    dtype = str(dtype).split('.')[-1]
  elif isinstance(dtype, np.dtype):
    dtype = np.dtype(dtype).name

  if framework is not None:
    framework = parse_framework(framework)
    if framework == np:
      np_dtype = True
      torch_dtype = False
      tf_dtype = False
    if framework == torch:
      torch_dtype = True
      np_dtype = False
      tf_dtype = False
    if framework == tf:
      tf_dtype = True
      torch_dtype = False
      np_dtype = False

  dtype = dtype.lower().strip()
  if torch_dtype:
    if dtype == 'float' or dtype == 'float32':
      return torch.float32
    if dtype == 'float64':
      return torch.float64
    if dtype == 'float16' or dtype == 'half':
      return torch.float16
    if dtype == 'int8':
      return torch.int8
    if dtype == 'uint8':
      return torch.uint8
    if dtype == 'int16' or dtype == 'short':
      return torch.int16
    if dtype == 'int' or dtype == 'int32':
      return torch.int32
    if dtype == 'int64' or dtype == 'long':
      return torch.int64
    if 'bool' in dtype:
      return torch.bool
  if tf_dtype:
    return tf.as_dtype(dtype)
  if np_dtype:
    return np.dtype(dtype)
  return dtype


def cast(x, dtype):
  if tf.is_tensor(x) or isinstance(dtype, tf.DType):
    return tf.cast(x, dtype=dtype_universal(dtype, tf_dtype=True))
  if torch.is_tensor(x) or isinstance(dtype, torch.dtype):
    if not torch.is_tensor(x):
      x = torch.tensor(x)
    return x.type(dtype_universal(dtype, torch_dtype=True))
  dtype = dtype_universal(dtype, np_dtype=True)
  return np.cast[dtype](x)


def array(x, framework=None, dtype=None):
  """ This function equal to `numpy.array` for numpy;
  `tensorflow.convert_to_tensor` for tensorflow;
  and `torch.tensor` for pytorch
  """
  in_framework = parse_framework(x)
  out_framework = parse_framework(framework)

  if in_framework != out_framework:
    # any conversion must go through numpy
    if in_framework != np:
      x = x.numpy()
    if out_framework == tf:
      return tf.convert_to_tensor(x, dtype=dtype)
    if out_framework == torch:
      if dtype is not None:
        x = cast(x, dtype)
      if torch.is_tensor(x):
        return x
      return torch.from_numpy(x)
  return x


# ===========================================================================
# Variable and gradients
# ===========================================================================
def variable(initial_value, framework=None, dtype=None, trainable=True):
  framework = parse_framework(framework)
  if framework == tf:
    return tf.Variable(initial_value=initial_value,
                       dtype=dtype_universal(dtype, tf_dtype=True),
                       trainable=trainable)
  elif framework == torch:
    return torch.nn.Parameter(data=torch.tensor(data=initial_value,
                                                dtype=dtype_universal(
                                                    dtype, torch_dtype=True),
                                                requires_grad=trainable),
                              requires_grad=trainable)
  raise RuntimeError("No variable support for framework: %s" % str(framework))


def grad(fn_outputs, inputs, grad_outputs=None, return_outputs=False):
  """ Compute and returns the sum of gradients of outputs w.r.t. the inputs.
  """
  if not callable(fn_outputs):
    raise ValueError('fn_outputs must be a callable return a list of tensors')
  inputs = nest.flatten(inputs)

  if torch.is_tensor(inputs[0]):
    outputs = nest.flatten(fn_outputs())
    gradients = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    grad_outputs=grad_outputs)
  elif tf.is_tensor(inputs[0]):
    with tf.GradientTape() as tape:
      tape.watch(inputs)
      outputs = nest.flatten(fn_outputs())
    gradients = tape.gradient(target=outputs,
                              sources=inputs,
                              output_gradients=grad_outputs)
  if not return_outputs:
    return gradients
  return gradients, outputs

  raise NotImplementedError(
      "gradient function only support pytorch or tensorflow")


def cumsum(x, axis=0, dtype=None):
  if tf.is_tensor(x):
    x = tf.cumsum(x, axis=axis)
    if dtype is not None:
      x = tf.cast(x, dtype=dtype)
    return x
  elif torch.is_tensor(x):
    return torch.cumsum(x, dim=axis, dtype=dtype)
  return np.cumsum(x, axis=axis, dtype=dtype)


# ===========================================================================
# Allocation and masking
# ===========================================================================
def ones_like(x, dtype=None):
  if tf.is_tensor(x):
    return tf.ones_like(x, dtype=dtype)
  if torch.is_tensor(x):
    return torch.ones_like(x, dtype=dtype)
  return np.ones_like(x, dtype=dtype)


def zeros_like(x, dtype=None):
  if tf.is_tensor(x):
    return tf.zeros_like(x, dtype=dtype)
  if torch.is_tensor(x):
    return torch.zeros_like(x, dtype=dtype)
  return np.zeros_like(x, dtype=dtype)


def ones(shape, dtype='float32', framework=None):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  if isinstance(shape, np.ndarray):
    shape = shape.tolist()
  return framework.ones(shape, dtype=dtype)


def zeros(shape, dtype='float32', framework=None):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  if isinstance(shape, np.ndarray):
    shape = shape.tolist()
  return framework.zeros(shape, dtype=dtype)


def eye(num_rows, num_cols=None, dtype=None, framework=None):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  if framework == tf:
    return tf.eye(num_rows, num_cols, dtype=dtype)
  if framework == torch:
    return torch.eye(num_rows, num_cols, dtype=dtype)
  return np.eye(N=num_rows, M=num_cols, dtype=dtype)


def arange(start, stop=None, dtype=None, framework=None):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  if framework == tf:
    return tf.range(start, stop, dtype=dtype)
  elif framework == torch:
    if stop is None:
      stop = start
      start = 0
    return torch.arange(start, stop, dtype=dtype)
  return np.arange(start, stop, dtype=dtype)


def nonzeros(x, value):
  """ Convert all zero entrities in `x` to a nonzeros `value`"""
  return where(equal(x, 0.), zeros_like(x) + value, x)


def tril_mask(shape, framework=None):
  """ Creates a lower-triangular boolean mask over the last 2 dimensions.

  """
  row_index = cumsum(ones(shape=shape, dtype='int32', framework=framework),
                     axis=-2)
  col_index = cumsum(ones(shape=shape, dtype='int32', framework=framework),
                     axis=-1)
  return greater_equal(row_index, col_index)


def tril(m, k=0):
  """
  Lower triangle of an array.

  Return a copy of an array with elements above the `k`-th diagonal zeroed.

  Parameters
  ----------
  m : array_like, shape (M, N)
      Input array.
  k : int, optional
      Diagonal above which to zero elements.  `k = 0` (the default) is the
      main diagonal, `k < 0` is below it and `k > 0` is above.

  Returns
  -------
  tril : ndarray, shape (M, N)
      Lower triangle of `m`, of same shape and data-type as `m`.
  """
  if k == 0:
    return tf.linalg.band_part(input=m, num_lower=-1, num_upper=0)
  if k < 0:
    return tf.subtract(
        m, tf.linalg.band_part(input=m, num_lower=np.abs(k) - 1, num_upper=-1))
  # k > 0
  return tf.linalg.band_part(input=m, num_lower=-1, num_upper=k)


def tril_indices(n, k=0):
  """ Similar as `numpy.tril_indices`
  @Author: avdrher
  https://github.com/GPflow/GPflow/issues/439

  Return the indices for the lower-triangle of an (n, m) array.

  Parameters
  ----------
  n : int
      The row dimension of the arrays for which the returned
      indices will be valid.
  k : int, optional
      Diagonal above which to zero elements.  `k = 0` (the default) is the
      main diagonal, `k < 0` is below it and `k > 0` is above.

  Returns
  -------
  inds : tuple of arrays
      The indices for the triangle. The returned tuple contains two arrays,
      each with the indices along one dimension of the array.

  """
  M1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
  M2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
  mask = tf.transpose((M1 - M2) >= -k)
  ix1 = tf.boolean_mask(M2, mask)
  ix2 = tf.boolean_mask(M1, mask)
  return ix1, ix2


def prior2weights(prior,
                  exponential=False,
                  min_value=0.1,
                  max_value=None,
                  norm=False):
  """ TODO: finish this

  Parameters
  ----------
  prior: numpy.ndarray [nb_classes,]
      probabilty values of each classes prior,
      sum of all prior must be equal to 1.
  exponential: bool
  min_value: bool
  max_value: bool
  norm: bool
      if True, normalize output weights to sum up to 1.
  """
  # idea is the one with highest prior equal to 1.
  # and all other classes is the ratio to this prior
  prior = np.array(prior).ravel()
  prior = 1 / prior * np.max(prior)
  # print(prior)
  if exponential:
    prior = sorted([(i, p) for i, p in enumerate(prior)],
                   key=lambda x: x[-1],
                   reverse=False)
    alpha = interp.expIn(n=len(prior), power=10)
    prior = {i: a * p for a, (i, p) in zip(alpha, prior)}
    prior = np.array([prior[i] for i in range(len(prior))]) + 1
  # ====== rescale everything within max_value ====== #
  if min_value is not None and max_value is not None:
    min_value = float(min_value)
    max_value = float(max_value)
    prior = (max_value - min_value) * (prior - np.min(prior)) \
        / (np.max(prior) - np.min(prior)) + min_value
  # ====== normaize by ====== #
  if norm:
    prior = prior / np.sum(prior)
  return prior


def entropy(p, name=None):
  """Return simple calculation of discrete Shanon entropy"""
  with tf.name_scope(name, "entropy"):
    return -tf.reduce_sum(p * tf.log(p))


def upsample(x, scale, axes, method='nn', name=None):
  """
  Parameters
  ----------
  scale: int, list of int
      scaling up factor
  axes: int, list of int
      the axes of tensor which the upsampling method will be applied
  method: str, int
      'nn' for nearest neighbor (e.g. [1, 2] => [1, 1, 2, 2]),
      'pad' for padding within the tensor. 'pad_margin' do padding
      in the margin of the tensor. 'repeat' simple algorithm for
      repeating the element (e.g. [1, 2] => [1, 2, 1, 2])
  """
  with tf.name_scope(name, "Upsample"):
    method = method.lower()
    input_shape = tf.shape(x)
    input_shape_int = x.shape.as_list()
    ndims = x.shape.ndims
    # normalize all negative axes
    if axes is None:
      raise ValueError("axes cannot be None.")
    axes = [1, 2] if axes is None else \
        [i % ndims for i in as_tuple(axes)]
    sorted(axes)
    # make scale a tuple
    scale = as_tuple(scale, N=len(axes), t=int)
    # mapping from axis -> scale
    scale_map = defaultdict(lambda: 1)
    scale_map.update([(i, j) for i, j in zip(axes, scale)])
    # create final output_shape
    output_shape = [input_shape[i] * scale_map[i] for i in range(ndims)]
    # ====== Nearest neighbor method ====== #
    if method == 'nn':
      # tensorflow only support for tile <= 6-D tensor
      if ndims >= 6:
        raise ValueError(
            'upsample with NN mode does not support rank >= 6 tensor.')
      elif ndims + len(axes) > 6:
        for a in axes:
          x = upsample(x, scale_map[a], axes=a, method='nn')
      else:
        # repeat the tensor
        x = transpose(x, pattern=list(range(ndims)) + ['x'] * len(axes))
        x = repeat(x, scale, axes=[i for i in range(ndims, ndims + len(axes))])
        # transpose it back to the right shape
        axes_map = {i: j for i, j in zip(axes, range(ndims, ndims + len(axes)))}
        new_axes = []
        for i in range(ndims):
          if i not in axes_map:
            new_axes.append(i)
          else:
            new_axes += [i, axes_map[i]]
        x = tf.transpose(x, perm=new_axes)
        x = reshape(x, output_shape)
    # ====== pading_margin ====== #
    elif method.lower() == 'pad_margin':
      paddings = [[0, 0] if i not in axes else [
          tf.cast(tf.ceil(input_shape[i] * (scale_map[i] - 1) / 2), 'int32'),
          tf.cast(tf.floor(input_shape[i] * (scale_map[i] - 1) / 2), 'int32')
      ] for i in range(ndims)]
      x = tf.pad(x, paddings=paddings, mode='CONSTANT')
    # ====== pading ====== #
    elif method == 'pad':
      raise NotImplementedError
      # x = tf.scatter_nd(indices, x, shape=output_shape)
    # ====== repeat ====== #
    elif method == 'repeat':
      x = repeat(x, n=scale, axes=axes)
    # ====== no support ====== #
    else:
      raise ValueError("No support for method='%s'" % method)
    # ====== add_shape ====== #
    return set_shape(x,
                     shape=[
                         s * scale_map[i] if is_number(s) else None
                         for i, s in enumerate(input_shape_int)
                     ])


# ===========================================================================
# Shape manipulation
# ===========================================================================
def reshape(x, shape):
  """ More flexible version of reshape operation

  Example
  -------
  x.shape = [25, 08, 12]
  reshape(shape=([1], [2], [0]))
  => x.shape = (08, 12, 25)
  """
  if tf.is_tensor(x):
    fn_reshape = tf.reshape
  elif torch.is_tensor(x):
    fn_reshape = lambda _, shape: _.view(shape)
  else:
    fn_reshape = np.reshape
  # start reshaping
  input_shape = x.shape
  new_shape = []
  for i in shape:
    if i is None:
      new_shape.append(-1)
    elif isinstance(i, (list, tuple)):
      new_shape.append(input_shape[i[0]])
    else:
      new_shape.append(i)
  new_shape = tuple([-1 if i is None else i for i in new_shape])
  return fn_reshape(x, new_shape)


def expand_dims(x, axis):
  if tf.is_tensor(x):
    return tf.expand_dims(x, axis)
  if torch.is_tensor(x):
    return torch.unsqueeze(x, axis)
  return np.expand_dims(x, axis)


def squeeze(x, axis):
  if tf.is_tensor(x):
    return tf.squeeze(x, axis)
  if torch.is_tensor(x):
    return torch.squeeze(x, axis)
  return np.squeeze(x, axis)


def concatenate(x, axis):
  if not isinstance(x, (tuple, list)):
    return x
  if len(x) == 1:
    return x[0]
  if tf.is_tensor(x[0]):
    return tf.concat(x, axis)
  if torch.is_tensor(x[0]):
    return torch.cat(x, axis)
  return np.concatenate(x, axis)


def swapaxes(x, axis1, axis2):
  """ Interchange two axes of an array. """
  if tf.is_tensor(x):
    ndim = x.shape.ndims
    axis1, axis2 = _normalize_axis((axis1, axis2), ndim)
    perm = list(range(ndim))
    perm[axis1] = axis2
    perm[axis2] = axis1
    x = tf.transpose(x, perm)
  elif torch.is_tensor(x):
    x = x.transpose(axis1, axis2)
  else:
    x = np.swapaxes(x, axis1, axis2)
  return x


def transpose(x, pattern):
  """ Reorder the dimensions of this variable, optionally inserting
  broadcasted dimensions.

  Parameters
  ----------
  pattern
      List/tuple of int mixed with 'x' for broadcastable dimensions.

  Examples
  --------
  For example, to create a 3D view of a [2D] matrix, call
  ``transpose([0,'x',1])``.  This will create a 3D view such that the
  middle dimension is an implicit broadcasted dimension.  To do the same
  thing on the transpose of that matrix, call ``transpose([1, 'x', 0])``.

  Notes
  -----
  This function supports the pattern passed as a tuple, or as a
  variable-length argument (e.g. ``a.transpose(pattern)`` is equivalent
  to ``a.transpose(*pattern)`` where ``pattern`` is a list/tuple of ints
  mixed with 'x' characters).

  @Author: Theano Authors
  """
  permute_pattern = [i for i in pattern if i != 'x']
  if tf.is_tensor(x):
    x = tf.transpose(x, perm=permute_pattern)
    # insert new dimension
    for i, p in enumerate(pattern):
      if p == 'x':
        x = tf.expand_dims(x, i)
  elif torch.is_tensor(x):
    x = x.permute(permute_pattern)
    for i, p in enumerate(pattern):
      if p == 'x':
        x = x.unsqueeze(i)
  else:
    x = np.transpose(x, permute_pattern)
    for i, p in enumerate(pattern):
      if p == 'x':
        x = np.expand_dims(x, i)
  return x


def flatten(x, outdim=1):
  """ Keep all the original dimension until `outdim - 1`
  """
  if tf.is_tensor(x):
    input_shape = tf.shape(x)
  elif torch.is_tensor(x):
    input_shape = x.shape
  else:
    input_shape = x.shape

  if outdim == 1:
    output_shape = [-1]
  else:
    other_shape = tuple([input_shape[i] for i in range(outdim - 1)])
    n = 1
    for i in input_shape[(outdim - 1):]:
      n = n * i
    output_shape = other_shape + (n,)

  return reshape(x, output_shape)


def tile(x, reps, axis=None):
  """ Construct an array by repeating `x` the number of times given by `reps`.

  If x has shape (s1, s2, s3) and axis=(1, -1), the output
  will have shape (s1, s2 * n[0], s3 * n[1]).

  Parameters
  ----------
  reps : {int, list of int}
    each number of repeatation according to the axes
  axis : {int, list or int}
    all axes for repeating
  """
  ndim = x.ndim
  if axis is not None:
    if not isinstance(axis, (tuple, list)):
      axis = (axis,)
    axis = _normalize_axis(axis, ndim)
    reps = as_tuple(reps, N=len(axis), t=int)
    multiples = [reps[axis.index(i)] if i in axis else 1 for i in range(ndim)]
  else:
    reps = as_tuple(reps, t=int)
    multiples = [reps[i] if i < len(reps) else 1 for i in range(ndim)]

  if tf.is_tensor(x):
    return tf.tile(x, multiples=multiples)
  elif torch.is_tensor(x):
    return x.repeat(multiples)
  return np.tile(x, reps=multiples)


def split(x, num_of_splits, axis=0):
  if tf.is_tensor(x):
    return tf.split(x, num_of_splits, axis=axis)
  if torch.is_tensor(x):
    return torch.split(x, num_of_splits, dim=axis)
  return np.split(x, num_of_splits, axis=axis)


# ===========================================================================
# Logical function
# ===========================================================================
def where(condition, x=None, y=None):
  if tf.is_tensor(condition) or tf.is_tensor(x) or tf.is_tensor(y):
    return tf.where(condition, x, y)
  if torch.is_tensor(condition) or torch.is_tensor(x) or torch.is_tensor(y):
    if not torch.is_tensor(x):
      x = torch.tensor(x, dtype=y.dtype)
    if not torch.is_tensor(y):
      y = torch.tensor(y, dtype=x.dtype)
    return torch.where(condition, x, y)
  return np.where(condition, x, y)


def equal(x, y):
  if tf.is_tensor(x) or tf.is_tensor(y):
    return tf.equal(x, y)
  if torch.is_tensor(x) or torch.is_tensor(y):
    return x == y
  return np.equal(x, y)


def not_equal(x, y):
  if tf.is_tensor(x) or tf.is_tensor(y):
    return tf.not_equal(x, y)
  if torch.is_tensor(x) or torch.is_tensor(y):
    return x != y
  return np.not_equal(x, y)


def greater_equal(x, y):
  if tf.is_tensor(x) or tf.is_tensor(y):
    return tf.greater_equal(x, y)
  if torch.is_tensor(x) or torch.is_tensor(y):
    return x >= y
  return np.greater_equal(x, y)


def switch(condition, then_expression, else_expression):
  condition = cast(condition, 'bool')
  x_shape = copy.copy(then_expression.shape)
  # tensorflow require the last dimension of 3 variables is equal, too
  # it is irrelevant since condition can have shape[-1] = 1
  cond_ndims = condition.shape.ndims
  if cond_ndims > 1 and condition.shape[-1] != x_shape[-1]:
    cond_shape = tf.shape(condition)
    condition = tf.reshape(condition,
                           [cond_shape[i] for i in range(cond_ndims - 1)])
  x = tf.where(condition, then_expression, else_expression)
  x.set_shape(x_shape)
  return x


# ===========================================================================
# Logical functions
# ===========================================================================
def logical_(fn, x, y):
  if x is None:
    return y
  if y is None:
    return x

  if fn == 'and':
    fn = lambda x, y: x & y
  elif fn == 'or':
    fn = lambda x, y: x | y
  else:
    raise NotImplementedError(str(fn))
  return fn(x, y)


def logical_and(x, y):
  r""" More flexible version of and operator that handle the case `x` or `y`
  might be `None` """
  return logical_('and', x, y)


def logical_or(x, y):
  r""" More flexible version of and operator that handle the case `x` or `y`
  might be `None` """
  return logical_('or', x, y)


def logical_not(x):
  if x is not None:
    x = ~x
  return x


def apply_mask(x, mask):
  """
  x : 3D tensor
  mask : 2D tensor

  Example
  -------
  >>> Input: [128, 500, 120]
  >>> Mask:  [1, 1, 0]
  >>> Output: [128, 500, 0]
  """
  return x * expand_dims(mask, -1)


# ===========================================================================
# Randomization helper
# ===========================================================================
def random_normal(*shape, mean=0.0, stddev=1.0, dtype='float32',
                  framework=None):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  shape = tf.nest.flatten(shape)

  if framework == tf:
    return tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype)
  if framework == torch:
    return mean + stddev * torch.randn(*shape, dtype=dtype)
  return mean + stddev * np.random.randn(*shape)


def random_uniform(*shape, minval=0, maxval=1, dtype='float32', framework=None):
  r""" [minval, maxval) """
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  shape = tf.nest.flatten(shape)

  if framework == tf:
    return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
  if framework == torch:
    return minval + torch.rand(*shape, dtype=dtype) * (maxval - minval)
  return np.random.uniform(low=minval, high=maxval, size=shape)


def random_binomial(*shape, p, dtype='float32', framework=None):
  r"""
  p : [0, 1], the probability of success (i.e. return 1)
  """
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  shape = tf.nest.flatten(shape)

  one = array(1, dtype=dtype, framework=framework)
  zero = array(0, dtype=dtype, framework=framework)
  return where(
      random_uniform(
          shape, minval=0., maxval=1., dtype=dtype, framework=framework) <= p,
      one, zero)


# ===========================================================================
# General neural network utilities
# ===========================================================================
def embedding(indices, weight, max_norm=None):
  R"""
  A simple lookup table that looks up embeddings in a fixed dictionary and size.

  This module is often used to retrieve word embeddings using indices. The input
  to the module is a list of indices, and the embedding matrix, and the output
  is the corresponding word embeddings.

  Arguments:
    indices: A `Tensor` with type `int32` or `int64` containing the ids to be
      looked up in `weight`.
    weight: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the 'div' `partition_strategy`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` with the same type as the tensors in `weights`.
  """
  if tf.is_tensor(indices) or tf.is_tensor(weight):
    return tf.nn.embedding_lookup(params=weight, ids=indices, max_norm=max_norm)
  if torch.is_tensor(indices) or torch.is_tensor(weight):
    return torch.nn.functional.embedding(input=indices,
                                         weight=weight,
                                         max_norm=max_norm)
  return tf.nn.embedding_lookup(params=weight, ids=indices,
                                max_norm=max_norm).numpy()


def _process_noise_dim(input_shape, dims):
  """
  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.
  Examples
  --------
  (None, 10, 10) with noise_dims=2
  => Noise mask: (None, 10, 1)
  """
  if dims is None:
    return input_shape
  ndims = len(input_shape)
  dims = [i % ndims for i in as_tuple(dims, t=int)]
  # ====== get noise shape ====== #
  return tuple([1 if i in dims else input_shape[i] for i in range(ndims)])


def dropout(x,
            p=0.5,
            axis=None,
            noise_type='uniform',
            rescale=True,
            training=True):
  r""" Computes dropout output for training or evaluation phase.
  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  Arguments:
    x: A tensor.
      input tensor
    p: float(0.-1.)
      probability of dropout (i.e. set a value to zero)
    axis: int or list(int)
      these dimensions will be setted to 1 in noise_shape, and
      used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform', 'binomial'
      distribution used for generating noise
    rescale: bool
      if `True`, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
      training. This means that during evaluation the module simply computes an
      identity function
    training: bool
      if `True`, return the dropout-ed tensor during training,
      otherwise rescaled tensor during evaluation.

  References:
    Hinton, G.E., et al., 2012. Improving neural networks by preventing
      co-adaptation of feature detectors. arXiv:1207.0580 [cs].
    Srivastava, N., et al., 2014. Dropout: A Simple Way to Prevent Neural Networks from
      Overﬁtting, JMLR

  Note:
    This function only apply noise on Variable when training is enable
  """
  framework = parse_framework(x)
  shape = x.shape
  retain_prob = 1. - p
  # ====== not a training variable NO dropout ====== #
  if 'normal' in noise_type or 'gaussian' in noise_type:
    randfunc = lambda shape: random_normal(shape,
                                           mean=1.0,
                                           stddev=np.sqrt((1.0 - retain_prob) /
                                                          retain_prob),
                                           dtype=x.dtype,
                                           framework=framework)
  elif 'uniform' in noise_type or 'binomial' in noise_type:
    randfunc = lambda shape: random_binomial(
        shape, p=retain_prob, dtype=x.dtype, framework=framework)
  else:
    raise ValueError('No support for noise_type=' + noise_type)

  # ====== Dropout ====== #
  if training:
    noise_shape = shape if axis is None else _process_noise_dim(shape, axis)
    noise_shape = [i for i in noise_shape]
    y = x * randfunc(noise_shape)
    if rescale:
      y /= retain_prob
    return y
  else:
    return x
