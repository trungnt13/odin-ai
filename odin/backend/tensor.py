# A collection of functions for processing, manipulating and calculating
# necessary information from tensors
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from __future__ import absolute_import, division, print_function

import copy
import inspect
import numbers
from collections import defaultdict
from functools import wraps

import numpy as np
import scipy as sp
import tensorflow as tf
import torch
from scipy.special import logsumexp
from six import string_types
from six.moves import builtins
from tensorflow.python.ops import init_ops

from odin.utils import as_tuple, is_number, is_same_shape, is_string, uuid


# ===========================================================================
# Helper
# ===========================================================================
def _normalize_axis(axis, ndim):
  if axis is None:
    return None
  if isinstance(axis, (tuple, list)):
    return tuple([a % ndim if a is not None else a for a in axis])
  return axis % ndim


def parse_framework(alias):
  """ Convert a string or object to appropriate framework module: numpy,
  tensorflow or torch """
  if not isinstance(alias, string_types):
    alias = str(alias)
  alias = alias.strip().lower()
  if any(i in alias for i in ['tf', 'tensorflow', 'tensor']):
    return tf
  if any(i in alias for i in ['torch', 'pytorch', 'pt', 'tr']):
    return torch
  return np


def dtype_universal(dtype,
                    torch_dtype=False,
                    tf_dtype=False,
                    np_dtype=False,
                    framework=None):
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
  if framework is None:
    framework = 'numpy'
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
      return torch.from_numpy(x)
  return x


# ===========================================================================
# Conversion
# ===========================================================================
def to_llh(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood
  :math:`LLH = log(prob(data|target))`
  '''
  with tf.name_scope(name, "log_likelihood", [x]):
    x /= tf.reduce_sum(x, axis=-1, keepdims=True)
    x = tf.clip_by_value(x, EPS, 1 - EPS)
    return tf.log(x)


def to_llr(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood ratio
  :math:`LLR = log(\\frac{prob(data|target)}{prob(data|non-target)})`
  '''
  with tf.name_scope(name, "log_likelihood_ratio", [x]):
    nb_classes = x.shape.as_list()[-1]
    new_arr = []
    for j in range(nb_classes):
      scores_copy = tf.transpose(
          tf.gather(tf.transpose(x), [i for i in range(nb_classes) if i != j]))
      scores_copy -= tf.expand_dims(x[:, j], axis=-1)
      new_arr.append(-logsumexp(scores_copy, 1))
    return tf.concat(new_arr, axis=-1) + np.log(13)


def to_nonzeros(x, value):
  x = tf.where(tf.equal(x, 0.), tf.zeros_like(x) + value, x)
  return x


def to_sample_weights(indices, weights, name=None):
  """ Convert indices or one-hot matrix and
  give weights to sample weights for training """
  with tf.name_scope(name, "to_sample_weights", [indices]):
    # ====== preprocess indices ====== #
    ndim = len(indices.shape)
    if ndim <= 1:  # indices vector
      indices = tf.cast(indices, dtype=tf.int64)
    else:
      indices = tf.argmax(indices, axis=-1)
    # ====== prior weights ====== #
    if isinstance(weights, (tuple, list, np.ndarray)):
      prior_weights = tf.constant(weights, dtype=floatX, name="prior_weights")
    # ====== sample weights ====== #
    weights = tf.gather(prior_weights, indices)
  return weights


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


def ones(shape, dtype='float32', framework='numpy'):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  return framework.ones(shape, dtype=dtype)


def zeros(shape, dtype='float32', framework='numpy'):
  framework = parse_framework(framework)
  dtype = dtype_universal(dtype, framework=framework)
  return framework.zeros(shape, dtype=dtype)


def tril_mask(shape, framework='numpy'):
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
        x = dimshuffle(x, pattern=list(range(ndims)) + ['x'] * len(axes))
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
# Linear Algebra
# ===========================================================================
def dot(x, y, name=None):
  """ Theano-style dot product
  For 2-D arrays it is equivalent to matrix multiplication,
  and for 1-D arrays to inner product of vectors (without complex conjugation).
  For N dimensions it is a sum product over the last axis of a and
  the second-to-last of b

  NOTE: this behavior is the same in `numpy.dot` as well but
  hasn't replicated in `tensorflow.matmul`

  Example
  -------
   (2, 3).(4, 3, 5) => (2, 4, 5)
   (2, 3, 4).(4, 5) => (2, 3, 5)
   (2, 3, 4).(5, 4, 6) => (2, 3, 5, 6)
  """
  with tf.name_scope(name, "DotProduct"):
    shapeX = [
        tf.shape(x)[i] if d is None else d
        for i, d in enumerate(x.shape.as_list())
    ]
    shapeY = [
        tf.shape(y)[i] if d is None else d
        for i, d in enumerate(y.shape.as_list())
    ]
    ndimX = x.shape.ndims
    ndimY = y.shape.ndims
    if ndimX > 2:
      x = tf.reshape(x, (-1, shapeX[-1]))
    if ndimY > 2:
      y_dims = list(range(ndimY))
      y_dims = [y_dims.pop(-2)] + y_dims
      y = tf.transpose(y, perm=y_dims)
      y = tf.reshape(y, (shapeY[-2], -1))
      outshapeY = [shapeY[i] for i in y_dims[1:]]
    else:
      outshapeY = [shapeY[-1]]
    # calculate dot product and desire shape
    output_shape = shapeX[:-1] + outshapeY
    output = tf.reshape(tf.matmul(x, y), output_shape)
  return output


def batched_dot(x, y, name=None):
  """Batchwise dot product.
  This function computes the dot product between the two tensors,
  by iterating over the first dimension.
  """
  with tf.name_scope(name, "BatchedDot"):
    shapeX = [
        tf.shape(x)[i] if d is None else d
        for i, d in enumerate(x.shape.as_list())
    ]
    shapeY = [
        tf.shape(y)[i] if d is None else d
        for i, d in enumerate(y.shape.as_list())
    ]
    ndimX = x.shape.ndims
    ndimY = y.shape.ndims
    # same as dot but one more batch dimension
    if ndimX > 2 + 1:
      x = tf.reshape(x, (-1, np.prod(shapeX[1:-1]), shapeX[-1]))
    if ndimY > 2 + 1:
      y_dims = list(range(ndimY))
      y_dims = [y_dims.pop(0), y_dims.pop(-2)] + y_dims
      y = tf.transpose(y, perm=y_dims)
      outshapeY = [shapeY[i] for i in y_dims[2:]]
      y = tf.reshape(y, (-1, shapeY[-2], np.prod(outshapeY)))
    else:
      outshapeY = [shapeY[-1]]
    # calculate dot product and desire shape
    output_shape = shapeX[:-1] + outshapeY
    output = tf.reshape(tf.matmul(x, y),
                        [i if i is not None else -1 for i in output_shape],
                        name=name)
    return output


def switch(condition, then_expression, else_expression, name=None):
  with tf.name_scope(name, 'switch'):
    if condition.dtype != tf.bool:
      condition = tf.cast(condition, 'bool')
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


def apply_mask(x, mask, name="ApplyMask"):
  """
  x : 3D tensor
  mask : 2D tensor

  Example
  -------
  >>> Input: [128, 500, 120]
  >>> Mask:  [1, 1, 0]
  >>> Output: [128, 500, 0]
  """
  return tf.mul(x, tf.expand_dims(mask, -1), name=name)


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
  if tf.is_tensor(x[0]):
    return tf.concat(x, axis)
  if torch.is_tensor(x[0]):
    return torch.cat(x, axis)
  return np.concatenate(x, axis)


def dimshuffle(x, pattern):
  """ Reorder the dimensions of this variable, optionally inserting
  broadcasted dimensions.

  Parameters
  ----------
  pattern
      List/tuple of int mixed with 'x' for broadcastable dimensions.

  Examples
  --------
  For example, to create a 3D view of a [2D] matrix, call
  ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
  middle dimension is an implicit broadcasted dimension.  To do the same
  thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

  Notes
  -----
  This function supports the pattern passed as a tuple, or as a
  variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
  to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
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


def repeat(x, n, axes=None, name="Repeat"):
  """ Repeat a N-D tensor.

  If x has shape (s1, s2, s3) and axes=(1, -1), the output
  will have shape (s1, s2 * n[0], s3 * n[1]).

  Parameters
  ----------
  n : {int, list of int}
    each number of repeatation according to the axes
  axes : {int, list or int}
    all axes for repeating
  """
  # TODO
  if axes is not None:
    ndim = x.shape.ndims
    if not isinstance(axes, (tuple, list)):
      axes = (axes,)
    axes = _normalize_axis(axes, ndim)
    n = as_tuple(n, len(axes))
    return tf.tile(
        x,
        multiples=[n[axes.index(i)] if i in axes else 1 for i in range(ndim)],
        name=name)
  else:
    n = int(n)
    return tf.tile(x, multiples=[n for i in range(ndim)], name=name)


# ===========================================================================
# Statistics
# ===========================================================================
def moments(x, axis=None, keepdims=False):
  """ Calculates the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.
  """
  if tf.is_tensor(x):
    mean, variance = tf.nn.moments(x, axes=axis, keepdims=keepdims)
  elif torch.is_tensor(x):
    mean = reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = square(x - mean)
    variance = reduce_mean(devs_squared, axis=axis, keepdims=keepdims)
    if not keepdims:
      mean = mean.squeeze(axis)
  else:
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    variance = np.var(x, axis=axis, keepdims=keepdims)
  return mean, variance


def reduce_var(x, axis=None, keepdims=False, mean=None):
  """ Calculate the variance of `x` along given `axis`
  if `mean` is given,
  """
  if isinstance(x, np.ndarray):
    return np.var(x, axis=axis, keepdims=keepdims)
  ndim = x.ndim
  axis = _normalize_axis(axis, ndim)
  m = reduce_mean(x, axis=axis, keepdims=True) if mean is None else mean
  devs_squared = square(x - m)
  return reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
  return sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def reduce_min(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_min(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.min(dim=axis, keepdim=keepdims)[0]
  return np.min(x, axis=axis, keepdims=keepdims)


def reduce_max(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_max(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.max(dim=axis, keepdim=keepdims)[0]
  return np.max(x, axis=axis, keepdims=keepdims)


def reduce_mean(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.mean(dim=axis, keepdim=keepdims)
  return np.mean(x, axis=axis, keepdims=keepdims)


def reduce_sum(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.sum(dim=axis, keepdim=keepdims)
  return np.sum(x, axis=axis, keepdims=keepdims)


def reduce_prod(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.prod(dim=axis, keepdim=keepdims)
  return np.prod(x, axis=axis, keepdims=keepdims)


def reduce_all(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_all(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.bool().all(dim=axis, keepdim=keepdims)
  return np.all(x, axis=axis, keepdims=keepdims)


def reduce_any(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_any(tf.cast(x, tf.bool), axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.bool().any(dim=axis, keepdim=keepdims)
  return np.any(x, axis=axis, keepdims=keepdims)


def reduce_logsumexp(x, axis=None, keepdims=False):
  if tf.is_tensor(x):
    return tf.reduce_logsumexp(x, axis=axis, keepdims=keepdims)
  if torch.is_tensor(x):
    return x.logsumexp(dim=axis, keepdim=keepdims)
  return logsumexp(x, axis=axis, keepdims=keepdims)


def reduce_log_exp(x, reduction_function=tf.reduce_mean, axis=None, name=None):
  """ log-reduction-exp over axis to avoid overflow and underflow

  Parameters
  ----------
  `x` : [nb_sample, feat_dim]
  `axis` should be features dimension
  """
  with tf.name_scope(name, "logreduceexp"):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    y = tf.log(reduction_function(tf.exp(x - x_max), axis=axis,
                                  keepdims=True)) + x_max
    return tf.squeeze(y)


def cumsum(x, axis):
  if tf.is_tensor(x):
    return tf.math.cumsum(x, axis=axis)
  if torch.is_tensor(x):
    return torch.cumsum(x, dim=axis)
  return np.cumsum(x, axis=axis)


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


def greater_equal(x, y):
  if tf.is_tensor(x) or tf.is_tensor(y):
    return tf.greater_equal(x, y)
  if torch.is_tensor(x) or torch.is_tensor(y):
    return x >= y
  return np.greater_equal(x, y)
