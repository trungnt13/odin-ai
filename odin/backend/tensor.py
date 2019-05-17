from __future__ import print_function, division, absolute_import

import copy
import numbers
import inspect
from functools import wraps
from six.moves import builtins
from collections import defaultdict

import numpy as np
import scipy as sp

import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin.autoconfig import (get_session, get_ngpu, get_ncpu, get_ncpu_native,
                         get_floatX, get_random_state, randint)
from odin.utils import as_tuple, uuid, is_number, is_string, is_same_shape
from odin.backend import role
from odin.backend.helpers import (set_shape, is_tensor, is_training, cond_training,
                                  get_all_variables, get_all_variables_or_tensors,
                                  get_all_tensors, is_variable, set_value)

floatX = get_floatX()
EPS = np.finfo(floatX).eps

def variable(value=None, shape=None, dtype=None, name=None, roles=[],
             initialize=False):
  '''Instantiates a tensor, automatically initialize the variable
  in tensorflow

  Parameters
  ----------
  value: numpy array
      initial value of the tensor.
      * string: interpreted as variable (or tensor) name,
        searching for variable (or tensor) with given name,
        then copy it value to new variable
      * callable: a function with input arguments include: shape and dtype
      * number:
      * numpy.ndarray:
      * Variable:
      * Tensor:
  dtype: dtype
      tensor type.
  name: str
      optional name string for the tensor.
  roles: {Role, list of Role}
      given Role for initialized Variable from `odin.backend.role`
  initialize : bool
      if True, call Session run to initialize the variable.

  Returns
  -------
      Tensor variable instance.
  '''
  if dtype is None:
    dtype = floatX
  #### Check if valid value or shape is given
  if value is not None or shape is not None:
    # 1. initializing function.
    if is_string(value):
      value = get_all_variables_or_tensors(name=value)
      if len(value) == 0:
        raise ValueError("Cannot find any variable or tensor with name: "
            "'%s' for the initializer." % value)
      value = value[0]
    # 2. initializing funciton
    elif hasattr(value, '__call__'):
      value = value(shape=shape)
      if dtype is not None:
        value = tf.cast(value, dtype=dtype)
    # 3. is a scalar value
    elif is_number(value):
      value = tf.constant(value=value, dtype=dtype, shape=shape)
    # 4. Numpy ndarray.
    elif isinstance(value, np.ndarray):
      pass
    # 5. Shared variable, just check the shape.
    elif is_variable(value):
      _shape = value.shape.as_list()
      if shape is not None and tuple(shape) != tuple(_shape):
        raise Exception('Require variable with shape=%s, but was given different '
                        'shape=%s' % (str(shape), str(_shape)))
    # 6. expression, we can only check number of dimension.
    elif is_tensor(value):
      if shape is not None and not is_same_shape(value.shape, shape):
        raise Exception("Expected shape: %s, given Tensor with shape: %s"
                        % (shape, value.shape))
    elif value is not None:
      raise ValueError("Unsupport for given set of arguments: value=%s; shape=%s" %
                       (type(value), type(shape)))
  #### matching the shape of `value` and given `shape`
  if value is not None:
    v_shape = value.shape
    if hasattr(v_shape, 'as_list'):
      v_shape = as_tuple(v_shape.as_list())
    if shape is not None:
      if not is_same_shape(v_shape, shape):
        raise ValueError("Given `value` with shape=%s, but require shape=%s" %
                        (v_shape, shape))
    else:
      shape = v_shape
  #### Try to find cached variable
  variable = None
  if name is not None:
    for v in get_all_variables(scope=tf.get_variable_scope().name,
                               name=name):
      if shape is not None and not is_same_shape(v.shape, shape):
        continue
      variable = v
      break
  if variable is None:
    #### try one more time getting the Variable
    if value is None:
      variable = tf.get_variable(name=name, shape=shape, dtype=dtype)
      if variable is None:
        raise RuntimeError("Cannot find or create Variable given following "
          "information: value=%s; shape=%s; dtype=%s; name=%s; roles=%s"
          % (str(value), str(shape), str(dtype), str(name), str(roles)))
    #### create totally new variable
    else:
      variable = tf.Variable(initial_value=value, dtype=dtype,
                             expected_shape=shape, name=name)
      # initialize variable
      if initialize:
        get_session(graph=variable.graph).run(variable.initializer)
  #### found exist Variable, set new value for it
  elif value is not None:
    set_value(x=variable, value=value)
  #### add roles and return
  return role.add_roles(variable, roles)

# ===========================================================================
# Helper
# ===========================================================================
def _normalize_axis(axis, ndim):
  if axis is None:
    return None
  if isinstance(axis, (tuple, list)):
    return tuple([a % ndim if a is not None else a
            for a in axis])
  return axis % ndim

# ===========================================================================
# Normalization
# ===========================================================================
def length_norm(x, axis=-1, epsilon=1e-12, ord=2):
  """ L2-normalization (or vector unit length normalization)

  Parameters
  ----------
  x : array
  axis : int
  ord : int
    order of norm (1 for L1-norm, 2 for Frobenius or Euclidean)
  """
  ord = int(ord)
  if ord not in (1, 2):
    raise ValueError("only support `ord`: 1 for L1-norm; 2 for Frobenius or Euclidean")
  # ====== tensorflow ====== #
  if is_tensor(x):
    if ord == 2:
      x_norm = tf.sqrt(tf.maximum(tf.reduce_sum(x ** 2, axis=axis, keepdims=True),
                                  epsilon))
    else:
      x_norm = tf.maximum(tf.reduce_sum(tf.abs(x), axis=axis, keepdims=True),
                          epsilon)
  # ====== numpy ====== #
  else:
    if ord == 2:
      x_norm = np.sqrt(np.maximum(np.sum(x ** 2, axis=axis, keepdims=True),
                                  epsilon))
    else:
      x_norm = np.maximum(np.sum(np.abs(x), axis=axis, keepdims=True),
                          epsilon)
  return x / x_norm

def calc_white_mat(X):
  """ calculates the whitening transformation for cov matrix X
  """
  # ====== tensorflow ====== #
  if is_tensor(X):
    W = tf.linalg.cholesky(tf.linalg.inv(X))
  # ====== numpy ====== #
  else:
    W = sp.linalg.cholesky(sp.linalg.inv(X), lower=True)
  return W

def log_norm(x, axis=1, scale_factor=10000):
  """ Seurat log-normalize
  y = log(X / (sum(X, axis) + epsilon) * scale_factor)

  where `log` is natural logarithm
  """
  if is_tensor(x):
    return tf.log1p(
        x / (tf.reduce_sum(x, axis=axis, keepdims=True) + EPS) * scale_factor)
  elif isinstance(x, np.ndarray):
    x = x.astype('float64')
    return np.log1p(
        x / (np.sum(x, axis=axis, keepdims=True) + np.finfo(x.dtype).eps) * scale_factor)
  else:
    raise ValueError(
        "Only support numpy.ndarray or tensorflow.Tensor, but given: %s" % str(type(x)))

def delog_norm(x, x_sum=1, scale_factor=10000):
  """ This perform de-log normalization of `log_norm` values
  if `x_sum` is not given (i.e. default value 1), then all the
  """
  if is_tensor(x):
    return (tf.exp(x) - 1) / scale_factor * (x_sum + EPS)
  elif isinstance(x, np.ndarray):
    return (np.exp(x) - 1) / scale_factor * (x_sum + EPS)
  else:
    raise ValueError("Only support numpy.ndarray or tensorflow.Tensor")

# ===========================================================================
# Conversion
# ===========================================================================
def logreduceexp(x, reduction_function=tf.reduce_mean, axis=None,
                 name="LogReduceExp"):
  """ log-reduction-exp over axis to avoid overflow and underflow

  Parameters
  ----------
  `x` : [nb_sample, feat_dim]
  `axis` should be features dimension
  """
  x_max = tf.reduce_max(x, axis=axis, keepdims=True)
  y = tf.log(
      reduction_function(tf.exp(x - x_max), axis = axis, keepdims=True)) + x_max
  return tf.squeeze(y)

def logsumexp(x, axis=-1, name=None):
  """
  `x` : [nb_sample, feat_dim]
  `axis` should be features dimension
  """
  # ====== tensorflow ====== #
  if is_tensor(x):
    with tf.name_scope(name, 'logsumexp', [x]):
      xmax = tf.reduce_max(x, axis=axis, keepdims=True)
      y = xmax + tf.log(tf.reduce_sum(tf.exp(x - xmax), axis=axis, keepdims=True))
  # ====== numpy ====== #
  elif isinstance(x, np.ndarray):
    xmax = np.max(x, axis=axis, keepdims=True)
    y = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
  else:
    raise ValueError("`x` must be tensorflow.Tensor or numpy.ndarray")
  return y

def to_llh(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood
  :math:`LLH = log(prob(data|target))`
  '''
  # ====== numpy ====== #
  if not is_tensor(x):
    x /= np.sum(x, axis=-1, keepdims=True)
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x)
  # ====== Tensorflow ====== #
  else:
    with tf.name_scope(name, "log_likelihood", [x]):
      x /= tf.reduce_sum(x, axis=-1, keepdims=True)
      x = tf.clip_by_value(x, EPS, 1 - EPS)
      return tf.log(x)

def to_llr(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood ratio
  :math:`LLR = log(\\frac{prob(data|target)}{prob(data|non-target)})`
  '''
  # ====== numpy ====== #
  if not is_tensor(x):
    new_arr = np.empty_like(x)
    nb_classes = x.shape[1]
    columns = np.arange(nb_classes)
    for j in range(nb_classes):
      scores_copy = x[:, np.delete(columns, j)] - x[:, j][:, None]
      new_arr[:, j] = -logsumexp(scores_copy, 1).T
    return new_arr + np.log(13)
  # ====== tensorflow ====== #
  else:
    with tf.name_scope(name, "log_likelihood_ratio", [x]):
      nb_classes = x.shape.as_list()[-1]
      new_arr = []
      for j in range(nb_classes):
        scores_copy = tf.transpose(tf.gather(tf.transpose(x),
                                   [i for i in range(nb_classes) if i != j]))
        scores_copy -= tf.expand_dims(x[:, j], axis=-1)
        new_arr.append(-logsumexp(scores_copy, 1))
      return tf.concat(new_arr, axis=-1) + np.log(13)

def to_nonzeros(x, value):
  if is_tensor(x):
    x = tf.where(tf.equal(x, 0.), tf.zeros_like(x) + value, x)
  else:
    x[x == 0] = value
  return x

def to_sample_weights(indices, weights, name=None):
  """ Convert indices or one-hot matrix and
  give weights to sample weights for training """
  with tf.name_scope(name, "to_sample_weights", [indices]):
    # ====== preprocess indices ====== #
    ndim = len(indices.shape)
    if ndim <= 1: # indices vector
      indices = tf.cast(indices, dtype=tf.int64)
    else:
      indices = tf.argmax(indices, axis=-1)
    # ====== prior weights ====== #
    if isinstance(weights, (tuple, list, np.ndarray)):
      prior_weights = tf.constant(weights, dtype=floatX,
                            name="prior_weights")
    # ====== sample weights ====== #
    weights = tf.gather(prior_weights, indices)
  return weights

# ===========================================================================
# Allocation
# ===========================================================================
def tril(m, k=0, name='LowerTriangle'):
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
    return tf.matrix_band_part(input=m, num_lower=-1, num_upper=0, name=name)
  if k < 0:
    return tf.subtract(m,
      tf.matrix_band_part(input=m, num_lower=np.abs(k) - 1, num_upper=-1),
      name=name)
  # k > 0
  return tf.matrix_band_part(input=m, num_lower=-1, num_upper=k, name=name)

def tril_indices(n, k=0, name='LowerTriangleIndices'):
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
  with tf.variable_scope(name):
    M1 = tf.tile(tf.expand_dims(tf.range(n), axis=0), [n, 1])
    M2 = tf.tile(tf.expand_dims(tf.range(n), axis=1), [1, n])
    mask = tf.transpose((M1 - M2) >= -k)
    ix1 = tf.boolean_mask(M2, mask)
    ix2 = tf.boolean_mask(M1, mask)
    return ix1, ix2

def prior2weights(prior, exponential=False,
                  min_value=0.1, max_value=None,
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
                   key=lambda x: x[-1], reverse=False)
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

def entropy(p):
  """Return simple calculation of discrete Shanon entropy"""
  return -tf.reduce_sum(p * tf.log(p))

def linear(x):
  return x

def upsample(x, scale, axes, method='nn', name="Upsample"):
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
  with tf.variable_scope(name):
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
        raise ValueError('upsample with NN mode does not support rank >= 6 tensor.')
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
      paddings = [[0, 0] if i not in axes else
          [tf.cast(tf.ceil(input_shape[i] * (scale_map[i] - 1) / 2), 'int32'),
           tf.cast(tf.floor(input_shape[i] * (scale_map[i] - 1) / 2), 'int32')]
          for i in range(ndims)]
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
    return set_shape(x, shape=[
        s * scale_map[i] if is_number(s) else None
        for i, s in enumerate(input_shape_int)])

def eye(n, m, dtype, name='eye'):
  x = tf.Variable(initial_value=np.eye(n, m, dtype=dtype), dtype=dtype,
                  name=name)
  get_session().run(x.initializer)
  return x

# ===========================================================================
# Linear Algebra
# ===========================================================================
def dot(x, y, name='Dot'):
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
  with tf.variable_scope(name):
    shapeX = [tf.shape(x)[i] if d is None else d
              for i, d in enumerate(x.shape.as_list())]
    shapeY = [tf.shape(y)[i] if d is None else d
              for i, d in enumerate(y.shape.as_list())]
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

def batched_dot(x, y, name='BatchedDot'):
  """Batchwise dot product.
  This function computes the dot product between the two tensors,
  by iterating over the first dimension.
  """
  with tf.variable_scope(name):
    shapeX = [tf.shape(x)[i] if d is None else d
              for i, d in enumerate(x.shape.as_list())]
    shapeY = [tf.shape(y)[i] if d is None else d
              for i, d in enumerate(y.shape.as_list())]
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


def switch(condition, then_expression, else_expression, name='switch'):
  with tf.variable_scope(name):
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


def argsort(x, k=None, name='argsort'):
  """ The indices in -1 axis will be sorted by the values in
  descending order.

  Parameters
  ----------
  x: 1-D or higher `Tensor`
      with last dimension at least `k`.
  k: 0-D `int32` `Tensor`.
      Number of top elements to look for along the last
      dimension (along each row for matrices).

  """
  # the return value contains (values, indices)
  # but we only take the indices
  if k is None:
    k = x.shape[-1]
  return tf.nn.top_k(x, k=k, sorted=True, name=name)[1]


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
def reshape(x, shape, name='Reshape'):
  """ More flexible version of reshape operation

  Example
  -------
  x.shape = [25, 08, 12]
  reshape(shape=([1], [2], [0]))
  => x.shape = (08, 12, 25)
  """
  input_shape = x.shape.as_list()
  new_shape = []
  for i in shape:
    if i is None:
      new_shape.append(-1)
    elif isinstance(i, (list, tuple)):
      new_shape.append(input_shape[i[0]])
    else:
      new_shape.append(i)
  new_shape = tuple([-1 if i is None else i
                     for i in new_shape])
  return tf.reshape(x, new_shape, name=name)

def dimshuffle(x, pattern, name='Dimshuffle'):
  """Transpose dimensions.

  pattern should be a tuple or list of
  dimension indices, e.g. [0, 2, 1].
  """
  with tf.variable_scope(name):
    x = tf.transpose(x, perm=[i for i in pattern if i != 'x'])
    # insert new dimension
    for i, p in enumerate(pattern):
      if p == 'x':
        x = tf.expand_dims(x, i)
  return x

def flatten(x, outdim=1, name='Flatten'):
  """ Keep all the original dimension until `outdim - 1`
  """
  with tf.variable_scope(name):
    if outdim == 1:
      return tf.reshape(x, [-1], name=name)
    input_shape = [tf.shape(x)[i] if d is None else d
                   for i, d in enumerate(x.shape.as_list())]
    other_shape = tuple([input_shape[i] for i in range(outdim - 1)])
    n = 1
    for i in input_shape[(outdim - 1):]:
      n = n * i
    output_shape = other_shape + (n,)
    return tf.reshape(x, output_shape)

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
  if axes is not None:
    ndim = x.shape.ndims
    if not isinstance(axes, (tuple, list)):
      axes = (axes,)
    axes = _normalize_axis(axes, ndim)
    n = as_tuple(n, len(axes))
    return tf.tile(x, multiples=[n[axes.index(i)] if i in axes else 1
                                 for i in range(ndim)],
                   name=name)
  else:
    n = int(n)
    return tf.tile(x, multiples=[n for i in range(ndim)],
                   name=name)


# ===========================================================================
# Statistics
# ===========================================================================
def var(x, axes=None, keepdims=False, name="Variance"):
  with tf.variable_scope(name):
    axes = _normalize_axis(axes, x.shape.ndims)
    x = tf.cast(x, floatX)
    m = tf.reduce_mean(x, axis=axes, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axes, keepdims=keepdims)


def std(x, axes=None, keepdims=False, name="Std"):
  with tf.variable_scope(name):
    return tf.sqrt(var(x, axes=axes, keepdims=keepdims), name=name)


def renorm_rms(X, axis=1, target_rms=1.0, name="RescaleRMS"):
  """ Scales the data such that RMS of the features dimension is 1.0
  scale = sqrt(x^t x / (D * target_rms^2)).

  NOTE
  ----
  by defaults, assume the features dimension is `1`
  """
  with tf.variable_scope(name):
    D = tf.sqrt(tf.cast(tf.shape(X)[axis], X.dtype.base_dtype))
    l2norm = tf.sqrt(tf.reduce_sum(X ** 2, axis=axis, keepdims=True))
    X_rms = l2norm / D
    X_rms = tf.where(tf.equal(X_rms, 0.),
                     x=tf.ones_like(X_rms, dtype=X_rms.dtype.base_dtype),
                     y=X_rms)
    return target_rms * X / X_rms

# ===========================================================================
# RNN and loop
# ===========================================================================
def map_tensors(fn):
  tf.map_fn

def scan_tensors(fn,
                 sequences=None, initializer=None,
                 mask=None, mask_value=None,
                 axis=0, n_steps=None,
                 backward=False, reverse=False,
                 reshape_outputs=False,
                 parallel_iterations=12,
                 name=None):
  """

  Parameters
  ----------
  fn: callable
    It accepts two arguments `(outputs, inputs)`.
    The first will have the same structure as `initializer` if one is provided,
    otherwise it will have the same structure as `elems`.  The second
    will have the same (possibly nested) structure as `elems`.
    Its returns must have the same structure as `initializer`
    if one is provided, otherwise it must have the same structure as `elems`.

  sequences : {Tensor, list of Tensor}

  mask : {Tensor, list of Tensor}
    binary tensors with shape [n_timestep, ...],
    with a zero for every element that is masked out.

  mask_value : {None, scalar}
    the replacement value of given time step is masked out,
    if None use previous time step values.

  initializer : {None, Tensor, list of Tensor}
    Containing the initial values for the states used in the step function.
    If initializer is None, `sequences` must contain at least one element,
    and its first element is used as the initializer.

  axis : {int, list of int} (default: 0)
    the axis to be unpacked (ravel) for iteration, if given a list,
    applying each value for each input in `sequences`
    For example, input to RNN is `[n_samples, n_timestep, n_features]`,
    and we want iterate over `time` dimension, hence, `axis=1`

  backward : bool (default: False)
    If True, reverse the input sequences, so the scan Op iterate
    from opposite order.

  reverse : bool (default: False)
    If True, do the iteration over the `axis` dimension in reverse
    order and return the reversed sequence.
    The difference between `reverse` and backward` is `reverse` also
    flip the outputs, so it returns reversed outputs.

  reshape_output : bool (default: False)
    reshape the output so the `axis` dimension (e.g. time dimension)
    back to original position.
    The `axis` dimension is be moved to the first dimension for
    iterating using scan

  n_steps : {None, integer}
    number of time steps

  Note
  ----
  backwards mode only invert sequences then iterate over them
  """
  single_input = False
  single_output = False
  with tf.name_scope(name=name, default_name='ScanTensors'):
    # ====== check sequences ====== #
    if sequences is None:
      sequences = []
    elif not isinstance(sequences, (tuple, list)):
      single_input = True
      sequences = [sequences]
    sequences = [tf.convert_to_tensor(x) for x in sequences]
    n_inputs = len(sequences)
    # ====== iterating axis ====== #
    axis = as_tuple(axis, t=int, N=n_inputs)
    new_sequences = []
    for x, a in zip(sequences, axis):
      if a != 0:
        extra_dims = [i for i in range(x.shape.ndims) if i != a]
        x = tf.transpose(x, perm=[a] + extra_dims)
      new_sequences.append(x)
    sequences = new_sequences
    n_timestep = sequences[0].shape.as_list()[0]
    # ====== check mask ====== #
    if mask is not None:
      mask = tf.cast(tf.convert_to_tensor(mask), tf.bool)
      if len(set(axis)) == 1:
        a = axis[0]
        if mask.shape[a].value == n_timestep:
          extra_dims = [i for i in range(mask.shape.ndims) if i != a]
          mask = tf.transpose(mask, perm=[a] + extra_dims)
      # special case for RNN
      elif mask.shape.ndims == 2:
        if mask.shape[1].value == n_timestep:
          mask = tf.transpose(mask, perm=(1, 0))
      # check match timestep for inputs and mask
      if n_timestep is not None:
        assert mask.shape.as_list()[0] == n_timestep,\
        "First dimension of `mask` must be %d, but given shape: %s" % \
        (n_timestep, str(mask.shape))
    # ====== check mask value ====== #
    if isinstance(mask_value, numbers.Number):
      pass
    elif mask_value is None:
      pass
    else:
      raise ValueError("No support for `mask_value`: %s" % str(mask_value))
    # ====== backward or reverse ====== #
    if backward:
      sequences = [tf.reverse(x, axis=(0,)) for x in sequences]
      if mask is not None:
        mask = tf.reverse(mask, axis=(0,))
    # ====== fixed number of steps ====== #
    if isinstance(n_steps, numbers.Number):
      sequences = [x[:int(n_steps)] for x in sequences]
      if mask is not None:
        mask = mask[:int(n_steps)]
    # ====== check output info ====== #
    if initializer is None:
      initializer = [x[0] for x in sequences]
      single_output = single_input
    elif not isinstance(initializer, (tuple, list)):
      single_output = True
      initializer = [initializer]

    # ====== modified step function ====== #
    def step_(outputs, inputs):
      mask_t = None
      if mask is not None:
        mask_t = inputs[-1]
        inputs = inputs[:-1]
      # applying the function
      new_outputs = fn(outputs[0] if single_output else outputs,
                       inputs[0] if single_input else inputs)
      # masking the output
      new_outputs = ([new_outputs]
                     if not isinstance(new_outputs, (tuple, list)) else
                     list(new_outputs))
      if mask_t is not None:
        _ = []
        for o, new_o in zip(outputs, new_outputs):
          if mask_t.shape.ndims == 0:
            tiled_mask_t = mask_t
          else:
            m = mask_t
            orig_ndims = m.shape.ndims
            for i in range(o.shape.ndims - mask_t.shape.ndims):
              m = tf.expand_dims(m, axis=-1)
            multiples = [1] * orig_ndims + [tf.shape(o)[i]
                                            for i in range(orig_ndims, m.shape.ndims)]
            tiled_mask_t = tf.tile(m, multiples)
          new_o = tf.where(tiled_mask_t,
                           new_o,
                           o if mask_value is None else mask_value)
          _.append(new_o)
        new_outputs = _
      return new_outputs
    # ====== call scan ====== #
    ret = tf.scan(step_,
                  elems=sequences + ([mask] if mask is not None else []),
                  initializer=initializer,
                  parallel_iterations=int(parallel_iterations),
                  back_prop=is_training(),
                  swap_memory=False, infer_shape=True, reverse=False)
    # ====== reshape output ====== #
    if reshape_outputs:
      assert len(axis) == len(ret), \
      "Number of outputs (%d) mismatch number of inputs (%d), cannot use `reshape_outputs`"\
      % (len(ret), len(axis))
      outputs = []
      for x, a in zip(ret, axis):
        if a != 0:
          extra_dims = [i for i in range(x.shape.ndims) if i != a]
          x = tf.transpose(x, perm=[a] + extra_dims)
        outputs.append(x)
    else:
      outputs = ret
    # ====== clean and return ====== #
    return outputs[0] if single_output else outputs

def rnn_decorator(*args, **kwargs):
  """Wraps any method (or function) to allow its iterative application.

  The decorator allows you to implement step_function and assign sequences
  arguments to the function in very flexible way.

  The idea behind this function is characterizing recursive function by
  3 primitive information:
   * `sequences`: the sequences to iterative over
   (i.e nb_samples, nb_time, nb_features)
   * `states`: describe output information (i.e. the initial value of
   output after each timestep)

  In the decorator, you are allowed to provide the `name` (in string) of
  above variables, the process of looking for these name are following:
   * If your `call-able` is a method (i.e bound to an object), then the
   variables will be searched in the attributes of the object.
   * If your `call-able` is a function (i.e the first argument is not an
   object but variable), then you have to specified all the information
   when you call the function.

  Parameters
  ----------
  sequences : list of strs
      Specifies which of the arguments are elements of input sequences.
      (batch_size, nb_time_step, trailing_dims)
  states : list of strs
      Specifies which of the arguments are the states.

  Sub-Parameters
  --------------
  iterate : bool
      If ``True`` iteration through whole sequence is made.
      By default ``True`` (i.e. False <=> stateful recurrent network)
  backwards : bool
      If ``True``, the sequences are processed in backward
      direction. ``False`` by default.
  n_steps: int
      number of timestep, required if not known in advance (i.e. the
      second dimension of sequences)
  batch_size: int
      batch size of input batch (i.e. the first dimension of sequences)
  repeat_states: bool
      repeat the states first dimension to match the batch_size
  name: str
      name for the scan operator

  Returns
  -------
  recurrent_apply : The new method that applies the RNN to sequences.

  Note
  --------
  sub-parameters is the addition parameters that the step funciton will
  accept
  The arguments inputed directly to the function will override the arguments
  in container object

  Example
  -------
  """
  #####################################
  # 0. Helper functions.
  def to_list(x):
    return [] if x is None else ([x] if not isinstance(x, (tuple, list))
                                 else list(x))

  def find_arg(name, type, container, kwargs):
    # if given name not found, return None
    if not isinstance(name, str):
      raise ValueError('Given sequences, states, contexts must be '
                       'string represent the name of variable in the '
                       'input arguments of step function or attributes '
                       'of container class, name="%s"' % str(name))
    # given name as string
    value = None
    if name in kwargs:
      value = kwargs[name]
    # if the variable is None, find it in the container
    if value is None:
      value = getattr(container, name, None)
    return value

  #####################################
  # 1. Getting all arguments.
  # Decorator can be used with or without arguments
  if len(args) > 1:
    raise Exception('You can use this "recurrent" function in 2 ways: \n'
                    ' - input the step_function directly to *arg, and '
                    'specify other parameters in **kwargs.\n'
                    ' - use this as a decorator, and only need to specify '
                    'the parameters in **kwargs.\n')
  sequences = to_list(kwargs.pop('sequences', []))
  states = to_list(kwargs.pop('states', []))
  if builtins.any(not isinstance(i, str) for i in sequences + states):
    raise Exception('"sequences", "contexts", and "states" must be '
                    'string, which specify the name of variable in '
                    'the container or in arguments of step_function.')

  #####################################
  # 2. Create wrapper.
  def recurrent_wrapper(step_function):
    arg_spec = inspect.signature(step_function)
    arg_names = []
    defaults_args = {}
    # all defaults arguments
    for n, p in arg_spec.parameters:
      if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD):
        continue
      arg_names.append(n)
      if p.default != inspect.Parameter.empty:
        defaults_args[n] = p.default
    nb_required_args = len(arg_names) - len(defaults_args)

    @wraps(step_function)
    def recurrent_apply(*args, **kwargs):
      """ Iterates a transition function. """
      # Extract arguments related to iteration and immediately relay the
      # call to the wrapped function if `iterate=False`
      iterate = kwargs.pop('iterate', True)
      # ====== not iterate mode, just return step_function ====== #
      if not iterate:
        return step_function(*args, **kwargs)
      # otherwise, continue, container is the object store all
      # necessary variables
      if is_tensor(args[0]) or len(args) == 0:
        container = None
      else:
        container = args[0]
      # ====== additional parameters ====== #
      backwards = kwargs.pop('backwards', False)
      n_steps = kwargs.pop('n_steps', None)
      batch_size = kwargs.pop('batch_size', None)
      repeat_states = kwargs.pop('repeat_states', False)
      name = kwargs.pop('name', None)
      # ====== Update the positional arguments ====== #
      step_args = dict(defaults_args)
      step_args.update(kwargs)
      # key -> positional_args
      for key, value in zip(arg_spec.args, args):
        step_args[key] = value
      # ====== looking for all variables ====== #
      sequences_given = [find_arg(i, 'sequences', container, step_args)
                         for i in sequences]
      states_given = [find_arg(i, 'states', container, step_args)
                      for i in states]
      # check all is variables
      if builtins.any(not is_tensor(i) and i is not None
             for i in sequences_given + states_given):
        raise ValueError('All variables provided to sequences, '
                         'contexts, or states must be Variables.'
                         'sequences:%s states:%s' %
                         (str(sequences_given), str(states_given)))
      # ====== configuraiton for iterations ====== #
      # Assumes time dimension is the second dimension
      shape = sequences_given[0].shape
      if n_steps is None:
        n_steps = shape[1]
      if batch_size is None:
        batch_size = shape[0]
      # ====== Ensure all initial states are with the right shape.
      _ = []
      for key, init_val in zip(states, states_given):
        shape = None if init_val is None else init_val.shape.as_list()
        # only one vector given for 1 batch matrix, should be repeated
        if init_val is not None and \
        (init_val.shape.ndims == 1 or shape[0] == 1):
          if repeat_states:
            init_val = (tf.expand_dims(init_val, axis=0)
                        if init_val.shape.ndims == 1 else init_val)
            init_val = repeat(init_val, batch_size, axes=0)
          else:
            print('[WARNING] The "states" should be initialized for all '
                  'samples in 1 batch (i.e. the first dimension, '
                  'should be equal to the batch_size, you can '
                  'repeat the first dimension of "%s"' % key)
        _.append(init_val)
      states_given = list(_)
      # ====== shuffle sequences variable to get time dimension first
      sequences_given = [
          dimshuffle(i, (1, 0) + tuple(range(2, i.shape.ndims)))
          if i is not None else i for i in sequences_given]
      # ====== create steps functions ====== #
      arg_order = ([i for i, j in zip(sequences, sequences_given)
                    if j is not None] +
                   [i for i, j in zip(states, states_given)
                    if j is not None])

      def scan_function(*args):
        # step args contains all kwargs for step function
        step_args.update(zip(arg_order, args))
        # kwargs = dict(step_args)
        kwargs = {i: j for i, j in step_args.items()
                  if i in arg_names}
        # check get all necessary parametesr for step fucntion
        if len(kwargs) < nb_required_args:
          raise Exception('Step function require %d arguments, but '
                          'only %d arguments given by Scan operator'
                          '.' % (len(arg_names), len(kwargs)))
        # Call step_function
        outputs = step_function(**kwargs)
        # check valid number of return
        if not isinstance(outputs, (tuple, list)):
          outputs = (outputs,)
        if len(outputs) != len(states):
          raise Exception('Given %d initial states but the step '
                          'function only return %d outputs'
                          '.' % (len(states), len(outputs)))
        return outputs
      # ====== run the scan function ====== #
      # print('Sequences:', sequences_given)
      # print('States:', states_given)
      # print('Gobackward:', backwards)
      # print('NSteps:', n_steps)
      # print('BatchSize:', batch_size)
      # print('Repeat:', repeat_states)
      # print('Name:', name)
      results = scan_tensors(scan_function,
                     sequences=[i for i in sequences_given if i is not None],
                     output_info=states_given,
                     n_steps=n_steps,
                     backwards=backwards,
                     name=name)
      # all the result in form (nb_time, nb_samples, trailing_dims)
      # we reshape them back to same as input
      results = [dimshuffle(i, [1, 0] + range(2, i.shape.ndims))
                 for i in to_list(results)]
      # Lasagne+blocks: if scan is backward reverse the output
      # but keras don't do this step (need to validate the performance)
      if backwards:
        results = [r[:, ::-1] for r in results]
      return results
    return recurrent_apply
  # NO arguments are passed, just decorator
  if args:
    step_function, = args
    return recurrent_wrapper(step_function)
  # other arguments are passes
  else:
    return recurrent_wrapper

# ===========================================================================
# CudnnRNN
# ===========================================================================
def params_to_cudnn(weights, biases, name=None):
  assert len(weights) == len(biases), \
  "number of weights is different from number of biases"
  with tf.name_scope(name, "ParamsToCUDNN", values=weights + biases):
    weights = list([tf.reshape(w, shape=(-1,)) for w in weights]) + list(biases)
    return tf.concat(values=weights, axis=0, name="OpaqueParams")

def sort_cudnn_params(weights, biases, rnn_mode):
  """ Sort the order of weights and biases according to
  `init_rnn` to be group into CudnnRNN params vector
  """
  def fn_key(v):
    name = v.name.split('/')[-1].split(':')[0]
    wtype = name.split('_')[0]
    gate = name.split('_')[1]
    layer = int(name.split('_')[-1]) * 1000
    # ====== weight type ====== #
    if 'W' == wtype or 'bw' == wtype:
      n1 = 100
    elif 'R' == wtype or 'br' == wtype:
      n1 = 200
    # ====== gate type ====== #
    if 'lstm' == rnn_mode:
      if gate == 'i':
        n2 = 10
      elif gate == 'f':
        n2 = 20
      elif gate == 'c':
        n2 = 30
      elif gate == 'o':
        n2 = 40
    elif 'gru' == rnn_mode:
      if gate == 'r':
        n2 = 10
      elif gate == 'i':
        n2 = 20
      elif gate == 'h':
        n2 = 30
    elif 'rnn_' in rnn_mode:
      n2 = 10
    else:
      n2 = int(gate[1:])
    return layer + n1 + n2
  weights = sorted(weights, key=fn_key)
  biases = sorted(biases, key=fn_key)
  return weights, biases

def _validate_number_of_params(params, shapes, initializer, N,
                               name, gate_name, roles):
  # params: [layer1_params_forward, layer1_params_backward,
  #          layer2_params_forward, ...]
  # layer1_params: [W_i, W_f, W_c, W_h, ...]
  num_gates = len(gate_name)
  try:
    if not isinstance(params, (tuple, list)):
      params = (params,)
    params = as_tuple(params, N=N)
    params = [as_tuple(layer, N=num_gates)
              for layer in params]
  except Exception:
    raise RuntimeError("Parameter name: '%s'. Expected: %d, but given: %d"
      % (str(name), int(N), len(params)))
  # exchange None for default initializer
  params = [[initializer if p is None else p
             for p in layer]
            for layer in params]
  # create the variable
  params = [[variable(value=p, shape=s, name=name + '%s_%d' % (n, i), roles=roles)
             for n, p in zip(gate_name, layer)]
            for i, (layer, s) in enumerate(zip(params, shapes))]
  return params

def _input_connection_shape(input_dim, hidden_dim, num_layers,
                            bidirectional, skip_input):
  shapes = []
  for i, l in enumerate(range(num_layers)):
    if i == 0: # first layers
      s = (hidden_dim, input_dim) if not skip_input else (0, 0)
    else: # other layers
      s = (hidden_dim, hidden_dim * 2) if bidirectional else (hidden_dim, hidden_dim)
    if bidirectional:
      shapes.append(s)
    shapes.append(s)
  return shapes

def init_rnn(input_dim, hidden_dim, num_layers=1, num_gates=1,
             W_init=init_ops.glorot_uniform_initializer(seed=randint()),
             b_init=init_ops.constant_initializer(value=0),
             W=None, b_w=None, R=None, b_r=None,
             skip_input=False, is_bidirectional=False,
             cudnn_vector=False, name=None):
  """ Fast initalize all Standard RNN weights

  Parameters
  ----------
  W_init:
    pass
  b_init:
    pass
  W:
    pass
  b_w:
    pass
  R:
    pass
  b_r:
    pass
  num_gates: {int, 'lstm', 'rnn_tanh', 'rnn_relu', 'gru'}
      'rnn_*': only has one gate
      'lstm': has four gates input, forget, cell, output
      'gru': has three gates reset, input, hidden
      if integer is given, custom number of gates is used
  cudnn_vector: bool
      if True, all the weights are flatten and concatenated into 1 big vector
  bidirectional: bool
      if True, return parameters for both forward and backward RNN

  Return
  ------
  [W_i, b_wi, R_h, b_wh]

  Note
  ----
  CudnnRNN parameters vector is ordered by
  weights: [W_gate1_forward, W_gate2_forward, ..., R_gate1_forward, R_gate2_forward, ..., # layer1
            W_gate1_backward, W_gate2_backward, ..., R_gate1_backward, R_gate2_backward, ...,
            ...] # layer2
  biases: [b_Wgate1_forward, b_Wgate2_forward, ..., b_Rgate1_forward, b_Rgate2_forward, ..., # layer1
           b_Wgate1_backward, b_Wgate2_backward, ..., b_Rgate1_backward, b_Rgate2_backward, ...,
           ...] # layer2
  """
  # ====== estimate number of weights and biases ====== #
  num_dirs = 2 if is_bidirectional else 1
  total_num_layers = num_dirs * num_layers
  if is_number(num_gates):
    assert num_gates > 0
    gate_name = ['g%d' % (i + 1) for i in range(num_gates)]
    default_name = 'RNN%d' % num_gates
  elif num_gates == 'rnn_relu' or num_gates == 'rnn_tanh':
    gate_name = ['i']
    default_name = 'RNNrelu' if num_gates == 'rnn_relu' else 'RNNtanh'
  elif num_gates == 'lstm':
    gate_name = ['i', 'f', 'c', 'o']
    default_name = 'LSTM'
  elif num_gates == 'gru':
    gate_name = ['r', 'i', 'h']
    default_name = 'GRU'
  else:
    raise ValueError("No support for `num_gates`=%s" % str(num_gates))
  num_gates = len(gate_name)
  num_weights = total_num_layers * num_gates * 2
  num_biases = total_num_layers * num_gates * 2
  weights = W_init if isinstance(W_init, (tuple, list)) else []
  biases = b_init if isinstance(b_init, (tuple, list)) else []
  # ====== initialize the Variables ====== #
  with tf.name_scope(name, "Initialize%s" % default_name):
    # initialize weights
    if len(weights) == 0:
      W = _validate_number_of_params(W,
          shapes=_input_connection_shape(input_dim, hidden_dim, num_layers,
                                         is_bidirectional, skip_input),
          initializer=W_init, N=total_num_layers,
          name='W_', gate_name=gate_name, roles=role.Weight)
      R = _validate_number_of_params(R,
          shapes=[(hidden_dim, hidden_dim) for i in range(total_num_layers)],
          initializer=W_init, N=total_num_layers,
          name='R_', gate_name=gate_name, roles=role.Weight)
      for w, r in zip(W, R):
        weights += w
        weights += r
    else:
      if len(weights) != num_weights:
        raise ValueError("Expected %d weights, given %d" %
          (num_weights, len(weights)))
      assert all(is_variable(w) for w in weights), "All weights must be Variables"
    # initialize biases
    if len(biases) == 0:
      b_w = _validate_number_of_params(b_w,
          shapes=[(hidden_dim,) for i in range(total_num_layers)],
          initializer=b_init, N=total_num_layers,
          name='bw_', gate_name=gate_name, roles=role.Bias)
      b_r = _validate_number_of_params(b_r,
          shapes=[(hidden_dim,) for i in range(total_num_layers)],
          initializer=b_init, N=total_num_layers,
          name='br_', gate_name=gate_name, roles=role.Bias)
      for bw, br in zip(b_w, b_r):
        biases += bw
        biases += br
    else:
      if len(biases) != num_biases:
        raise ValueError("Expected %d biases, given %d" %
          (num_biases, len(biases)))
      assert all(is_variable(b) for b in biases), "All biases must be Variables"
    # ====== organize into single list ====== #
    assert len(weights) == num_weights
    assert len(biases) == num_biases
    if cudnn_vector:
      return params_to_cudnn(weights, biases)
    return weights, biases

def cudnn_rnn(X, num_units, rnn_mode,
              num_layers=1, parameters=None,
              h0=None, c0=None,
              skip_input=False, is_bidirectional=False,
              dropout=0., is_training=None,
              name=None):
  """CuDNN v7 RNN implementation.

  Parameters
  ----------
  X : input varialbe or placeholder
      shape=(batch_size, timesteps, input_dims)
  num_units : int
      the number of units within the RNN model.
  rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
      See cudnn documentation for ``cudnnRNNMode_t``.
  num_layers : int
      the number of layers for the RNN model.
  h0: tensor
      h0 with shape [num_layers, batch_size, hidden_size]
  c0: tensor
      c0 (lstm) with shape [num_layers, batch_size, hidden_size]
  parameters: vector
      vector contain all flatten weights and bias
      check `backend.init.lstm`, `backend.init.gru`, and `backend.init.rnn`
      for more information
  skip_input : bool (default: False)
      if False, `linear_input`: input will be multiplied by a bias matrix
      if True, `skip_input`: No operation is performed on the input.
      The size must match the hidden size (`num_units`).
      (CuDNN docs: cudnnRNNInputMode_t)
  is_bidirectional : {'unidirectional', 'bidirectional'}
      if True, bidirectional: The network operates from first to last then from last
      to first and concatenates the results at each layer.
      otherwise, unidirectional: The network operates recurrently from the
      first input to the last.
  dropout: float (0.0-1.0)
      whether to enable dropout. With it is 0, dropout is disabled.
  is_training : {None, boolean}
      if None, is_training is conditioned on `odin.backend.is_training()`

  Returns
  -------
  [output, hidden_states, cell_states] for lstm
  [output, hidden_states] for gru and rnn

  output_shape: (batch_size, timesteps, hidden_size)
  hidden_shape: (num_layers, batch_size, hidden_size)
  cell_shape: (num_layers, batch_size, hidden_size)

  Note
  ----
  dropout is turn off if K.set_training(False) or K.is_training() == False

  """
  if get_ngpu() == 0:
    raise Exception('This opt is not supported with CPU.')
  # ====== Check arguments ====== #
  rnn_mode = str(rnn_mode)
  if rnn_mode not in ('rnn_relu', 'rnn_tanh', 'lstm', 'gru'):
    raise ValueError("rnn_mode=%s must be: 'rnn_relu', 'rnn_tanh', 'lstm', 'gru'"
                     % rnn_mode)
  input_mode = 'skip_input' if skip_input else 'linear_input'
  direction = 'bidirectional' if is_bidirectional else 'unidirectional'

  # ====== helper function ====== #
  def check_init_states(s0, nb_layers, batch_size):
    if s0 is None: return None
    if is_number(s0):
      s0 = tf.fill(dims=(nb_layers, batch_size, num_units), value=s0)
    if s0.shape.ndims < 3:
      s0 = tf.expand_dims(s0, dim=0)
    s0shape = s0.shape.as_list()
    if s0shape[0] == 1 and s0shape[0] != nb_layers:
      s0 = repeat(s0, n=nb_layers, axes=0)
    if s0shape[1] == 1:
      s0 = repeat(s0, n=batch_size, axes=1)
    if s0.dtype != X.dtype:
      s0 = tf.cast(s0, X.dtype)
    return s0
  # ====== create RNNBlock ====== #
  from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
  input_shape = X.shape.as_list()
  if X.shape.ndims != 3:
    raise ValueError('Input must be 3-D tensor, but X is %d-D tensor' % X.ndim)
  if input_shape[-1] != num_units and skip_input:
    raise ValueError('In skip_input mode, input size must be equal to hidden size'
                     ', but input_size=%d != hidden_size=%d' %
                     (input_shape[-1], num_units))
  # IF we dimshuffle here, a lot of error concern GPUarray,
  # and cudnn will happen
  batch_size = tf.shape(X)[0] # native shape
  input_size = input_shape[-1]
  with tf.device('/cpu:0'):
    num_params = cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
        rnn_mode=rnn_mode, num_layers=num_layers, num_units=num_units,
        input_size=input_size, input_mode=input_mode, direction=direction)
    num_params = num_params.eval(session=get_session())
  # layer info (note in case of bidirectional, output from previous
  # layers are concatenated).
  with tf.name_scope(name, "CudnnRNN"):
    # ====== create parameters ====== #
    # check parameters
    if parameters is None:
      weights, biases = init_rnn(input_dim=input_size, hidden_dim=num_units,
                                 num_layers=num_layers, num_gates=rnn_mode,
                                 skip_input=skip_input,
                                 is_bidirectional=is_bidirectional,
                                 cudnn_vector=False)
      parameters = params_to_cudnn(weights, biases)
    assert num_params == parameters.shape.as_list()[0], \
        "Require %d parameters but %d provided" % \
        (num_params, parameters.shape[0])
    # check initial states
    total_num_layers = num_layers * 2 if is_bidirectional else num_layers
    h0 = tf.zeros(shape=(total_num_layers, batch_size, num_units),
                  dtype=floatX, name='h0') if h0 is None else h0
    h0 = check_init_states(h0, total_num_layers, batch_size)
    args = {'input_h': h0}
    # check initial memory
    if rnn_mode == 'lstm':
      c0 = (tf.zeros(shape=(total_num_layers, batch_size, num_units),
                     dtype=floatX, name='c0')
            if rnn_mode == 'lstm' and c0 is None else c0)
      c0 = check_init_states(c0, total_num_layers, batch_size)
      args['input_c'] = c0
    else:
      c0 = tf.constant([], dtype=floatX)
    # ====== get output ====== #
    kwargs = {
        'inputs': tf.transpose(X, (1, 0, 2)), # [?, batch_size, input_size]
        'input_h': h0, # [num_layers, batch_size, num_units]
        'input_c': c0, # same shape as input_h
        'params': parameters,
        'rnn_mode': rnn_mode,
        'input_mode': input_mode,
        'direction': direction,
        'dropout': dropout,
        'seed': get_random_state().randint(low=0, high=5218, dtype='int32'),
        'name': name
    }
    if is_training is None:
      (output, output_h, output_c) = cond_training(
          train_fn=lambda: cudnn_rnn_ops._cudnn_rnn(is_training=True, **kwargs),
          infer_fn=lambda: cudnn_rnn_ops._cudnn_rnn(is_training=False, **kwargs))
    else:
      (output, output_h, output_c) = cudnn_rnn_ops._cudnn_rnn(
          is_training=bool(is_training), **kwargs)
    # ====== post processing ====== #
    output = set_shape(tf.transpose(output, (1, 0, 2)),
      shape=(input_shape[0], input_shape[1], num_units * (2 if is_bidirectional else 1)))
    output_h = set_shape(output_h,
      shape=(total_num_layers, input_shape[0], num_units))
    rets = [output, output_h]
    if rnn_mode == 'lstm':
      output_c = set_shape(output_c,
        shape=(total_num_layers, input_shape[0], num_units))
      rets.append(output_c)
  return tuple(rets)
