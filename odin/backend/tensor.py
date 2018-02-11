from __future__ import print_function, division, absolute_import

import copy
import numbers
import inspect
from functools import wraps
from six.moves import builtins
from collections import defaultdict

import numpy as np

import tensorflow as tf
from odin.config import get_session, get_ngpu, get_floatX
from odin.utils import as_tuple, uuid, is_number
from .helpers import set_shape, is_tensor, is_training

floatX = get_floatX()
EPS = np.finfo(floatX).eps

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
# Conversion
# ===========================================================================
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
      x /= tf.reduce_sum(x, axis=-1, keep_dims=True)
      x = tf.clip_by_value(x, EPS, 1 - EPS)
      return tf.log(x)

def to_llr(x, name=None):
  ''' Convert a matrix of probabilities into log-likelihood ratio
  :math:`LLR = log(\\frac{prob(data|target)}{prob(data|non-target)})`
  '''
  # ====== numpy ====== #
  if not is_tensor(x):
    x /= np.sum(x, axis=-1, keepdims=True)

    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))
  # ====== tensorflow ====== #
  else:
    with tf.name_scope(name, "log_likelihood_ratio", [x]):
      x /= tf.reduce_sum(x, axis=-1, keep_dims=True)
      x = tf.clip_by_value(x, EPS, 1 - EPS)
      return tf.log(x / (tf.cast(1., x.dtype.base_dtype) - x))

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
    ndim = len(indices.get_shape())
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
      axes is [1, 2] the width and height of an `channel last` image.
  method: str, int
      'nn' for nearest neighbor (e.g. [1, 2] => [1, 1, 2, 2]),
      'pad' for padding within the tensor. 'pad_margin' do padding
      in the margin of the tensor. 'repeat' simple algorithm for
      repeating the element (e.g. [1, 2] => [1, 2, 1, 2])
  """
  with tf.variable_scope(name):
    method = method.lower()
    input_shape = tf.shape(x)
    input_shape_int = x.get_shape().as_list()
    ndims = x.get_shape().ndims
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
  When attempting to multiply a ND tensor
  with a ND tensor, reproduces the Theano behavior

  Example
  -------
   (2, 3).(4, 3, 5) => (2, 4, 5)
   (2, 3, 4).(4, 5) => (2, 3, 5)
  """
  with tf.variable_scope(name):
    shapeX = x.get_shape().as_list()
    shapeY = y.get_shape().as_list()
    ndimX = x.get_shape().ndims
    ndimY = y.get_shape().ndims
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
    output_shape = [-1 if i is None else i
                    for i in shapeX[:-1] + outshapeY]
    output = tf.reshape(tf.matmul(x, y), output_shape)
  return output

def batched_dot(x, y, name='BatchedDot'):
  """Batchwise dot product.
  This function computes the dot product between the two tensors,
  by iterating over the first dimension.
  """
  with tf.variable_scope(name):
    shapeX = x.get_shape().as_list()
    shapeY = y.get_shape().as_list()
    ndimX = x.get_shape().ndims
    ndimY = y.get_shape().ndims
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
    x_shape = copy.copy(then_expression.get_shape())
    # tensorflow require the last dimension of 3 variables is equal, too
    # it is irrelevant since condition can have shape[-1] = 1
    cond_ndims = condition.get_shape().ndims
    if cond_ndims > 1 and condition.get_shape()[-1] != x_shape[-1]:
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
    k = x.get_shape()[-1]
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
  """ x.shape = [25, 08, 12]
  reshape(shape=([1], [2], [0]))
  => x.shape = (08, 12, 25)
  """
  input_shape = x.get_shape().as_list()
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
  with tf.variable_scope(name):
    if outdim == 1:
      return tf.reshape(x, [-1], name=name)
    input_shape = x.get_shape().as_list()
    other_shape = tuple([input_shape[i] for i in range(outdim - 1)])
    n = np.prod(input_shape[(outdim - 1):])
    output_shape = [-1 if i is None else i
                    for i in other_shape + (n,)]
    return tf.reshape(x, output_shape)


def repeat(x, n, axes=None, name="Repeat"):
  """Repeat a N-D tensor.

  If x has shape (s1, s2, s3) and axes=(1, -1), the output
  will have shape (s1, s2 * n[0], s3 * n[1]).
  """
  if axes is not None:
    ndim = x.get_shape().ndims
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
    axes = _normalize_axis(axes, x.get_shape().ndims)
    x = tf.cast(x, floatX)
    m = tf.reduce_mean(x, axis=axes, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axes, keep_dims=keepdims)


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
    l2norm = tf.sqrt(tf.reduce_sum(X ** 2, axis=axis, keep_dims=True))
    X_rms = l2norm / D
    X_rms = tf.where(tf.equal(X_rms, 0.),
                     x=tf.ones_like(X_rms, dtype=X_rms.dtype.base_dtype),
                     y=X_rms)
    return target_rms * X / X_rms


# ===========================================================================
# RNN and loop
# ===========================================================================
def Scan(fn,
         sequences=None,
         outputs_info=None,
         n_steps=None,
         backwards=False,
         name=None):
  """
  Note
  ----
  backwards mode only invert sequences then iterate over them
  """
  # ====== check sequences ====== #
  if sequences is None:
    sequences = []
  elif not isinstance(sequences, (tuple, list)):
    sequences = [sequences]
  if backwards:
    sequences = [tf.reverse(seq, axis=(0,)) for seq in sequences]
  if isinstance(n_steps, numbers.Number):
    sequences = [seq[:n_steps] for seq in sequences]
  # ====== check output info ====== #
  if outputs_info is None:
    outputs_info = []
  elif not isinstance(outputs_info, (tuple, list)):
    outputs_info = [outputs_info]
  else:
    outputs_info = list(outputs_info)
  nb_outputs = len(outputs_info)

  # ====== modified step function ====== #
  def step_(outputs, inputs):
    inputs = inputs + outputs
    outputs = fn(*inputs)
    if not isinstance(outputs, (tuple, list)):
      outputs = [outputs]
    else:
      outputs = list(outputs)
    return outputs
  outputs = tf.scan(step_,
              elems=sequences,
              initializer=outputs_info,
              parallel_iterations=32, back_prop=is_training(),
              swap_memory=False, infer_shape=True,
              name=name)
  # consistent return as theano
  if nb_outputs == 1:
    outputs = outputs[0]
  return outputs


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
    arg_spec = inspect.getargspec(step_function)
    arg_names = arg_spec.args
    # all defaults arguments
    if arg_spec.defaults is not None:
      defaults_args = dict(zip(
          reversed(arg_spec.args),
          reversed(arg_spec.defaults)
      ))
    else:
      defaults_args = dict()
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
      shape = sequences_given[0].get_shape()
      if n_steps is None:
        n_steps = shape[1]
      if batch_size is None:
        batch_size = shape[0]
      # ====== Ensure all initial states are with the right shape.
      _ = []
      for key, init_val in zip(states, states_given):
        shape = None if init_val is None else init_val.get_shape().as_list()
        # only one vector given for 1 batch matrix, should be repeated
        if init_val is not None and \
        (init_val.get_shape().ndims == 1 or shape[0] == 1):
          if repeat_states:
            init_val = (tf.expand_dims(init_val, axis=0)
                        if init_val.get_shape().ndims == 1 else init_val)
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
          dimshuffle(i, (1, 0) + tuple(range(2, i.get_shape().ndims)))
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
      results = Scan(scan_function,
                     sequences=[i for i in sequences_given if i is not None],
                     outputs_info=states_given,
                     n_steps=n_steps,
                     backwards=backwards,
                     name=name)
      # all the result in form (nb_time, nb_samples, trailing_dims)
      # we reshape them back to same as input
      results = [dimshuffle(i, [1, 0] + range(2, i.get_shape().ndims))
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


def rnn_dnn(X, hidden_size, rnn_mode,
            num_layers=1,
            parameters=None,
            h0=None, c0=None,
            input_mode='linear',
            direction_mode='unidirectional',
            dropout=0., name=None):
  """CuDNN v5 RNN implementation.

  Parameters
  ----------
  X : input varialbe or placeholder
      shape=(batch_size, timesteps, input_dims)
  hidden_size : int
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
  input_mode : {'linear', 'skip'}
      linear: input will be multiplied by a biased matrix
      skip: No operation is performed on the input.  The size must
      match the hidden size.
      (CuDNN docs: cudnnRNNInputMode_t)
  direction_mode : {'unidirectional', 'bidirectional'}
      unidirectional: The network operates recurrently from the
                      first input to the last.
      bidirectional: The network operates from first to last then from last
                     to first and concatenates the results at each layer.
  dropout: float (0.0-1.0)
      whether to enable dropout. With it is 0, dropout is disabled.

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
  if name is None: name = uuid()
  # ====== Check arguments ====== #
  if rnn_mode not in ('rnn_relu', 'rnn_tanh', 'lstm', 'gru'):
    raise ValueError("rnn_mode=%s must be: 'rnn_relu', 'rnn_tanh', 'lstm', 'gru'"
                     % rnn_mode)
  if input_mode not in ('linear', 'skip'):
    raise ValueError("input_mode=%s must be: 'linear', 'skip'" % input_mode)
  input_mode = 'linear_input' if input_mode == 'linear' else 'skip_input'
  if direction_mode not in ('unidirectional', 'bidirectional'):
    raise ValueError("direction_mode=%s must be: 'unidirectional', 'bidirectional'"
                     % direction_mode)
  is_bidirectional = direction_mode == 'bidirectional'

  # ====== helper function ====== #
  def check_init_states(s0, nb_layers, batch_size):
    if s0 is None: return None
    if s0.get_shape().ndims < 3:
      s0 = tf.expand_dims(s0, dim=0)
    s0shape = s0.get_shape().as_list()
    if s0shape[0] == 1 and s0shape[0] != nb_layers:
      s0 = repeat(s0, n=nb_layers, axes=0)
    if s0shape[1] == 1:
      s0 = repeat(s0, n=batch_size, axes=1)
    return s0
  # ====== create RNNBlock ====== #
  from tensorflow.contrib import cudnn_rnn
  input_shape = X.get_shape().as_list()
  if X.get_shape().ndims != 3:
    raise ValueError('Input must be 3-D tensor, but X is %d-D tensor' % X.ndim)
  if input_shape[-1] != hidden_size and 'skip' in input_mode:
    raise ValueError('In skip_input mode, input size must be equal to hidden size'
                     ', but input_size=%d != hidden_size=%d' %
                     (input_shape[-1], hidden_size))
  # IF we dimshuffle here, a lot of error concern GPUarray,
  # and cudnn will happen
  batch_size = tf.shape(X)[0] # native shape
  if rnn_mode == 'lstm':
    rnn = cudnn_rnn.CudnnLSTM(num_layers=num_layers,
                              num_units=hidden_size,
                              input_size=input_shape[-1],
                              input_mode=input_mode,
                              direction=direction_mode,
                              dropout=dropout,
                              seed=0)
  else:
    if rnn_mode == 'gru':
      rnn_class = cudnn_rnn.CudnnGRU
    elif rnn_mode == 'rnn_relu':
      rnn_class = cudnn_rnn.CudnnRNNRelu
    elif rnn_mode == 'rnn_tanh':
      rnn_class = cudnn_rnn.CudnnRNNTanh
    rnn = rnn_class(num_layers=num_layers,
                    num_units=hidden_size,
                    input_size=input_shape[-1],
                    input_mode=input_mode,
                    direction=direction_mode,
                    dropout=dropout,
                    seed=0)
  # layer info (note in case of bidirectional, output from previous
  # layers are concatenated).
  layer_info = [input_shape[-1], hidden_size] + \
               [hidden_size * (2 if is_bidirectional else 1),
                hidden_size] * (num_layers - 1)
  with tf.device('/cpu:0'):
    nb_params = rnn.params_size().eval(session=get_session())
  # ====== create parameters ====== #
  # check parameters
  if parameters is None:
    if rnn_mode == 'lstm':
      from odin.backend.init import lstm as init_func
    elif rnn_mode == 'gru':
      from odin.backend.init import gru as init_func
    else:
      from odin.backend.init import rnn as init_func
    parameters = np.concatenate([init_func(layer_info[i * 2], layer_info[i * 2 + 1],
                                 one_vector=True, return_variable=False,
                                 bidirectional=True if is_bidirectional else False)
                                 for i in range(num_layers)]).astype(floatX)
    parameters = tf.Variable(parameters, name=name)
  assert nb_params == parameters.get_shape().as_list()[0], \
      "Require %d parameters but only %d provided" % \
      (nb_params, parameters.get_shape()[0])
  # check initial states
  num_layers = num_layers * 2 if is_bidirectional else num_layers
  h0 = tf.zeros(shape=(num_layers, batch_size, hidden_size), dtype=floatX, name='h0') \
      if h0 is None else h0
  h0 = check_init_states(h0, num_layers, batch_size)
  c0 = (tf.zeros(shape=(num_layers, batch_size, hidden_size), dtype=floatX, name='c0')
        if rnn_mode == 'lstm' and c0 is None else c0)
  c0 = check_init_states(c0, num_layers, batch_size)
  # preprocess arguments
  args = {'input_h': h0}
  if rnn_mode == 'lstm':
    args['input_c'] = c0
  # ====== get output ====== #
  output = rnn(input_data=tf.transpose(X, (1, 0, 2)),
               params=parameters, is_training=bool(is_training()),
               **args)
  output = [tf.transpose(output[0], (1, 0, 2))] + list(output[1:])
  set_shape(output[0], (input_shape[0], input_shape[1],
                        hidden_size * (2 if is_bidirectional else 1)))
  for o in output[1:]:
    set_shape(o, (num_layers, input_shape[0], hidden_size))
  return output
