from __future__ import division, absolute_import, print_function
from __builtin__ import min as min_

import os
import copy
import math
import numbers
import cPickle
from collections import OrderedDict

import numpy as np

import tensorflow as tf

from odin.config import CONFIG, RNG_GENERATOR
from odin.utils import as_tuple, as_shape_tuple, dict_union, uuid
from odin.utils.shape_calculation import (get_conv_output_shape,
                                          get_pool_output_shape)
from odin.basic import (add_role, PARAMETER, ACTIVATION_PARAMETER,
                        add_shape, get_shape, is_training)

from .helpers import (get_session, as_tensor_variable, ComputationGraph,
                      variable)
FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
NPROCESSORS = CONFIG['device_info']['n']
_RNG = np.random.RandomState(seed=RNG_GENERATOR.randint(10e8))


def _normalize_axis(axis, ndim):
    if axis is None:
        return None
    if isinstance(axis, (tuple, list)):
        return tuple([a % ndim if a is not None else a
                for a in axis])
    return axis % ndim


def eval(x):
    '''Evaluates the value of a tensor.
    Returns a Numpy array.
    '''
    return x.eval(session=get_session())


# ===========================================================================
# Basic ops
# ===========================================================================
def backend_ops_relu(x, alpha=0.):
    # Adapted implementation from theano
    if alpha == 0.:
        return tf.nn.relu(x)
    else:
        # We can't use 0.5 and 1 for one and half.  as if alpha is a
        # numpy dtype, they will be considered as float64, so would
        # cause upcast to float64.
        alpha = as_tensor_variable(alpha, dtype=x.dtype.base_dtype)
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * tf.abs(x)


def backend_ops_elu(x, alpha):
    res = tf.nn.elu(x)
    if alpha != 1:
        res = tf.select(x > 0, res, alpha * res)
    return res


def backend_ops_hard_sigmoid(x):
    slope = tf.constant(0.2, dtype=x.dtype.base_dtype)
    shift = tf.constant(0.5, dtype=x.dtype.base_dtype)
    x = (x * slope) + shift
    x = tf.clip_by_value(x, 0., 1.)
    return x

backend_ops_softmax = tf.nn.softmax
backend_ops_softplus = tf.nn.softplus
backend_ops_softsign = tf.nn.softsign
backend_ops_sigmoid = tf.nn.sigmoid
backend_ops_tanh = tf.nn.tanh

backend_ops_square = tf.square
backend_ops_abs = tf.abs
backend_ops_sign = tf.sign
backend_ops_inv = tf.inv
backend_ops_sqrt = tf.sqrt
backend_ops_exp = tf.exp
backend_ops_log = tf.log
backend_ops_round = tf.round
backend_ops_pow = tf.pow
backend_ops_clip = tf.clip_by_value

backend_ops_diag = tf.diag_part

# backend_ops_categorical_crossentropy = tf.nn.softmax_cross_entropy_with_logits
backend_ops_categorical_crossentropy = \
    lambda x, y: - tf.reduce_sum(y * tf.log(x),
                                 reduction_indices=x.get_shape().ndims - 1)
backend_ops_binary_crossentropy = \
    lambda x, y: tf.nn.sigmoid_cross_entropy_with_logits(tf.log(x / (1. - x)), y)


def backend_ops_eye(n, m, dtype):
    x = tf.Variable(initial_value=np.eye(n, m, dtype=dtype), dtype=dtype)
    get_session().run(x.initializer)
    return x


# Comparator
backend_ops_eq = tf.equal
backend_ops_neq = tf.not_equal
backend_ops_gt = tf.greater
backend_ops_ge = tf.greater_equal
backend_ops_lt = tf.less
backend_ops_le = tf.less_equal


def switch(condition, then_expression, else_expression):
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
    x = tf.select(condition, then_expression, else_expression)
    x.set_shape(x_shape)
    return x


# ===========================================================================
# Shape operator
# ===========================================================================
def broadcastable(x):
    return x


def addbroadcast(x, *axes):
    return x


# ===========================================================================
# Predefined data
# ===========================================================================
def zeros(shape, dtype=FLOATX, name=None):
    """Instantiate an all-zeros variable.
    """
    x = tf.zeros(shape, dtype=dtype, name=name)
    return x


def ones(shape, dtype=FLOATX, name=None):
    """Instantiate an all-ones variable.
    """
    x = tf.ones(shape, dtype=dtype, name=name)
    return x


def ones_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype.base_dtype
    x = tf.ones_like(x, dtype=dtype, optimize=True)
    return x


def zeros_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype.base_dtype
    x = tf.zeros_like(x, dtype=dtype, optimize=True)
    return x


def cast(x, dtype):
    if 'tensorflow.' in str(x.__class__):
        return tf.cast(x, dtype)
    if hasattr(dtype, 'as_numpy_dtype'):
        dtype = dtype.as_numpy_dtype()
    return np.cast[dtype](x)


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    '''Multiplies 2 tensors.
    When attempting to multiply a ND tensor
    with a ND tensor, reproduces the Theano behavior
    e.g.
    (2, 3).(4, 3, 5) = (2, 4, 5)
    (2, 3, 4).(4, 5) = (2, 3, 5)
    '''
    shapeX = get_shape(x)
    shapeY = get_shape(y)
    ndimX = x.get_shape().ndims
    ndimY = y.get_shape().ndims
    if ndimX > 2:
        x = tf.reshape(x, (-1, shapeX[-1]))
    if ndimY > 2:
        y_dims = list(range(ndimY))
        y_dims = [y_dims.pop(-2)] + y_dims
        y = tf.transpose(y, perm=y_dims)
        y = tf.reshape(y, (shapeY[-2], -1))
        outshapeY = tuple([shapeY[i] for i in y_dims[1:]])
    else:
        outshapeY = (shapeY[-1],)
    # calculate dot product and desire shape
    output_shape = [-1 if i is None else i
                    for i in shapeX[:-1] + outshapeY]
    output = tf.reshape(tf.matmul(x, y), output_shape)
    return output


def batched_dot(x, y):
    """Batchwise dot product.
    This function computes the dot product between the two tensors,
    by iterating over the first dimension.
    """
    shapeX = get_shape(x)
    shapeY = get_shape(y)
    ndimX = x.get_shape().ndims
    ndimY = y.get_shape().ndims
    # same as dot but one more batch dimension
    if ndimX > 2 + 1:
        x = tf.reshape(x, (-1, np.prod(shapeX[1:-1]), shapeX[-1]))
    if ndimY > 2 + 1:
        y_dims = list(range(ndimY))
        y_dims = [y_dims.pop(0), y_dims.pop(-2)] + y_dims
        y = tf.transpose(y, perm=y_dims)
        outshapeY = tuple([shapeY[i] for i in y_dims[2:]])
        y = tf.reshape(y, (-1, shapeY[-2], np.prod(outshapeY)))
    else:
        outshapeY = (shapeY[-1],)
    # calculate dot product and desire shape
    output_shape = shapeX[:-1] + outshapeY
    output = tf.reshape(tf.batch_matmul(x, y, adj_x=None, adj_y=None),
                        [i if i is not None else -1 for i in output_shape])
    return output


def transpose(x, axes=None):
    """ Transposes a matrix. """
    return tf.transpose(x, perm=axes)


def gather(reference, indices):
    """Retrieves the vectors of indices `indices`
    in the 2D tensor `reference`.

    # Arguments
        reference: a 2D tensor.
        indices: an int tensor of indices.

    # Returns
        A 3D tensor of same type as `reference`.
    """
    return tf.gather(reference, indices)


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================
def var(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    x = tf.cast(x, FLOATX)
    m = tf.reduce_mean(x, reduction_indices=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared,
                          reduction_indices=axis,
                          keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    dtype = x.dtype.base_dtype
    if 'int' in str(dtype) or 'bool' in str(dtype):
        x = tf.cast(x, FLOATX)
    return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def max(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.
    """
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis.
    """
    axis = _normalize_axis(axis, x.get_shape().ndims)
    return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).
    """
    axis = _normalize_axis(axis, x.get_shape().ndims)
    original_dtype = x.dtype
    x = tf.cast(x, tf.bool)
    x = tf.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)
    return tf.cast(x, original_dtype)


def argmax(x, axis=-1, keepdims=False):
    axis %= x.get_shape().ndims
    x = tf.argmax(x, axis)
    if keepdims:
        x = tf.expand_dims(x, axis)
    return x


def argmin(x, axis=-1, keepdims=False):
    axis %= x.get_shape().ndims
    x = tf.argmin(x, axis)
    if keepdims:
        x = tf.expand_dims(x, axis)
    return x


def arange(start, stop=None, step=1, dtype=None):
    x = tf.range(start, limit=stop, delta=step)
    if dtype is not None:
        x = tf.cast(x, dtype)
    return x


def argsort(x):
    """ The indices in -1 axis will be sorted by the values in
    descending order.
    """
    # the return value contains (values, indices)
    # but we only take the indices
    return tf.nn.top_k(x, k=x.get_shape()[-1], sorted=True)[1]
    # raise NotImplementedError


def argtop_k(x, k=1):
    # top-k accuracy
    return tf.nn.top_k(x, k=k, sorted=True)


# ===========================================================================
# Primitive ops
# ===========================================================================
def add(x, y):
    return tf.add(x, y)


def sub(x, y):
    return tf.sub(x, y)


def mul(x, y):
    return tf.mul(x, y)


def div(x, y):
    return tf.divide(x, y)


def mod(x, y):
    return tf.mod(x, y)


def maximum(x, y):
    return tf.maximum(x, y)


def minimum(x, y):
    return tf.minimum(x, y)


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axes=-1):
    """Apply [::-1] to appropriate axis"""
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    ndim = x.get_shape().ndims
    axes = _normalize_axis(axes, ndim)
    dims = [True if i in axes else False for i in range(ndim)]
    return tf.reverse(x, dims)


def concatenate(tensors, axis=-1):
    axis = _normalize_axis(axis, tensors[0].get_shape().ndims)
    return tf.concat(axis, tensors)


def tile(x, n):
    # TODO: error here
    ndim = x.get_shape().ndims
    return tf.tile(x, [1 for i in range(ndim - 1)] + [n])


def stack(tensors):
    """ (5, 2) and (5, 2) => (2, 5, 2) """
    return tf.pack(tensors)


def expand_dims(x, dim=-1):
    """ Add a 1-sized dimension at index "dim". """
    return tf.expand_dims(x, dim)


def reshape(x, shape):
    """ x.shape = [25, 08, 12]
    reshape(shape=([1], [2], [0]))
    => x.shape = (08, 12, 25)
    """
    input_shape = get_shape(x)
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
    return tf.reshape(x, new_shape)


def dimshuffle(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    x = tf.transpose(x, perm=[i for i in pattern if i != 'x'])
    # insert new dimension
    for i, p in enumerate(pattern):
        if p == 'x':
            x = tf.expand_dims(x, i)
    return x


def flatten(x, outdim=1):
    if outdim == 1:
        return tf.reshape(x, [-1])
    input_shape = x.get_shape().as_list()
    other_shape = tuple([input_shape[i] for i in range(outdim - 1)])
    n = np.prod(input_shape[(outdim - 1):])
    output_shape = [-1 if i is None else i
                    for i in other_shape + (n,)]
    return tf.reshape(x, output_shape)


def repeat(x, n, axes=None):
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
        return tf.tile(x, [n[axes.index(i)] if i in axes else 1
                           for i in range(ndim)])
    else:
        return tile(x, n)


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    axis = axis % x.get_shape().ndims
    return tf.squeeze(x, [axis])


def pad(x, axes=1, padding=1):
    """Pad the all dimension given in axes` of a N-D tensor
    with "padding" zeros left and right.

    Example
    -------
    >>> X = [[1, 1, 1],
             [1, 1, 1]]
    >>> Y1 = pad(X, axes=1, padding=1)
    >>> Y1 = [[0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0]]
    >>> Y2 = pad(X, axes=(0, 1), padding=1)
    >>> Y2 = [[0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0]]
    """
    ndim = x.get_shape().ndims
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    axes = tuple([i % ndim for i in axes])
    padding = as_tuple(padding, len(axes), int)
    return tf.pad(x, [[padding[axes.index(i)], padding[axes.index(i)]] if i in axes
                      else [0, 0]
                      for i in range(ndim)])


# ===========================================================================
# Graph manipulation
# ===========================================================================
def gradients(loss, variables, consider_constant=None):
    """
    Return symbolic gradients for one or more variables with respect to some
    cost.

    For more information about how automatic differentiation works in Theano,
    see :mod:`gradient`. For information on how to implement the gradient of
    a certain Op, see :func:`grad`.

    Parameters
    ----------
    cost : scalar (0-dimensional) tensor variable or None
        Value with respect to which we are differentiating.  May be
        `None` if known_grads is provided.
    wrt : variable or list of variables
        term[s] for which we want gradients
    consider_constant : list of expressions(variables)
        expressions not to backpropagate through
    Returns
    -------
    variable or list/tuple of variables (matches `wrt`)
        symbolic expression of gradient of `cost` with respect to each
        of the `wrt` terms.  If an element of `wrt` is not
        differentiable with respect to the output, then a zero
        variable is returned.

    Example
    -------
    >>> # For consider_constant:
    >>> a = T.variable(1.2)
    >>> b = T.variable(1.3)
    >>> x = a * b
    >>>
    >>> y = T.variable(2.)
    >>> z = T.variable(1.)
    >>>
    >>> z_pred = x * y
    >>> loss = T.pow((z - z_pred), 2)
    >>>
    >>> G = T.gradients(loss, [a, b, y], consider_constant=[x])
    >>>
    >>> for g in G:
    >>>     print(g.eval())
    >>> # a_grad=0. b_grad=0. y_grad=6.614
    """
    return tf.gradients(loss, variables, colocate_gradients_with_ops=True)


def stop_gradient(vars):
    return tf.stop_gradient(vars)


def jacobian(loss, variables):
    raise NotImplementedError


def hessian(loss, variables):
    raise NotImplementedError


class Function(object):
    """ Two way to call this Function
    f(x1, x2, x3)
    or f('x1'=x1, 'x2'=x2, 'x3'=x3)
    """

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        # ====== validate input ====== #
        if isinstance(inputs, dict):
            self.inputs_name = inputs.keys()
            inputs = inputs.values()
        elif not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        self.inputs = inputs
        if not hasattr(self, 'inputs_name'):
            self.inputs_name = [i.name for i in self.inputs]
        # ====== validate outputs ====== #
        return_list = True
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
            return_list = False
        self.outputs = list(outputs)
        self.return_list = return_list
        # ====== validate updates ====== #
        if isinstance(updates, dict):
            updates = updates.items()
        updates = updates + ComputationGraph(outputs).updates.items()
        # create updates ops
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if isinstance(update, (tuple, list)):
                    p, new_p = update
                    updates_ops.append(tf.assign(p, new_p))
                else: # assumed already an assign op
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)

    def __call__(self, *inputs, **kwargs):
        # dictionary as inputs
        if len(kwargs) == len(self.inputs_name):
            inputs = [kwargs[i] for i in self.inputs_name]
        # ====== create feed_dict ====== #
        feed_dict = {}
        for tensor, value in zip(self.inputs, inputs):
            feed_dict[tensor] = value
        # ====== run the output ====== #
        session = get_session()
        updated = session.run(self.outputs + [self.updates_op],
                              feed_dict=feed_dict)
        # ====== get the results ====== #
        outputs = updated[:len(self.outputs)]
        if not self.return_list:
            outputs = outputs[0]
        return outputs


# ===========================================================================
# utilities
# ===========================================================================
def one_hot(x, nb_class):
    '''Input: nD integer tensor of shape (batch_size, dim1, dim2, ... dim(n-1))
    Output: (n + 1)D one hot representation of the input
    with shape (batch_size, dim1, dim2, ... dim(n-1), nb_classes)
    '''
    return tf.one_hot(x, depth=nb_class, axis=-1)


def confusion_matrix(y_pred, y_true, labels=None):
    """
    Computes the confusion matrix of given vectors containing
    actual observations and predicted observations.
    Parameters
    ----------
    pred : 1-d or 2-d tensor variable
    actual : 1-d or 2-d tensor variable
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    """
    from tensorflow.contrib.metrics import confusion_matrix
    if y_true.get_shape().ndims == 2:
        y_true = argmax(y_true, -1)
    elif y_true.get_shape().ndims != 1:
        raise ValueError('actual must be 1-d or 2-d tensor variable')

    if y_pred.get_shape().ndims == 2:
        y_pred = argmax(y_pred, -1)
    elif y_pred.get_shape().ndims != 1:
        raise ValueError('pred must be 1-d or 2-d tensor variable')

    return confusion_matrix(y_pred, y_true,
                            num_classes=None if labels is None else len(labels))


def one_hot_max(x, axis=-1):
    """
    Example
    -------
    >>> Input: [[0.0, 0.0, 0.5],
    >>>         [0.0, 0.3, 0.1],
    >>>         [0.6, 0.0, 0.2]]
    >>> Output: [[0.0, 0.0, 1.0],
    >>>         [0.0, 1.0, 0.0],
    >>>         [1.0, 0.0, 0.0]]
    """
    dtype = x.dtype.base_dtype
    return tf.cast(
        tf.equal(tf.cast(arange(x.get_shape()[axis])[None, :], 'int32'),
                 tf.cast(argmax(x, axis=axis, keepdims=True), 'int32')
                ),
        dtype
    )


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
    return tf.mul(x, tf.expand_dims(mask, -1))


# ===========================================================================
# RANDOMNESS
# ===========================================================================
_RNG = np.random.RandomState(seed=CONFIG['seed'])


def set_rng(seed):
    global _RNG
    _RNG = np.random.RandomState(seed=seed)


def random_normal(shape, mean=0.0, std=1.0, dtype=FLOATX):
    return tf.random_normal(shape, mean=mean, stddev=std,
                            dtype=dtype.base_dtype if hasattr(dtype, 'base_dtype') else dtype,
                            seed=_RNG.randint(10e6))


def random_uniform(shape, low=0.0, high=1.0, dtype=FLOATX):
    return tf.random_uniform(shape, minval=low, maxval=high,
                             dtype=dtype.base_dtype if hasattr(dtype, 'base_dtype') else dtype,
                             seed=_RNG.randint(10e6))


def random_binomial(shape, p, dtype=FLOATX, seed=None):
    if hasattr(dtype, 'base_dtype'):
        dtype = dtype.base_dtype
    return tf.select(tf.random_uniform(shape, dtype=dtype, seed=_RNG.randint(10e6)) <= p,
                     tf.ones(shape, dtype=dtype),
                     tf.zeros(shape, dtype=dtype))


# ===========================================================================
# Convolution
# ===========================================================================
def __validate_strides_padding_dilation(strides, border_mode, filter_dilation, ndim):
    if border_mode == 'same' or border_mode == 'valid' or border_mode == 'full':
        border_mode = border_mode.upper()
    elif isinstance(border_mode, (tuple, list, int)):
        border_mode = as_tuple(border_mode, N=ndim, t=int)
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))
    # strides shape
    if strides is None:
        strides = (1,) * ndim
    else:
        strides = as_tuple(strides, N=ndim, t=int)
    # dilation shape
    if filter_dilation is None:
        filter_dilation = (1,) * ndim
    else:
        filter_dilation = as_tuple(filter_dilation, N=ndim, t=int)
    return strides, border_mode, filter_dilation


def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           filter_dilation=(1, 1)):
    """ Dimension is ordered by
    TH input shape: (samples, input_depth, rows, cols)
    TH kernel shape: (depth, input_depth, rows, cols)

    Parameters
    ----------
    border_mode: string
        "same", "valid" or "full".

    Note
    ----
    dim_ordering : tf-tensorflow (defaults), th-theano
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        ---
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
    Only support float32 on CPU

    """
    # store original information for calculating output_shape
    image_shape = get_shape(x)
    kernel_shape = get_shape(kernel)
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=2)
    # convert to TF order
    is_float64 = False
    if 'float64' in x.dtype.name: # only conv in float32
        x = tf.cast(x, 'float32')
        is_float64 = True
    if 'float64' in kernel.dtype.name:
        kernel = tf.cast(kernel, 'float32')

    if filter_dilation == (1, 1):
        x = tf.nn.conv2d(x, kernel, strides=(1,) + strides + (1,),
                         padding=border_mode)
    else:
        assert filter_dilation[0] == filter_dilation[1]
        assert strides == (1, 1), 'Invalid strides for dilated convolution'
        x = tf.nn.atrous_conv2d(x, kernel, filter_dilation[0], padding=border_mode)
    # ====== estimate output shape ====== #
    if is_float64: x = tf.cast(x, 'float64')
    add_shape(x, get_conv_output_shape(image_shape, kernel_shape,
                                       border_mode, strides, filter_dilation))
    return x


def deconv2d(x, kernel, output_shape, strides=(1, 1), border_mode='valid',
             filter_dilation=(1, 1)):
    """
    Note
    ----
    dim_ordering : tf-tensorflow (defaults), th-theano
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        ---
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
    """
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=2)
    x = tf.nn.conv2d_transpose(x, kernel, output_shape, (1,) + strides + (1,),
                               padding=border_mode)
    add_shape(x, output_shape)
    return x


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid',
           filter_dilation=(1, 1, 1)):
    """
    Note
    ----
    dim_ordering : tf-tensorflow (defaults), th-theano
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        ---
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
    """
    volume_shape = get_shape(x)
    kernel_shape = get_shape(kernel)
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=3)
    # no dilation for tensorflow
    if filter_dilation != (1, 1, 1):
        raise Exception("tensorflow has not supported 3D-dilation yet.")
    # convert to TF order
    is_float64 = False
    if 'float64' in x.dtype.name: # only conv in float32
        x = tf.cast(x, 'float32')
        is_float64 = True
    if 'float64' in kernel.dtype.name:
        kernel = tf.cast(kernel, 'float32')
    # ====== estimate output shape ====== #
    if is_float64: x = tf.cast(x, 'float64')
    x = tf.nn.conv3d(x, kernel, (1,) + strides + (1,), border_mode)
    add_shape(x, get_conv_output_shape(volume_shape, kernel_shape,
                                       border_mode, strides, filter_dilation))
    return x


def deconv3d(x, kernel, output_shape, strides=(1, 1, 1), border_mode='valid',
             filter_dilation=(1, 1, 1)):
    """
    Note
    ----
    dim_ordering : tf-tensorflow (defaults), th-theano
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        ---
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
    """
    raise Exception('tensorflow has not supported deconv3d.')


# ===========================================================================
# Pooling
# ===========================================================================
def __validate_pool_stride_border(pool_size, strides, border_mode, mode, ndim):
    # pool_size
    if pool_size is None:
        pool_size = (2,) * ndim
    else:
        pool_size = as_tuple(pool_size, ndim, int)
    # strides
    if strides is None:
        strides = pool_size
    else:
        strides = as_tuple(strides, ndim, int)
    # border_mode
    if border_mode is None or border_mode == 'valid':
        border_mode = 'VALID'
    elif border_mode == 'same':
        border_mode = 'SAME'
    else:
        border_mode = as_tuple(border_mode, ndim, int)
    # pooling mode
    if 'max' in mode.lower():
        mode = 'max'
    elif 'avg' in mode.lower():
        mode = 'avg'
    return pool_size, strides, border_mode, mode


def pool2d(x, pool_size=(2, 2), strides=None, border_mode='valid',
           ignore_border=True, mode='max'):
    """
    Parameters
    ----------
    x : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_size : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    strides : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'avg'}
        Operation executed on each window. `max` or `average`

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """
    input_shape = get_shape(x)
    pool_size, strides, border_mode, mode = __validate_pool_stride_border(
        pool_size, strides, border_mode, mode, ndim=2)
    ndim = len(input_shape) - 3
    # ====== pooling ====== #
    if mode == 'max':
        x = tf.nn.max_pool(x, ksize=(1,) * ndim + pool_size + (1,),
                           strides=(1,) * ndim + strides + (1,),
                           padding=border_mode)
    elif mode == 'avg':
        x = tf.nn.avg_pool(x, ksize=(1,) * ndim + pool_size + (1,),
                           strides=(1,) * ndim + strides + (1,),
                           padding=border_mode)
    output_shape = get_pool_output_shape(input_shape, pool_size,
        ignore_border=ignore_border, strides=strides, pad=border_mode)
    add_shape(x, tuple(output_shape))
    return x


def pool3d(x, pool_size=(2, 2), strides=None, border_mode=(0, 0),
           ignore_border=True, mode='max'):
    """
    Parameters
    ----------
    x : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_size : tuple of length 3
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2,2) will halve the image in each dimension.
    strides : tuple of 3 ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5,5) input with ds=(2,2,2) will generate a (2,2,2) output.
        (3,3,3) otherwise.
    padding : tuple of 3 ints
        (pad_h, pad_w, pad_l), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'avg'}
        Operation executed on each window. `max` or `average`

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """
    input_shape = get_shape(x)
    pool_size, strides, border_mode, mode = __validate_pool_stride_border(
        pool_size, strides, border_mode, mode, ndim=3)
    ndim = len(input_shape) - 4
    # ====== pooling ====== #
    if mode == 'max':
        x = tf.nn.max_pool3d(x, ksize=(1,) * ndim + strides + (1,),
                             strides=(1,) * ndim + pool_size + (1,),
                             padding=border_mode)
    elif mode == 'avg':
        x = tf.nn.avg_pool3d(x, ksize=(1,) * ndim + strides + (1,),
                             strides=(1,) * ndim + pool_size + (1,),
                             padding=border_mode)
    output_shape = get_pool_output_shape(input_shape, pool_size,
        ignore_border=ignore_border, strides=strides, pad=border_mode)
    add_shape(x, tuple(output_shape))
    return x


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
        sequences = [tf.reverse(seq, [True] + [False] * (seq.get_shape().ndims - 1))
                     for seq in sequences]
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
    if CONFIG['device'] == 'cpu':
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
            s0 = expand_dims(s0, dim=0)
        s0shape = get_shape(s0)
        if s0shape[0] == 1 and s0shape[0] != nb_layers:
            s0 = repeat(s0, n=nb_layers, axes=0)
        if s0shape[1] == 1:
            s0 = repeat(s0, n=batch_size, axes=1)
        return s0
    # ====== create RNNBlock ====== #
    from tensorflow.contrib import cudnn_rnn
    input_shape = get_shape(X)
    if X.get_shape().ndims != 3:
        raise ValueError('Input must be 3-D tensor, but X is %d-D tensor' % X.ndim)
    if input_shape[-1] != hidden_size and 'skip' in input_mode:
        raise ValueError('In skip_input mode, input size must be equal to hidden size'
                         ', but input_size=%d != hidden_size=%d' %
                         (input_shape[-1], hidden_size))
    # IF we dimshuffle here, a lot of error concern GPUarray,
    # and cudnn will happen
    batch_size = get_shape(X, native=True)[0]
    if rnn_mode == 'lstm':
        rnn = cudnn_rnn.CudnnLSTM(num_layers=num_layers,
                                  num_units=hidden_size,
                                  input_size=input_shape[-1],
                                  input_mode=input_mode,
                                  direction=direction_mode,
                                  dropout=dropout,
                                  seed=0,
                                  seed2=0)
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
                        seed=0,
                        seed2=0)
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
                                     for i in range(num_layers)]).astype(FLOATX)
        parameters = variable(parameters, name=name)
    assert nb_params == get_shape(parameters)[0], \
        "Require %d parameters but only %d provided" % (nb_params, get_shape(parameters)[0])
    # check initial states
    num_layers = num_layers * 2 if is_bidirectional else num_layers
    h0 = zeros((num_layers, batch_size, hidden_size)) if h0 is None else h0
    h0 = check_init_states(h0, num_layers, batch_size)
    c0 = (zeros((num_layers, batch_size, hidden_size))
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
    add_shape(output[0], (input_shape[0], input_shape[1],
                          hidden_size * (2 if is_bidirectional else 1)))
    for o in output[1:]:
        add_shape(o, (num_layers, input_shape[0], hidden_size))
    return output
