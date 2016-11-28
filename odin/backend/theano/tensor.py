# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import division, absolute_import, print_function

import os
import math
import numbers
from collections import OrderedDict
from __builtin__ import (any as _any, sum as _sum)

import numpy as np

import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.nnet import softsign as T_softsign
if theano.gpuarray.pygpu is not None and theano.gpuarray.pygpu_activated:
    from theano.gpuarray import dnn

from odin.config import CONFIG, RNG_GENERATOR
from odin.utils import as_tuple, as_shape_tuple, dict_union, package_installed, uuid
from odin.utils.shape_calculation import (get_conv_output_shape,
                                          get_pool_output_shape)
from odin.basic import (add_role, PARAMETER, ACTIVATION_PARAMETER,
                        add_shape, get_shape)

from .helpers import (auto_infer_shape, _check_target, variable,
                      is_trainable_variable, is_variable, is_placeholder,
                      ComputationGraph)

FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
NPROCESSORS = CONFIG['device_info']['n']

# store simple theano ops
backend_ops_relu = T.nnet.relu
backend_ops_elu = T.nnet.elu
backend_ops_softmax = T.nnet.softmax
backend_ops_softplus = T.nnet.softplus
backend_ops_softsign = T_softsign
backend_ops_sigmoid = T.nnet.sigmoid
backend_ops_hard_sigmoid = T.nnet.hard_sigmoid
backend_ops_tanh = T.tanh

backend_ops_square = T.sqr
backend_ops_abs = T.abs_
backend_ops_sign = T.sgn
backend_ops_inv = T.inv
backend_ops_sqrt = T.sqrt
backend_ops_exp = T.exp
backend_ops_log = T.log
backend_ops_round = T.round
backend_ops_pow = T.pow
backend_ops_clip = T.clip

backend_ops_diag = T.diag
backend_ops_eye = T.eye

# Comparator
backend_ops_switch = T.switch
backend_ops_eq = T.eq
backend_ops_neq = T.neq
backend_ops_gt = T.gt
backend_ops_ge = T.ge
backend_ops_lt = T.lt
backend_ops_le = T.le

backend_ops_categorical_crossentropy = T.nnet.categorical_crossentropy
backend_ops_binary_crossentropy = T.nnet.binary_crossentropy


# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
def eval(x):
    """ Run a graph. """
    return x.eval()


# ===========================================================================
# Shape operator
# ===========================================================================
def broadcastable(x):
    return x.broadcastable


def addbroadcast(x, *axes):
    input_shape = get_shape(x)
    x = T.addbroadcast(x, *axes)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


# ===========================================================================
# Predefined data
# ===========================================================================
def zeros(shape, dtype=FLOATX, name=None):
    """Instantiate an all-zeros variable.
    """
    x = T.zeros(shape=shape, dtype=dtype)
    add_shape(x, shape)
    return x


def ones(shape, dtype=FLOATX, name=None):
    """Instantiate an all-ones variable.
    """
    x = T.ones(shape=shape, dtype=dtype)
    add_shape(x, shape)
    return x


def ones_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    input_shape = get_shape(x)
    x = T.ones_like(x, dtype=dtype, opt=True)
    add_shape(x, input_shape)
    return x


def zeros_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype
    input_shape = get_shape(x)
    x = T.zeros_like(x, dtype=dtype, opt=True)
    add_shape(x, input_shape)
    return x


def cast(x, dtype):
    if 'theano.' in str(x.__class__):
        input_shape = get_shape(x)
        x = T.cast(x, dtype)
        add_shape(x, input_shape)
        return x
    return np.cast[dtype](x)


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    """ 2 special cases:
    (2, 3).(4, 3, 5) = (2, 4, 5)
    (2, 3, 4).(4, 5) = (2, 3, 5)
    """
    output = T.dot(x, y)
    shapeX = get_shape(x)
    shapeY = get_shape(y)
    if isinstance(shapeX, (tuple, list)) and isinstance(shapeY, (tuple, list)):
        if y.ndim > 2:
            outshapeY = tuple([shapeY[i] for i in range(y.ndim)
                               if i != y.ndim - 2])
        else:
            outshapeY = (shapeY[-1],)
        add_shape(output, shapeX[:-1] + outshapeY)
    return output


def batched_dot(x, y):
    """Batchwise dot product.
    This function computes the dot product between the two tensors,
    by iterating over the first dimension.
    """
    output = T.batched_dot(x, y)
    shapeX = get_shape(x)
    shapeY = get_shape(y)
    if isinstance(shapeX, (tuple, list)) and isinstance(shapeY, (tuple, list)):
        if y.ndim > 2:
            outshapeY = tuple([shapeY[i] for i in range(1, y.ndim)
                               if i != y.ndim - 2])
        else:
            outshapeY = (shapeY[-1],)
        add_shape(output, shapeX[:-1] + outshapeY)
    return output


def transpose(x, axes=None):
    output_shape = get_shape(x)
    x = T.transpose(x, axes=axes)
    if isinstance(output_shape, (tuple, list)):
        if axes is None:
            output_shape = output_shape[::-1]
        else:
            output_shape = [output_shape[i] for i in axes]
        add_shape(x, tuple(output_shape))
    return x


def gather(reference, indices):
    """reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    """
    return reference[indices]


# ===========================================================================
# ELEMENT-WISE OPERATIONS
# ===========================================================================
def var(x, axis=None, keepdims=False):
    y = T.var(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.var, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def mean(x, axis=None, keepdims=False):
    dtype = x.dtype
    if 'int' in str(dtype) or 'bool' in str(dtype):
        dtype = FLOATX
    y = T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.mean, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def std(x, axis=None, keepdims=False):
    y = T.std(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.std, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def max(x, axis=None, keepdims=False):
    y = T.max(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.max, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def min(x, axis=None, keepdims=False):
    y = T.min(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.min, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.
    """
    y = T.sum(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.sum, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def prod(x, axis=None, keepdims=False):
    """Multiply the values in a tensor, alongside the specified axis.
    """
    y = T.prod(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.prod, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).
    """
    y = T.any(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.any, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def argmax(x, axis=-1, keepdims=False):
    y = T.argmax(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.argmax, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def argmin(x, axis=-1, keepdims=False):
    y = T.argmin(x, axis=axis, keepdims=keepdims)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.argmin, x, axis=axis, keepdims=keepdims)
        add_shape(y, output_shape)
    return y


def arange(start, stop=None, step=1, dtype=None):
    x = T.arange(start=start, stop=stop, step=step, dtype=dtype)
    if stop is None:
        stop = start
        start = 0
    add_shape(x, (int(np.ceil((stop - start) / step)),))
    return x


def argsort(x):
    """ The indices in -1 axis will be sorted by the values in
    descending order.
    """
    axis = -1
    _ = [slice(None) for i in range(x.ndim - 1)] + [slice(None, None, -1)]
    y = T.argsort(x, axis)[_]
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.argsort, x, axis=axis)
        add_shape(y, output_shape)
    return y


def argtop_k(x, k=1):
    # top-k accuracy
    top = T.argsort(x, axis=-1)
    # (Theano cannot index with [..., -top_k:], we need to simulate that)
    top = top[[slice(None) for _ in range(top.ndim - 1)] +
              [slice(-k, None)]]
    top = top[(slice(None),) * (top.ndim - 1) + (slice(None, None, -1),)]
    return top


# ===========================================================================
# Primitive ops
# ===========================================================================
def add(x, y):
    z = T.add(x, y)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.add, x, y)
        add_shape(z, output_shape)
    return z


def sub(x, y):
    z = T.sub(x, y)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.sub, x, y)
        add_shape(z, output_shape)
    return z


def mul(x, y):
    z = T.mul(x, y)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.mul, x, y)
        add_shape(z, output_shape)
    return z


def div(x, y):
    z = T.true_div(x, y)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.true_div, x, y)
        add_shape(z, output_shape)
    return z


def mod(x, y):
    z = T.mod(x, y)
    if isinstance(get_shape(x), (tuple, list)):
        output_shape = auto_infer_shape(T.mod, x, y)
        add_shape(z, output_shape)
    return z


def maximum(x, y):
    x = T.maximum(x, y)
    output_shape = get_shape(y)
    if isinstance(output_shape, (tuple, list)):
        add_shape(x, output_shape)
    return x


def minimum(x, y):
    x = T.minimum(x, y)
    output_shape = get_shape(y)
    if isinstance(output_shape, (tuple, list)):
        add_shape(x, output_shape)
    return x


# ===========================================================================
# SHAPE OPERATIONS
# ===========================================================================
def reverse(x, axes=-1):
    """Apply [::-1] to appropriate axis"""
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    axes = [i % x.ndim for i in axes]
    input_shape = get_shape(x)
    x = x[tuple([slice(None, None, -1) if i in axes
                 else slice(None)
                 for i in range(x.ndim)])]
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def concatenate(tensors, axis=-1):
    x = T.concatenate(tensors, axis=axis)
    add_shape(x,
        auto_infer_shape(T.concatenate, *tensors, axis=axis, group_inputs=True))
    return x


def tile(x, n):
    y = T.tile(x, n)
    add_shape(y, auto_infer_shape(T.tile, x, reps=n))
    return y


def stack(tensors):
    """ (5, 2) and (5, 2) => (2, 5, 2) """
    y = T.stack(*tensors)
    add_shape(y, auto_infer_shape(T.stack, *tensors))
    return y


def expand_dims(x, dim=-1):
    """Add a 1-sized dimension at index "dim".
    """
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    return dimshuffle(x, pattern)


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
    new_shape = tuple(new_shape)
    _ = auto_infer_shape(T.reshape, x, newshape=new_shape)
    x = T.reshape(x, new_shape)
    add_shape(x, _)
    return x


def dimshuffle(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    input_shape = get_shape(x)
    new_shape = tuple([1 if i == 'x' else input_shape[i] for i in pattern])
    x = x.dimshuffle(pattern)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, new_shape)
    return x


def flatten(x, outdim=1):
    input_shape = get_shape(x)
    ndim = x.ndim
    x = T.flatten(x, outdim)
    if isinstance(input_shape, (tuple, list)):
        remove_dims = input_shape[:(ndim - outdim + 1)]
        remain_dims = input_shape[(ndim - outdim + 1):]
        n = None
        if all(i is not None for i in remove_dims):
            n = np.prod(remove_dims)
        output_shape = (n,) + remain_dims
        add_shape(x, output_shape)
    return x


def repeat(x, n, axes=None):
    """Repeat a N-D tensor.

    If x has shape (s1, s2, s3) and axis=(1, -1), the output
    will have shape (s1, s2 * n[0], s3 * n[1]).
    """
    input_shape = get_shape(x)
    if axes is not None:
        if not isinstance(axes, (tuple, list)):
            axes = (axes,)
        axes = tuple([i % x.ndim for i in axes])
        n = as_tuple(n, len(axes))
        for i, j in zip(n, axes):
            x = T.extra_ops.repeat(x, repeats=i, axis=j)
        if isinstance(input_shape, (tuple, list)):
            if axes is None and None not in input_shape:
                add_shape(x, int(np.prod(input_shape) * n))
            else:
                add_shape(x, tuple([j if i not in axes or j is None
                                    else j * n[axes.index(i)]
                                    for i, j in enumerate(input_shape)]))
        return x
    else:
        return tile(x, n)


def squeeze(x, axis):
    """Remove a 1-dimension from the tensor at index "axis".
    """
    input_shape = get_shape(x)
    axis = axis % x.ndim
    x = T.addbroadcast(x, axis)
    x = T.squeeze(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([j for i, j in enumerate(input_shape) if i != axis]))
    return x


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
    if not isinstance(axes, (tuple, list)):
        axes = (axes,)
    axes = tuple([i % x.ndim for i in axes])
    padding = as_tuple(padding, len(axes), int)

    input_shape = x.shape
    output_shape = tuple([input_shape[i] if i not in axes
                         else input_shape[i] + 2 * padding[axes.index(i)]
                         for i in range(x.ndim)])
    output = T.zeros(output_shape)
    indices = tuple([slice(None) if i not in axes
                    else slice(padding[axes.index(i)], input_shape[i] + padding[axes.index(i)])
                    for i in range(x.ndim)])
    input_shape = get_shape(x)
    x = T.set_subtensor(output[indices], x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, tuple([input_shape[i] if i not in axes or input_shape[i] is None
                            else input_shape[i] + 2 * padding[axes.index(i)]
                            for i in range(x.ndim)]))
    return x


# ===========================================================================
# GRAPH MANIPULATION
# ===========================================================================
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
        if not hasattr(self, 'inputs_name'):
            self.inputs_name = [i.name for i in inputs]
        # ====== validate updates ====== #
        if not isinstance(updates, OrderedDict):
            updates = OrderedDict(updates)
        updates = dict_union(updates, ComputationGraph(outputs).updates)
        updates = updates.items()
        # ====== add and reset global update ====== #
        self.function = theano.function(
            inputs=inputs, outputs=outputs,
            updates=updates,
            on_unused_input='raise', # TODO: remove this when stop testing
            allow_input_downcast=True,
            **kwargs)

    def __call__(self, *inputs, **kwargs):
        if len(kwargs) == len(self.inputs_name):
            inputs = [kwargs[i] for i in self.inputs_name]
        return self.function(*inputs)


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
    # TODO: float16 overflow, unsupport DeepCopyOps
    return T.grad(loss, wrt=variables, consider_constant=consider_constant,
        disconnected_inputs='raise')


def stop_gradient(vars):
    return theano.gradient.disconnected_grad(vars)


def jacobian(loss, variables):
    return theano.gradient.jacobian(loss, variables, disconnected_inputs='warn')


def hessian(loss, variables):
    return theano.gradient.hessian(loss, variables, disconnected_inputs='warn')


def one_hot(x, nb_class):
    """ x: 1D-integer vector """
    ret = T.zeros((x.shape[0], nb_class), dtype=FLOATX)
    x = T.cast(x, 'int32')
    ret = T.set_subtensor(ret[T.arange(x.shape[0]), x], 1)
    return ret


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

    Returns
    -------
    conf_mat : Confusion matrix of actual and predictions observations as shown below.
               | Predicted
    ___________|___________
       Actual  |
               |
    Examples
    --------
    >>> import theano
    >>> from theano.tensor.nnet import confusion_matrix
    >>> x = theano.tensor.vector()
    >>> y = theano.tensor.vector()
    >>> f = theano.function([x, y], confusion_matrix(x, y))
    >>> a = [0, 1, 2, 1, 0]
    >>> b = [0, 0, 2, 1, 2]
    >>> print(f(a, b))
    [array([[0, 0, 1],
            [2, 1, 0],
            [0, 0, 1]]), array([ 0.,  1.,  2.])]
    """
    if y_true.ndim == 2:
        y_true = T.argmax(y_true, axis=-1)
    elif y_true.ndim != 1:
        raise ValueError('actual must be 1-d or 2-d tensor variable')
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=-1)
    elif y_pred.ndim != 1:
        raise ValueError('pred must be 1-d or 2-d tensor variable')

    if labels is None:
        labels = T.extra_ops.Unique(False, False, False)(T.concatenate([y_true, y_pred]))

    colA = y_true.dimshuffle(0, 'x')
    colP = y_pred.dimshuffle(0, 'x')

    oneHotA = T.eq(colA, labels).astype('int64')
    oneHotP = T.eq(colP, labels).astype('int64')

    conf_mat = T.dot(oneHotA.T, oneHotP)
    return conf_mat


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
    input_shape = get_shape(x)
    x = T.cast(
        T.eq(T.arange(x.shape[axis])[None, :],
             T.argmax(x, axis=axis, keepdims=True)),
        FLOATX
    )
    add_shape(x, input_shape)
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
    input_shape = get_shape(x)
    x = T.mul(x, expand_dims(mask, -1))
    add_shape(x, input_shape)
    return x


# ===========================================================================
# RANDOMNESS
# ===========================================================================
_RNG = RandomStreams(seed=CONFIG['seed'])


def set_rng(seed):
    global _RNG
    _RNG = RandomStreams(seed=seed)


def random_normal(shape, mean=0.0, std=1.0, dtype=FLOATX):
    return _RNG.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=FLOATX):
    return _RNG.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p, dtype=FLOATX, seed=None):
    return _RNG.binomial(size=shape, n=1, p=p, dtype=dtype)


# ===========================================================================
# Convolution
# ===========================================================================
def __validate_strides_padding_dilation(strides, border_mode, filter_dilation, ndim):
    # border_mode or padding
    if border_mode == 'same':
        border_mode = 'half'
    elif border_mode == 'valid' or border_mode == 'full':
        pass
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


def __img_theano_format(x):
    return x.dimshuffle((0, x.ndim - 1,) + tuple(range(1, x.ndim - 1)))


def __img_tensorflow_format(x):
    return x.dimshuffle((0,) + tuple(range(2, x.ndim)) + (1,))


def __ker_theano_format(x):
    return x.dimshuffle((x.ndim - 1, x.ndim - 2) + tuple(range(x.ndim - 2)))


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
    Warning
    -------
    For "same" or "half" border_mode, the shape of output only equal
    to input if kernel shape is odd.
    """
    image_shape = get_shape(x)
    kernel_shape = get_shape(kernel)
    if _any(i % 2 == 0 for i in kernel_shape[:-2]):
        print('[WARNING] Kernel shape %s contains even values, the output shape is '
              'different from the input shape in "same"-border_mode.' % str(kernel_shape[:-2]))
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=2)
    # ====== convert input to theano format ====== #
    x = __img_theano_format(x)
    kernel = __ker_theano_format(kernel)
    conv_out = T.nnet.conv2d(x, kernel,
        border_mode=border_mode,
        subsample=strides,
        input_shape=(image_shape[0], image_shape[-1]) + image_shape[1:-1],
        filter_shape=(kernel_shape[-1], kernel_shape[-2]) + kernel_shape[:-2],
        filter_dilation=filter_dilation)
    # if border_mode == 'half':
    #     if kernel_shape[2] % 2 == 0:
    #         conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
    #     if kernel_shape[3] % 2 == 0:
    #         conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]
    # ====== estimate output shape ====== #
    conv_out = __img_tensorflow_format(conv_out)
    add_shape(conv_out, get_conv_output_shape(image_shape, kernel_shape,
                                              border_mode, strides,
                                              filter_dilation))
    return conv_out


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
    Warining
    --------
    Transposed_conv won't procedure the same shape as original image if kernel
    value is even (i.e. x % 2 == 0).
    """
    # ====== convert to theano formated shapes ====== #
    x = __img_theano_format(x)
    kernel = __ker_theano_format(kernel)
    # ====== params ====== #
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=2)
    # ====== grad_wrt_inputs ====== #
    x = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(x, kernel,
        input_shape=(output_shape[0], output_shape[-1]) + tuple(output_shape[1:-1]),
        subsample=strides, border_mode=border_mode,
        filter_flip=True,
        filter_dilation=filter_dilation)
    # back to tf-shape
    x = __img_tensorflow_format(x)
    add_shape(x, output_shape)
    return x


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid',
           filter_dilation=(1, 1, 1)):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".

    Note
    ----
    dim_ordering : tf-tensorflow (__img_theano_format(x)
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        ---
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
    """
    # get and convert volume_shape to theano format
    volume_shape = get_shape(x)
    # get and convert filter_shape to theano format
    kernel_shape = get_shape(kernel)
    if _any(i % 2 == 0 for i in kernel_shape[:-2]):
        print('[WARNING] Kernel shape %s contains even values, the output shape is '
              'different from the input shape in "same"-border_mode.' % str(kernel_shape[:-2]))
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=3)
    # ====== convert input to theano format ====== #
    x = __img_theano_format(x)
    kernel = __ker_theano_format(kernel)
    # call convolution
    conv_out = T.nnet.conv3d(x, kernel,
        border_mode=border_mode,
        subsample=strides,
        input_shape=(volume_shape[0], volume_shape[-1]) + volume_shape[1:-1],
        filter_shape=(kernel_shape[-1], kernel_shape[-2]) + kernel_shape[:-2],
        filter_dilation=filter_dilation)
    # if border_mode == 'half':
    #     if kernel_shape[2] % 2 == 0:
    #         conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :, :]
    #     if kernel_shape[3] % 2 == 0:
    #         conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1], :]
    #     if kernel_shape[4] % 2 == 0:
    #         conv_out = conv_out[:, :, :, :, :(x.shape[4] + strides[2] - 1) // strides[2]]
    # back to theano form
    # convert back to tensorflow shape
    conv_out = __img_tensorflow_format(conv_out)
    # infer output shape
    output_shape = get_conv_output_shape(volume_shape, kernel_shape,
                                         border_mode, strides, filter_dilation)
    add_shape(conv_out, output_shape)
    return conv_out


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
    Warining
    --------
    Transposed_conv won't procedure the same shape as original image if kernel
    value is even (i.e. x % 2 == 0).
    """
    if len(output_shape) != 5:
        raise ValueError('output_shape for deconvolution operator must be 4-D')
    # ====== convert to theano formated shapes ====== #
    x = __img_theano_format(x)
    kernel = __ker_theano_format(kernel)
    # ====== params ====== #x
    strides, border_mode, filter_dilation = __validate_strides_padding_dilation(
        strides, border_mode, filter_dilation, ndim=3)
    # ====== grad_wrt_inputs ====== #
    x = T.nnet.abstract_conv.conv3d_grad_wrt_inputs(x, kernel,
        input_shape=(output_shape[0], output_shape[-1]) + tuple(output_shape[1:-1]),
        subsample=strides, border_mode=border_mode,
        filter_flip=True,
        filter_dilation=filter_dilation)
    # back to tf-shape
    x = __img_tensorflow_format(x)
    add_shape(x, output_shape)
    return x


# ===========================================================================
# Pooling
# ===========================================================================
def __validate_pool_stride_border(x, pool_size, strides, border_mode, mode, ndim):
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
        border_mode = (0,) * ndim
    elif border_mode == 'same':
        # pad x by hand
        input_shape = get_shape(x)[-ndim - 1:-1]
        native_shape = get_shape(x, native=True)
        output_shape = [math.ceil(float(i) / float(j)) for i, j in zip(input_shape, strides)]
        pad_size = [int((out - 1) * st + ps - ins)
                    for ins, out, ps, st in zip(input_shape, output_shape, pool_size, strides)]
        # pad if necessary
        if _sum(pad_size) > 0:
            padded_x = T.zeros(tuple([native_shape[i] for i in range(x.ndim - ndim - 1)]) +
                               tuple([i + j for i, j in zip(input_shape, pad_size)]) +
                               (native_shape[-1],))
            indices = [slice(None) for i in range(x.ndim - ndim - 1)] + \
                [slice(i // 2, i // 2 + j) for i, j in zip(pad_size, input_shape)] + \
                [slice(None)]
            x = T.set_subtensor(padded_x[indices], x)
    else:
        border_mode = as_tuple(border_mode, ndim, int)
    # pooling mode
    if 'max' in mode.lower():
        mode = 'max'
    elif 'avg' in mode.lower():
        mode = 'average_exc_pad'
    return x, pool_size, strides, border_mode, mode


def pool2d(x, pool_size=(2, 2), strides=None, border_mode=(0, 0),
           ignore_border=True, mode='max'):
    """
    Parameters
    ----------
    x : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    pool_size : tuple of length 2 or theano vector of ints of size 2.
        Factor by which to downscale (vertical ws, horizontal ws).
        (2, 2) will halve the image in each dimension.
    strides : tuple of two ints or theano vector of ints of size 2.
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    border_mode : tuple of two ints or theano vector of ints of size 2.
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ws=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    mode : {'max', 'avg'}
        Operation executed on each window. `max` or `average`

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """
    # ====== convert to theano formated shapes ====== #
    input_shape = get_shape(x)
    # pool_size
    x, pool_size, strides, border_mode, mode = __validate_pool_stride_border(
        x, pool_size, strides, border_mode, mode, ndim=2)
    x = __img_theano_format(x)
    # ====== On GPU: use CuDNN ====== #
    pool_out = pool.pool_2d(x, ws=pool_size, stride=strides,
                            ignore_border=ignore_border,
                            pad=(0, 0) if isinstance(border_mode, str) else border_mode,
                            mode=mode)
    # ====== Estimate output shape ====== #
    pool_out = __img_tensorflow_format(pool_out)
    output_shape = get_pool_output_shape(input_shape, pool_size,
        ignore_border=ignore_border, strides=strides, pad=border_mode)
    add_shape(pool_out, tuple(output_shape))
    return pool_out


def pool3d(x, pool_size=(2, 2), strides=None, border_mode=(0, 0, 0),
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
    # ====== convert to theano formated shapes ====== #
    input_shape = get_shape(x)
    # pool_size
    x, pool_size, strides, border_mode, mode = __validate_pool_stride_border(
        x, pool_size, strides, border_mode, mode, ndim=3)
    x = __img_theano_format(x)
    # ====== On GPU: use CuDNN ====== #
    pool_out = pool.pool_3d(x, ws=pool_size, stride=strides,
                            ignore_border=ignore_border,
                            pad=(0, 0, 0) if isinstance(border_mode, str) else border_mode,
                            mode=mode)
    # ====== Estimate output shape ====== #
    pool_out = __img_tensorflow_format(pool_out)
    output_shape = get_pool_output_shape(input_shape, pool_size,
        ignore_border=ignore_border, strides=strides, pad=border_mode)
    add_shape(pool_out, tuple(output_shape))
    return pool_out


# ===========================================================================
# RNN
# ===========================================================================
def Scan(fn,
         sequences=None,
         outputs_info=None,
         n_steps=None,
         truncate_gradient=-1,
         backwards=False,
         name=None):
    """
    Note
    ----
    backwards mode only invert sequences then iterate over them
    """
    return theano.scan(fn,
                       sequences=sequences,
                       outputs_info=outputs_info,
                       non_sequences=None,
                       n_steps=n_steps,
                       truncate_gradient=truncate_gradient,
                       go_backwards=backwards,
                       mode=None,
                       name=name,
                       profile=False,
                       allow_gc=None,
                       strict=False)


def rnn_dnn(X, hidden_size, rnn_mode,
            num_layers=1,
            initial_states=None,
            parameters=None,
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
    initial_states: list of tensor
        pass
    parameters: list of tensor
        pass
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

    """
    if name is None: name = uuid()
    # ====== Check arguments ====== #
    if rnn_mode not in ('rnn_relu', 'rnn_tanh', 'lstm', 'gru'):
        raise ValueError("rnn_mode=%s must be: 'rnn_relu', 'rnn_tanh', 'lstm', 'gru'"
                         % rnn_mode)
    if input_mode not in ('linear', 'skip'):
        raise ValueError("input_mode=%s must be: 'linear', 'skip'" % input_mode)
    if direction_mode not in ('unidirectional', 'bidirectional'):
        raise ValueError("direction_mode=%s must be: 'unidirectional', 'bidirectional'"
                         % direction_mode)
    # ====== create RNNBlock ====== #
    input_shape = get_shape(X)
    if X.ndim != 3:
        raise ValueError('Input must be 3-D tensor, but X is %d-D tensor' % X.ndim)
    if input_shape[-1] != hidden_size and 'skip' in input_mode:
        raise ValueError('In skip_input mode, input size must be equal to hidden size'
                         ', but input_size=%d != hidden_size=%d' %
                         (input_shape[-1], hidden_size))
    # IF we dimshuffle here, a lot of error concern GPUarray,
    # and cudnn will happen
    batch_size = X.shape[0]
    rnnb = dnn.RNNBlock(dtype=theano.config.floatX, hidden_size=hidden_size,
                        num_layers=num_layers, rnn_mode=rnn_mode,
                        input_mode=input_mode, direction_mode=direction_mode,
                        context_name=None)
    # layer info (note in case of bidirectional, output from previous
    # layers are concatenated).
    layer_info = [input_shape[-1], hidden_size] + \
                 [hidden_size * (2 if direction_mode == 'bidirectional' else 1),
                  hidden_size] * (num_layers - 1)
    nb_params = rnnb.get_param_size([12, input_shape[-1]])
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
                                     bidirectional=True if direction_mode == 'bidirectional' else False)
                                     for i in range(num_layers)]).astype(FLOATX)
        parameters = variable(parameters, name=name)
    else:
        if get_shape(parameters)[0] != nb_params:
            raise ValueError('parameters must be 1-D vector of length %d' % nb_params)
    assert nb_params == get_shape(parameters)[0], \
        "Require %d parameters but only %d provided" % (nb_params, get_shape(parameters)[0])
    # check initial states
    num_layers = num_layers * 2 if direction_mode == 'bidirectional' else num_layers
    if initial_states is None:
        h0 = zeros((num_layers, batch_size, hidden_size))
        if rnn_mode == 'lstm':
            c0 = zeros((num_layers, batch_size, hidden_size))
        else:
            c0 = None
    else:
        if rnn_mode == 'lstm':
            h0, c0 = initial_states
        else:
            h0 = initial_states[0] if isinstance(initial_states, (list, tuple)) \
                else initial_states
            c0 = None
    # ====== get output ====== #
    output = rnnb.apply(w=parameters, x=X.dimshuffle(1, 0, 2),
                        hx=h0, cx=c0)
    output = [output[0].dimshuffle(1, 0, 2)] + list(output[1:])
    add_shape(output[0], (input_shape[0], input_shape[1],
                          hidden_size * (2 if direction_mode == 'bidirectional' else 1)))
    for o in output[1:]:
        add_shape(o, (num_layers, input_shape[0], hidden_size))
    return output
