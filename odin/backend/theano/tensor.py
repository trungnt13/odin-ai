# ===========================================================================
# This module is adpated from: https://github.com/fchollet/keras
# Original work Copyright (c) 2014-2015 keras contributors
# Some idea are also borrowed from Lasagne library
# Original work Copyright (c) 2014-2015 Lasagne contributors
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import division, absolute_import

import os
import math
import numbers
from collections import OrderedDict

import numpy as np

import theano
from theano import tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.nnet import softsign as T_softsign

from odin.config import CONFIG, RNG_GENERATOR
from odin.utils import as_tuple, as_shape_tuple, dict_union
from odin.basic import (add_role, TRAINING, PARAMETER,
                        ACTIVATION_PARAMETER, DEPLOYING,
                        add_shape, get_shape)

from .helpers import (auto_infer_shape, _check_target,
                      is_trainable_variable, is_variable, is_placeholder,
                      is_training, ComputationGraph)

FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
NPROCESSORS = CONFIG['device_info']['n']
_RNG = RandomStreams(seed=RNG_GENERATOR.randint(10e8))

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


# ===========================================================================
# INTERNAL UTILS
# ===========================================================================
def on_gpu():
    """Return whether the session is set to
    run on GPU or not (i.e. on CPU).
    """
    import theano.sandbox.cuda

    return 'gpu' in theano.config.device or \
    'cuda' in theano.config.device or \
    'gpu' in theano.config.contexts or \
    'cuda' in theano.config.contexts or \
    theano.sandbox.cuda.cuda_enabled


if on_gpu():
    # dummy initialization to remove the overhead of running libgpuarray backend
    if CONFIG['multigpu']:
        _ = theano.shared(value=np.asarray(1., dtype='float32'),
                         name='temporary_var', target='dev0')
    else:
        _ = theano.shared(value=np.asarray(1., dtype='float32'),
                         name='temporary_var')
    T.grad(2 * _, _).eval()
    _.set_value(None)
    del _


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
        return T.cast(x, dtype)
    return np.cast[dtype](x)


def castX(x):
    return cast(x, FLOATX)


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    # TODO: float16 overflow
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


def mean(x, axis=None, keepdims=False):
    dtype = x.dtype
    if 'int' in dtype:
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


def argsort(x, axis=-1):
    y = T.argsort(x, axis)
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
def reverse(x, axis=-1):
    """Apply [::-1] to appropriate axis"""
    if axis < 0:
        axis += x.ndim
    input_shape = get_shape(x)
    x = x[(slice(None),) * axis + (slice(None, None, -1),)]
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape, )
    return x


def concatenate(tensors, axis=-1):
    x = T.concatenate(tensors, axis=axis)
    add_shape(x,
        auto_infer_shape(T.concatenate, *tensors,
                          axis=axis, group_inputs=True))
    return x


def tile(x, n):
    y = T.tile(x, n)
    add_shape(y, auto_infer_shape(T.tile, x, reps=n))
    return y


def stack(*x):
    y = T.stack(*x)
    add_shape(y, auto_infer_shape(T.stack, *x))
    return y


def reshape(x, shape_):
    """ x.shape = [25, 08, 12]
    reshape(shape=([1], [2], [0]))
    => x.shape = (08, 12, 25)
    """
    input_shape = get_shape(x)
    new_shape = []
    for i in shape_:
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
    else:
        x = T.extra_ops.repeat(x, n, None)
    if isinstance(input_shape, (tuple, list)):
        if axes is None and None not in input_shape:
            add_shape(x, int(np.prod(input_shape) * n))
        else:
            add_shape(x, tuple([j if i not in axes or j is None
                                else j * n[axes.index(i)]
                                for i, j in enumerate(input_shape)]))
    return x


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
# VALUE MANIPULATION
# ===========================================================================
def get_value(x, borrow=False):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value(borrow=borrow)


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def set_subtensor(x, y):
    return T.set_subtensor(x, y)


# ===========================================================================
# GRAPH MANIPULATION
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
    return T.cast(
        T.eq(T.arange(x.shape[axis])[None, :],
             T.argmax(x, axis=axis, keepdims=True)),
        FLOATX
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
    return T.mul(x, expand_dims(mask, -1))


# ===========================================================================
# Ops
# ===========================================================================
def antirectify(x):
    """
    This is the combination of a sample-wise L2 normalization with the
    concatenation of:
        - the positive part of the input
        - the negative part of the input
    The result is a tensor of samples that are twice as large as
    the input samples.
    It can be used in place of a ReLU.
        - Input shape: 2D tensor of shape (samples, n)
        - Output shape: 2D tensor of shape (samples, 2*n)

    Notes
    -----
    When applying ReLU, assuming that the distribution of the previous
    output is approximately centered around 0., you are discarding half of
    your input. This is inefficient.
    Antirectifier allows to return all-positive outputs like ReLU, without
    discarding any data.
    Tests on MNIST show that Antirectifier allows to train networks with
    twice less parameters yet with comparable classification accuracy
    as an equivalent ReLU-based network.

    """
    if x.ndim != 2:
        raise Exception('This Ops only support 2D input.')
    input_shape = get_shape(x)
    x -= T.mean(x, axis=1, keepdims=True)
    # l2 normalization
    x /= T.sqrt(T.sum(T.square(x), axis=1, keepdims=True))
    x = T.concatenate([T.nnet.relu(x, 0), T.nnet.relu(-x, 0)], axis=1)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (input_shape[0], input_shape[1] * 2))
    return x


def randrectify(x, lower=0.3, upper=0.8, shared_axes='auto'):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Applies a randomized leaky rectify activation to x.

    The randomized leaky rectifier was first proposed and used in the Kaggle
    NDSB Competition, and later evaluated in [1]_. Compared to the standard
    leaky rectifier :func:`leaky_rectify`, it has a randomly sampled slope
    for negative input during training, and a fixed slope during evaluation.

    Equation for the randomized rectifier linear unit during training:
    :math:`\\varphi(x) = \\max((\\sim U(lower, upper)) \\cdot x, x)`

    During evaluation, the factor is fixed to the arithmetic mean of `lower`
    and `upper`.

    Parameters
    ----------
    lower : Theano shared variable, expression, or constant
        The lower bound for the randomly chosen slopes.

    upper : Theano shared variable, expression, or constant
        The upper bound for the randomly chosen slopes.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the random slopes of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the random slope over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, thus using a single random slope.

     References
    ----------
    .. [1] Bing Xu, Naiyan Wang et al. (2015):
       Empirical Evaluation of Rectified Activations in Convolutional Network,
       http://arxiv.org/abs/1505.00853
    """
    input_shape = get_shape(x)
    # ====== check lower and upper ====== #
    if is_trainable_variable(lower):
        add_role(lower, ACTIVATION_PARAMETER)
        lower.name = 'lower'
    if is_trainable_variable(upper):
        add_role(upper, ACTIVATION_PARAMETER)
        upper.name = 'upper'
    if not is_variable(lower > upper) and lower > upper:
        raise ValueError("Upper bound for Randomized Rectifier needs "
                         "to be higher than lower bound.")
    # ====== check shared_axes ====== #
    if shared_axes == 'auto':
        shared_axes = (0,) + tuple(range(2, len(input_shape)))
    elif shared_axes == 'all':
        shared_axes = tuple(range(len(input_shape)))
    elif isinstance(shared_axes, int):
        shared_axes = (shared_axes,)
    else:
        shared_axes = shared_axes
    # ====== main logic ====== #
    if not is_training(x) or upper == lower:
        x = T.nnet.relu(x, (upper + lower) / 2.0)
    else: # Training mode
        shape = list(input_shape)
        if any(s is None for s in shape):
            shape = list(x.shape)
        for ax in shared_axes:
            shape[ax] = 1

        rnd = _RNG.uniform(tuple(shape),
                           low=lower,
                           high=upper,
                           dtype=FLOATX)
        rnd = addbroadcast(rnd, *shared_axes)
        x = T.nnet.relu(x, rnd)
    add_shape(x, input_shape)
    return x


def categorical_crossentropy(output, target):
    input_shape = get_shape(output)
    # scale preds so that the class probas of each sample sum to 1
    output /= output.sum(axis=-1, keepdims=True)
    output = T.clip(output, EPSILON, 1.0 - EPSILON)
    x = T.nnet.categorical_crossentropy(output, target)
    add_shape(x, input_shape[0])
    return x


def squared_error(output, target):
    return T.square(output - target)


def binary_crossentropy(output, target):
    input_shape = get_shape(output)
    if output.ndim > 1: output = output.ravel()
    if target.ndim > 1: target = target.ravel()
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, EPSILON, 1.0 - EPSILON)
    x = T.nnet.binary_crossentropy(output, target)
    add_shape(x, input_shape[0])
    return x


# ===========================================================================
# Helper
# ===========================================================================
def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """ Copyright (c) 2014-2015 Lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


def conv2d(x, kernel, strides=(1, 1),
           border_mode='valid', image_shape=None, filter_shape=None):
    """
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, rows, cols)
        TH kernel shape: (depth, input_depth, rows, cols)
    """
    if border_mode == 'same':
        th_border_mode = 'half'
        np_kernel = kernel.eval()
    elif border_mode == 'valid':
        th_border_mode = 'valid'
    elif border_mode == 'full':
        th_border_mode = 'full'
    elif isinstance(border_mode, (tuple, list)):
        th_border_mode = border_mode
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))

    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None

    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)

    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)

    conv_out = T.nnet.conv2d(x, kernel,
                             border_mode=th_border_mode,
                             subsample=strides,
                             input_shape=image_shape,
                             filter_shape=filter_shape)

    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

    return conv_out


def deconv2d(x, kernel, image_shape, filter_shape=None,
    strides=(1, 1), border_mode='valid', flip_filters=True):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    img_shape: (n, channels, width, height) of original image
    filter_shape: (n_filter, channels, w, h) of original filters
    """
    if len(image_shape) != 4:
        raise ValueError('img_shape for deconvolution operator must be 4-D')
    border_mode = 'half' if border_mode == 'same' else border_mode
    op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
        imshp=tuple([int(i) if isinstance(i, (long, float, int)) else None
                     for i in image_shape]),
        kshp=filter_shape,
        subsample=strides, border_mode=border_mode,
        filter_flip=flip_filters)
    transposed_x = op(kernel, x, image_shape[2:])
    if isinstance(image_shape, (tuple, list)):
        add_shape(transposed_x, image_shape)
    return transposed_x


def conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid',
           image_shape=None, filter_shape=None):
    """
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    dim_ordering : th (defaults)
        TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
    """
    if False and on_gpu(): # Using DNN on GPU
        from theano.sandbox.cuda import dnn
        if border_mode == 'same':
            border_mode = 'half'
        conv_out = dnn.dnn_conv3d(img=x,
                                kerns=kernel,
                                subsample=strides,
                                border_mode=border_mode,
                                conv_mode='conv')
    else: # Using default implementation of Theano
        if border_mode not in {'same', 'valid', 'full'} and not isinstance(border_mode, (tuple, list)):
            raise Exception('Invalid border mode: ' + str(border_mode))

        if border_mode == 'same':
            assert(strides == (1, 1, 1))
            pad_dim1 = (kernel.shape[2] - 1)
            pad_dim2 = (kernel.shape[3] - 1)
            pad_dim3 = (kernel.shape[4] - 1)
            output_shape = (x.shape[0], x.shape[1],
                            x.shape[2] + pad_dim1,
                            x.shape[3] + pad_dim2,
                            x.shape[4] + pad_dim3)
            output = T.zeros(output_shape)
            indices = (slice(None), slice(None),
                       slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                       slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                       slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
            x = T.set_subtensor(output[indices], x)
            border_mode = 'valid'

        border_mode_3d = (border_mode, border_mode, border_mode)
        conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                                   filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                                   border_mode=border_mode_3d,
                                   signals_shape=None,
                                   filters_shape=None)
        conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

        # support strides by manually slicing the output
        if strides != (1, 1, 1):
            conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]
    return conv_out


def pool2d(x, pool_size=(2, 2), ignore_border=True,
           strides=(1, 1), pad=(0, 0), mode='max'):
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
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """
    if strides is None:
        strides = pool_size
    pool_size = as_tuple(pool_size, 2, int)
    strides = as_tuple(strides, 2, int)
    pad = as_tuple(pad, 2, int)
    # ====== On GPU: use CuDNN ====== #
    if mode != 'sum' and on_gpu():
        from theano.sandbox.cuda import dnn
        if not ignore_border:
            raise ValueError('CuDNN does not support ignore_border = False.')
        pool_out = dnn.dnn_pool(x, ws=pool_size, stride=strides, mode=mode, pad=pad)
    # ====== Use default Theano implementation ====== #
    else:
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=ignore_border,
                                padding=pad,
                                mode=mode)
    # ====== Estimate output shape ====== #
    input_shape = get_shape(x)
    output_shape = list(input_shape)  # copy / convert to mutable list
    output_shape[2] = pool_output_length(input_shape[2],
                                         pool_size=pool_size[0],
                                         stride=strides[0],
                                         pad=pad[0],
                                         ignore_border=ignore_border)
    output_shape[3] = pool_output_length(input_shape[3],
                                         pool_size=pool_size[1],
                                         stride=strides[1],
                                         pad=pad[1],
                                         ignore_border=ignore_border)
    add_shape(pool_out, tuple(output_shape))
    return pool_out


def pool3d(x, pool_size=(2, 2, 2), ignore_border=True,
           strides=(1, 1, 1), pad=(0, 0, 0), mode='max'):
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
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """
    if strides is None:
        strides = pool_size
    pool_size = as_tuple(pool_size, 3, int)
    strides = as_tuple(strides, 3, int)
    pad = as_tuple(pad, 3, int)
    # ====== On GPU: use CuDNN ====== #
    if mode != 'sum' and on_gpu():
        from theano.sandbox.cuda import dnn
        if not ignore_border:
            raise ValueError('CuDNN does not support ignore_border = False.')
        pool_out = dnn.dnn_pool(x, ws=pool_size, stride=strides, mode=mode, pad=pad)
    # ====== Use default Theano implementation ====== #
    else:
        if len(set(pad)) > 1:
            raise ValueError('Only support same padding on CPU.')
        padding = (pad[0], pad[0])
        output = pool.pool_2d(input=dimshuffle(x, (0, 1, 4, 3, 2)),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode=mode)
        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=dimshuffle(output, (0, 1, 4, 3, 2)),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode=mode)
    # ====== Estimate output shape ====== #
    input_shape = get_shape(x)
    output_shape = list(input_shape)  # copy / convert to mutable list
    output_shape[2] = pool_output_length(input_shape[2],
                                         pool_size=pool_size[0],
                                         stride=strides[0],
                                         pad=pad[0],
                                         ignore_border=ignore_border)
    output_shape[3] = pool_output_length(input_shape[3],
                                         pool_size=pool_size[1],
                                         stride=strides[1],
                                         pad=pad[1],
                                         ignore_border=ignore_border)
    output_shape[4] = pool_output_length(input_shape[4],
                                         pool_size=pool_size[2],
                                         stride=strides[2],
                                         pad=pad[2],
                                         ignore_border=ignore_border)
    add_shape(pool_out, tuple(output_shape))
    return pool_out


def poolWTA(x, pool_size=(2, 2), axis=1):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    'Winner Take All' layer

    This layer performs 'Winner Take All' (WTA) across feature maps: zero out
    all but the maximal activation value within a region.

    Parameters
    ----------
    pool_size : integer
        the number of feature maps per region.

    axis : integer
        the axis along which the regions are formed.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer requires that the size of the axis along which it groups units
    is a multiple of the pool size.
    """
    input_shape = get_shape(x)
    num_feature_maps = input_shape[axis]
    num_pools = num_feature_maps // pool_size

    if input_shape[axis] % pool_size != 0:
        raise ValueError("Number of input feature maps (%d) is not a "
                         "multiple of the region size (pool_size=%d)" %
                         (num_feature_maps, pool_size))

    pool_shape = ()
    arange_shuffle_pattern = ()
    for k in range(axis):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    pool_shape += (num_pools, pool_size)
    arange_shuffle_pattern += ('x', 0)

    for k in range(axis + 1, x.ndim):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    input_reshaped = reshape(x, pool_shape)
    max_indices = argmax(input_reshaped, axis=axis + 1, keepdims=True)

    arange = T.arange(pool_size).dimshuffle(*arange_shuffle_pattern)
    mask = reshape(T.eq(max_indices, arange), input_shape)
    output = x * mask
    add_shape(output, input_shape)
    return output


def poolGlobal(x, pool_function=mean):
    """ Global pooling

    This layer pools globally across all trailing dimensions beyond the 2nd.

    Parameters
    ----------
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.

    Note
    ----
    output_shape = input_shape[:2]
    """
    input_shape = get_shape(x)
    x = pool_function(T.flatten(x, 3), axis=2)
    add_shape(x, input_shape[:2])
    return x


# ===========================================================================
# RANDOMNESS
# ===========================================================================
class _RandomWrapper(object):

    def __init__(self, rng):
        super(_RandomWrapper, self).__init__()
        self._rng = rng

    def normal(self, shape, mean, std, dtype=FLOATX):
        return self._rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

    def uniform(self, shape, low, high, dtype=FLOATX):
        return self._rng.uniform(size=shape, low=low, high=high, dtype=dtype)

    def binomial(self, shape, p, dtype=FLOATX):
        return self._rng.binomial(size=shape, n=1, p=p, dtype=dtype)


def rng(seed=None):
    if seed is None:
        seed = RNG_GENERATOR.randint(10e8)
    return _RandomWrapper(RandomStreams(seed=seed))


def random_normal(shape, mean=0.0, std=1.0, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p, dtype=FLOATX, seed=None):
    rng = _RNG
    if seed is not None:
        rng = RandomStreams(seed=seed)
    return rng.binomial(size=shape, n=1, p=p, dtype=dtype)


# ===========================================================================
# Noise
# ===========================================================================
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
    => (None, 10, 1)
    """
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    # ====== get noise shape ====== #
    if dims is None:
        noise_shape = input_shape
    else:
        return tuple([1 if i in dims else j
                      for i, j in enumerate(input_shape)])
    return noise_shape


def apply_dropout(x, level=0.5, noise_dims=None, rescale=True, seed=None):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
    level: float(0.-1.)
        probability dropout values in given tensor
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    seed: random seed or `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = get_shape(x)
    if not isinstance(seed, _RandomWrapper):
        seed = rng(seed=seed)
    # ====== not a training variable NO dropout ====== #
    if not is_training(x):
        return x
    # ====== Dropout ====== #
    retain_prob = 1. - level
    shape = x.shape
    if noise_dims is None:
        x = x * seed.binomial(shape=shape, p=retain_prob, dtype=x.dtype)
    else:
        noise_shape = _process_noise_dim(shape, noise_dims)
        # auto select broadcast shape
        broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
        if len(broadcast) > 0:
            x = x * addbroadcast(seed.binomial(shape=noise_shape,
                                               p=retain_prob,
                                               dtype=x.dtype), *broadcast)
        else:
            x = x * seed.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype)
    if rescale:
        x /= retain_prob
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def apply_noise(x, sigma=0.075, noise_dims=None, noise_type='gaussian', seed=None):
    """
    Parameters
    ----------
    x: A tensor.
    sigma : float or tensor scalar
        Standard deviation of added Gaussian noise
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    seed: random seed or `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = get_shape(x)
    noise_type = noise_type.lower()
    if not isinstance(seed, _RandomWrapper):
        seed = rng(seed=seed)
    # ====== not a training variable NO dropout ====== #
    if not is_training(x):
        return x
    # ====== applying noise ====== #
    shape = x.shape
    noise_shape = (shape if noise_dims is None
                   else _process_noise_dim(shape, noise_dims))
    if 'normal' in noise_type or 'gaussian' in noise_type:
        noise = seed.normal(shape=noise_shape, mean=0.0, std=sigma, dtype=x.dtype)
    elif 'uniform' in noise_type:
        noise = seed.uniform(shape=noise_shape, low=-sigma, high=sigma, dtype=x.dtype)
        # no idea why uniform does not give any broadcastable dimensions
        if noise_dims is not None:
            broadcastable = [i for i, j in enumerate(noise_shape) if j == 1]
            if len(broadcastable) > 0:
                noise = addbroadcast(noise, *broadcastable)
    x = x + noise
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x
