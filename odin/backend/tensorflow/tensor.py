from __future__ import division, absolute_import

import os
import math
import numbers
import cPickle
from collections import OrderedDict

import numpy as np

import tensorflow as tf

from odin.config import CONFIG, RNG_GENERATOR
from odin.utils import as_tuple, as_shape_tuple, dict_union
from odin.basic import (add_role, TRAINING, PARAMETER,
                        ACTIVATION_PARAMETER, DEPLOYING,
                        add_shape, get_shape)

from .helpers import (get_session, as_tensor_variable)
FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
NPROCESSORS = CONFIG['device_info']['n']
_RNG = np.random.RandomState(seed=RNG_GENERATOR.randint(10e8))


def eval(x):
    '''Evaluates the value of a tensor.
    Returns a Numpy array.
    '''
    return x.eval(session=get_session())

# if alpha == 0:
#     return 0.5 * (x + abs(x))
# else:
#     # We can't use 0.5 and 1 for one and half.  as if alpha is a
#     # numpy dtype, they will be considered as float64, so would
#     # cause upcast to float64.
#     alpha = tensor.as_tensor_variable(alpha)
#     f1 = 0.5 * (1 + alpha)
#     f2 = 0.5 * (1 - alpha)
#     return f1 * x + f2 * abs(x)


# ===========================================================================
# Basic ops
# ===========================================================================
def backend_ops_relu(x, alpha=0.):
    # Adapted implementation from theano
    if alpha == 0:
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


def backend_ops_eye(n, m, dtype):
    x = tf.Variable(initial_value=np.eye(n, m, dtype=dtype), dtype=dtype)
    get_session().run(x.initializer)
    return x

# Comparator
backend_ops_switch = tf.select
backend_ops_eq = tf.equal
backend_ops_neq = tf.not_equal
backend_ops_gt = tf.greater
backend_ops_ge = tf.greater_equal
backend_ops_lt = tf.less
backend_ops_le = tf.less_equal


# ===========================================================================
# LINEAR ALGEBRA
# Assumed overridden:
# +, -, /, *, +=, -=, *=, /=
# ===========================================================================
def dot(x, y):
    '''Multiplies 2 tensors.
    When attempting to multiply a ND tensor
    with a ND tensor, reproduces the Theano behavior
    (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
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
    output_shape = shapeX[:-1] + outshapeY
    output = tf.reshape(tf.matmul(x, y), output_shape)
    add_shape(output, output_shape)
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
    add_shape(output, output_shape)
    return output


def transpose(x, axes=None):
    '''Transposes a matrix.
    '''
    output_shape = get_shape(x)
    x = tf.transpose(x, perm=axes)
    if isinstance(output_shape, (tuple, list)):
        if axes is None:
            output_shape = output_shape[::-1]
        else:
            output_shape = [output_shape[i] for i in axes]
        add_shape(x, tuple(output_shape))
    return x


def gather(reference, indices):
    '''Retrieves the vectors of indices `indices`
    in the 2D tensor `reference`.

    # Arguments
        reference: a 2D tensor.
        indices: an int tensor of indices.

    # Returns
        A 3D tensor of same type as `reference`.
    '''
    return tf.gather(reference, indices)


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
    add_shape(x, shape)
    return x


def ones(shape, dtype=FLOATX, name=None):
    """Instantiate an all-ones variable.
    """
    x = tf.ones(shape, dtype=dtype, name=name)
    add_shape(x, shape)
    return x


def ones_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype.base_dtype
    input_shape = get_shape(x)
    x = tf.ones_like(x, dtype=dtype, optimize=True)
    add_shape(x, input_shape)
    return x


def zeros_like(x, dtype=None):
    if dtype is None:
        dtype = x.dtype.base_dtype
    input_shape = get_shape(x)
    x = tf.zeros_like(x, dtype=dtype, optimize=True)
    add_shape(x, input_shape)
    return x


def cast(x, dtype):
    if 'tensorflow.' in str(x.__class__):
        return tf.cast(x, dtype)
    return np.cast[dtype](x)


def castX(x):
    return cast(x, FLOATX)


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
    return tf.gradients(loss, variables=variables, colocate_gradients_with_ops=True)


def stop_gradient(vars):
    return tf.stop_gradient(vars)
