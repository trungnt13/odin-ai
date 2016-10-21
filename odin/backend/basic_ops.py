from __future__ import print_function, division, absolute_import

import __builtin__

from odin.config import auto_config
config = auto_config()

if config['backend'] == 'theano':
    from .theano import *
elif config['backend'] == 'tensorflow':
    from .tensorflow import *


# ===========================================================================
# Some useful general helper
# ===========================================================================
def _copy_shape(input, ops, *args, **kwargs):
    """ Simple shortcut, perfrom the ops on input, then,
    copy the shape of input to output """
    input_shape = get_shape(input)
    output = ops(input, *args, **kwargs)
    add_shape(output, input_shape)
    return output


def ndim(x):
    if hasattr(x, 'ndim'):
        return x.ndim
    else:
        return x.get_shape().ndims


# ==================== activations ==================== #
def relu(x, alpha=0.):
    return _copy_shape(x, backend_ops_relu, alpha)


def elu(x, alpha=1.0):
    """ Exponential linear unit """
    return _copy_shape(x, backend_ops_elu, alpha)


def softmax(x):
    return _copy_shape(x, backend_ops_softmax)


def softplus(x):
    return _copy_shape(x, backend_ops_softplus)


def softsign(x):
    return _copy_shape(x, backend_ops_softsign)


def linear(x):
    return x


def sigmoid(x):
    return _copy_shape(x, backend_ops_sigmoid)


def hard_sigmoid(x):
    return _copy_shape(x, backend_ops_hard_sigmoid)


def tanh(x):
    return _copy_shape(x, backend_ops_tanh)


# ==================== arthimetic ==================== #
def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return _copy_shape(x, backend_ops_clip, min_value, max_value)


def square(x):
    return _copy_shape(x, backend_ops_square)


def abs(x):
    return _copy_shape(x, backend_ops_abs)


def inv(x):
    return _copy_shape(x, backend_ops_inv)


def sqrt(x):
    x = clip(x, 0., np.inf)
    return _copy_shape(x, backend_ops_sqrt)


def exp(x):
    return _copy_shape(x, backend_ops_exp)


def log(x):
    return _copy_shape(x, backend_ops_log)


def round(x):
    return _copy_shape(x, backend_ops_round)


def pow(x, a):
    return _copy_shape(x, backend_ops_pow, a)


def sign(x):
    return _copy_shape(x, backend_ops_sign)


# ==================== others ==================== #
def diag(x):
    input_shape = get_shape(x)
    x = backend_ops_diag(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (__builtin__.min(input_shape),))
    return x


def eye(n, m=None, dtype=FLOATX):
    """ Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
      Number of rows in the output.
    m : int, optional
      Number of columns in the output. If None, defaults to `N`.
    dtype : data-type, optional
      Data-type of the returned array.

    """
    x = backend_ops_eye(n, m, dtype=dtype)
    add_shape(x, (n, n if m is None else m))
    return x


# ==================== comparators ==================== #
def switch(condition, then_expression, else_expression):
    return _copy_shape(condition, backend_ops_switch,
                       then_expression, else_expression)


def neq(a, b):
    """a != b"""
    return _copy_shape(a, backend_ops_neq, b)


def eq(a, b):
    """a == b"""
    return _copy_shape(a, backend_ops_eq, b)


def gt(a, b):
    """a > b"""
    return _copy_shape(a, backend_ops_gt, b)


def ge(a, b):
    """a >= b"""
    return _copy_shape(a, backend_ops_ge, b)


def lt(a, b):
    """a < b"""
    return _copy_shape(a, backend_ops_lt, b)


def le(a, b):
    """a <= b"""
    return _copy_shape(a, backend_ops_le, b)


# ===========================================================================
# Graph creator helper
# ===========================================================================
def function(inputs, outputs, updates=[]):
    f = Function(inputs, outputs, updates=updates)
    return f
