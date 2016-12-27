from __future__ import print_function, division, absolute_import

import __builtin__
import platform

from odin.config import auto_config
from odin.basic import set_training, is_training
from odin.utils import package_installed, exec_commands

config = auto_config()

if config['backend'] == 'tensorflow':
    from .tensorflow import *
elif config['backend'] == 'theano':
    from .theano import *


def backend():
    return config['backend']


def floatX():
    return config['floatX']


def device():
    return config['device']


def cudnn_available():
    """ return True if running on GPU with cuDNN available """
    if config['device'] == 'gpu':
        # theano backend
        if config['backend'] == 'theano':
            try:
                if package_installed(name='pygpu'):
                    from theano.gpuarray import dnn
                    from theano.gpuarray.type import list_contexts
                    return dnn.dnn_available(list_contexts()[0])
                else:
                    from theano.sandbox.cuda import dnn
                    return dnn.dnn_available()
            except ImportError:
                return False
        # tensorflow backend
        else:
            import commands
            if platform.system() == "Darwin":
                x = commands.getstatusoutput('ls /usr/local/cuda/lib')
                x = x[-1].split('\n')
            elif platform.version() == "Windows":
                raise Exception('No support for Windows')
            else:
                x = commands.getstatusoutput('ldconfig -p')
                x = x[-1].split('=>')
            if __builtin__.any('libcudnn' in i for i in x):
                return True
            else:
                return False
    return False


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
    x = clip(x, as_tensor_variable(0., dtype=x.dtype),
             as_tensor_variable(np.inf, dtype=x.dtype))
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


def castX(x):
    return cast(x, FLOATX)
