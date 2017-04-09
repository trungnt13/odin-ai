from __future__ import print_function, division, absolute_import

import math
import platform
from six.moves import builtins

from odin.config import auto_config
from odin.basic import set_training, is_training, output_roles
from odin.utils import package_installed, exec_commands, is_number

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
            if builtins.any('libcudnn' in i for i in x):
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


def get_dtype(x, numpy=False, string=False):
    dtype = x.dtype
    if (numpy or string) and hasattr(dtype, 'as_numpy_dtype'):
        dtype = dtype.as_numpy_dtype
    # ====== convert to normalized string ====== #
    if string: dtype = np.dtype(dtype).name
    return dtype


# ==================== activations ==================== #
@output_roles
def relu(x, alpha=0.):
    return _copy_shape(x, backend_ops_relu, alpha)


@output_roles
def elu(x, alpha=1.0):
    """ Exponential linear unit """
    return _copy_shape(x, backend_ops_elu, alpha)


@output_roles
def softmax(x):
    return _copy_shape(x, backend_ops_softmax)


@output_roles
def softplus(x):
    return _copy_shape(x, backend_ops_softplus)


@output_roles
def softsign(x):
    return _copy_shape(x, backend_ops_softsign)


def linear(x):
    return x


@output_roles
def sigmoid(x):
    return _copy_shape(x, backend_ops_sigmoid)


@output_roles
def hard_sigmoid(x):
    return _copy_shape(x, backend_ops_hard_sigmoid)


@output_roles
def tanh(x):
    return _copy_shape(x, backend_ops_tanh)


# ==================== arthimetic ==================== #
@output_roles
def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    min_value = as_tensor_variable(min_value, dtype=get_dtype(x, numpy=True))
    max_value = as_tensor_variable(max_value, dtype=get_dtype(x, numpy=True))
    return _copy_shape(x, backend_ops_clip, min_value, max_value)


@output_roles
def square(x):
    if x.__class__.__name__ == 'IndexedSlices':
        x = add(x, 0.)
    return _copy_shape(x, backend_ops_square)


@output_roles
def abs(x):
    return _copy_shape(x, backend_ops_abs)


@output_roles
def inv(x):
    return _copy_shape(x, backend_ops_inv)


@output_roles
def sqrt(x):
    x = clip(x, 0., np.inf)
    return _copy_shape(x, backend_ops_sqrt)


@output_roles
def exp(x):
    return _copy_shape(x, backend_ops_exp)


@output_roles
def log(x):
    return _copy_shape(x, backend_ops_log)


@output_roles
def round(x):
    return _copy_shape(x, backend_ops_round)


@output_roles
def pow(x, a):
    return _copy_shape(x, backend_ops_pow, a)


@output_roles
def sign(x):
    return _copy_shape(x, backend_ops_sign)


@output_roles
def ceil(x):
    if is_number(x): return math.ceil(x)
    return _copy_shape(x, backend_ops_ceil)


@output_roles
def floor(x):
    if is_number(x): return math.floor(x)
    return _copy_shape(x, backend_ops_floor)


# ==================== others ==================== #
@output_roles
def diag(x):
    input_shape = get_shape(x)
    x = backend_ops_diag(x)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (builtins.min(input_shape),))
    return x


@output_roles
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
@output_roles
def neq(a, b):
    """a != b"""
    return _copy_shape(a, backend_ops_neq, b)


@output_roles
def eq(a, b):
    """a == b"""
    return _copy_shape(a, backend_ops_eq, b)


@output_roles
def gt(a, b):
    """a > b"""
    return _copy_shape(a, backend_ops_gt, b)


@output_roles
def ge(a, b):
    """a >= b"""
    return _copy_shape(a, backend_ops_ge, b)


@output_roles
def lt(a, b):
    """a < b"""
    return _copy_shape(a, backend_ops_lt, b)


@output_roles
def le(a, b):
    """a <= b"""
    return _copy_shape(a, backend_ops_le, b)


# ===========================================================================
# Graph creator helper
# ===========================================================================
def function(inputs, outputs, updates=[], **kwargs):
    # ====== check inputs ====== #
    if inputs is None or len(as_tuple(inputs)) == 0:
        inputs = ComputationGraph(outputs).inputs
        print("[WARNING] inputs haven't specified, auto-inferred from Graph of "
              "outputs, graph inputs: %s" % ', '.join([str(i) for i in inputs]))
    f = Function(inputs=inputs, outputs=outputs,
                 updates=updates, **kwargs)
    return f


@output_roles
def castX(x):
    return cast(x, FLOATX)
