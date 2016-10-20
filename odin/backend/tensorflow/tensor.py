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

FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
NPROCESSORS = CONFIG['device_info']['n']
_RNG = np.random.RandomState(seed=RNG_GENERATOR.randint(10e8))

# with tf.Session() as sess:
#   with tf.device("/gpu:1"):
#     matrix1 = tf.constant([[3., 3.]])
#     matrix2 = tf.constant([[2.],[2.]])
#     product = tf.matmul(matrix1, matrix2


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
    if alpha != 0.:
        negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if alpha != 0.:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x

def backend_ops_elu(x, alpha):
    res = tf.nn.elu(x)
    if alpha != 1:
        res = tf.select(x > 0, res, alpha * res)
    return res
# backend_ops_softmax = T.nnet.softmax
# backend_ops_softplus = T.nnet.softplus
# backend_ops_softsign = T_softsign
# backend_ops_sigmoid = T.nnet.sigmoid
# backend_ops_hard_sigmoid = T.nnet.hard_sigmoid
# backend_ops_tanh = T.tanh

# backend_ops_square = T.sqr
# backend_ops_abs = T.abs_
# backend_ops_inv = T.inv
# backend_ops_sqr = T.sqr
# backend_ops_sqrt = T.sqrt
# backend_ops_exp = T.exp
# backend_ops_log = T.log
# backend_ops_round = T.round
# backend_ops_pow = T.pow
# backend_ops_clip = T.clip
