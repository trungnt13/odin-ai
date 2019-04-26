from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin.utils import uuid, as_tuple, flatten_list, is_string, is_number
from odin.autoconfig import get_rng, CONFIG, randint
from odin.backend.helpers import is_training
from odin.backend.tensor import variable
from odin.backend.role import add_roles, Weight, Bias, Parameter

floatX = CONFIG.floatX

def random_binomial(shape, p, dtype=floatX, seed=None, name="RandomBinomal"):
  with tf.variable_scope(name):
    return tf.where(
        tf.random_uniform(shape, minval=0., maxval=1., dtype=dtype, seed=seed) <= p,
        tf.ones(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype))

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
  => Noise mask: (None, 10, 1)
  """
  if dims is None:
    return input_shape
  ndims = input_shape.shape[0].value
  dims = [i % ndims for i in as_tuple(dims, t=int)]
  # ====== get noise shape ====== #
  return tuple([1 if i in dims else input_shape[i]
                for i in range(ndims)])


def apply_dropout(x, level=0.5, noise_dims=None, noise_type='uniform',
                  rescale=True, name="ApplyDropout"):
  """Computes dropout.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.


  Parameters
  ----------
  x: A tensor.
      input tensor
  level: float(0.-1.)
      probability dropout values in given tensor
  noise_dims: int or list(int)
      these dimensions will be setted to 1 in noise_shape, and
      used to broadcast the dropout mask.
  noise_type: 'gaussian' (or 'normal'), 'uniform'
      distribution used for generating noise
  rescale: bool
      whether rescale the outputs by dividing the retain probablity
  seed: random seed or `tensor.rng`
      random generator from tensor class

  References
  ----------
  [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

  Note
  ----
  This function only apply noise on Variable when training is enable
  """
  shape = tf.shape(x)
  retain_prob = 1. - level
  # ====== not a training variable NO dropout ====== #
  if 'normal' in noise_type or 'gaussian' in noise_type:
    randfunc = lambda shape: tf.random_normal(shape=shape,
        mean=1.0, stddev=np.sqrt((1.0 - retain_prob) / retain_prob),
        dtype=x.dtype.base_dtype, seed=randint())
  elif 'uniform' in noise_type:
    randfunc = lambda shape: random_binomial(shape=shape,
        p=retain_prob, dtype=x.dtype.base_dtype, seed=randint())
  else:
    raise ValueError('No support for noise_type=' + noise_type)

  # ====== Dropout ====== #
  def training_fn():
    noise_shape = shape if noise_dims is None else \
        _process_noise_dim(shape, noise_dims)
    y = x * randfunc(shape=noise_shape)
    if rescale:
      y /= retain_prob
    return y

  def inference_fn():
    return x
  with tf.variable_scope(name):
    return tf.cond(is_training(), training_fn, inference_fn)


def apply_noise(x, level=0.075, noise_dims=None, noise_type='gaussian',
                name="ApplyNoise"):
  """
  Parameters
  ----------
  x: A tensor.
  level : float or tensor scalar
      Standard deviation of added Gaussian noise
  noise_dims: int or list(int)
      these dimensions will be setted to 1 in noise_shape, and
      used to broadcast the dropout mask.
  noise_type: 'gaussian' (or 'normal'), 'uniform'
      distribution used for generating noise
  seed: random seed or `tensor.rng`
      random generator from tensor class

  Note
  ----
  This function only apply noise on Variable when training is enable
  """
  noise_type = noise_type.lower()
  shape = tf.shape(x)

  # ====== applying noise ====== #
  def training_fn():
    noise_shape = (shape if noise_dims is None
                   else _process_noise_dim(shape, noise_dims))
    if 'normal' in noise_type or 'gaussian' in noise_type:
      noise = tf.random_normal(shape=noise_shape,
          mean=0.0, stddev=level, dtype=x.dtype.base_dtype, seed=randint())
    elif 'uniform' in noise_type:
      noise = tf.random_uniform(shape=noise_shape,
          minval=-level, maxval=level,
          dtype=x.dtype.base_dtype, seed=randint())
    else:
      raise ValueError('No support for noise_type=' + noise_type)
    return x + noise

  # ====== inference_fn ====== #
  def inference_fn():
    return x
  with tf.variable_scope(name):
    return tf.cond(is_training(), training_fn, inference_fn)
