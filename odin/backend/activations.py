from __future__ import print_function, division, absolute_import

from six.moves import builtins

import tensorflow as tf

from odin.autoconfig import randint
from odin.backend.helpers import *

def softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.expm1(x))

def relu(x, alpha=0., name='ReLu'):
  if alpha == 0.:
    return tf.nn.relu(x, name=name)
  else:
    with tf.variable_scope(name):
      # We can't use 0.5 and 1 for one and half.  as if alpha is a
      # numpy dtype, they will be considered as float64, so would
      # cause upcast to float64.
      alpha = tf.convert_to_tensor(x, name='alpha',
          dtype=x.dtype.base_dtype)
      f1 = 0.5 * (1 + alpha)
      f2 = 0.5 * (1 - alpha)
      return f1 * x + f2 * tf.abs(x)

def elu(x, alpha=1., name="ELu"):
  with tf.variable_scope(name):
    res = tf.nn.elu(x)
    if alpha != 1.:
      alpha = tf.convert_to_tensor(x, name='alpha',
          dtype=x.dtype.base_dtype)
      res = tf.where(x > 0, res, alpha * res)
    return res

def hard_sigmoid(x, name='HardSigmoid'):
  with tf.variable_scope(name):
    slope = tf.constant(0.2, dtype=x.dtype.base_dtype)
    shift = tf.constant(0.5, dtype=x.dtype.base_dtype)
    x = (x * slope) + shift
    x = tf.clip_by_value(x, 0., 1.)
    return x

def antirectify(x, name="AntiRectify"):
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
  with tf.variable_scope(name):
    if x.shape.ndims != 2:
      raise Exception('This Ops only support 2D input.')
    x -= tf.reduce_mean(x, axis=1, keepdims=True)
    # l2 normalization
    x /= tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))
    return tf.concat([relu(x, 0), relu(-x, 0)], axis=1)


def randrectify(x, lower=0.3, upper=0.8, shared_axes='auto', name="RandRectify"):
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
  ndims = x.shape.ndims
  # ====== check lower and upper ====== #
  if is_variable(lower):
    add_roles(lower, ActivationParameter)
  if is_variable(upper):
    add_roles(upper, ActivationParameter)
  if not is_tensor(lower > upper) and lower > upper:
    raise ValueError("Upper bound for Randomized Rectifier needs "
                     "to be higher than lower bound.")
  # ====== check shared_axes ====== #
  if shared_axes == 'auto':
    shared_axes = (0,) + tuple(range(2, ndims))
  elif shared_axes == 'all':
    shared_axes = tuple(range(ndims))
  elif isinstance(shared_axes, int):
    shared_axes = (shared_axes,)
  else:
    shared_axes = shared_axes
  # ====== main logic ====== #
  if not is_training() or upper == lower:
    x = relu(x, alpha=(upper + lower) / 2.0)
  else: # Training mode
    shape = list(input_shape)
    if builtins.any(s is None for s in shape):
      shape = list(x.shape)
    for ax in shared_axes:
      shape[ax] = 1
    rnd = tf.random_uniform(tuple(shape),
               minval=lower,
               maxval=upper,
               dtype=x.dtype.base_dtype,
               seed=randint())
    x = relu(x, rnd)
  return x
