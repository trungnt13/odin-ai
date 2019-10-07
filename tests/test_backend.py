from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import torch

from odin import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
torch.manual_seed(8)

# ===========================================================================
# Prepare
# ===========================================================================
x = np.random.rand(12, 25, 8).astype('float32')
y = torch.Tensor(x)
z = tf.convert_to_tensor(x)


# ===========================================================================
# Shape manipulation
# ===========================================================================
def _equal(info, a: np.ndarray, b: torch.Tensor, c: tf.Tensor):
  assert all(
      int(i) == int(j) == int(k) for i, j, k in zip(a.shape, b.shape, c.shape)),\
        "Input shape: %s, info: %s, output shapes mismatch: %s, %s and %s" % \
          (str(x.shape), str(info), str(a.shape), str(b.shape), str(c.shape))
  assert np.all(
    np.logical_and(np.allclose(a, b.numpy()), np.allclose(a, c.numpy()))),\
    "info: %s, output value mismatch, \n%s\n%s\n%s" % \
      (info, str(a[0]), str(b.numpy()[0]), str(c.numpy()[0]))


def test_reshape():

  def reshape_and_test(newshape):
    a = K.reshape(x, newshape)
    b = K.reshape(y, newshape)
    c = K.reshape(z, newshape)
    _equal(newshape, a, b, c)

  reshape_and_test((-1, 8))
  reshape_and_test((8, 12, 25))
  reshape_and_test((-1, [1]))
  reshape_and_test(([-1], -1))
  reshape_and_test(([-1], [1], -1))


def test_dimshuffle():

  def dimshuffle_and_test(pattern):
    a = K.dimshuffle(x, pattern)
    b = K.dimshuffle(y, pattern)
    c = K.dimshuffle(z, pattern)
    _equal(pattern, a, b, c)

  dimshuffle_and_test((0, 2, 1))
  dimshuffle_and_test((0, 2, 1, 'x'))
  dimshuffle_and_test((1, 0, 'x', 2))
  dimshuffle_and_test((1, 'x', 0, 'x', 2))
  dimshuffle_and_test(('x', 1, 'x', 0, 'x', 2, 'x'))


def test_flatten():

  def flatten_and_test(n):
    a = K.flatten(x, n)
    b = K.flatten(y, n)
    c = K.flatten(z, n)
    _equal(n, a, b, c)

  flatten_and_test(1)
  flatten_and_test(2)


# ===========================================================================
# Reduction
# ===========================================================================
def test_stats_and_reduce():
  for name, fn in (
      ("min_keepdims", lambda _: K.reduce_min(_, 1, keepdims=True)),
      ("min", lambda _: K.reduce_min(_, 1, keepdims=False)),
      ("max_keepdims", lambda _: K.reduce_max(_, 1, keepdims=True)),
      ("max", lambda _: K.reduce_max(_, 1, keepdims=False)),
      ("mean_keepdims", lambda _: K.reduce_mean(_, 1, keepdims=True)),
      ("mean", lambda _: K.reduce_mean(_, 1, keepdims=False)),
      ("var_keepdims", lambda _: K.reduce_var(_, 1, keepdims=True)),
      ("var", lambda _: K.reduce_var(_, 1, keepdims=False)),
      ("std_keepdims", lambda _: K.reduce_std(_, 1, keepdims=True)),
      ("std", lambda _: K.reduce_std(_, 1, keepdims=False)),
      ("sum_keepdims", lambda _: K.reduce_sum(_, 1, keepdims=True)),
      ("sum", lambda _: K.reduce_sum(_, 1, keepdims=False)),
      ("prod_keepdims", lambda _: K.reduce_prod(_, 1, keepdims=True)),
      ("prod", lambda _: K.reduce_prod(_, 1, keepdims=False)),
      ("all_keepdims", lambda _: K.reduce_all(_, 1, keepdims=True)),
      ("all", lambda _: K.reduce_all(_, 1, keepdims=False)),
      ("any_keepdims", lambda _: K.reduce_any(_, 1, keepdims=True)),
      ("any", lambda _: K.reduce_any(_, 1, keepdims=False)),
      ("logsumexp_keepdims", lambda _: K.reduce_logsumexp(_, 1, keepdims=True)),
      ("logsumexp", lambda _: K.reduce_logsumexp(_, 1, keepdims=False)),
  ):
    a = fn(x)
    b = fn(y)
    c = fn(z)
    _equal(name, a, b, c)

  a1, a2 = K.moments(x, axis=1)
  b1, b2 = K.moments(y, axis=1)
  c1, c2 = K.moments(z, axis=1)
  _equal("moments_mean", a1, b1, c1)
  _equal("moments_var", a2, b2, c2)


test_reshape()
test_dimshuffle()
test_flatten()
test_stats_and_reduce()
