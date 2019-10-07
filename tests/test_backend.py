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
      int(i) == int(j) == int(k) for i, j, k in zip(a.shape, b.shape, c.shape)) \
        and np.all(np.logical_and(a == b.numpy(), a == c.numpy())),\
        "Input shape: %s, info: %s, output shapes mismatch: %s, %s and %s" % \
          (str(x.shape), str(info), str(a.shape), str(b.shape), str(c.shape))


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


test_reshape()
test_dimshuffle()
test_flatten()
