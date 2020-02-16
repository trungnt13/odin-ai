from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from odin.bay import vi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(8)
tf.random.set_seed(8)


class VAETest(unittest.TestCase):

  def test_permute_dims(self):
    x = tf.reshape(tf.range(8), (4, 2))
    z = vi.permute_dims(x)
    w = tf.convert_to_tensor([[2, 5], [4, 7], [6, 3], [0, 1]], dtype=tf.int32)
    self.assertTrue(np.all(z.numpy() == w.numpy()))

    x = tf.random.uniform((128, 64), dtype=tf.float64)
    z = vi.permute_dims(x)
    self.assertTrue(np.any(x.numpy() != z.numpy()))
    self.assertTrue(np.all(np.any(i != j) for i, j in zip(x, z)))
    self.assertTrue(
        all(i == j for i, j in zip(sorted(x.numpy().ravel()),
                                   sorted(z.numpy().ravel()))))


if __name__ == '__main__':
  unittest.main()
