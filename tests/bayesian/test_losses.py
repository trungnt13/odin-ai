from __future__ import absolute_import, division, print_function

import os
import unittest
from tempfile import mkstemp

import numpy as np
import tensorflow as tf

from odin.bay.vi import losses

np.random.seed(1)
tf.random.set_seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class BayesianLossesTest(unittest.TestCase):

  def test_pairwise_distances(self):
    for shape1, shape2, output_shape in [
        [(2, 5), (3, 5), (2, 3, 5)],
        [(3, 5), (2, 5), (3, 2, 5)],
        [(4, 3, 5), (2, 5), (4, 3, 2, 5)],
        [(4, 3, 5), (1, 2, 5), (4, 3, 1, 2, 5)],
    ]:
      x = tf.random.uniform(shape1)
      y = tf.random.uniform(shape2)
      z = losses.pairwise_distances(x, y)
      self.assertEqual(output_shape, z.shape)


if __name__ == '__main__':
  unittest.main()
