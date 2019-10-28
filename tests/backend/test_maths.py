from __future__ import absolute_import, division, print_function

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
from tempfile import mkstemp

import numpy as np

from odin import backend as bk
from tests.backend.utils import assert_equal, x, y, z

np.random.seed(8)

FRAMEWORKS = ('numpy', 'torch', 'tensorflow')


class BackendMathTest(unittest.TestCase):

  def test_matmul(self):
    for shape1, shape2, outshape in [
        [(2, 3), (4, 3, 5), (4, 2, 5)],
        [(2, 3, 4), (4, 5), (2, 3, 5)],
        [(5, 3, 4), (5, 4, 6), (5, 3, 6)],
    ]:
      x = np.random.rand(*shape1)
      y = np.random.rand(*shape2)
      for fw in FRAMEWORKS:
        a = bk.array(x, fw)
        b = bk.array(y, fw)
        c = bk.matmul(a, b)
        self.assertEqual(c.shape, outshape, msg=fw)

  def test_norm(self):
    for p in [1, 2, 'fro', np.inf]:
      for axis in [None, 0, 1, (0, 1)]:
        a = bk.norm(bk.flatten(x, 2), p=p, axis=axis, keepdims=True)
        b = bk.norm(bk.flatten(y, 2), p=p, axis=axis, keepdims=True)
        c = bk.norm(bk.flatten(z, 2), p=p, axis=axis, keepdims=True)
        assert_equal(self, (p, axis), a, b, c)

        a = bk.norm(bk.flatten(x, 2), p=p, axis=axis, keepdims=False)
        b = bk.norm(bk.flatten(y, 2), p=p, axis=axis, keepdims=False)
        c = bk.norm(bk.flatten(z, 2), p=p, axis=axis, keepdims=False)
        assert_equal(self, (p, axis), a, b, c)


if __name__ == '__main__':
  unittest.main()
