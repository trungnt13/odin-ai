# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import unittest

import numpy as np

from odin.utils.mpi import MPI
from odin.utils import batching


class UtilsTest(unittest.TestCase):

  def test_mpi(self):
    X = batching(n=512, batch_size=np.random.randint(low=12000, high=80000))

    def map_func(batch):
      for b in batch:
        yield b
    mpi = MPI(X, map_func=map_func, ncpu=12, buffer_size=8,
        maximum_queue_size=12 * 8)

    Y = [i for i in mpi]
    self.assertEqual(len(X), len(Y))
    self.assertEqual(sum(j - i for i, j in X), sum(j - i for i, j in Y))
    self.assertTrue(all(i == j for i, j in zip(
        sorted(X, key=lambda x: x[0]),
        sorted(Y, key=lambda x: x[0])
    )))

if __name__ == '__main__':
  print(' odin.tests.run() to run these tests ')
