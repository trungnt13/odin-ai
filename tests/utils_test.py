# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import unittest

import numpy as np

from odin.utils.mpi import MPI
from odin.utils import batching
from odin.utils import async, async_mpi, UnitTimer


class UtilsTest(unittest.TestCase):

  def test_async_task(self):
    def test_fn(idx):
      with open('/tmp/tmp%d.txt' % idx, 'w') as f:
        r = 0
        for i in range(idx, idx + 25000):
          j = i**2 - i**3 + i**4 - i**5 + \
              i**6 - i**7 + i**8
          r += j
          f.write(str(j))
      print("Finish:", idx)
      return r

    thread = async(test_fn)
    mpi = async_mpi(test_fn)

    with UnitTimer(name="Sequential"):
      tmp = [test_fn(i) for i in range(8)]

    with UnitTimer(name="Threading"):
      tmp1 = [thread(i) for i in range(8)]
      while not all(i.finished for i in tmp1):
        pass
      result1 = [i.get() for i in tmp1]
    print(tmp1[0])

    with UnitTimer(name="Processing"):
      tmp2 = [mpi(i) for i in range(8)]
      while not all(i.finished for i in tmp2):
        pass
      result2 = [i.get() for i in tmp2]
    print(tmp2[0])
    self.assertTrue(all(i == j for i, j in zip(result1, result2)))

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
