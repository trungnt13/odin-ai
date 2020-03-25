from __future__ import absolute_import, division, print_function

import os
import unittest
from tempfile import mkstemp

import numpy as np

from odin.bay import RandomVariable

np.random.seed(8)


class RandomVariableTest(unittest.TestCase):

  def test_posterior(self):
    pass


if __name__ == '__main__':
  unittest.main()
