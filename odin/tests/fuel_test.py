# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range

import numpy as np

from odin import fuel as F, utils


class FuelTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_feeders(self):
        with utils.TemporaryDirectory() as temppath:
            ds = F.Dataset(os.path.join(temppath, 'ds'))
            ds['X'] = np.arange(0, 1000).reshape(-1, 5)
            for i in range(0, 1000, 20):
                pass

    def test_dataset(self):
        pass


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
