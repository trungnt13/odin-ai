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
            np.random.seed(1208251813)
            transcription_test = {}
            # ====== create fake dataset ====== #
            ds = F.Dataset(os.path.join(temppath, 'ds'))
            ds['X'] = np.arange(0, 10000).reshape(-1, 5)
            # generate fake indices
            indices = []
            for i, j in enumerate(range(0, ds['X'].shape[0], 20)):
                indices.append(['name_%d' % i, j, j + 20])
            np.savetxt(os.path.join(ds.path, 'indices.csv'), indices,
                       fmt='%s', delimiter=' ')
            # generate fake transcription
            transcription = F.MmapDict(os.path.join(ds.path, 'transcription.dict'))
            for name, start, end in indices:
                trans = np.random.randint(0, 4, size=(20,)).tolist()
                transcription[name] = trans
                for i, j in zip(ds['X'][start:end], trans):
                    transcription_test[str(i.tolist())] = j
            transcription.flush()
            transcription.close()
            ds.flush()
            ds.close()
            # ====== test feeder ====== #
            ds = F.Dataset(os.path.join(temppath, 'ds'), read_only=True)
            feeder = F.Feeder(ds['indices.csv'],
                              ncpu=2, buffer_size=2, maximum_queue_size=12)
            feeder.set_recipes([
                F.recipes.DataLoader(ds['X']),
                F.recipes.CreateBatch()
            ])
            # ==================== No recipes ==================== #
            # ====== NO shuffle ====== #
            n = 0
            for i in feeder.set_batch(12, seed=None, shuffle_level=0):
                x = i[1:] - i[:-1]
                self.assertTrue(np.all(x == 5)) # must always be True
                n += i.shape[0]
            self.assertEqual(n, 2000)
            # ====== shuffle 0 ====== #
            n = 0
            s = 0
            for i in feeder.set_batch(12, seed=1203, shuffle_level=0):
                x = i[1:] - i[:-1]
                s += np.sum(x)
                n += i.shape[0]
            # always equal to this value because we only shuffle the order
            # of the indices
            self.assertEqual(s, 177000)
            self.assertEqual(n, 2000)
            # ====== shuffle 2 ====== #
            n = 0
            for i in feeder.set_batch(12, seed=1203, shuffle_level=2):
                n += i.shape[0]
            self.assertEqual(n, 2000)
            # ==================== Convert indices ==================== #
            n = 0
            feeder.set_recipes([
                F.recipes.DataLoader(ds['X']),
                F.recipes.Name2Trans(
                    converter_func=lambda name, x: [int(name.split('_')[-1])] * x[0].shape[0]),
                F.recipes.CreateBatch()
            ])
            # ====== NO shuffle ====== #
            n = 0
            y = 0
            for i, j in feeder.set_batch(12, seed=None, shuffle_level=0):
                x = i[1:] - i[:-1]
                self.assertTrue(np.all(x == 5)) # must always be True
                n += i.shape[0]
                y += np.sum(j)
            self.assertEqual(n, 2000)
            self.assertEqual(y, 99000)
            # ====== shuffle 0 ====== #
            n = 0
            s = 0
            y = 0
            for i, j in feeder.set_batch(12, seed=1203, shuffle_level=0):
                x = i[1:] - i[:-1]
                s += np.sum(x)
                n += i.shape[0]
                y += np.sum(j)
            # always equal to this value because we only shuffle the order
            # of the indices
            self.assertEqual(s, 177000)
            self.assertEqual(y, 99000)
            self.assertEqual(n, 2000)
            # ====== shuffle 2 ====== #
            n = 0
            y = 0
            for i, j in feeder.set_batch(12, seed=1203, shuffle_level=2):
                n += i.shape[0]
                y += np.sum(j)
            self.assertEqual(n, 2000)
            self.assertEqual(y, 99000)
            # ==================== Transcription ==================== #
            del feeder
            ds = F.Dataset(os.path.join(temppath, 'ds'))
            feeder = F.Feeder(indices=ds['indices.csv'],
                              ncpu=2, buffer_size=2, maximum_queue_size=12)
            feeder.set_recipes([
                F.recipes.DataLoader(ds['X']),
                F.recipes.TransLoader(ds['transcription.dict'], dtype='int32'),
                F.recipes.CreateBatch()
            ])
            for i, j in feeder.set_batch(12, seed=1208251813, shuffle_level=2):
                for x, y in zip(i, j):
                    self.assertTrue(transcription_test[str(x.tolist())] == y)

    def test_dataset(self):
        pass


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
