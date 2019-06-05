# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range

import numpy as np

from odin import fuel as F, utils

test_speech_features = {
"mfcc_std": 182.59453,
"phase_sum2": 178783.16,
"qmfcc_std": 207.53845,
"energy": -91121.508,
"qmfcc_pca": 19.485082005965275,
"spec_sum1": -1.2290379e+08,
"energy_std": 2.7323465,
"qphase_sum2": 216875.97,
"mfcc_sum2": 1.1214674e+08,
"qphase_pca": 19.662871222598081,
"qmfcc_sum2": 2.1506418e+08,
"mfcc_sum1": 106488.96,
"qmspec": -45861200.0,
"mspec_std": 623.85437,
"qmspec_sum2": 1.9336586e+09,
"energy_pca": 2.9886061735628831,
"qmspec_sum1": -45861192.0,
"mspec_mean": -1632.179,
"qphase": 1751.0167,
"qmspec_pca": -6.6926648354591727,
"mspec_pca": 4.5802233044162932,
"pitch_sum2": 261.27863,
"phase_std": 25.289757,
"qspec_mean": -1766.666,
"spec_sum2": 4.5609349e+09,
"qspec_sum1": -57625108.0,
"mfcc": 106491.7,
"qspec_std": 1138.7697,
"phase": -1061.4377,
"qmfcc_mean": -94.225319,
"mspec": -53238400.0,
"phase_sum1": -1061.4102,
"mfcc_mean": 3.2647274,
"energy_mean": -2.7935946,
"pitch_sum1": 12.862646,
"qmfcc_sum1": -3073441.2,
"energy_sum1": -91121.461,
"mfcc_pca": 30.924762713486047,
"qmspec_mean": -1406.0087,
"mspec_sum2": 2.4402621e+09,
"pitch_mean": 0.00039434194,
"spec_mean": -3767.9749,
"phase_mean": -0.032540679,
"qphase_sum1": 1750.8629,
"pitch_std": 0.4675571,
"qphase_std": 23.571823,
"spec_pca": 13.403272873193941,
"qspec": -57625136.0,
"spec_std": 1837.2585,
"pitch_pca": 23.899668727039654,
"spec": -1.2290374e+08,
"qspec_pca": 8.3138976790867538,
"qmspec_std": 559.00739,
"mspec_sum1": -53238408.0,
"qmfcc": -3073441.2,
"qspec_sum2": 1.9416402e+09,
"energy_sum2": 444549.72,
"phase_pca": 26.387505780161359,
"qphase_mean": 0.05367782,
"pitch": 12.863468,
}


class FuelTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_feeder_recipes_shape(self):
        X = np.arange(0, 3600).reshape(-1, 3)
        indices = [("name" + str(i), j, j + 10) for i, j in enumerate(range(0, X.shape[0], 10))]
        vadids = [("name" + str(i), [(2, 7), (8, 10)]) for i in range(len(indices))]

        feeder = F.Feeder(X, indices, dtype='float32', ncpu=1,
                          buffer_size=2, maximum_queue_size=12)
        feeder.set_batch(batch_size=12, seed=None, shuffle_level=2)
        feeder.set_recipes([
            F.recipes.VADindex(vadids, frame_length=2, padding=None),
            F.recipes.Sequencing(frame_length=3, hop_length=2, end='cut'),
            F.recipes.Slice([slice(0, 3), 0, slice(10, 12)], axis=-1),
            F.recipes.Slice(0, axis=0),
            F.recipes.CreateFile()
        ])
        X = np.concatenate([x for x in feeder], axis=0)
        self.assertEqual(feeder.shape, X.shape)

    def test_speech_processor(self):
        try:
            datapath = F.load_digit_wav()
        except Exception as e:
            print('Error (skip this test):', str(e))
            return
        output_path = utils.get_datasetpath(name='digit', override=True)
        feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', sr_new=8000,
                         win=0.02, hop=0.01, nb_melfilters=40, nb_ceps=13,
                         get_delta=2, get_energy=True, pitch_threshold=0.8,
                         get_spec=True, get_mspec=True, get_mfcc=True,
                         get_pitch=True, get_vad=True,
                         save_stats=True, substitute_nan=None,
                         dtype='float32', datatype='memmap',
                         n_cache=0.12, ncpu=4)
        feat.run()
        ds = F.Dataset(output_path)

        def is_equal(x1, x2):
            x1 = repr(np.array(x1, 'float32').tolist())
            x2 = repr(np.array(x2, 'float32').tolist())
            n = 0
            for i, j in zip(x1, x2):
                if i == j:
                    n += 1
            return n >= max(len(x1), len(x2)) // 2
        # these numbers are highly numerical instable
        for i in ds.keys():
            if i == 'indices.csv':
                self.assertTrue(isinstance(ds[i], str))
            elif '_' not in i:
                pca = i + '_pca'
                if pca in ds:
                    self.assertTrue(
                        is_equal(np.sum(ds[i][:], dtype='float32'),
                        test_speech_features[i]))
            elif '_pca' not in i:
                self.assertTrue(
                    is_equal(np.sum(ds[i][:], dtype='float32'),
                     test_speech_features[i]))
            else:
                self.assertTrue(
                    is_equal(np.sum(ds[i].components_),
                     test_speech_features[i]))

    def test_feeders(self):
        with utils.TemporaryDirectory() as temppath:
            np.random.seed(1234)
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
            REF = ds['X'][:].ravel().tolist()
            feeder = F.Feeder(ds['X'], ds['indices.csv'],
                              ncpu=2, buffer_size=2)

            # ==================== No recipes ==================== #
            def test_iter_no_trans(it):
                x = []
                n = 0
                for i in it:
                    x += i.ravel().tolist()
                    n += i.shape[0]
                x = np.sort(x).tolist()
                self.assertEqual(x, REF)
                self.assertEqual(n, ds['X'].shape[0])
            # ====== NO shuffle ====== #
            test_iter_no_trans(feeder.set_batch(12, seed=None, shuffle_level=0))
            # ====== shuffle 0 ====== #
            test_iter_no_trans(feeder.set_batch(12, seed=1203, shuffle_level=0))
            # ====== shuffle 2 ====== #
            test_iter_no_trans(feeder.set_batch(12, seed=1203, shuffle_level=2))
            # ==================== Convert name to indices ==================== #
            feeder.set_recipes([
                F.recipes.Name2Label(converter_func=
                    lambda name: int(name.split('_')[-1])),
                F.recipes.CreateBatch()
            ])

            def test_iter_trans(it):
                x = []
                y = 0
                n = 0
                for i, j in it:
                    x += i.ravel().tolist()
                    n += i.shape[0]
                    y += np.sum(j)
                x = np.sort(x).tolist()
                self.assertEqual(x, REF)
                self.assertEqual(y, 99000)
                self.assertEqual(n, ds['X'].shape[0])
            # ====== NO shuffle ====== #
            test_iter_trans(feeder.set_batch(12, seed=None, shuffle_level=0))
            # ====== shuffle 0 ====== #
            test_iter_trans(feeder.set_batch(12, seed=1203, shuffle_level=0))
            # ====== shuffle 2 ====== #
            test_iter_trans(feeder.set_batch(12, seed=1203, shuffle_level=2))
            # ==================== Transcription ==================== #
            del feeder
            ds = F.Dataset(os.path.join(temppath, 'ds'))
            feeder = F.Feeder(ds['X'], indices=ds['indices.csv'],
                              ncpu=2, buffer_size=2)
            feeder.set_recipes([
                F.recipes.TransLoader(ds['transcription.dict'], dtype='int32'),
                F.recipes.CreateBatch()
            ])
            n = 0
            X = []
            for i, j in feeder.set_batch(12, seed=1234, shuffle_level=2):
                X += i.ravel().tolist()
                n += i.shape[0]
                for x, y in zip(i, j):
                    self.assertTrue(transcription_test[str(x.tolist())] == y)
            X = np.sort(X).tolist()
            self.assertEqual(X, REF)
            self.assertEqual(n, ds['X'].shape[0])

    def test_dataset(self):
        pass


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
