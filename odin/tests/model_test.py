# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
import cPickle
from six.moves import zip, range

import numpy as np

from odin import backend as K
from odin import fuel
from odin import nnet as N
from odin import model
from odin.utils import get_file, TemporaryDirectory, urlretrieve


class ModelTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_complex_transform(self):
        with TemporaryDirectory() as temp:
            from sklearn.pipeline import Pipeline
            path = os.path.join(temp, 'audio.sph')
            urlretrieve(filename=path,
                        url='https://s3.amazonaws.com/ai-datasets/sw02001.sph')
            f = Pipeline([
                ('step1', model.SpeechTransform('mspec', fs=8000, vad=True)),
                ('step2', model.Transform(lambda x: (x[0][:, :40],
                                                     x[1].astype(str)))),
                ('step3', model.Transform(lambda x: (np.sum(x[0]),
                                                    ''.join(x[1].tolist()))))
            ])
            x = f.transform(path)
            f = cPickle.loads(cPickle.dumps(f))
            y = f.transform(path)
            self.assertEqual(x[0], y[0])
            self.assertEqual(y[0], -3444229.0)
            self.assertEqual(x[1], y[1])

    def test_mnist(self):
        ds = fuel.load_mnist()
        m = model.SequentialClassifier(
            N.FlattenRight(outdim=2),
            N.Dense(64, activation=K.relu),
            N.Dense(10, activation=K.softmax)
        )
        m.set_inputs(
            K.placeholder(shape=(None, 28, 28), name='X', dtype='float32')
        ).set_outputs(
            K.placeholder(shape=(None,), name='y', dtype='int32')
        )
        # ====== query info ====== #
        m.path
        self.assertEqual(m.is_initialized, True)
        self.assertEqual(m.input_shape, (None, 28, 28))
        self.assertEqual(m.output_shape, (None, 10))
        # ====== training test ====== #
        m.set_training_info(learning_rate=0.001, n_epoch=3)
        m.fit(X=(ds['X_train'],
                 ds['y_train']),
              X_valid=(ds['X_valid'],
                       ds['y_valid'])
        )
        score = m.score(ds['X_test'][:], ds['y_test'][:])
        self.assertEqual(score > 0.8, True,
                         msg='Test if the model get reasonable results: %f accuracy' % score)
        # ====== make prediction and transform test ====== #
        np.random.seed(12)
        _ = np.random.rand(8, 28, 28)
        self.assertEqual(m.transform(_).shape, (8, 10))
        self.assertEqual(np.isclose(m.predict_proba(_).sum(-1), 1.).sum() == 8,
                         True)
        self.assertEqual(len(m.predict(_)), 8)
        # ====== pickling test ====== #
        str_old = str(m)
        p_old = m.get_params(True)

        m = cPickle.loads(
            cPickle.dumps(m, protocol=cPickle.HIGHEST_PROTOCOL)
        )
        str_new = str(m)
        p_new = m.get_params(True)
        # ====== test same configurations ====== #
        self.assertEqual(str_new, str_old)
        # ====== test same params ====== #
        for i, j in p_new.iteritems():
            k = p_old[i]
            for a, b in zip(j, k):
                self.assertEqual(np.array_equal(a, b), True)
        # ====== test set params ====== #
        params = m.get_params(deep=True)
        params_new = {}
        for n, p in params.iteritems():
            params_new[n] = [np.random.rand(*i.shape).astype('float32')
                             for i in p]
        m.set_params(**params_new)
        # test if equal new onces
        for i, j in m.get_params(deep=True).iteritems():
            k = params_new[i]
            for a, b in zip(j, k):
                self.assertEqual(np.array_equal(a, b), True)
        # ====== training test ====== #
        print('Re-train the model second time:')
        m.fit(X=(ds['X_train'],
                 ds['y_train']),
              X_valid=(ds['X_valid'],
                       ds['y_valid'])
        )
        score = m.score(ds['X_test'][:], ds['y_test'][:])
        self.assertEqual(score > 0.8, True,
                         msg='Test if the model get reasonable results: %f accuracy' % score)


if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
