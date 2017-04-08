# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import unittest
from six.moves import zip, range

import numpy as np

from odin.basic import add_updates, add_auxiliary_variable, add_role, Auxiliary
from odin import backend as K
from odin import nnet as N
from odin.utils import segment_list


class BackendTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_shape(self):
        var = K.variable(np.random.rand(8, 12))
        inp = K.placeholder((None, 1, 20))
        self.assertEquals(K.get_shape(var), (8, 12))
        self.assertEquals(K.get_shape(inp), (None, 1, 20))

    def test_ops(self):
        x = K.variable(np.random.rand(8, 12))
        y = K.variable(np.random.rand(12, 25))
        z = K.placeholder((25, 18, 13))
        w = K.placeholder((18, 18))

        # ====== dot ====== #
        t = K.dot(x, y)
        self.assertEquals(K.get_shape(t), (8, 25))
        self.assertEquals(K.get_shape(t), K.eval(t).shape)
        t = K.dot(t, K.dimshuffle(z, (1, 0, 2)))
        self.assertEquals(K.get_shape(t), (8, 18, 13))

        # ====== transpose ====== #
        self.assertEquals(K.get_shape(K.transpose(z)), (13, 18, 25))
        self.assertEquals(K.get_shape(K.transpose(t, axes=(2, 0, 1))),
                         (13, 8, 18))

        # ====== eye ====== #
        self.assertEquals(K.get_shape(K.eye(5)),
                          K.eval(K.eye(5)).shape)
        # ====== diag ====== #
        self.assertEquals(K.get_shape(K.diag(w)), (18,))
        # self.assertEquals(K.get_shape(K.diag(x)),
        # K.eval(K.diag(y)).shape)
        self.assertEquals(K.get_shape(K.square(x)),
                          K.eval(K.square(x)).shape)
        self.assertEquals(K.get_shape(K.abs(x)),
                          K.eval(K.abs(x)).shape)
        self.assertEquals(K.get_shape(K.sqrt(x)),
                          K.eval(K.sqrt(x)).shape)
        self.assertEquals(K.get_shape(K.exp(x)),
                          K.eval(K.exp(x)).shape)
        self.assertEquals(K.get_shape(K.log(x)),
                          K.eval(K.log(x)).shape)
        self.assertEquals(K.get_shape(K.round(x)),
                          K.eval(K.round(x)).shape)
        self.assertEquals(K.get_shape(K.pow(x, 2)),
                          K.eval(K.pow(x, 2)).shape)
        self.assertEquals(K.get_shape(K.clip(x, -1, 1)),
                          K.eval(K.clip(x, -1, 1)).shape)
        self.assertEquals(K.get_shape(K.inv(x)),
                          K.eval(K.inv(x)).shape)

    def test_auto_infer_shape(self):
        x = K.variable(np.random.rand(8, 25, 12))
        y = K.placeholder((None, 25, 12))

        def test_func(func):
            self.assertEquals(K.get_shape(func(x, 0)),
                              K.eval(func(x, 0)).shape)
            self.assertEquals(K.get_shape(func(x, -1)),
                              K.eval(func(x, -1)).shape)
            self.assertEquals(K.get_shape(func(x, 1, True)),
                              K.eval(func(x, 1, True)).shape)

            self.assertEquals(K.get_shape(func(x, 0)),
                              K.get_shape(func(y, 0)))
            self.assertEquals(K.get_shape(func(x, 0, True)),
                              K.get_shape(func(y, 0, True)))

            if func != K.argmax and func != K.argmin:
                self.assertEquals(K.get_shape(func(x, (1, -1))),
                                  K.eval(func(x, (1, -1))).shape)
                self.assertEquals(K.get_shape(func(x, (0, 1))),
                                  K.eval(func(x, (0, 1))).shape)
                self.assertEquals(K.get_shape(func(x, (0, 1), True)),
                                  K.eval(func(x, (0, 1), True)).shape)

        test_func(K.var)
        test_func(K.max)
        test_func(K.min)
        test_func(K.any)
        test_func(K.sum)
        test_func(K.prod)
        test_func(K.mean)
        test_func(K.std)
        test_func(K.any)
        test_func(K.argmax)
        test_func(K.argmin)

        self.assertEquals(K.get_shape(K.argsort(x)),
                          K.eval(K.argsort(x)).shape)

    def test_basic_ops_value(self):
        np.random.seed(12082518)
        x = K.variable(np.random.randn(8, 8))
        y = K.variable(np.random.randn(8, 8))
        z = K.variable(np.random.randint(0, 2, size=(8, 8)),
                       dtype=np.bool)
        w = K.variable(np.random.randint(0, 2, size=(8, 8)),
                       dtype=np.bool)

        self.assertEqual(round(np.sum(K.eval(K.relu(x, alpha=0.12))) * 10000), 276733)
        self.assertEqual(round(np.sum(K.eval(K.elu(x, alpha=0.12))) * 10000), 289202)
        self.assertEqual(np.sum(K.eval(K.softmax(x))), 8.0)
        self.assertEqual(round(np.sum(K.eval(K.softplus(x))) * 10000), 554564)
        self.assertEqual(round(np.sum(K.eval(K.softsign(x))) * 100000), 211582)
        self.assertEqual(round(np.sum(K.eval(K.sigmoid(x))) * 10000), 330427)
        self.assertEqual(round(np.sum(K.eval(K.hard_sigmoid(x))) * 10000), 330836)
        self.assertEqual(round(np.sum(K.eval(K.tanh(x))) * 100000), 290165)
        self.assertEqual(round(np.sum(K.eval(K.square(x))) * 10000), 744492)
        self.assertEqual(round(np.sum(K.eval(K.sqrt(x))) * 10000), 300212)
        self.assertEqual(round(np.sum(K.eval(K.abs(x))) * 10000), 559979)
        self.assertEqual(np.sum(K.eval(K.sign(x))), 6.0)
        self.assertEqual(round(np.sum(K.eval(K.inv(x))) * 1000), 495838)
        self.assertEqual(round(np.sum(K.eval(K.exp(x))) * 1000), 122062)
        self.assertEqual(round(np.sum(K.eval(K.log(K.abs(x)))) * 10000), -344491)
        self.assertEqual(np.sum(K.eval(K.round(x))), 5.0)
        self.assertEqual(round(np.sum(K.eval(K.pow(x, 8))) * 100), 398153)
        self.assertEqual(round(np.sum(K.eval(K.clip(x, -0.12, 0.12))) * 1000000), 620529)
        # TODO: pygpu (libgpuarray) still not support diag
        # self.assertEqual(round(np.sum(K.eval(K.diag(x))) * 100000), 325289)
        self.assertEqual(np.sum(K.eval(K.eye(12, 8))), 8.0)

        self.assertEqual(np.sum(K.eval(K.eq(z, w))), 38)
        self.assertEqual(np.sum(K.eval(K.neq(z, w))), 26)
        self.assertEqual(np.sum(K.eval(K.gt(x, y))), 33)
        self.assertEqual(np.sum(K.eval(K.ge(x, y))), 33)
        self.assertEqual(np.sum(K.eval(K.lt(x, y))), 31)
        self.assertEqual(np.sum(K.eval(K.le(x, y))), 31)
        self.assertEqual(round(np.sum(K.eval(K.switch(z, x, y))) * 100000), 139884)

    def test_linear_algebra_value(self):
        np.random.seed(1208)
        x = K.variable(np.random.randn(2, 4, 3))
        y = K.variable(np.random.rand(1, 2, 3, 5))

        z = K.dot(x, y)
        self.assertEqual(K.get_shape(z), (2, 4, 1, 2, 5))
        self.assertEqual(repr(np.sum(K.eval(z)))[:8],
                         "-1.0198305134529524"[:8])

        np.random.seed(1208)
        x = K.variable(np.random.randn(100, 3, 4, 5))
        y = K.variable(np.random.rand(100, 12, 5, 6))
        z = K.batched_dot(x, y)
        self.assertEqual(K.get_shape(z), K.eval(z).shape)
        self.assertEqual(repr(K.eval(z).sum())[:7],
                         "1655.44")

    def test_simple_ops_shape(self):
        x = K.variable(np.random.rand(25, 8, 12))
        y = K.variable(18)
        z = K.variable(np.random.rand(25, 8, 12))
        v = K.variable(np.random.rand(12, 8))
        w = K.variable(np.random.rand(1, 12))
        w = K.addbroadcast(w, 0)

        def test_func(x, y, func):
            self.assertEquals(K.get_shape(func(x, y)),
                              K.eval(func(x, y)).shape)
        test_func(x, y, K.add)
        test_func(x, y, K.sub)
        test_func(x, y, K.mul)
        test_func(x, y, K.div)
        test_func(x, y, K.mod)

        test_func(x, w, K.add)
        test_func(x, w, K.sub)
        test_func(x, w, K.mul)
        test_func(x, w, K.div)
        test_func(x, w, K.mod)

        test_func(x, z, K.minimum)
        test_func(x, z, K.maximum)

        # test_func(x, z, K.concatenate)
        test_func(x, z, lambda *x: K.stack(x))

        test_func(v, v, K.categorical_crossentropy)

    def test_complex_ops_shape(self):
        x = K.variable(np.random.rand(25, 8, 12))
        y = K.variable(np.random.rand(8, 12))

        def test_func(x, func, *args, **kwargs):
            self.assertEquals(K.get_shape(func(x, *args, **kwargs)),
                              K.eval(func(x, *args, **kwargs)).shape)

        test_func(x, K.reverse, 0)
        test_func(x, K.reverse, -1)
        test_func(x, K.repeat, 2, -1)
        test_func(x, K.dimshuffle, (2, 0, 1))
        test_func(x, K.expand_dims, 1)
        test_func(x, K.pad, [[0, 0], [2, 1], [3, 0]], 'constant')
        test_func(x, K.reshape, (-1, 12))

        test_func(y, K.antirectify)
        test_func(y, K.randrectify, 0.3, 0.8, 'auto')
        test_func(x, K.elu, 1.0)
        test_func(x, K.relu, 0.)
        test_func(x, K.tanh)
        test_func(x, K.softplus)
        test_func(y, K.softmax)
        test_func(x, K.softsign)
        test_func(x, K.linear)
        test_func(x, K.sigmoid)
        test_func(x, K.hard_sigmoid)

    def test_computational_graph1(self):
        X = K.placeholder(shape=(None, 32), name='input')
        z = K.variable(np.random.rand(10, 10), name='z')
        f = N.Sequence([
            N.Dense(16, activation=K.relu),
            N.Dense(8, activation=K.softmax)
        ])
        y = f(X)
        add_auxiliary_variable(y, K.constant(10, name='aux_const'))
        add_updates(y, z, z * 2)

        tmp = K.ComputationGraph(y)
        self.assertEqual(len(tmp.placeholders), 1)
        self.assertEqual(len(tmp.trainable_variables), 4)
        self.assertEqual(len(tmp.parameters), 4)
        self.assertEqual(len(tmp.dict_of_placeholders), 1)
        self.assertEqual(len(tmp.auxiliary_variables), 1)
        tmp.intermediary_variables # no idea how to test this
        self.assertEqual(len(tmp.updates), 1)
        self.assertEqual(K.ComputationGraph(y), tmp)

    def test_computational_graph2(self):
        np.random.seed(1208)

        X = K.variable(np.zeros((8, 12)), name='X')
        Y = K.variable(np.random.rand(12, 8), name='Y')
        Z = K.placeholder(shape=(8, 8), name='Z')
        a = K.dot(X, Y)
        add_role(a, Auxiliary)
        add_updates(a, X, X + 12)
        a = a + Z
        g1 = K.ComputationGraph(a)

        self.assertEqual(len(g1.trainable_variables), 2)
        self.assertEqual(len(g1.placeholders), 1)
        self.assertEqual(len(g1.updates), 1)
        self.assertEqual(len(g1.auxiliary_variables), 1)

        f = K.function(Z, [a] + g1.auxiliary_variables)

        output = f(np.random.rand(8, 8))
        self.assertEqual(repr(np.sum(output[0]))[:5], "32.20")
        self.assertEqual(np.sum(output[1]), 0)
        self.assertEqual(np.unique(K.eval(X)).tolist(), [12.])

    def test_computational_graph3(self):
        # validate the number of updates found by ComputationGraph
        X = K.placeholder(shape=(None, 28, 28, 3))
        f = N.Sequence([
            N.Conv(32, 3, pad='same', activation=K.linear),
            N.BatchNorm(activation=K.relu),
            N.Flatten(outdim=2),
            N.Dense(16),
            N.BatchNorm(),
            N.Dense(10)
        ])
        K.set_training(True); y_train = f(X)
        K.set_training(False); y_score = f(X)
        self.assertTrue(K.get_shape(y_train) == K.get_shape(y_score) and
                        K.get_shape(y_score) == (None, 10))
        cc_train = K.ComputationGraph(y_train)
        cc_score = K.ComputationGraph(y_score)
        self.assertTrue(len(cc_score.updates) == 0)
        self.assertTrue(len(cc_train.updates) == 4)
        # create real function
        fn_train = K.function(X, y_train)
        fn_score = K.function(X, y_score)
        shape1 = fn_train(np.random.rand(12, 28, 28, 3)).shape
        shape2 = fn_score(np.random.rand(12, 28, 28, 3)).shape
        self.assertTrue(shape1 == shape2 and
                        shape1 == (12, 10))

    def test_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        y1 = np.random.randint(0, 8, size=100)
        y2 = np.random.randint(0, 8, size=100)
        y_pred = K.variable(y1)
        y_true = K.variable(y2)
        confusion = K.confusion_matrix(y_pred, y_true)

        r1 = K.eval(confusion)
        r2 = confusion_matrix(y1, y2)
        self.assertEqual(np.sum(r1 - r2), 0.)

    def test_rnn_decorator(self):
        @K.rnn_decorator(sequences='X', states='out')
        def rnn(X, out):
            return K.relu(X + out)

        y = rnn(K.ones(shape=(25, 12, 18, 8)),
                K.zeros(shape=(25, 18, 8))
        )
        f = K.function([], y)
        self.assertEqual(f()[0].shape, (25, 12, 18, 8))

    def test_segments_list(self):
        for i in np.random.randint(120, 12082518, size=12):
            tmp = segment_list(list(range(i)), n_seg=12)
            # must be 12 segments
            self.assertEqual(len(tmp), 12)
            # sum of all segments equal to original length
            self.assertEqual(sum(len(j) for j in tmp), i)
            # none of the segments length = 0
            self.assertTrue(all(len(j) > 0 for j in tmp))

    def test_randomization(self):
        # same rng generate the same random
        K.set_rng(12)
        a1 = K.eval(K.random_normal(shape=(10, 10), mean=0, std=1))
        b1 = K.eval(K.random_uniform(shape=(10, 10), low=-1, high=1))
        c1 = K.eval(K.random_binomial(shape=(10, 10), p=0.7))

        K.set_rng(12)
        a2 = K.eval(K.random_normal(shape=(10, 10), mean=0, std=1))
        b2 = K.eval(K.random_uniform(shape=(10, 10), low=-1, high=1))
        c2 = K.eval(K.random_binomial(shape=(10, 10), p=0.7))

        self.assertTrue(np.array_equal(a1, a2))
        self.assertTrue(np.array_equal(b1, b2))
        self.assertTrue(np.array_equal(c1, c2))

        self.assertTrue(np.abs(np.mean(a1)) < 0.1)
        self.assertTrue(np.std(a1) >= 0.8 and np.std(a1) <= 1.2)
        self.assertTrue(np.sum(c1) <= 80)
        # direct random generation
        a = K.eval(K.random_normal(shape=(10, 10)))
        b = K.eval(K.random_uniform(shape=(10, 10)))
        c = K.eval(K.random_binomial(shape=(10, 10), p=0.1))
        self.assertTrue(np.sum(c) <= 20)

    def test_flatten(self):
        x = K.placeholder(shape=(None, 8, 12, 25, 18))
        for i in range(1, 5):
            y = K.flatten(x, outdim=i)
            f = K.function(x, y)
            shape1 = K.get_shape(y)
            shape2 = f(np.random.rand(16, 8, 12, 25, 18)).shape
            self.assertEqual(len(shape1), len(shape2))
            self.assertTrue(all(i == j for i, j in zip(shape1, shape2)
                                if i is not None))

    def test_upsample(self):
        X = K.variable(np.arange(1, 24 + 1).reshape(2, 2, 3, 2))
        self.assertEqual(K.eval(K.sum(X)), 300.)
        self.assertEqual(K.eval(K.upsample(X, 2, axes=(1, 2), method='nn')).sum(),
                         1200.)
        self.assertEqual(K.eval(K.upsample(X, 2, axes=(1, 2), method='pad_margin')).sum(),
                         300.)
        self.assertEqual(K.eval(K.upsample(X, 2, axes=(1, 2), method='repeat')).sum(),
                         1200.)
        # print(K.eval(K.upsample(X, 2, axes=(1, 2), method='pad')).sum())

if __name__ == '__main__':
    print(' odin.tests.run() to run these tests ')
