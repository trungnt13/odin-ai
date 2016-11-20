from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'theano,float32,cpu,seed=12082518'
import cPickle

import numpy as np

from odin import backend as K
from odin import nnet as N
from odin import utils as U

np.random.seed(1208)
X = K.placeholder((None, 28, 28, 1))
x = np.random.rand(12, 28, 28, 1)
K.set_training(True)


def create():
    f = N.Sequence([
        N.Conv(8, (3, 3), strides=1, pad='same'),
        N.Dimshuffle(pattern=(0, 3, 1, 2)),
        N.FlattenLeft(outdim=2),
        N.Noise(level=0.3, noise_dims=None, noise_type='gaussian'),
        N.Dense(128, activation=K.relu),
        N.Dropout(level=0.3, noise_dims=None),
        N.Dense(10, activation=K.softmax)
    ], debug=True)
    y = f(X)
    yT = f.T(y)
    f1 = K.function(X, y)
    f2 = K.function(X, yT)
    cPickle.dump(f, open(U.get_modelpath('dummy.ai', override=True), 'w'))

    _ = f1(x)
    print(_.shape, _.sum())
    _ = f2(x)
    print(_.shape, _.sum())


def load():
    f = cPickle.load(open(U.get_modelpath('dummy.ai'), 'r'))
    y = f(X)
    yT = f.T(y)
    f1 = K.function(X, y)
    f2 = K.function(X, yT)

    _ = f1(x)
    print(_.shape, _.sum())
    _ = f2(x)
    print(_.shape, _.sum())

# call create() then call load()
# create()
# load()
