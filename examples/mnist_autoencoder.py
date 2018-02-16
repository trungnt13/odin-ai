#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np

from odin.utils import get_modelpath, ArgController, stdio, get_logpath
stdio(get_logpath('mnist.log', override=True))

import os
os.environ['ODIN'] = 'float32,gpu,seed=12'

import tensorflow as tf

from odin import backend as K
from odin import nnet as N
from odin import fuel as F, training
from six.moves import cPickle

# ===========================================================================
# Load data
# ===========================================================================
ds = F.MNIST.get_dataset()
print(ds)
nb_classes = 10
X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X')
y = K.placeholder(shape=(None,), name='y', dtype='int32')
# ===========================================================================
# Build network
# ===========================================================================
ops = N.Sequence([
    N.Flatten(outdim=2, name='F0'),
    N.Dense(512, activation=tf.nn.relu, name='D1'),
    N.Dense(256, activation=tf.nn.relu, name='D2'),
    N.Dense(10, activation=K.linear, name='Out')
], debug=True, name="Net")
y_logit = ops(X)
y_prob = tf.nn.softmax(y_logit)
y_onehot = tf.one_hot(y, depth=nb_classes)
# ====== objectives ====== #
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_true=y, y_pred=y_prob)
cm = K.metrics.confusion_matrix(y_true=y, y_pred=y_prob, labels=nb_classes)
# ====== training ====== #
f_pred = training.train(X=X, y_true=y_onehot, y_pred=y_prob,
                        train_data=(ds['X_train'], ds['y_train']),
                        valid_data=(ds['X_valid'], ds['y_valid']),
                        objectives=loss,
                        metrics=(loss, acc, cm),
                        batch_size=256, epochs=8, verbose=4)
