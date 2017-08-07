# POSTER for the digisami datasets, for publishing the datasets (Deadline: 30 of July)
# Detail of transportation and accomodation.
# If there is big change in the laughter density, it is sign for changing the topic

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
from odin import fuel, training
from six.moves import cPickle

# ===========================================================================
# Load data
# ===========================================================================
ds = fuel.load_mnist()
print(ds)
X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X')
y = K.placeholder(shape=(None,), name='y', dtype='int32')

# ===========================================================================
# Build network
# ===========================================================================
ops = N.Sequence([
    N.Dimshuffle((0, 1, 2, 'x')),
    N.BatchNorm(axes='auto'),
    N.Conv(32, (3, 3), strides=(1, 1), pad='same',
           activation=tf.nn.relu),
    N.Pool(pool_size=(2, 2), strides=None),
    # N.Conv(64, (3, 3), strides=(1, 1), pad='same',
    #        activation=tf.nn.relu),
    N.Pool(pool_size=(2, 2), strides=None),
    N.Dropout(level=0.5),
    N.Flatten(outdim=2),
    N.Dense(256, activation=tf.nn.relu),
    N.Dense(10, activation=K.linear)
], debug=True)
y_pred = ops(X)

y_onehot = K.one_hot(y, nb_classes=10)
cost_ce = tf.identity(tf.losses.softmax_cross_entropy(y_onehot, y_pred), name='CE')
cost_acc = K.metrics.categorical_accuracy(y_pred, y, name="Acc")
cost_cm = K.metrics.confusion_matrix(y_pred, y, labels=10)

parameters = ops.parameters
optimizer = K.optimizers.SGD(lr=0.05)
updates = optimizer(cost_ce, parameters)
print('Building training functions ...')
f_train = K.function([X, y], [cost_ce, optimizer.norm, cost_cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function([X, y], [cost_ce, cost_acc, cost_cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(X, y_pred, training=False)


# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=12, shuffle_level=2)
task.set_save(get_modelpath(name='mnist_ai', override=True), ops)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost_ce, threshold=1)
])
task.set_train_task(f_train, (ds['X_train'], ds['y_train']), epoch=4,
                    name='train')
task.set_valid_task(f_test, (ds['X_test'], ds['y_test']),
                    freq=training.Timer(percentage=0.6), name='valid')
task.set_eval_task(f_test, (ds['X_test'], ds['y_test']), name='test')
task.run()

# ===========================================================================
# Evaluate
# ===========================================================================
ACC = []
CM = []
for X, y in zip(ds['X_test'].set_batch(128, seed=None),
                ds['y_test'].set_batch(128, seed=None)):
    ce, acc, cm = f_test(X, y)
    ACC.append(acc)
    CM.append(cm)
print("Accuracy:", np.mean(ACC))
print("CM:")
print(sum(cm for cm in CM))
