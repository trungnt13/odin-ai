#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np

import os
os.environ['ODIN'] = 'float32,cpu,theano,seed=12'

from odin import backend as K
from odin import nnet as N
from odin import fuel, training
from odin.utils import one_hot, get_modelpath
import cPickle

import lasagne
# ===========================================================================
# Load data
# ===========================================================================
ds = fuel.load_mnist()
print(ds)

X_train = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X_train',
                        for_training=True)
X_score = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X_score',
                        for_training=False)
y = K.placeholder(shape=(None,), name='y', dtype='int32')

# ===========================================================================
# Build network
# ===========================================================================
ops = N.Sequence([
    # N.Dimshuffle((0, 'x', 1, 2)),
    # N.Conv2D(8, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    # N.Pool2D(strides=None),
    N.FlattenRight(outdim=2),
    N.Dense(64, activation=K.relu),
    N.Dense(10, activation=K.softmax)
])
ops = cPickle.loads(cPickle.dumps(ops)) # test if the ops is pickle-able

y_pred_train = ops(X_train)
y_pred_score = ops(X_score)
cost_train = K.mean(K.categorical_crossentropy(y_pred_train, y))
cost_test = K.mean(K.categorical_accuracy(y_pred_score, y))

parameters = ops.parameters
updates = K.optimizers.SGD(momentum=None)(cost_train, parameters)
print('Building training functions ...')
f_train = K.function([X_train, y], cost_train, updates=updates)
print('Building testing functions ...')
f_test = K.function([X_score, y], cost_test)

# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=64, seed=12, shuffle_level=2)
task.set_save(get_modelpath(name='mnist.ai', override=True), ops)
task.set_task(f_train, (ds['X_train'], ds['y_train']), epoch=3, name='train')
task.set_subtask(f_test, (ds['X_valid'], ds['y_valid']), freq=0.6, name='valid')
task.set_subtask(f_test, (ds['X_test'], ds['y_test']), when=-1, name='test')
task.set_callback([
    training.ProgressMonitor(name='train', format='Results: %.2f'),
    training.ProgressMonitor(name='valid', format='Results: %.2f'),
    training.ProgressMonitor(name='test', format='Results: %.2f'),
    training.History(),
    # training.EarlyStopGeneralizationLoss(5, 'valid', lambda x: 1 - np.mean(x)),
    # training.EarlyStopPatience(0, 'valid', lambda x: 1 - np.mean(x)),
])
task.run()

# ====== plot the training process ====== #
# task['History'].print_epoch('train')
# task['History'].print_epoch('valid')
# task['History'].print_epoch('test')
