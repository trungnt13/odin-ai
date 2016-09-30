#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np

import os
os.environ['ODIN'] = 'float32,gpu,theano,seed=12'

from odin import backend as K
from odin import nnet as N
from odin import fuel, training
from odin.utils import get_modelpath, ArgController, stdio, get_logpath
import cPickle

stdio(get_logpath('tmp.log', override=True))

arg = ArgController(version=0.12
).add('-ds', 'dataset cifar10, or mnist', 'mnist'
).add('-epoch', 'number of epoch', 3
).add('-lr', 'learning rate', 0.01
).parse()

# ===========================================================================
# Load data
# ===========================================================================
USE_MNIST_DATA = True if 'mnist' in arg['ds'].lower() else False

if USE_MNIST_DATA:
    ds = fuel.load_mnist()
else:
    ds = fuel.load_cifar10()
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
    N.Dimshuffle((0, 'x', 1, 2)) if USE_MNIST_DATA else None,
    N.BatchNorm(axes='auto'),
    N.Conv2D(32, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    N.Pool2D(pool_size=(2, 2), strides=None),
    N.Conv2D(64, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    N.Pool2D(pool_size=(2, 2), strides=None),
    N.Conv2D(128, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    N.Pool2D(pool_size=(2, 2), strides=None),
    N.FlattenRight(outdim=2),
    N.Dense(512, activation=K.relu),
    N.Dense(128, activation=K.relu),
    N.Dense(10, activation=K.softmax)
])
ops = cPickle.loads(cPickle.dumps(ops)) # test if the ops is pickle-able

y_pred_train = ops(X_train)
y_pred_score = ops(X_score)
cost_train = K.mean(K.categorical_crossentropy(y_pred_train, y))
cost_test_1 = K.mean(K.categorical_crossentropy(y_pred_score, y))
cost_test_2 = K.mean(K.categorical_accuracy(y_pred_score, y))
cost_test_3 = K.confusion_matrix(y_pred_score, y, labels=range(10))

parameters = ops.parameters
optimizer = K.optimizers.SGD(lr=arg['lr'])
updates = optimizer(cost_train, parameters)
print('Building training functions ...')
f_train = K.function([X_train, y], [cost_train, optimizer.norm],
                     updates=updates)
print('Building testing functions ...')
f_test = K.function([X_score, y], [cost_test_1, cost_test_2, cost_test_3])


# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=64, seed=12, shuffle_level=2)
task.set_save(get_modelpath(name='mnist.ai', override=True), ops)
task.set_task(f_train, (ds['X_train'], ds['y_train']), epoch=arg['epoch'], name='train')
task.set_subtask(f_test, (ds['X_test'], ds['y_test']), freq=0.6, name='valid')
task.set_subtask(f_test, (ds['X_test'], ds['y_test']), when=-1, name='test')
task.set_callback([
    training.ProgressMonitor(name='train', format='Results: {:.4f}-{:.4f}'),
    training.ProgressMonitor(name='valid', format='Results: {:.4f}-{:.4f}',
                             tracking={2: lambda x: sum(x)}),
    training.ProgressMonitor(name='test', format='Results: {:.4f}-{:.4f}'),
    training.History(),
    training.EarlyStopGeneralizationLoss('valid', threshold=5, patience=3),
    training.NaNDetector(('train', 'valid'), patience=3, rollback=True)
])
task.run()

# ====== plot the training process ====== #
task['History'].print_info()
task['History'].print_batch('train')
task['History'].print_batch('valid')
task['History'].print_epoch('test')
