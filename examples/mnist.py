#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import numpy as np

import os
os.environ['ODIN'] = 'float32,cpu,theano,seed=12'

from odin import backend as K
from odin import nnet as N
from odin import fuel, training, visual
from odin.roles import add_role, DEPLOYING
from odin.utils import one_hot
import cPickle

ds = fuel.load_mnist()

X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X',
                  for_training=True)
y = K.placeholder(shape=ds['y_train'].shape, name='y', dtype='int32')

ops = N.Sequence([
    # N.Dimshuffle((0, 'x', 1, 2)),
    # N.Conv2D(8, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    # K.pool2d,
    N.FlattenRight(outdim=2),
    N.Dense(64, activation=K.relu),
    N.Dense(10, activation=K.softmax)
])
ops = cPickle.loads(cPickle.dumps(ops)) # test if the ops is pickle-able

y_pred = ops(X)
cost_train = K.mean(K.categorical_crossentropy(y_pred, y))
cost_test = K.mean(K.metrics.categorical_accuracy(y_pred, y))

parameters = ops.parameters
updates = K.optimizers.sgd(cost_train, parameters, learning_rate=0.1)

print('Building functions ...')
f_train = K.function([X, y], cost_train, updates=updates)
f_test = K.function([X, y], cost_test)

# ====== Main task ====== #
print('Start training ...')
task = training.MainLoop(dataset=ds, batch_size=128, seed=12)
task.set_callback(
    training.ProgressMonitor(title='Results: %.2f'),
    training.History(),
    training.EarlyStopGeneralizationLoss(5, 'valid', lambda x: 1 - np.mean(x)),
    # training.EarlyStopPatience(0, 'valid', lambda x: 1 - np.mean(x)),
    training.Checkpoint('/Users/trungnt13/tmp/tmp.ops').set_obj(ops)
)
# task = cPickle.loads(cPickle.dumps(task))
task.set_task(f_train, ('X_train', 'y_train'), epoch=3, name='train')
task.add_subtask(f_test, ('X_valid', 'y_valid'), freq=0.6, name='valid')
task.add_subtask(f_test, ('X_test', 'y_test'), freq=0.99, name='test')
task.run()

train = task['History'].get(task='train', event='batch_end')
print(visual.print_bar(train, bincount=20))


valid = task['History'].get(task='valid', event='epoch_end')
valid = [np.mean(i) for i in valid]
# print(valid)
print(visual.print_bar(valid, bincount=len(valid)))


test = task['History'].get(task='test', event='epoch_end')
test = [np.mean(i) for i in test]
# print(test)
print(visual.print_bar(test, bincount=len(test)))
