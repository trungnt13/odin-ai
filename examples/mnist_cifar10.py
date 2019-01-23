#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'float32,gpu,seed=12,log'
import shutil

import numpy as np
import tensorflow as tf

from odin.utils import ArgController
from odin import backend as K
from odin import nnet as N
from odin import fuel as F, training

arg = ArgController(
).add('-ds', 'dataset cifar10, mnist, or fmnist', 'mnist'
).add('--rnn', 'using RNN network', False
).parse()
# ===========================================================================
# Load data
# ===========================================================================
USE_MNIST_DATA = True
if arg.ds.lower() == 'mnist':
  ds = F.MNIST_original.get_dataset()
elif arg.ds.lower() == 'fmnist':
  ds = F.FMNIST_original.get_dataset()
else:
  ds = F.CIFAR10.get_dataset()
  USE_MNIST_DATA = False
print(ds)

X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X')
y = K.placeholder(shape=(None,), name='y', dtype='int32')
y_onehot = tf.one_hot(y, depth=10)
# ===========================================================================
# Build network
# ===========================================================================
if not arg.rnn:
  ops = N.Sequence([
      N.Dimshuffle((0, 1, 2, 'x')) if USE_MNIST_DATA else N.Dimshuffle((0, 2, 3, 1)),

      N.BatchNorm(axes='auto'),
      N.Conv(32, (3, 3), strides=(1, 1), pad='same',
             activation=tf.nn.relu),
      N.Pool(pool_size=(2, 2), strides=None),
      N.Conv(64, (3, 3), strides=(1, 1), pad='same',
             activation=tf.nn.relu),
      N.Pool(pool_size=(2, 2), strides=None),
      N.Dropout(level=0.5),

      N.Flatten(outdim=2),
      N.Dense(256, activation=tf.nn.relu),
      N.Dense(10, activation=K.linear)
  ], debug=True)
else:
  ops = N.Sequence([
      N.Dimshuffle((0, 1, 2, 'x')) if USE_MNIST_DATA else N.Dimshuffle((0, 2, 3, 1)),

      N.Conv(32, filter_size=3, strides=1, pad='same', activation=K.linear),
      N.BatchNorm(axes='auto', activation=K.relu),
      N.Pool(pool_size=2, strides=None),

      N.Dimshuffle(pattern=(0, 3, 1, 2)),
      N.Flatten(outdim=3),
      N.CudnnRNN(18, initial_states=None, rnn_mode='gru',
                 num_layers=2,
                 input_mode='linear', direction_mode='unidirectional',
                 params_split=False),

      N.Flatten(outdim=2),
      N.Dense(128, activation=K.relu),
      N.Dense(10, activation=tf.nn.softmax)
  ], debug=True)
# ====== applying the NNOps ====== #
y_pred = ops(X)
if arg.rnn:
  loss = tf.losses.softmax_cross_entropy(y_onehot, ops(X, training=True))
else:
  loss = tf.losses.softmax_cross_entropy(y_onehot, y_pred)
acc = K.metrics.categorical_accuracy(y, y_pred, name="Acc")
cm = K.metrics.confusion_matrix(y_pred=y_pred, y_true=y, labels=10)
# ====== optimizer ====== #
optimizer = K.optimizers.Adam(lr=0.001)
updates = optimizer.minimize(loss, verbose=True)
# ====== initialize all variable ====== #
K.initialize_all_variables()
# ====== function ====== #
print('Building training functions ...')
f_train = K.function([X, y], [loss, optimizer.norm, cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function([X, y], [loss, acc, cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(X, y_pred, training=False)
# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
# ====== some configurations ====== #
model_save_path = '/tmp/EXP_MNIST'
if os.path.exists(model_save_path):
  shutil.rmtree(model_save_path)
os.mkdir(model_save_path)

print("Save path:", model_save_path)
N_EPOCH = 120
BATCH_SIZE = 512

# ====== run the training ====== #
task = training.MainLoop(batch_size=BATCH_SIZE, seed=12, shuffle_level=2,
                         allow_rollback=True, verbose=1)
task.set_checkpoint(os.path.join(model_save_path, 'checkpoint'),
                    ops, max_checkpoint=-1)
task.set_callbacks([
    training.NaNDetector(),
    training.CheckpointEpoch('train', epoch_percent=1.),
    training.EarlyStopGeneralizationLoss('valid', loss,
                                         threshold=5, patience=3),
    training.LambdaCallback(fn=lambda t:print(str(t)),
                            task_name='train',
                            signal=training.TaskSignal.EpochEnd),
    training.LambdaCallback(fn=lambda t:print(str(t)),
                            task_name='valid',
                            signal=training.TaskSignal.EpochEnd),
    training.EpochSummary(task_name=('train', 'valid'),
                          output_name=loss, print_plot=True,
                          save_path=os.path.join(model_save_path, 'summary.pdf'))
])
task.set_train_task(f_train, (ds['X_train'], ds['y_train']), epoch=N_EPOCH,
                    name='train')
task.set_valid_task(f_test, (ds['X_test'], ds['y_test']),
                    freq=training.Timer(percentage=1.), name='valid')
task.set_eval_task(f_test, (ds['X_test'], ds['y_test']), name='test')
task.run()
# ===========================================================================
# Evaluate
# ===========================================================================
ACC = []
CM = []
for X, y in zip(F.as_data(ds['X_test']).set_batch(128, seed=None),
                F.as_data(ds['y_test']).set_batch(128, seed=None)):
  ce, acc, cm = f_test(X, y)
  ACC.append(acc)
  CM.append(cm)
print("Accuracy:", np.mean(ACC))
print("CM:")
print(sum(cm for cm in CM))
