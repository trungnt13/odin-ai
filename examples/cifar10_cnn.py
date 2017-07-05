from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu,seed=12082518'

import numpy as np
import tensorflow as tf

from odin import fuel as F, nnet as N, backend as K, training, utils

MODEL_PATH = utils.get_modelpath(name='cifar10_ai', override=True)
REPORT_PATH = utils.get_logpath(name='cifar10.pdf', override=True)

# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.load_cifar10()
nb_labels = 10
print(ds)
X_train = ds['X_train'][:].astype('float32') / 255.
y_train = ds['y_train'][:]
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = ds['y_test'][:]

# ===========================================================================
# Create network
# ===========================================================================
X = K.placeholder(shape=(None,) + X_train.shape[1:], name='X')
y_true = K.placeholder(shape=(None,), name='y_true', dtype='int32')

ops = N.get_model_descriptor('cifar_cnn', prefix='models_')
y_pred = ops(X, nb_labels=nb_labels)

y_onehot = K.one_hot(y_true, nb_classes=nb_labels)
cost_ce = tf.losses.softmax_cross_entropy(y_onehot, y_pred)
cost_acc = K.metrics.categorical_accuracy(y_pred, y_true, name="Acc")
cost_cm = K.metrics.confusion_matrix(y_pred, y_true, labels=10)

optz = K.optimizers.Adam(lr=0.001)
parameters = ops.parameters
updates = optz(cost_ce, parameters)
print('Building training functions ...')
f_train = K.function([X, y_true], [cost_ce, optz.norm, cost_cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function([X, y_true], [cost_ce, cost_acc, cost_cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(X, y_pred, training=False)

# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=12, shuffle_level=2,
                         print_progress=True, confirm_exit=True)
task.set_save(MODEL_PATH, ops)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost_ce, threshold=1)
])
task.set_train_task(f_train, (X_train, y_train), epoch=8, name='train')
task.set_valid_task(f_test, (X_test, y_test),
    freq=training.Timer(percentage=0.6), name='valid')
task.set_eval_task(f_test, (X_test, y_test), name='test')
task.run()

# ===========================================================================
# Evaluate
# ===========================================================================
ce, acc, cm = f_test(X_test[:256], y_test[:256])
print("Accuracy:", np.mean(acc))
print("CM:")
print(cm)
