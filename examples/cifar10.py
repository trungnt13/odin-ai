from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

from odin.utils import ArgController, stdio, one_hot
args = ArgController(
).add('-model', 'model name, specified in models_cifar.py', 'test'
).parse()

import os
os.environ['ODIN'] = 'float32,gpu,seed=12082518'

import numpy as np
import tensorflow as tf

from odin import fuel as F, nnet as N, backend as K, training, utils
from odin.stats import train_valid_test_split

MODEL_NAME = args.model
MODEL_PATH = utils.get_modelpath(name='cifar10_%s' % MODEL_NAME, override=True)
LOG_PATH = utils.get_logpath(name='cifar10_%s.log' % MODEL_NAME, override=True)
stdio(LOG_PATH)

# ===========================================================================
# Some handmade constants
# ===========================================================================
NB_EPOCH = 10
LEARNING_RATE = 0.001

# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.CIFAR10.get_dataset()
nb_labels = 10
print(ds)
X_train = ds['X_train'][:].astype('float32') / 255.
y_train = one_hot(ds['y_train'][:], nb_classes=nb_labels)
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = one_hot(ds['y_test'][:], nb_classes=nb_labels)
# ===========================================================================
# Create network
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + X_train.shape[1:], name='X', dtype='float32'),
          K.placeholder(shape=(None, nb_labels), name='y', dtype='float32')]
print("Inputs:", inputs)
model = N.Lambda.search(MODEL_NAME, prefix='models_cifar')
outputs = model(*inputs)
# ====== create losses ====== #
ce = tf.losses.softmax_cross_entropy(inputs[-1], outputs['logit'])
acc = K.metrics.categorical_accuracy(outputs['prob'], inputs[-1])
cm = K.metrics.confusion_matrix(outputs['prob'], inputs[-1], labels=nb_labels)
# ====== create optimizer ====== #
optz = K.optimizers.Adam(lr=LEARNING_RATE)
parameters = model.parameters
updates = optz(ce, parameters)
print('Building training functions ...')
f_train = K.function(inputs, [ce, optz.norm, cm], updates=updates, training=True)
print('Building testing functions ...')
f_test = K.function(inputs, [ce, acc, cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(inputs[0], outputs['prob'], training=False)
# ===========================================================================
# Build trainer
# ===========================================================================
# ====== spliting the data ====== #
idx = np.arange(len(X_train), dtype='int32')
idx_train, idx_valid = train_valid_test_split(idx, train=0.8,
                                              inc_test=False, seed=5218)
X_valid = X_train[idx_valid]
y_valid = y_train[idx_valid]
X_train = X_train[idx_train]
y_train = y_train[idx_train]
print("#Train:", X_train.shape, y_train.shape)
print("#Valid:", X_valid.shape, y_valid.shape)
print("#Test:", X_test.shape, y_test.shape)
# ====== trainign ====== #
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=120825, shuffle_level=2,
                         allow_rollback=True)
task.set_checkpoint(MODEL_PATH, model)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce, threshold=5, patience=3)
])
task.set_train_task(f_train, (X_train, y_train), epoch=NB_EPOCH, name='train')
task.set_valid_task(f_test, (X_valid, y_valid),
                   freq=training.Timer(percentage=0.6), name='valid')
task.set_eval_task(f_test, (X_test, y_test), name='eval')
task.run()

# ===========================================================================
# Exsternal validation
# ===========================================================================
script = r"""
import os
os.environ['ODIN'] = 'float32,gpu,seed=12082518'
import pickle

import numpy as np
from odin import fuel as F, nnet as N, backend as K
from odin.utils import one_hot, batching, Progbar
from odin.visual import print_confusion

from sklearn.metrics import confusion_matrix

ds = F.CIFAR10.get_dataset()
nb_labels = 10
print(ds)
X_train = ds['X_train'][:].astype('float32') / 255.
y_train = one_hot(ds['y_train'][:], nb_classes=nb_labels)
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = one_hot(ds['y_test'][:], nb_classes=nb_labels)

path = '/home/trung/.odin/models/cifar10_cnn'
f = N.deserialize(path)
outputs = f()
inputs = f.placeholders
f_pred = K.function(inputs[0], outputs['prob'], training=False)

y_pred = []
for i, (s, e) in enumerate(batching(n=X_test.shape[0], batch_size=256)):
    X = X_test[s:e]
    y_pred.append(f_pred(X))

y_pred = np.concatenate(y_pred, axis=0)
cm = confusion_matrix(y_true=np.argmax(y_test, axis=-1),
                      y_pred=np.argmax(y_pred, axis=-1))
print(print_confusion(cm))
"""
with open('/tmp/cifar10_tmp.py', 'w') as f:
    f.write(script)
os.system('python /tmp/cifar10_tmp.py')
