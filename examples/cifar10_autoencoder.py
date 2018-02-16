from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

from odin.utils import ArgController, stdio, one_hot

import os
os.environ['ODIN'] = 'float32,gpu,seed=12082518'

import numpy as np
import tensorflow as tf

from odin import fuel as F, nnet as N, backend as K, training, utils
from odin.stats import train_valid_test_split, freqcount, prior2weights

from sklearn.metrics import classification_report

MODEL_PATH = utils.get_modelpath(name='cifar10_ae', override=True)
LOG_PATH = utils.get_logpath(name='cifar10_ae.log', override=True)
stdio(LOG_PATH)
# ===========================================================================
# Constants
# ===========================================================================
NB_EPOCH = 10
LEARNING_RATE = 0.001

ds = F.CIFAR10.get_dataset()
nb_classes = 10
print(ds)
X_train = ds['X_train'][:].astype('float32') / 255.
y_train = one_hot(ds['y_train'][:], nb_classes=nb_classes)
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = one_hot(ds['y_test'][:], nb_classes=nb_classes)
weights = prior2weights(np.sum(y_train, axis=0),
                        min_value=None, max_value=None,
                        norm=False)
# ===========================================================================
# Basic variables
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + X_train.shape[1:], name='X', dtype='float32'),
          K.placeholder(shape=(None, nb_classes), name='y', dtype='float32')]
X, y = inputs
print("Inputs:", inputs)

fn = N.Sequence([
    N.Dimshuffle(pattern=(0, 2, 3, 1)),
    N.Conv(32, (3, 3), b_init=None, pad='same', stride=(1, 1), name='Conv1'),
    N.BatchNorm(activation=K.relu),
    N.Conv(32, (3, 3), pad='same', stride=(1, 1),
           b_init=0, activation=K.relu, name='Conv2'),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),
    #
    N.Conv(64, (3, 3), b_init=None, pad='same', stride=(1, 1), name='Conv3'),
    N.BatchNorm(activation=K.relu),
    N.Conv(64, (3, 3), pad='same', stride=(1, 1),
           b_init=0, activation=K.relu, name='Conv4'),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),
    #
    N.Flatten(outdim=2),
    N.Dense(512, activation=K.relu),
    N.Dropout(level=0.5),
    N.Dense(nb_classes, activation=K.linear)
], debug=1, name="Cifar10")
y_logits = fn(X)

f_pred = training.train(X=X, y_true=y, y_pred=y_logits,
                        objectives=tf.losses.softmax_cross_entropy,
                        metrics=(0, K.metrics.categorical_accuracy,
                                 (K.metrics.confusion_matrix, {'labels': nb_classes})),
                        training_metrics=2,
                        prior_weights=weights, epochs=NB_EPOCH, batch_size=256,
                        train_data=(X_train, y_train), valid_data=0.2,
                        optimizer='rmsprop', optz_kwargs={'lr': LEARNING_RATE},
                        labels=['l%d' % i for i in range(nb_classes)],
                        verbose=4)

y_test_pred = np.concatenate(
    [f_pred(x) for x in F.as_data(X_test).set_batch(1024, seed=None)],
    axis=0)
print(classification_report(y_true=np.argmax(y_test, -1),
                            y_pred=np.argmax(y_test_pred, -1)))
