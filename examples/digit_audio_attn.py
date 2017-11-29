# ===========================================================================
# Should reach: > 0.9861111% on test set with default configuration
# ===========================================================================
from __future__ import print_function, absolute_import, division

import matplotlib
matplotlib.use("Agg")

# ====== import ====== #
import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np
import tensorflow as tf
np.random.seed(1208)

from odin import nnet as N, backend as K, fuel as F, stats
from odin.utils import get_modelpath, stdio, get_logpath, get_datasetpath
from odin import training

# set log path
stdio(path=get_logpath('digit_audio.log', override=True))

# ===========================================================================
# Get wav and process new dataset configuration
# ===========================================================================
ds = F.load_digit_feat()
nb_classes = 10
indices = [(name, start, end)
           for name, (start, end) in ds['indices'].items(True)]
longest_utterances = max(int(end) - int(start) - 1
                         for i, start, end in indices)
longest_vad = max(end - start
                  for name, vad in ds['vadids']
                  for (start, end) in vad)

np.random.shuffle(indices)
n = len(indices)
train = indices[:int(0.6 * n)]
valid = indices[int(0.6 * n):int(0.8 * n)]
test = indices[int(0.8 * n):]
print('Nb train:', len(train), stats.freqcount([int(i[0][0]) for i in train]))
print('Nb valid:', len(valid), stats.freqcount([int(i[0][0]) for i in valid]))
print('Nb test:', len(test), stats.freqcount([int(i[0][0]) for i in test]))

train_feeder = F.Feeder(ds['mspec'], train, ncpu=1)
test_feeder = F.Feeder(ds['mspec'], test, ncpu=2)
valid_feeder = F.Feeder(ds['mspec'], valid, ncpu=2)

recipes = [
    F.recipes.Name2Trans(converter_func=lambda x: int(x[0])),
    F.recipes.Normalization(
        mean=ds['mspec_mean'], std=ds['mspec_std'],
        local_normalize=False
    ),
    F.recipes.Sequencing(frame_length=longest_utterances, hop_length=1,
                         end='pad', endvalue=0, endmode='post',
                         label_transform=lambda x: x[-1]),
]

train_feeder.set_recipes(recipes + [F.recipes.CreateBatch()])
valid_feeder.set_recipes(recipes + [F.recipes.CreateBatch()])
test_feeder.set_recipes(recipes + [F.recipes.CreateFile(return_name=True)])
print('Feature shape:', train_feeder.shape)
feat_shape = (None,) + train_feeder.shape[1:]

X = K.placeholder(shape=feat_shape, name='X')
y = K.placeholder(shape=(None,), dtype='int32', name='y')
# ===========================================================================
# Create network
# ===========================================================================
f = N.Sequence([
    # ====== CNN ====== #
    # N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    # N.Conv(num_filters=32, filter_size=3, pad='same', strides=1,
    #        activation=K.linear),
    # N.BatchNorm(activation=K.relu),
    # N.Conv(num_filters=64, filter_size=3, pad='same', strides=1,
    #        activation=K.linear),
    # N.BatchNorm(activation=K.relu),
    # N.Pool(pool_size=2, strides=None, pad='valid', mode='max'),
    # N.Flatten(outdim=3),
    # ====== RNN ====== #
    # N.CudnnRNN(32, rnn_mode='gru', num_layers=1, direction_mode='unidirectional'),
    N.GRU(num_units=32, attention=True, num_layers=1, bidirectional=False),
    # ====== Dense ====== #
    N.Flatten(outdim=2),
    # N.Dropout(level=0.2), # adding dropout does not help
    # N.Dense(num_units=512, activation=K.relu),
    N.Dense(num_units=nb_classes)
], debug=True)
y_logits = f(X)
y_prob = tf.nn.softmax(y_logits)
y_onehot = tf.one_hot(y, depth=nb_classes)
# ====== create cost ====== #
cost_ce = tf.losses.softmax_cross_entropy(y_onehot, y_logits)
cost_acc = K.metrics.categorical_accuracy(y_prob, y)
cost_cm = K.metrics.confusion_matrix(y_pred=y_prob, y_true=y,
                                     labels=nb_classes)

# ====== create optimizer ====== #
parameters = [p for p in f.parameters]
optimizer = K.optimizers.Adam(lr=0.0001)
updates = optimizer.get_updates(cost_ce, parameters)

print('Building training functions ...')
f_train = K.function([X, y], [cost_ce, optimizer.norm, cost_cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_score = K.function([X, y], [cost_ce, cost_acc, cost_cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(X, y_prob, training=False)

# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=64, seed=12, shuffle_level=2,
                         print_progress=True, confirm_exit=True)
task.set_checkpoint(get_modelpath(name='digit_audio_ai', override=True), f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost_ce, threshold=5)
])
task.set_train_task(f_train, train_feeder, epoch=8,
                    name='train')
task.set_valid_task(f_score, valid_feeder, freq=training.Timer(percentage=0.6),
                    name='valid')
task.run()

# ===========================================================================
# Eval
# ===========================================================================
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
name = []
y_pred = []
y_true = []
for n, X, y in test_feeder.set_batch(1, seed=None):
    name.append(n)
    y_pred.append(f_pred(X))
    y_true.append(y[0])
y_pred = np.argmax(np.concatenate(y_pred, axis=0), axis=-1)
print("Misfiles:")
for n, i, j in zip(name, y_pred, y_true):
    if i != j:
        print(n, "pred:%d" % i, "true:%d" % j)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred, average='micro'))
print("Confusion:",)
print(confusion_matrix(y_true, y_pred))
