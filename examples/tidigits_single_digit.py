# ===========================================================================
# Using TIDIGITS dataset to predict gender (Boy, Girl, Woman, Man)
# ===========================================================================
# Saved WAV file format:
#     0) [train|test]
#     1) [m|w|b|g] (alias for man, women, boy, girl)
#     2) [age]
#     3) [dialectID]
#     4) [speakerID]
#     5) [production]
#     6) [digit_sequence]
#     => "train_g_08_17_as_a_4291815"
# ===========================================================================
from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'gpu,float32,seed=12082518'
from collections import defaultdict

import numpy as np
import tensorflow as tf

from odin import backend as K, nnet as N, fuel as F
from odin.stats import train_valid_test_split, freqcount
from odin import training
from odin.visual import print_dist, print_confusion, print_hist
from odin.utils import (get_logpath, Progbar, get_modelpath, unique_labels,
                        as_tuple_of_shape, stdio, ctext)
# ===========================================================================
# Const
# ===========================================================================
FEAT = ['mspec', 'vad']
DS_PATH = '/home/trung/data/tidigits'
BATCH_SIZE = 32

MODEL_PATH = get_modelpath('tidigit', override=True)
LOG_PATH = get_logpath('tidigit.log', override=True)
stdio(LOG_PATH)


# ===========================================================================
# Helper
# ===========================================================================
def is_train(x):
    return x.split('_')[0] == 'train'


def extract_spk(x):
    return x.split('_')[4]


def extract_gender(x):
    return x.split('_')[1]


def extract_digit(x):
    return x.split('_')[6]

# ===========================================================================
# Load and visual the dataset
# ===========================================================================
ds = F.Dataset(DS_PATH, read_only=True)
print(ds)
train = {}
test = {}
for name, (start, end) in ds['indices'].iteritems():
    assert end - start > 0
    if len(extract_digit(name)) == 1:
        if is_train(name):
            train[name] = (start, end)
        else:
            test[name] = (start, end)
print(ctext("#Train:", 'yellow'), len(train))
print(ctext("#Test:", 'yellow'), len(test))
# ====== gender and single digit distribution ====== #
print(print_dist(
    freqcount(train.items(),
              key=lambda x: extract_gender(x[0]) + '-' + extract_digit(x[0])),
    show_number=True,
    title="Training distribution"))
print(print_dist(
    freqcount(test.items(),
              key=lambda x: extract_gender(x[0]) + '-' + extract_digit(x[0])),
    show_number=True,
    title="Testing distribution"))
# ====== length distribution ====== #
length = [(end - start) for name, (start, end) in train.iteritems()]
length += [(end - start) for name, (start, end) in test.iteritems()]
print(print_hist(length, bincount=30, showSummary=True, title="#Frames"))
length = max(length)
# ====== genders ====== #
f_digits, digits = unique_labels([i[0] for i in train.items() + test.items()],
                                 extract_digit, True)
print(ctext("All digits:", 'yellow'), digits)
# ===========================================================================
# SPlit dataset
# ===========================================================================
# split by speaker ID
train, valid = train_valid_test_split(
    train.items(), train=0.6,
    cluster_func=lambda x: extract_digit(x[0]),
    idfunc=lambda x: extract_spk(x[0]),
    inc_test=False)
test = test.items()
print(ctext("#File train:", 'yellow'), len(train))
print(ctext("#File valid:", 'yellow'), len(valid))
print(ctext("#File test:", 'yellow'), len(test))
recipes = [
    F.recipes.Slice(slices=slice(80), axis=-1, data_idx=0),
    F.recipes.Name2Trans(converter_func=f_digits),
    F.recipes.LabelOneHot(nb_classes=len(digits), data_idx=-1),
    F.recipes.Sequencing(frame_length=length, hop_length=1,
                         end='pad', endmode='post', endvalue=0,
                         label_transform=F.recipes.last_seen,
                         data_idx=None, label_idx=-1)
]
data = [ds[f] for f in FEAT]
train = F.Feeder(F.DataDescriptor(data=data, indices=train),
                 dtype='float32', ncpu=8,
                 buffer_size=len(digits),
                 batch_mode='batch')
valid = F.Feeder(F.DataDescriptor(data=data, indices=valid),
                 dtype='float32', ncpu=1,
                 buffer_size=len(digits),
                 batch_mode='batch')
test = F.Feeder(F.DataDescriptor(data=data, indices=test),
                dtype='float32', ncpu=4,
                buffer_size=1,
                batch_mode='file')
train.set_recipes(recipes)
valid.set_recipes(recipes)
test.set_recipes(recipes)
print(train)
print(valid)
print(test)
# just to evaluate if we correctly estimate the right amount
# of data in the FeederRecipes
n = 0
for X, vad, y in train:
    n += X.shape[0]
assert n == len(train)

n = 0
for name, idx, X, vad, y in test:
    n += X.shape[0]
assert n == len(test)

n = 0
for X, vad, y in valid:
    n += X.shape[0]
assert n == len(valid)
# ===========================================================================
# Create model
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:],
                        dtype='float32',
                        name='input%d' % i)
          for i, shape in enumerate(as_tuple_of_shape(train.shape))]
print("Inputs:", inputs)
with N.nnop_scope(ops=['Conv', 'Dense'], b_init=None, activation=K.linear,
                  pad='same'):
    with N.nnop_scope(ops=['BatchNorm'], activation=K.relu):
        f = N.Sequence([
            N.Dimshuffle(pattern=(0, 1, 2, 'x')),
            N.Conv(num_filters=32, filter_size=(7, 7)), N.BatchNorm(),
            N.Pool(pool_size=(3, 2), strides=2),
            N.Conv(num_filters=32, filter_size=(3, 3)), N.BatchNorm(),
            N.Pool(pool_size=(3, 2), strides=2),
            N.Conv(num_filters=64, filter_size=(3, 3)), N.BatchNorm(),
            N.Pool(pool_size=(3, 2), strides=2),
            N.Conv(num_filters=128, filter_size=(3, 3)), N.BatchNorm(),
            # N.Pool(pool_size=(3, 2), strides=2, mode='avg'),
            N.Flatten(outdim=2),
            N.Dense(1024, b_init=0, activation=K.relu),
            N.Dropout(0.5),
            N.Dense(len(digits))
        ], debug=True)
y_logit = f(inputs[0])
y_prob = tf.nn.softmax(y_logit)
# ====== create loss ====== #
y = inputs[-1]
ce = tf.losses.softmax_cross_entropy(y, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_prob, y)
cm = K.metrics.confusion_matrix(y_prob, y, labels=len(digits))
# ====== params and optimizing ====== #
params = [p for p in f.parameters
          if K.role.has_roles(p, K.role.Parameter)]
print("Parameters:", params)
optz = K.optimizers.RMSProp(lr=0.0001)
updates = optz.get_updates(ce, params)
# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs=inputs,
                     outputs=[ce, acc, optz.norm, cm],
                     updates=updates,
                     training=True)
print('Building testing functions ...')
f_test = K.function(inputs=inputs,
                    outputs=[ce, acc, cm],
                    training=False)
print('Building predicting functions ...')
f_pred = K.function(inputs=inputs,
                    outputs=y_prob,
                    training=False)
# ===========================================================================
# Training
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=BATCH_SIZE,
                         seed=120825,
                         shuffle_level=2,
                         allow_rollback=True)
task.set_save(MODEL_PATH, f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce,
                                         threshold=5, patience=12)
])
task.set_train_task(f_train, train, epoch=25, name='train',
                    labels=digits)
task.set_valid_task(f_test, valid,
                    freq=training.Timer(percentage=0.5),
                    name='valid', labels=digits)
task.run()
# ===========================================================================
# Prediction
# ===========================================================================
y_true = []
y_pred = []
for outputs in Progbar(test, name="Evaluating",
                       count_func=lambda x: x[-1].shape[0]):
    name = str(outputs[0])
    idx = int(outputs[1])
    data = outputs[2:]
    if idx >= 1:
        raise ValueError("NOPE")
    y_true.append(f_digits(name))
    y_pred.append(f_pred(*data))
y_true = np.array(y_true, dtype='int32')
y_pred = np.argmax(np.array(y_pred, dtype='float32'), axis=-1)
# ====== Acc ====== #
from sklearn.metrics import confusion_matrix, accuracy_score
print()
print("Acc:", accuracy_score(y_true, y_pred))
print("Confusion matrix:")
print(print_confusion(confusion_matrix(y_true, y_pred), digits))
print(LOG_PATH)
