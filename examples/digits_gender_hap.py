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
from odin.stats import train_valid_test_split, freqcount, prior2weights
from odin import training
from odin.visual import print_dist, print_confusion, print_hist
from odin.utils import (get_logpath, Progbar, get_modelpath, unique_labels,
                        as_tuple_of_shape, stdio, get_datasetpath,
                        ArgController, ctext)

args = ArgController(
).add('path', 'path to acoustic features of TIDIGITS'
).parse()
# ===========================================================================
# Const
# ===========================================================================
FEAT = 'mspec'
MODEL_PATH = get_modelpath(name='digit', override=True)
LOG_PATH = get_logpath(name='digit.log', override=True)
stdio(LOG_PATH)

if not os.path.isdir(args.path):
  raise ValueError("`path` at '%s' must be a folder" % args.path)
ds = F.Dataset(args.path, read_only=True)
if FEAT not in ds:
  print(ds)
  raise RuntimeError("Cannot find feature with name: %s" % FEAT)
indices = list(ds['indices'].items())
K.get_rng().shuffle(indices)
print("#Files:", ctext(len(indices), 'cyan'))
# ====== genders ====== #
fn_gen, label_gen = unique_labels([i[0] for i in indices],
                                  lambda x: x.split('_')[1], True)
gen_old = {'m': 'old', 'w': 'old',
           'b': 'young', 'g': 'young'}
fn_old, label_old = unique_labels([i[0] for i in indices],
                                 lambda x: gen_old[x.split('_')[1]], True)
print("Label Gender:", ctext(label_gen, 'cyan'))
print("Label Old:", ctext(label_old, 'cyan'))
weight_gen = list(freqcount([fn_gen(name)
                             for name, (s, e) in indices]).values())
weight_gen = prior2weights(prior=weight_gen, exponential=False,
                           min_value=0.1, max_value=8.0,
                           norm=False)
weight_old = list(freqcount([fn_old(name)
                             for name, (s, e) in indices]).values())
weight_old = prior2weights(prior=weight_old, exponential=False,
                           min_value=None, max_value=None,
                           norm=True)
print("Weight Gender:", ctext(weight_gen, 'cyan'))
print("Weight Old:", ctext(weight_old, 'cyan'))
# ====== split train, test ====== #
train = []
test = []
for name, (start, end) in indices:
  if name.split('_')[0] == 'train':
    train.append((name, start, end))
  else:
    test.append((name, start, end))
# split by speaker ID
train, valid = train_valid_test_split(train, train=0.8,
    cluster_func=None,
    idfunc=lambda x: x[0].split('_')[4],
    inc_test=False)
print("#File train:", len(train))
print("#File valid:", len(valid))
print("#File test:", len(test))
LONGEST_UTT = max(e - s for name, (s, e) in indices)
print("Longest utterances:", ctext(LONGEST_UTT, 'cyan'))
# ===========================================================================
# Create feeder
# ===========================================================================
recipes = [
    F.recipes.Name2Trans(converter_func=fn_gen),
    F.recipes.Name2Trans(converter_func=fn_old),
    F.recipes.Sequencing(frame_length=LONGEST_UTT, step_length=1,
                         end='pad', pad_mode='post', pad_value=0,
                         label_idx=(-1, -2)),
    F.recipes.LabelOneHot(nb_classes=len(label_gen), data_idx=-2),
    F.recipes.LabelOneHot(nb_classes=len(label_old), data_idx=-1),
]
train = F.Feeder(data_desc=F.DataDescriptor(data=ds[FEAT], indices=train),
                 ncpu=8, buffer_size=12, dtype='float32',
                 batch_mode='batch')
valid = F.Feeder(data_desc=F.DataDescriptor(data=ds[FEAT], indices=valid),
                 ncpu=6, buffer_size=1, dtype='float32',
                 batch_mode='batch')
test = F.Feeder(data_desc=F.DataDescriptor(data=ds[FEAT], indices=test),
                ncpu=1, buffer_size=1, dtype='float32',
                batch_mode='file')
train.set_recipes(recipes)
valid.set_recipes(recipes)
test.set_recipes(recipes)
print(train)
# ===========================================================================
# Create model
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:], dtype='float32', name='input%d' % i)
     for i, shape in enumerate(as_tuple_of_shape(train.shape))]
print("Inputs:")
for x in inputs:
  print(' * ', x)
X = inputs[0]
y_gen = inputs[-2]
y_old = inputs[-1]
with N.args_scope(ops=['Conv', 'Dense'], b_init=None, activation=K.linear,
                  pad='same'):
  with N.args_scope(ops=['BatchNorm'], activation=K.relu):
    f = N.Sequence([
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),
        N.Conv(num_filters=32, filter_size=(7, 7)), N.BatchNorm(),
        N.Pool(pool_size=(3, 2), strides=2),
        N.Conv(num_filters=32, filter_size=(3, 3)), N.BatchNorm(),
        N.Pool(pool_size=(3, 2), strides=2),
        N.Conv(num_filters=64, filter_size=(3, 3)), N.BatchNorm(),
        N.Pool(pool_size=(3, 2), strides=2),
        N.Flatten(outdim=2),
        N.Dense(512), N.BatchNorm(),
        N.Dense(len(label_old))
    ], debug=True)
y_logit = f(X)
y_prob = tf.nn.softmax(y_logit)
f_pred = training.train(X=X, y_true=y_old, y_pred=y_logit,
               train_data=train, valid_data=valid,
               valid_freq=0.8, patience=3, threshold=5, rollback=True,
               metrics=[0, K.metrics.categorical_accuracy, K.metrics.confusion_matrix],
               training_metrics=[1, 2], prior_weights=weight_old,
               batch_size=256, epochs=8, shuffle=True,
               optimizer='rmsprop', optz_kwargs={'lr': 0.0001},
               labels=label_old, verbose=4)
# ===========================================================================
# Prediction
# ===========================================================================
y_true_gen = []
y_true_old = []
y_pred = []
for outputs in Progbar(test, name="Evaluating",
                       count_func=lambda x: x[-1].shape[0]):
  name = str(outputs[0])
  idx = int(outputs[1])
  data = outputs[2:]
  if idx >= 1:
    raise ValueError("NOPE")
  y_true_gen.append(fn_gen(name))
  y_true_old.append(fn_old(name))
  y_pred.append(f_pred(*data))
y_true_gen = np.array(y_true_gen, dtype='int32')
y_true_old = np.array(y_true_old, dtype='int32')
y_pred = np.array(y_pred, dtype='float32')
nb_classes = y_pred.shape[1]
y_pred = np.argmax(y_pred, axis=-1)

from sklearn.metrics import confusion_matrix, accuracy_score
if nb_classes == len(label_gen):
  y_true = y_true_gen
  labels = label_gen
else:
  y_true = y_true_old
  labels = label_old
print()
print("Acc:", accuracy_score(y_true, y_pred))
print("Confusion matrix:")
print(print_confusion(confusion_matrix(y_true, y_pred), labels))
print(LOG_PATH)
