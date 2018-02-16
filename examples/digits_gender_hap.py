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
indices = list(ds['indices'].items())
K.get_rng().shuffle(indices)
print("#Files:", ctext(len(indices), 'cyan'))
# ====== genders ====== #
fn_gender, label_gen = unique_labels([i[0] for i in indices],
                                     lambda x: x.split('_')[1], True)
gen_old = {'m': 'old', 'w': 'old',
           'b': 'young', 'g': 'young'}
fn_old, label_old = unique_labels([i[0] for i in indices],
                                 lambda x: gen_old[x.split('_')[1]], True)
print("Label Gender:", ctext(label_gen, 'cyan'))
print("Label Old:", ctext(label_old, 'cyan'))
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
    F.recipes.Name2Trans(converter_func=fn_gender),
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
        N.Dense(1024), N.BatchNorm(),
        N.Dense(128), # bottleneck
        N.Dense(512), N.BatchNorm(),
        N.Dense(len(label_gen))
    ], debug=True)
y_logit = f(inputs[0])
y_prob = tf.nn.softmax(y_logit)

# ====== create loss ====== #
ce = tf.losses.softmax_cross_entropy(y_gen, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_prob, y_gen)
cm = K.metrics.confusion_matrix(y_pred=y_prob, y_true=y_gen,
                                labels=len(label_gen))
# ====== params and optimizing ====== #
params = [p for p in f.parameters
         if K.role.has_roles(p, K.role.Parameter)]
print("Parameters:", params)
optz = K.optimizers.RMSProp(lr=0.0001)
updates = optz.get_updates(ce, params)
# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs, [ce, acc, optz.norm, cm], updates=updates,
                     training=True)
print('Building testing functions ...')
f_test = K.function(inputs, [ce, acc, cm],
                    training=False)
print('Building predicting functions ...')
f_pred = K.function(inputs, y_prob, training=False)
# ===========================================================================
# Training
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=8, seed=120825, shuffle_level=2,
                         allow_rollback=True)
task.set_checkpoint(MODEL_PATH, f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce,
                                         threshold=5, patience=5)
])
task.set_train_task(f_train, train, epoch=25, name='train')
task.set_valid_task(f_test, valid,
                    freq=training.Timer(percentage=0.5), name='valid')
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
  y_true.append(fn_gender(name))
  y_pred.append(f_pred(*data))
y_true = np.array(y_true, dtype='int32')
y_pred = np.argmax(np.array(y_pred, dtype='float32'), -1)

from sklearn.metrics import confusion_matrix, accuracy_score
print()
print("Acc:", accuracy_score(y_true, y_pred))
print("Confusion matrix:")
print(print_confusion(confusion_matrix(y_true, y_pred), label_gen))
print(LOG_PATH)
