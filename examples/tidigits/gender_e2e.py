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
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'gpu,float32,seed=12082518'
from collections import defaultdict

import numpy as np
import tensorflow as tf

from odin import backend as K, nnet as N, fuel as F
from odin.stats import train_valid_test_split, freqcount
from odin import training
from odin.ml import evaluate
from odin.visual import (print_dist, print_confusion, print_hist, plot_spectrogram,
                         plot_save, figure, plot_multiple_features)
from odin.utils import (Progbar, unique_labels, as_tuple_of_shape, stdio,
                        get_datasetpath, ctext, args_parse, get_formatted_datetime,
                        get_exppath)

args = args_parse(descriptions=[
    ('-feat', 'Input feature for training', None, 'mspec')
])
FEAT = args.feat
# ===========================================================================
# Const
# ===========================================================================
# ====== general path ====== #
EXP_DIR = get_exppath(tag='TIDIGITS_e2e',
                      name='%s' % FEAT)
if not os.path.exists(EXP_DIR):
  os.mkdir(EXP_DIR)
# ====== start logging ====== #
LOG_PATH = os.path.join(EXP_DIR,
                        'log_%s.txt' % get_formatted_datetime(only_number=True))
stdio(LOG_PATH)
print("Exp-dir:", ctext(EXP_DIR, 'cyan'))
print("Log path:", ctext(LOG_PATH, 'cyan'))
stdio(LOG_PATH)
DEBUG = False
# ====== training ====== #
BATCH_SIZE = 64
NB_EPOCH = 20
NB_SAMPLES = 8
# ===========================================================================
# Load dataset
# ===========================================================================
path = get_datasetpath(name='TIDIGITS_feats', override=False)
assert os.path.isdir(path), \
    "Cannot find preprocessed feature at: %s, try to run 'odin/examples/features'" % path
ds = F.Dataset(path, read_only=True)
assert FEAT in ds, "Cannot find feature with name: %s" % FEAT
indices = list(ds['indices'].items())
K.get_rng().shuffle(indices)
# ===========================================================================
# Helper
# ===========================================================================
def is_train(x):
  return x.split('_')[0] == 'train'

def extract_gender(x):
  return x.split('_')[1]

print("Task:", ctext("gender", 'cyan'))
fn_extract = extract_gender
fn_label, labels = unique_labels([i[0] for i in indices],
                                 key_func=fn_extract,
                                 return_labels=True)
print("Labels:", ctext(labels, 'cyan'))
# ===========================================================================
# Preparing data
# ===========================================================================
train_files = [] # (name, (start, end)) ...
test_files = []
for name, (start, end) in indices:
  if is_train(name):
    train_files.append((name, (start, end)))
  else:
    test_files.append((name, (start, end)))
# name for each dataset, useful for later
data_name = ['train', 'test']
print("#Train:", len(train_files))
print("#Test:", len(test_files))
# ====== length ====== #
length = [(end - start) for _, (start, end) in indices]
max_length = max(length)
print("Max length:", ctext(max_length, 'cyan'))
# ===========================================================================
# SPlit dataset
# ===========================================================================
# split by speaker ID
train_files, valid_files = train_valid_test_split(train_files, train=0.8,
    cluster_func=None,
    idfunc=lambda x: x[0].split('_')[4], # splitted by speaker
    inc_test=False)
print("#File train:", ctext(len(train_files), 'cyan'))
print("#File valid:", ctext(len(valid_files), 'cyan'))
print("#File test:", ctext(len(test_files), 'cyan'))

recipes = [
    F.recipes.Name2Trans(converter_func=fn_label),
    F.recipes.Sequencing(frame_length=max_length, step_length=1,
                         end='pad', pad_mode='post', pad_value=0,
                         label_mode='last', label_idx=-1),
    # F.recipes.StackingSequence(length=max_length, data_idx=0),
    F.recipes.LabelOneHot(nb_classes=len(labels), data_idx=-1)
]
feeder_train = F.Feeder(F.DataDescriptor(ds[FEAT], indices=train_files),
                        ncpu=6, batch_mode='batch')
feeder_valid = F.Feeder(F.DataDescriptor(ds[FEAT], indices=valid_files),
                        ncpu=4, batch_mode='batch')
feeder_test = F.Feeder(F.DataDescriptor(ds[FEAT], indices=test_files),
                       ncpu=2, batch_mode='file')

feeder_train.set_recipes(recipes)
feeder_valid.set_recipes(recipes)
feeder_test.set_recipes(recipes)
print(feeder_train)
# ====== testing ====== #
# prog = Progbar(target=len(feeder_train),
#                print_report=False, print_summary=True,
#                name='Iterate training set')
# for X, y in feeder_train:
#   prog.add(X.shape[0])
# ===========================================================================
# Create model
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:], dtype='float32', name='input%d' % i)
          for i, shape in enumerate(as_tuple_of_shape(feeder_train.shape))]
X = inputs[0]
y = inputs[1]
print("Inputs:", ctext(inputs, 'cyan'))
# ====== create the networks ====== #
with N.args_scope(
    [('Conv', 'Dense'), dict(b_init=None, activation=K.linear, pad='same')],
        ['BatchNorm', dict(activation=K.relu)]):
  f = N.Sequence([
      N.Dimshuffle(pattern=(0, 1, 2, 'x')),

      N.Conv(num_filters=32, filter_size=(9, 7)), N.BatchNorm(),
      N.Pool(pool_size=(3, 2), strides=2),
      N.Conv(num_filters=64, filter_size=(5, 3)), N.BatchNorm(),
      N.Pool(pool_size=(3, 2), strides=2, name='PoolOutput'),

      N.Flatten(outdim=2),

      N.Dense(1024), N.BatchNorm(),
      N.Dense(128),
      N.Dense(512), N.BatchNorm(),

      N.Dense(len(labels))
  ], debug=1)
# ====== create outputs ====== #
y_logit = f(X)
y_proba = tf.nn.softmax(y_logit)
z = K.flatten(K.ComputationGraph(y_proba).get(roles=N.Pool, scope='PoolOutput')[0],
              outdim=3)
print('Latent:', ctext(z, 'cyan'))
# ====== create loss ====== #
ce = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_true=y, y_pred=y_proba)
cm = K.metrics.confusion_matrix(y_true=y, y_pred=y_proba, labels=len(labels))
# ====== params and optimizing ====== #
updates = K.optimizers.Adam(lr=0.001).minimize(
    loss=ce, roles=[K.role.TrainableParameter],
    exclude_roles=[K.role.InitialState],
    verbose=True)
K.initialize_all_variables()
# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs, [ce, acc, cm], updates=updates,
                     training=True)
print('Building testing functions ...')
f_score = K.function(inputs, [ce, acc, cm],
                    training=False)
print('Building predicting functions ...')
f_pred_proba = K.function(X, y_proba, training=False)
f_z = K.function(inputs=X, outputs=z, training=False)
# ===========================================================================
# Training
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=BATCH_SIZE, seed=120825, shuffle_level=2,
                         allow_rollback=True, labels=labels)
task.set_checkpoint(os.path.join(EXP_DIR, 'model.ai'), f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce,
                                         threshold=5, patience=5)
])
task.set_train_task(func=f_train, data=feeder_train,
                    epoch=NB_EPOCH, name='train')
task.set_valid_task(func=f_score, data=feeder_valid,
                    freq=training.Timer(percentage=0.8),
                    name='valid')
task.run()
# ===========================================================================
# Prediction
# ===========================================================================
y_true = []
y_pred = []
samples = []
for outputs in Progbar(feeder_test, name="Evaluating",
                       count_func=lambda x: x[-1].shape[0]):
  name = str(outputs[0])
  idx = int(outputs[1])
  data = outputs[2:]
  if idx >= 1:
    raise ValueError("NOPE")
  y_true.append(f_gender(name))
  y_pred.append(f_pred_proba(*data))
  if np.random.random(1) < NB_SAMPLES / len(feeder_test):
    samples.append([name, data])
y_true = np.array(y_true, dtype='int32')
y_pred = np.concatenate(y_pred, axis=0)
evaluate(y_true=y_true, y_pred_proba=y_pred, labels=genders)

for name, data in samples:
  z = f_z(data[0])
  plot_multiple_features(features={
      'spec': data[0][0],
      'z': z[0]
  }, title=name)
plot_save(os.path.join(FIG_PATH, 'latent.pdf'))
