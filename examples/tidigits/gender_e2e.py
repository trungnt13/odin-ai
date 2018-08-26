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

from odin import backend as K, nnet as N, fuel as F, visual as V
from odin.stats import train_valid_test_split, freqcount
from odin import training
from odin.ml import evaluate, fast_pca, PLDA, Scorer
from odin.utils import (Progbar, unique_labels, as_tuple_of_shape, stdio,
                        get_datasetpath, ctext, args_parse, get_formatted_datetime,
                        get_exppath, batching)

args = args_parse(descriptions=[
    ('-feat', 'Input feature for training', None, 'mspec'),
    ('-batch', 'batch size', None, 32),
    ('-epoch', 'Number of training epoch', None, 12),
])
FEAT = args.feat
# ===========================================================================
# Const
# ===========================================================================
# ====== general path ====== #
EXP_DIR = get_exppath(tag='TIDIGITS_e2e', name='%s' % FEAT)
if not os.path.exists(EXP_DIR):
  os.mkdir(EXP_DIR)
# ====== start logging ====== #
LOG_PATH = os.path.join(EXP_DIR,
                        'log_%s.txt' % get_formatted_datetime(only_number=True))
stdio(LOG_PATH)
print("Exp-dir:", ctext(EXP_DIR, 'cyan'))
print("Log path:", ctext(LOG_PATH, 'cyan'))
stdio(LOG_PATH)
# ====== training ====== #
BATCH_SIZE = args.batch
NB_EPOCH = args.epoch
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.TIDIGITS_feat.load()
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
frame_length = max_length // 2
step_length = frame_length
print("Max length  :", ctext(max_length, 'yellow'))
print("Frame length:", ctext(frame_length, 'yellow'))
print("Step length :", ctext(step_length, 'yellow'))
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
print("#File test :", ctext(len(test_files), 'cyan'))

recipes = [
    F.recipes.Name2Trans(converter_func=fn_label),
    F.recipes.Sequencing(frame_length=frame_length, step_length=step_length,
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
                       ncpu=4, batch_mode='file')
feeder_train.set_recipes(recipes)
feeder_valid.set_recipes(recipes)
feeder_test.set_recipes(recipes)
print(feeder_train)
# ====== process X_test, y_test in advance for faster evaluation ====== #
prog = Progbar(target=len(feeder_test),
               print_summary=True, name="Preprocessing test set")
X_test = defaultdict(list)
for name, idx, X, y in feeder_test:
  # validate everything as expected
  assert fn_label(name) == np.argmax(y), name # label is right
  # save to list
  X_test[name].append((idx, X))
  prog.add(X.shape[0])
X_test = {name: np.concatenate([x[1] for x in sorted(X, key=lambda i: i[0])],
                               axis=0)
          for name, X in X_test.items()}
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
      N.Pool(pool_size=(3, 1), strides=(2, 1), name='PoolOutput1'),
      N.Conv(num_filters=64, filter_size=(5, 3)), N.BatchNorm(),
      N.Pool(pool_size=(3, 2), strides=(2, 2), name='PoolOutput2'),

      N.Flatten(outdim=2),

      N.Dense(512, name="LatentDense"), N.BatchNorm(),
      N.Dense(512), N.BatchNorm(),

      N.Dense(len(labels))
  ], debug=1)
# ====== create outputs ====== #
y_logit = f(X)
y_proba = tf.nn.softmax(y_logit)
z1 = K.ComputationGraph(y_proba).get(roles=N.Pool, scope='PoolOutput1')[0]
z2 = K.ComputationGraph(y_proba).get(roles=N.Pool, scope='PoolOutput2')[0]
z3 = K.ComputationGraph(y_proba).get(scope='LatentDense')[0]
print('Latent space:', ctext([z1, z2, z3], 'cyan'))
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
# Latent spaces
f_z1 = K.function(inputs=X, outputs=z1, training=False)
f_z2 = K.function(inputs=X, outputs=z2, training=False)
f_z3 = K.function(inputs=X, outputs=z3, training=False)
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
# ====== making prediction for Deep Net ====== #
y_true = []
y_pred_proba = []
Z1_test = []
Z2_test = []
Z3_test = []
prog = Progbar(target=len(X_test), print_summary=True,
               name="Making prediction for test set")
for name, X in X_test.items():
  y_pred_proba.append(np.mean(f_pred_proba(X), axis=0, keepdims=True))
  Z1_test.append(np.mean(f_z1(X), axis=0, keepdims=True))
  Z3_test.append(np.mean(f_z3(X), axis=0, keepdims=True))
  y_true.append(fn_label(name))
  prog.add(1)
Z1_test = np.concatenate(Z1_test, axis=0)
Z3_test = np.concatenate(Z3_test, axis=0)
y_pred_proba = np.concatenate(y_pred_proba, axis=0)
y_pred = np.argmax(y_pred_proba, axis=-1)
evaluate(y_true=y_true, y_pred_proba=y_pred_proba, labels=labels,
         title="Test set (Deep prediction)",
         path=os.path.join(EXP_DIR, 'test_deep.pdf'))
# ====== make a streamline classifier ====== #
Z3_train = []
y_train = []
prog = Progbar(target=len(feeder_train) + len(feeder_valid),
               print_summary=True,
               name="Extracting training latent space")
# for training set
for X, y in feeder_train:
  prog.add(X.shape[0])
  Z3_train.append(f_z3(X))
  y_train.append(np.argmax(y, axis=-1))
# for validation set
for X, y in feeder_valid:
  prog.add(X.shape[0])
  Z3_train.append(f_z3(X))
  y_train.append(np.argmax(y, axis=-1))
Z3_train = np.concatenate(Z3_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
# training PLDA
plda = PLDA(n_phi=256, random_state=K.get_rng().randint(10e8),
            n_iter=12, labels=labels, verbose=2)
plda.fit(Z3_train, y_train)
y_pred_log_proba = plda.predict_log_proba(Z3_test)
evaluate(y_true=y_true, y_pred_log_proba=y_pred_log_proba, labels=labels,
         title="Test set (Latent prediction)",
         path=os.path.join(EXP_DIR, 'test_latent.pdf'))
# ====== evaluation of the latent space ====== #
ids = K.get_rng().permutation(X_test.shape[0])
n_channels = 25 # only plot first 25 channels
for i in ids[:8]:
  x = X_test[i]
  z = Z1_test[i]
  name = name_test[i]
  V.plot_figure(nrow=20, ncol=8)
  # plot original acoustic
  V.plot_spectrogram(x.T, ax=(n_channels + 2, 1, 1),
                     title='X')
  # plot the mean
  V.plot_spectrogram(np.mean(z, axis=-1).T, ax=(n_channels + 2, 1, 2),
                     title='Zmean')
  # plot first 25 channels
  for i in range(n_channels):
    V.plot_spectrogram(z[:, :, i].T, ax=(n_channels + 2, 1, i + 3), title='Z%d' % i)
  V.plot_title(name)
V.plot_save('/tmp/tmp.pdf', tight_plot=True)
