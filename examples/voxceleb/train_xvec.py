from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'

import scipy.io
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import training
from odin.utils import (args_parse, ctext, Progbar, as_tuple_of_shape,
                        crypto, stdio)
from odin import fuel as F, visual as V, nnet as N, backend as K

from utils import prepare_dnn_data, get_model_path, csv2mat
# ===========================================================================
# Configs
# ===========================================================================
args = args_parse([
    ('recipe', 'the name of function defined in feature_recipes.py', None),
    ('-feat', "Acoustic feature", ('mspec', 'mfcc'), 'mspec'),
    ('-batch', "batch size", None, 64),
    ('-epoch', "number of epoch", None, 25),
    ('-l', "audio segmenting length in second", None, 3),
    ('--debug', "enable debug mode", None, False),
    ('--train', "force continue training the saved model", None, False),
])
FEAT = args.feat
TRAIN_MODEL = args.train
DEBUG = bool(args.debug)
(EXP_DIR, MODEL_PATH, LOG_PATH,
 TRAIN_PATH, TEST_PATH) = get_model_path('xvec', args)
stdio(LOG_PATH)
# ===========================================================================
# Create data feeder
# ===========================================================================
(train, valid,
 test_ids, test_dat,
 all_speakers) = prepare_dnn_data(
    recipe=args.recipe, feat=FEAT, utt_length=args.l)
n_speakers = len(all_speakers) + 1
# ===========================================================================
# Create the network
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:],
                        dtype='float32',
                        name='input%d' % i)
          for i, shape in enumerate(as_tuple_of_shape(train.shape))]
X = inputs[0]
y = inputs[1]
print("Inputs:", ctext(inputs, 'cyan'))
# ====== the network ====== #
if os.path.exists(MODEL_PATH):
  x_vec = N.deserialize(path=MODEL_PATH, force_restore_vars=True)
else:
  TRAIN_MODEL = True
  with N.args_scope(
      ['TimeDelayedConv', dict(time_pool='none', activation=K.relu)],
      ['Dense', dict(activation=K.linear, b_init=None)],
      ['BatchNorm', dict(activation=K.relu)]
  ):
    x_vec = N.Sequence([
        N.Dropout(level=0.3),

        N.TimeDelayedConv(n_new_features=512, n_time_context=5),
        N.TimeDelayedConv(n_new_features=512, n_time_context=5),
        N.TimeDelayedConv(n_new_features=512, n_time_context=7),

        N.Dense(512), N.BatchNorm(),
        N.Dense(1500), N.BatchNorm(),

        N.StatsPool(axes=1, output_mode='concat'),
        N.Flatten(outdim=2),

        N.Dense(512, name="LatentOutput"), N.BatchNorm(),
        N.Dense(512), N.BatchNorm(),

        N.Dense(n_speakers, activation=K.linear,
                b_init=init_ops.constant_initializer(value=0))
    ], debug=1)
# ====== create outputs ====== #
y_logit = x_vec(X)
y_proba = tf.nn.softmax(y_logit)
z = K.ComputationGraph(y_proba).get(roles=N.Dense, scope='LatentOutput',
                                    beginning_scope=False)[0]
print('Latent space:', ctext(z, 'cyan'))
# ====== create loss ====== #
ce = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_true=y, y_pred=y_proba)
# ====== params and optimizing ====== #
updates = K.optimizers.Adam(lr=0.0001, name='XAdam').minimize(
    loss=ce,
    roles=[K.role.TrainableParameter],
    exclude_roles=[K.role.InitialState],
    verbose=True)
K.initialize_all_variables()
# # ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs, [ce, acc], updates=updates,
                     training=True)
print('Building testing functions ...')
f_score = K.function(inputs, [ce, acc],
                    training=False)
# Latent spaces
f_z = K.function(inputs=X, outputs=z, training=False)
# ===========================================================================
# Create trainer
# ===========================================================================
if TRAIN_MODEL:
  print('Start training ...')
  task = training.MainLoop(batch_size=args.batch, seed=120825, shuffle_level=2,
                           allow_rollback=True)
  task.set_checkpoint(MODEL_PATH, x_vec)
  task.set_callbacks([
      training.NaNDetector(),
      training.EarlyStopGeneralizationLoss('valid', ce,
                                           threshold=5, patience=5)
  ])
  task.set_train_task(func=f_train, data=train,
                      epoch=args.epoch, name='train')
  task.set_valid_task(func=f_score, data=valid,
                      freq=training.Timer(percentage=0.8),
                      name='valid')
  task.run()
# ===========================================================================
# Saving the test data
# CSV separated by tab
# ===========================================================================
sep = '\t'
prog = Progbar(target=len(test_ids) + len(train) + len(valid),
               print_summary=True, print_report=True,
               name="Extracting x-vector")
with open(TRAIN_PATH, 'w') as f_train, open(TEST_PATH, 'w') as f_test:
  # ====== save training set ====== #
  for name, idx, X, y in train.set_batch(batch_size=8000,
                                         batch_mode='file', seed=None):
    assert idx == 0
    y = np.argmax(y, axis=-1)
    assert len(set(y)) == 1
    y = y[0]
    z = np.mean(f_z(X), axis=0, keepdims=False).astype('float32')
    f_train.write(sep.join([str(y)] + [str(i) for i in z]) + '\n')
    prog.add(X.shape[0])
  # ====== save validation set ====== #
  for name, idx, X, y in valid.set_batch(batch_size=8000,
                                         batch_mode='file', seed=None):
    assert idx == 0
    y = np.argmax(y, axis=-1)
    assert len(set(y)) == 1
    y = y[0]
    z = np.mean(f_z(X), axis=0, keepdims=False).astype('float32')
    f_train.write(sep.join([str(y)] + [str(i) for i in z]) + '\n')
    prog.add(X.shape[0])
  # ====== save test set ====== #
  for name, (start, end) in sorted(test_ids.items(),
                                   key=lambda x: x[0]):
    y = test_dat[start:end]
    z = np.mean(f_z(y), axis=0, keepdims=False).astype('float32')
    f_test.write(sep.join([name] + [str(i) for i in z]) + '\n')
    prog.add(1)
# convert everything to matlab format
csv2mat(exp_dir=EXP_DIR)
# ===========================================================================
# Evaluate and save the log
# ===========================================================================
np.random.seed(52181208)
shape = inputs[0].shape
X = np.random.rand(64, shape[1].value, shape[2].value).astype('float32')
Z = f_z(X)
# ====== make sure model has the same identity ====== #
print(Z.shape, Z.sum(), (Z**2).sum(), Z.std())
print(ctext(crypto.md5_checksum(Z), 'cyan'))
