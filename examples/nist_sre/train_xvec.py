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

from helpers import (get_model_path, prepare_dnn_data,
                     IS_TRAINING, BATCH_SIZE, EPOCH)
# ===========================================================================
# Create data feeder
# ===========================================================================
(EXP_DIR, MODEL_PATH, LOG_PATH) = get_model_path(system_name='xvec',
                                                 args_name=['utt'])
stdio(LOG_PATH)
# ====== load the data ====== #
(train, valid,
 all_speakers) = prepare_dnn_data()
n_speakers = len(all_speakers)
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
  x_vec = N.deserialize(path=MODEL_PATH,
                        force_restore_vars=True)
else:
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
    ], debug=1, name='XNetwork')
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
if not os.path.exists(MODEL_PATH) or IS_TRAINING:
  print('Start training ...')
  task = training.MainLoop(batch_size=BATCH_SIZE, seed=120825,
                           shuffle_level=2, allow_rollback=True,
                           verbose=4)
  task.set_checkpoint(MODEL_PATH, x_vec,
                      increasing=True, max_checkpoint=-1)
  task.set_callbacks([
      training.NaNDetector(),
      training.Checkpoint(task_name='train', epoch_percent=0.5),
      # training.EarlyStopGeneralizationLoss('valid', ce,
      #                                      threshold=5, patience=3)
  ])
  task.set_train_task(func=f_train, data=train,
                      epoch=EPOCH, name='train')
  task.set_valid_task(func=f_score, data=valid,
                      freq=training.Timer(percentage=1.),
                      name='valid')
  task.run()
