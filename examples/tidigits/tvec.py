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
from odin.stats import train_valid_test_split, freqcount, sampling_iter
from odin import training
from odin.ml import evaluate, fast_pca, PLDA, Scorer
from odin.utils import (Progbar, unique_labels, as_tuple_of_shape, stdio,
                        ctext, args_parse, get_formatted_datetime,
                        batching)

from utils import (prepare_data, make_dnn_prediction, visualize_latent_space,
                   get_exp_path)

args = args_parse(descriptions=[
    ('-feat', 'Input feature for training', None, 'mspec24'),
    ('-task', 'gender, age, dialect, speaker, digit', None, 'gender'),
    ('-batch', 'batch size', None, 32),
    ('-epoch', 'Number of training epoch', None, 12),
    ('--retrain', "deleted trained model, and re-train everything", None, False)
])
# ===========================================================================
# Const
# ===========================================================================
EXP_DIR, MODEL_PATH, LOG_PATH = get_exp_path('tvec', args, override=args.retrain)
stdio(LOG_PATH)
# ====== load data feeder ====== #
(train, valid,
 X_test_name, X_test_true, X_test_data,
 labels) = prepare_data(feat=args.feat, label=args.task)
n_classes = len(labels)
# ===========================================================================
# Create model
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:], dtype='float32', name='input%d' % i)
          for i, shape in enumerate(as_tuple_of_shape(train.shape))]
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

      N.Dense(n_classes)
  ], debug=1)
# ====== create outputs ====== #
y_logit = f(X)
y_proba = tf.nn.softmax(y_logit)
z1 = K.ComputationGraph(y_proba).get(roles=N.Pool, scope='PoolOutput1',
                                     beginning_scope=False)[0]
z2 = K.ComputationGraph(y_proba).get(roles=N.Pool, scope='PoolOutput2',
                                     beginning_scope=False)[0]
z3 = K.ComputationGraph(y_proba).get(scope='LatentDense',
                                     beginning_scope=False)[0]
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
task = training.MainLoop(batch_size=args.batch, seed=120825, shuffle_level=2,
                         allow_rollback=True, labels=labels)
task.set_checkpoint(MODEL_PATH, f)
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
# Prediction
# ===========================================================================
y_pred_proba, Z1_test, Z2_test, Z3_test = make_dnn_prediction(
    functions=[f_pred_proba, f_z1, f_z2, f_z3], X=X_test_data, title='TEST')
print("Test Latent:", Z1_test.shape, Z2_test.shape, Z3_test.shape)
y_pred = np.argmax(y_pred_proba, axis=-1)
evaluate(y_true=X_test_true, y_pred_proba=y_pred_proba, labels=labels,
         title="Test set (Deep prediction)",
         path=os.path.join(EXP_DIR, 'test_deep.pdf'))
# ====== make a streamline classifier ====== #
# training PLDA
Z3_train, y_train = make_dnn_prediction(f_z3, X=train, title="TRAIN")
print("Z3_train:", Z3_train.shape, y_train.shape)
Z3_valid, y_valid = make_dnn_prediction(f_z3, X=valid, title="VALID")
print("Z3_valid:", Z3_valid.shape, y_valid.shape)
plda = PLDA(n_phi=200, random_state=K.get_rng().randint(10e8),
            n_iter=12, labels=labels, verbose=0)
plda.fit(np.concatenate([Z3_train, Z3_valid], axis=0),
         np.concatenate([y_train, y_valid], axis=0))
y_pred_log_proba = plda.predict_log_proba(Z3_test)
evaluate(y_true=X_test_true, y_pred_log_proba=y_pred_log_proba, labels=labels,
         title="Test set (PLDA - Latent prediction)",
         path=os.path.join(EXP_DIR, 'test_latent.pdf'))
# ====== visualize ====== #
visualize_latent_space(X_org=X_test_data, X_latent=Z1_test,
                       name=X_test_name, labels=X_test_true,
                       title="latent1")
visualize_latent_space(X_org=X_test_data, X_latent=Z2_test,
                       name=X_test_name, labels=X_test_true,
                       title="latent2")
V.plot_save(os.path.join(EXP_DIR, 'latent.pdf'))
