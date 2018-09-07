from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import ml, training
from odin.stats import (train_valid_test_split, sampling_iter)
from odin.utils import (args_parse, ctext, Progbar, as_tuple_of_shape)
from odin import fuel as F, visual as V, nnet as N, backend as K

from const import PATH_ACOUSTIC_FEAT, TRAIN_DATA, PATH_EXP

# ===========================================================================
# Configs
# ===========================================================================
args = args_parse([
    ('-feat', "Acoustic feature", ('mspec', 'mfcc'), 'mspec'),
    ('-batch', "batch size", None, 32),
    ('-epoch', "number of epoch", None, 12),
    ('-l', "audio segmenting length in second", None, 3),
    ('--debug', "enable debug mode", None, False),
])
FEAT = args.feat
DEBUG = bool(args.debug)
NAME = '_'.join(['tvec', args.feat, str(int(args.l))])
SAVE_PATH = os.path.join(PATH_EXP, NAME)
print('Model name:', ctext(NAME, 'cyan'))
print("Save path:", ctext(SAVE_PATH, 'cyan'))
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.Dataset(PATH_ACOUSTIC_FEAT, read_only=True)
print(ds)
X = ds[FEAT]
train_indices = {name: ds['indices'][name]
                 for name in TRAIN_DATA.keys()}
test_indices = {name: start_end
                for name, start_end in ds['indices'].items()
                if name not in TRAIN_DATA}
train_indices, valid_indices = train_valid_test_split(
    x=list(train_indices.items()), train=0.9, inc_test=False, seed=52181208)
all_speakers = sorted(set(TRAIN_DATA.values()))
n_speakers = max(all_speakers) + 1
print("#Train files:", ctext(len(train_indices), 'cyan'))
print("#Valid files:", ctext(len(valid_indices), 'cyan'))
print("#Test files:", ctext(len(test_indices), 'cyan'))
print("#Speakers:", ctext(n_speakers, 'cyan'))
# ===========================================================================
# Create data feeder
# ===========================================================================
FRAME_SHIFT = 0.005
recipes = [
    F.recipes.Name2Label(lambda name:TRAIN_DATA[name], ref_idx=0),
    F.recipes.Sequencing(frame_length=int(args.l / FRAME_SHIFT),
                         step_length=int(args.l / FRAME_SHIFT),
                         end='pad', pad_value=0, pad_mode='post',
                         data_idx=0, label_idx=1),
    F.recipes.LabelOneHot(nb_classes=n_speakers, data_idx=1)
]
train_feeder = F.Feeder(
    data_desc=F.DataDescriptor(data=X, indices=train_indices),
    batch_mode='batch', ncpu=6, buffer_size=12)
valid_feeder = F.Feeder(
    data_desc=F.DataDescriptor(data=X, indices=valid_indices),
    batch_mode='batch', ncpu=2, buffer_size=4)
test_feeder = F.Feeder(
    data_desc=F.DataDescriptor(data=X, indices=test_indices),
    batch_mode='file', ncpu=8, buffer_size=4)
train_feeder.set_recipes(recipes)
valid_feeder.set_recipes(recipes)
test_feeder.set_recipes(recipes)
print(train_feeder)
# ====== check data is valid ====== #
if DEBUG:
  prog = Progbar(target=len(train_feeder), print_report=True,
                 progress_func=lambda x: x[0].shape[0],
                 name="Validating the data")
  V.plot_figure(nrow=9, ncol=6)
  for i, (x, y) in enumerate(sampling_iter(train_feeder.set_batch(batch_size=1024),
                                           k=8,
                                           seed=5218, progress_bar=prog)):
    y = np.argmax(y[0], axis=-1)
    V.plot_spectrogram(x[0].T, ax=(8, 1, i + 1), title=str(y))
# ===========================================================================
# Create the network
# ===========================================================================
inputs = [K.placeholder(shape=(None,) + shape[1:],
                        dtype='float32',
                        name='input%d' % i)
          for i, shape in enumerate(as_tuple_of_shape(train_feeder.shape))]
X = inputs[0]
y = inputs[1]
print("Inputs:", ctext(inputs, 'cyan'))
# ====== the network ====== #
with N.args_scope(
    ['TimeDelayedConv', dict(time_pool='none', activation=K.relu)],
    ['Dense', dict(activation=K.linear)]
):
  x_vec = N.Sequence([
      N.TimeDelayedConv(n_new_features=512, n_time_context=5),
      N.TimeDelayedConv(n_new_features=512, n_time_context=5),
      N.TimeDelayedConv(n_new_features=512, n_time_context=7),

      N.Dense(num_units=512, b_init=None),
      N.BatchNorm(activation=K.relu),
      N.Dense(num_units=1500, b_init=None),
      N.BatchNorm(activation=K.relu),

      N.StatsPool(axes=1, output_mode='concat'),
      N.Flatten(outdim=2),

      N.Dense(512, b_init=None, name="LatentOutput"),
      N.BatchNorm(activation=K.relu),
      N.Dense(512, b_init=None),
      N.BatchNorm(activation=K.relu),

      N.Dense(n_speakers, activation=K.linear)
  ], debug=1)
# ====== create outputs ====== #
y_logit = x_vec(X)
y_proba = tf.nn.softmax(y_logit)
z = K.ComputationGraph(y_proba).get(roles=N.Flatten, scope='StatsPooling')[0]
print('Latent space:', ctext(z, 'cyan'))
# ====== create loss ====== #
ce = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_logit)
acc = K.metrics.categorical_accuracy(y_true=y, y_pred=y_proba)
# ====== params and optimizing ====== #
updates = K.optimizers.Adam(lr=0.001).minimize(
    loss=ce, roles=[K.role.TrainableParameter],
    exclude_roles=[K.role.InitialState],
    verbose=True)
K.initialize_all_variables()
# ====== Functions ====== #
print('Building training functions ...')
f_train = K.function(inputs, [ce, acc], updates=updates,
                     training=True)
print('Building testing functions ...')
f_score = K.function(inputs, [ce, acc],
                    training=False)
print('Building predicting functions ...')
f_pred_proba = K.function(X, y_proba, training=False)
# Latent spaces
f_z = K.function(inputs=X, outputs=z, training=False)
# ===========================================================================
# Create trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=args.batch, seed=120825, shuffle_level=2,
                         allow_rollback=True)
task.set_checkpoint(os.path.join(SAVE_PATH, 'model.ai'), x_vec)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', ce,
                                         threshold=5, patience=5)
])
task.set_train_task(func=f_train, data=train_feeder,
                    epoch=args.epoch, name='train')
task.set_valid_task(func=f_score, data=valid_feeder,
                    freq=training.Timer(percentage=0.8),
                    name='valid')
# task.run()
# ===========================================================================
# Evaluate and save the log
# ===========================================================================
