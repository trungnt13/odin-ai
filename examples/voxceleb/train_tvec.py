from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import ml
from odin.stats import (train_valid_test_split, sampling_iter)
from odin.utils import (args_parse, ctext, Progbar, as_tuple_of_shape)
from odin import fuel as F, visual as V, nnet as N, backend as K

from const import PATH_ACOUSTIC_FEAT, TRAIN_DATA, PATH_EXP

# ===========================================================================
# Configs
# ===========================================================================
args = args_parse([
    ('-feat', "Acoustic feature", ('mspec', 'mfcc'), 'mspec'),
    ('-pool', "max or avg pooling", ('max', 'avg'), 'max'),
    ('-l', "audio segmenting length in second", None, 3),
    ('--debug', "enable debug mode", None, False),
])
FEAT = args.feat
DEBUG = bool(args.debug)
NAME = '_'.join(['tvec', args.feat, args.pool, str(int(args.l))])
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
    [('Conv', 'Dense'), dict(b_init=None, activation=K.linear, pad='same')],
        ['BatchNorm', dict(activation=K.relu)],
        ['Pool', dict(mode=args.pool)]):
  t_vec = N.Sequence([
      N.Dimshuffle(pattern=(0, 1, 2, 'x')),

      N.Conv(num_filters=5, filter_size=(5, 1), strides=1),
      N.BatchNorm(),
      N.Pool(pool_size=(5, 1), strides=None),

      N.Conv(num_filters=5, filter_size=(5, 1), strides=1),
      N.BatchNorm(),
      N.Pool(pool_size=(5, 1), strides=None),

      N.Flatten(outdim=2),
      N.Dense(512), N.BatchNorm(),
      N.Dense(512), N.BatchNorm(),
      N.Dense(n_speakers, b_init=init_ops.constant_initializer(0),
              name='OutputSpeaker')
  ], debug=1, name='Tvector')
# ====== create outputs ====== #
y_logit = t_vec(X)
y_proba = tf.nn.softmax(y_logit)

# ===========================================================================
# Evaluate and save the log
# ===========================================================================
