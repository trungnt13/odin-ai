# ===========================================================================
# Should reach: > 0.9861111% on test set with default configuration
# Note: without CNN the performance is decreased by 10%
# One titan X:
#  Benchmark TRAIN-batch: 0.11614914765 (s)
#  Benchmark TRAIN-epoch: 5.98400415693 (s)
#  Benchmark PRED-batch: 0.183033730263 (s)
#  Benchmark PRED-epoch: 3.5595933524 (s)
# NOTE: the performance of float16 and float32 dataset are identical
# ===========================================================================
from __future__ import print_function, absolute_import, division

import matplotlib
matplotlib.use("Agg")

from odin.utils import ArgController

# ====== parse arguments ====== #
args = ArgController(
).add('-dev', 'gpu or cpu', 'gpu'
).add('-dt', 'dtype: float32 or float16', 'float16'
).add('-feat', 'feature type: mfcc, mspec, spec, qspec, qmspec, qmfcc', 'mspec'
).add('-cnn', 'enable CNN or not', True
).add('-vad', 'number of GMM component for VAD', 2
# for training
).add('-lr', 'learning rate', 0.001
).add('-epoch', 'number of epoch', 5
).add('-bs', 'batch size', 8
).parse()

# ====== import ====== #
import os
os.environ['ODIN'] = 'float32,%s' % (args['dev'])

import numpy as np
import tensorflow as tf
np.random.seed(1208)

from odin import nnet as N, backend as K, fuel as F, stats
from odin.utils import get_modelpath, stdio, get_logpath, get_datasetpath
from odin import training

# set log path
stdio(path=get_logpath('digit_audio.log', override=True))

# ===========================================================================
# Get wav and process new dataset configuration
# ===========================================================================
# ====== process new features ====== #
if False:
    datapath = F.load_digit_wav()
    output_path = get_datasetpath(name='digit_audio', override=True)
    feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', sr_new=8000,
                             win=0.025, hop=0.005, nb_melfilters=40, nb_ceps=13,
                             get_spec=True, get_qspec=False, get_phase=False,
                             get_pitch=True, get_f0=True,
                             get_vad=args['vad'], get_energy=True, get_delta=2,
                             fmin=64, fmax=None, preemphasis=None,
                             pitch_threshold=0.3, pitch_fmax=260,
                             vad_smooth=3, vad_minlen=0.1,
                             pca=True, pca_whiten=False,
                             save_stats=True, substitute_nan=None,
                             dtype='float16', datatype='memmap',
                             ncache=0.12, ncpu=12)
    feat.run()
    ds = F.Dataset(output_path, read_only=True)
# ====== use online features ====== #
else:
    ds = F.load_digit_audio()
nb_classes = 10
# ===========================================================================
# Create feeder
# ===========================================================================
indices = [(name, start, end) for name, (start, end) in ds['indices'].iteritems(True)]
longest_utterances = max(int(end) - int(start) - 1
                         for i, start, end in indices)
longest_vad = max(end - start
                  for name, vad in ds['vadids']
                  for (start, end) in vad)
print("Longest Utterance:", longest_utterances)
print("Longest Vad:", longest_vad)

np.random.shuffle(indices)
n = len(indices)
train = indices[:int(0.6 * n)]
valid = indices[int(0.6 * n):int(0.8 * n)]
test = indices[int(0.8 * n):]
print('Nb train:', len(train), stats.freqcount([int(i[0][0]) for i in train]))
print('Nb valid:', len(valid), stats.freqcount([int(i[0][0]) for i in valid]))
print('Nb test:', len(test), stats.freqcount([int(i[0][0]) for i in test]))

train_feeder = F.Feeder(ds[args['feat']], train, ncpu=1)
test_feeder = F.Feeder(ds[args['feat']], test, ncpu=2)
valid_feeder = F.Feeder(ds[args['feat']], valid, ncpu=2)

recipes = [
    F.recipes.Name2Trans(converter_func=lambda x: int(x[0])),
    F.recipes.Normalization(
        mean=ds[args['feat'] + '_mean'],
        std=ds[args['feat'] + '_std'],
        local_normalize=False
    ),
    F.recipes.Sequencing(frame_length=longest_utterances, hop_length=1,
                         end='pad', endvalue=0, endmode='post',
                         transcription_transform=lambda x: x[-1]),
    F.recipes.CreateBatch(),
]

train_feeder.set_recipes(recipes)
test_feeder.set_recipes(recipes)
valid_feeder.set_recipes(recipes)
print('Feature shape:', train_feeder.shape)
feat_shape = (None,) + train_feeder.shape[1:]

X = K.placeholder(shape=feat_shape, name='X')
y = K.placeholder(shape=(None,), dtype='int32', name='y')
# ===========================================================================
# Create network
# ===========================================================================
f = N.Sequence([
    # ====== CNN ====== #
    N.Dimshuffle(pattern=(0, 1, 2, 'x')),
    N.Conv(num_filters=32, filter_size=3, pad='same', strides=1,
           activation=K.linear),
    N.BatchNorm(activation=K.relu),
    N.Conv(num_filters=64, filter_size=3, pad='same', strides=1,
           activation=K.linear),
    N.BatchNorm(activation=K.relu),
    N.Pool(pool_size=2, strides=None, pad='valid', mode='max'),
    N.Flatten(outdim=3),
    # ====== RNN ====== #
    # N.AutoRNN(128, rnn_mode='lstm', num_layers=3,
    # direction_mode='bidirectional', prefer_cudnn=True),
    # ====== Dense ====== #
    N.Flatten(outdim=2),
    # N.Dropout(level=0.2), # adding dropout does not help
    N.Dense(num_units=1024, activation=K.relu),
    N.Dense(num_units=512, activation=K.relu),
    N.Dense(num_units=nb_classes)
], debug=True)
y_pred_logits = f(X)
y_pred_prob = tf.nn.softmax(y_pred_logits)
y_onehot = tf.one_hot(y, depth=nb_classes)
# ====== create cost ====== #
cost_ce = tf.losses.softmax_cross_entropy(y_onehot, y_pred_logits)
cost_acc = K.metrics.categorical_accuracy(y_pred_prob, y)
cost_cm = K.metrics.confusion_matrix(y_pred_prob, y, labels=nb_classes)

# ====== create optimizer ====== #
parameters = [p for p in f.parameters
              if K.role.has_roles(p, [K.role.Weight,
                                      K.role.Bias])]
optimizer = K.optimizers.Adam(lr=args['lr'])
updates = optimizer.get_updates(cost_ce, parameters)

print('Building training functions ...')
f_train = K.function([X, y], [cost_ce, optimizer.norm, cost_cm],
                     updates=updates, training=True)
print('Building testing functions ...')
f_score = K.function([X, y], [cost_ce, cost_acc, cost_cm], training=False)
print('Building predicting functions ...')
f_pred = K.function(X, y_pred_prob, training=False)

# ===========================================================================
# Build trainer
# ===========================================================================
print('Start training ...')
task = training.MainLoop(batch_size=128, seed=12, shuffle_level=2,
                         print_progress=True, confirm_exit=True)
task.set_save(get_modelpath(name='digit_audio_ai', override=True), f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost_ce, threshold=5)
])
task.set_train_task(f_train, train_feeder, epoch=4,
                    name='train')
task.set_valid_task(f_score, valid_feeder, freq=training.Timer(percentage=0.6),
                    name='valid')
task.set_eval_task(f_score, test_feeder, name='test')
task.run()
