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

import os
os.environ['ODIN'] = 'float32,%s' % ('gpu')
from six.moves import cPickle

import numpy as np
import tensorflow as tf
np.random.seed(1208)

from odin import nnet as N, backend as K, fuel as F, stats
from odin.stats import train_valid_test_split
from odin.utils import get_modelpath, stdio, get_logpath, get_datasetpath
from odin import training
from odin.visual import print_dist

# set log path
stdio(path=get_logpath('digit_audio.log', override=True))

# ===========================================================================
# Fun constant
# ===========================================================================
nb_vad_mixture = 3
nb_classes = 10
hop_length = 0.005
features = ['mspec']

# ===========================================================================
# Get wav and process new dataset configuration
# ===========================================================================
# ====== process new features ====== #
output_path = get_datasetpath(name='digit', override=False)
ds = F.Dataset(output_path, read_only=True)
print(ds)

# ===========================================================================
# Create feeder
# ===========================================================================
indices = ds['indices'].items()
longest_utterances = max(int(end) - int(start) - 1
                         for _, (start, end) in indices)
print("Longest Utterance:", longest_utterances)
np.random.shuffle(indices)
n = len(indices)
train, valid, test = train_valid_test_split(indices, train=0.6, inc_test=True,
                                            cluster_func=lambda x: x[0][0],
                                            seed=12082518)
print("Traning set")
print(print_dist(stats.freqcount([int(i[0][0]) for i in train]), show_number=True))
print("Validation set")
print(print_dist(stats.freqcount([int(i[0][0]) for i in valid]), show_number=True))
print("Test set")
print(print_dist(stats.freqcount([int(i[0][0]) for i in test]), show_number=True))
# ===========================================================================
# Create feeders
# ===========================================================================
data = [ds[i] for i in features]
train = F.Feeder(F.DataDescriptor(data=data, indices=train),
                 ncpu=4)
valid = F.Feeder(F.DataDescriptor(data=data, indices=valid),
                 ncpu=2)
test = F.Feeder(F.DataDescriptor(data=data, indices=test),
                ncpu=2)

recipes = [
    F.recipes.Slice(slices=slice(0, 40), axis=-1, data_idx=0),
    F.recipes.Name2Trans(converter_func=lambda x: int(x[0])),
    F.recipes.Sequencing(frame_length=longest_utterances, hop_length=1,
                         end='pad', endvalue=0, endmode='post',
                         label_transform=F.recipes.last_seen,
                         label_idx=-1),
]
train.set_recipes(recipes)
test.set_recipes(recipes)
valid.set_recipes(recipes)
print(train)

with open('/tmp/test_feeder', 'w') as f:
    cPickle.dump(test, f, protocol=2)

inputs = [K.placeholder(shape=(None,) + shape[1:], name='input_%d' % i)
          for i, shape in enumerate(train.shape)]
print("Inputs:", str(inputs))
X = inputs[:-1]
y = inputs[-1]
# ===========================================================================
# Create network
# ===========================================================================
with N.nnop_scope(ops=['Pool'], mode='max', pool_size=2):
    f = N.Sequence([
        # ====== CNN ====== #
        N.Dimshuffle(pattern=(0, 1, 2, 'x')),
        N.Conv(num_filters=32, filter_size=3, pad='same', strides=1,
               b_init=None, activation=K.linear),
        N.BatchNorm(activation=K.relu),

        N.Conv(num_filters=64, filter_size=3, pad='same', strides=1,
               b_init=None, activation=K.linear),
        N.BatchNorm(activation=K.relu),
        N.Pool(strides=None, pad='valid'),
        # ====== RNN ====== #
        N.Flatten(outdim=3),
        N.CudnnRNN(128, rnn_mode='lstm', num_layers=1,
                   bidirectional=True),
        # ====== Dense ====== #
        N.Flatten(outdim=2),
        N.Dense(num_units=1024, activation=K.relu),
        N.Dropout(level=0.5), # adding dropout does not help
        N.Dense(num_units=512, activation=K.relu),
        N.Dense(num_units=nb_classes)
    ], debug=True)
y_pred_logits = f(X)
y_pred_prob = tf.nn.softmax(y_pred_logits)
y_onehot = tf.one_hot(tf.cast(y, dtype='int32'), depth=nb_classes)
# ====== create cost ====== #
cost_ce = tf.losses.softmax_cross_entropy(y_onehot, y_pred_logits)
cost_acc = K.metrics.categorical_accuracy(y_pred_prob, y)
cost_cm = K.metrics.confusion_matrix(y_pred=y_pred_prob, y_true=y,
                                     labels=nb_classes)

# ====== create optimizer ====== #
parameters = [p for p in f.parameters
              if K.role.has_roles(p, [K.role.Weight,
                                      K.role.Bias])]
optimizer = K.optimizers.Adam(lr=0.0001)
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
task = training.MainLoop(batch_size=8, seed=12, shuffle_level=2)
task.set_checkpoint(get_modelpath(name='digit_audio_ai', override=True), f)
task.set_callbacks([
    training.NaNDetector(),
    training.EarlyStopGeneralizationLoss('valid', cost_ce, threshold=5)
])
task.set_train_task(f_train, train, epoch=6, name='train')
task.set_valid_task(f_score, valid, freq=training.Timer(percentage=0.6),
                    name='valid')
task.set_eval_task(f_score, test, name='test')
task.run()
