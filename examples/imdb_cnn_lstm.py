from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,seed=5218'
from six.moves import cPickle
from itertools import chain
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from odin import (fuel as F,
                  nnet as N,
                  backend as K,
                  training, utils)

# ===========================================================================
# Configuration
# ===========================================================================
# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 2

# ===========================================================================
# TMP
# ===========================================================================
import json
import shutil
import pickle

def _remove_long_seq(maxlen, seq, label):
  """Removes sequences that exceed the maximum length.
  # Arguments
      maxlen: Int, maximum length of the output sequences.
      seq: List of lists, where each sublist is a sequence.
      label: List where each element is an integer.
  # Returns
      new_seq, new_label: shortened lists for `seq` and `label`.
  """
  new_seq, new_label = [], []
  for x, y in zip(seq, label):
    if len(x) < maxlen:
      new_seq.append(x)
      new_label.append(y)
  return new_seq, new_label

 # num_words: max number of words to include. Words are ranked
 #            by how often they occur (in the training set) and only
 #            the most frequent words are kept
 # skip_top: skip the top N most frequently occurring words
 #     (which may not be informative).
 # maxlen: sequences longer than this will be filtered out.
 # seed: random seed for sample shuffling.
 # start_char: The start of a sequence will be marked with this character.
 #     Set to 1 because 0 is usually the padding character.
 # oov_char: words that were cut out because of the `num_words`
 #     or `skip_top` limit will be replaced with this character.
 # index_from: index actual words with this index and higher.

DATASET_DIR = '/home/trung/data/IMDB_org'
OUTPUT_DIR = '/home/trung/data/IMDB'
if os.path.isdir(OUTPUT_DIR):
  shutil.rmtree(OUTPUT_DIR)
os.mkdir(OUTPUT_DIR)
# ====== test dataset ====== #
ds = F.Dataset(path=DATASET_DIR, read_only=True)
print(ds)

x_train, labels_train = np.array(ds['X_train']), np.array(ds['y_train'])
x_test, labels_test = np.array(ds['X_test']), np.array(ds['y_test'])

num_words = None
skip_top = 0
maxlen = None
seed = 113
start_char = 1
oov_char = 2
index_from = 3

np.random.seed(seed)
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
labels_train = labels_train[indices]

indices = np.arange(len(x_test))
np.random.shuffle(indices)
x_test = x_test[indices]
labels_test = labels_test[indices]

xs = np.concatenate([x_train, x_test])
labels = np.concatenate([labels_train, labels_test])

if start_char is not None:
  xs = [[start_char] + [w + index_from for w in x] for x in xs]
elif index_from:
  xs = [[w + index_from for w in x] for x in xs]

if maxlen:
  xs, labels = _remove_long_seq(maxlen, xs, labels)
  if not xs:
    raise ValueError('After filtering for sequences shorter than maxlen=' +
                     str(maxlen) + ', no sequence was kept. '
                     'Increase maxlen.')
if not num_words:
  num_words = max([max(x) for x in xs])

# by convention, use 2 as OOV word
# reserve 'index_from' (=3 by default) characters:
# 0 (padding), 1 (start), 2 (OOV)
if oov_char is not None:
  xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
else:
  xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

idx = len(x_train)
x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

ds.close()

with open(os.path.join(OUTPUT_DIR, 'README'), 'wb') as f:
  f.write(
"""Preprocessed IMDB dataset
The dataset have been
"""
)
exit()
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.load_imdb(nb_words=max_features, maxlen=maxlen)

X_train = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_train')
X_score = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_score')
y = K.placeholder(shape=(None,), name='y', dtype='int32')
exit()
# ===========================================================================
# Build model
# ===========================================================================
f = N.Sequence([
    N.Embedding(max_features, embedding_size),
    N.Dropout(0.25),
    N.Dimshuffle(pattern=(0, 1, 'x', 2)), # convolution on time dimension
    N.Conv(nb_filter,
           filter_size=(filter_length, 1),
           pad='valid',
           stride=(1, 1),
           activation=K.relu),
    N.Pool(pool_size=(pool_length, 1), mode='max'),
    N.Flatten(outdim=3),
    N.Merge([
        N.Dense(lstm_output_size, activation=K.linear, name='ingate'), # input-gate
        N.Dense(lstm_output_size, activation=K.linear, name='forgetgate'), # forget-gate
        N.Dense(lstm_output_size, activation=K.linear, name='cellupdate'), # cell-update
        N.Dense(lstm_output_size, activation=K.linear, name='outgate') # output-gate
    ], merge_function=tf.concat),
    N.LSTM(num_units=lstm_output_size, input_mode='skip')[:, -1],
    N.Dense(1, activation=K.sigmoid)
], debug=True)
K.set_training(True); y_pred_train = f(X_train)
K.set_training(False); y_pred_score = f(X_score)

cost_train = K.mean(K.binary_crossentropy(y_pred_train, y))
cost_score = K.mean(K.binary_accuracy(y_pred_score, y))

parameters = f.parameters
print('Params:', [p.name for p in parameters])

updates = K.optimizers.Adam(lr=0.001).get_updates(cost_train, parameters)

print('Building training function ...')
f_train = K.function([X_train, y], cost_train, updates)
print('Building scoring function ...')
f_score = K.function([X_score, y], cost_score)

# ===========================================================================
# Test
# ===========================================================================
trainer = training.MainLoop(batch_size=batch_size,
                            seed=12082518, shuffle_level=1)
trainer.set_checkpoint(utils.get_modelpath(name='imdb.ai', override=True), f)
trainer.set_task(f_train, (ds['X_train'], ds['y_train']), epoch=nb_epoch, name='Train')
trainer.set_subtask(f_score, (ds['X_test'], ds['y_test']), freq=1., name='Valid')
trainer.set_callback([
    training.History(),
    training.ProgressMonitor('Train', format='Results: {:.4f}'),
    training.ProgressMonitor('Valid', format='Results: {:.4f}')
])
trainer.run()

# ===========================================================================
# Visualization
# ===========================================================================
trainer['History'].print_batch('Train')
trainer['History'].print_epoch('Valid')
