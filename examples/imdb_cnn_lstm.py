from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,cpu,theano,seed=12082518'
import cPickle
from itertools import chain

import numpy as np

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
nb_epoch = 3


# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.load_imdb(nb_words=max_features, maxlen=maxlen)

X_train = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_train', for_training=True)
X_score = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_score', for_training=False)
y = K.placeholder(shape=(None,), name='y', dtype='int32')

# ===========================================================================
# Build model
# ===========================================================================
f = N.Sequence([
    N.Embedding(max_features, embedding_size),
    N.Dropout(0.25),
    N.Dimshuffle(pattern=(0, 2, 1, 'x')), # convolution on time dimension
    N.Conv2D(nb_filter,
             filter_size=(filter_length, 1),
             pad='valid',
             stride=(1, 1),
             activation=K.relu),
    N.Pool2D(pool_size=(pool_length, 1), mode='max'),
    N.Squeeze(axis=-1),
    N.Dimshuffle(pattern=(0, 2, 1)),
    N.Merge([
        N.Dense(lstm_output_size, activation=K.linear, name='ingate'), # input-gate
        N.Dense(lstm_output_size, activation=K.linear, name='forgetgate'), # forget-gate
        N.Dense(lstm_output_size, activation=K.linear, name='cellupdate'), # cell-update
        N.Dense(lstm_output_size, activation=K.linear, name='outgate') # output-gate
    ], merge_function=K.concatenate),
    N.LSTM(num_units=lstm_output_size)[:, -1],
    N.Dense(1, activation=K.sigmoid)
])
y_pred_train = f(X_train)[0]
y_pred_score = f(X_score)[0]

cost_train = K.mean(K.binary_crossentropy(y_pred_train, y))
cost_score = K.mean(K.binary_accuracy(y_pred_score, y))

parameters = f.parameters

updates = K.optimizers.adam(cost_train, parameters, learning_rate=0.001)

print('Building training function ...')
f_train = K.function([X_train, y], cost_train, updates)
print('Building scoring function ...')
f_score = K.function([X_score, y], cost_score)

# ===========================================================================
# Test
# ===========================================================================
trainer = training.MainLoop(batch_size=batch_size,
                            seed=12082518, shuffle_level=1)
trainer.set_save(utils.get_modelpath(name='imdb.ai', override=True), f)
trainer.set_task(f_train, (ds['X_train'], ds['y_train']), epoch=nb_epoch, name='Train')
trainer.set_subtask(f_score, (ds['X_test'], ds['y_test']), freq=0.5, name='Valid')
trainer.set_callback([
    training.History(),
    training.ProgressMonitor('Train', format='Results: %.4f'),
    training.ProgressMonitor('Valid', format='Results: %.4f')
])
trainer.run()

# ===========================================================================
# Visualization
# ===========================================================================
trainer['History'].print_batch('Train')
trainer['History'].print_epoch('Valid')
