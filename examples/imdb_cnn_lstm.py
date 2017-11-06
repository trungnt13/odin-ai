from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,cpu,seed=12082518'
from six.moves import cPickle
from itertools import chain

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
# Load dataset
# ===========================================================================
ds = F.load_imdb(nb_words=max_features, maxlen=maxlen)

X_train = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_train')
X_score = K.placeholder(shape=(None,) + ds['X_train'].shape[1:],
                        name='X_score')
y = K.placeholder(shape=(None,), name='y', dtype='int32')

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
