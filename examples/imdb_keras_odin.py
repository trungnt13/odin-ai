from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,tensorflow,gpu'
import cPickle

import numpy as np

from odin import backend as K, nnet as N, utils, fuel as F, training
from odin.basic import has_roles, INITIAL_STATE
from sklearn.metrics import accuracy_score

# ===========================================================================
# Constants
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

ds = F.load_imdb(nb_words=max_features, maxlen=maxlen)
print(ds)

# ===========================================================================
# ODIN
# ===========================================================================
X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X', dtype='int32')
y = K.placeholder(shape=(None,), name='y', dtype='int32')

net_odin = N.Sequence([
    N.Embedding(input_size=max_features, output_size=embedding_size),
    N.Dropout(level=0.25),
    N.Dimshuffle(pattern=(0, 1, 'x', 2)),
    N.Conv(nb_filter, (filter_length, 1), strides=1, pad='valid',
           activation=K.relu),
    N.Pool(pool_size=(pool_length, 1), pad='valid', mode='max'),
    N.Flatten(outdim=3),
    # ====== LSTM ====== #
    N.Merge([
        N.Dense(lstm_output_size, activation=K.linear, name='ingate'), # input-gate
        N.Dense(lstm_output_size, activation=K.linear, name='forgetgate'), # forget-gate
        N.Dense(lstm_output_size, activation=K.linear, name='cellupdate'), # cell-update
        N.Dense(lstm_output_size, activation=K.linear, name='outgate') # output-gate
    ], merge_function=K.concatenate),
    N.LSTM(num_units=lstm_output_size, input_mode='skip')[:, -1],
    N.Dense(1, activation=K.sigmoid)
], debug=True)

print('Building ODIN network ...')
K.set_training(True); y_odin_train = net_odin(X)
K.set_training(False); y_odin_score = net_odin(X)

cost_train = K.mean(K.binary_crossentropy(y_odin_train, y))
cost_score = K.mean(K.binary_accuracy(y_odin_score, y))
parameters = [p for p in net_odin.parameters if not has_roles(p, INITIAL_STATE)]
print('Params:', [p.name for p in parameters])

opt = K.optimizers.RMSProp()
updates = opt.get_updates(cost_train, parameters)

print('Build training function ODIN ...')
f_train = K.function([X, y], cost_train, updates)
print('Build scoring function ODIN ...')
f_score = K.function([X, y], cost_score)
print('Build predicting function ODIN ...')
f_pred = K.function(X, y_odin_score)

trainer = training.MainLoop(batch_size=batch_size, seed=12082518,
                            shuffle_level=2)
trainer.set_task(f_train, (ds['X_train'], ds['y_train']),
                 epoch=nb_epoch, name='Train')
trainer.set_subtask(f_score, (ds['X_test'], ds['y_test']),
                    freq=1.0, name='Valid')
trainer.set_callback([
    training.ProgressMonitor('Train', 'Result: {:.4f}'),
    training.ProgressMonitor('Valid', 'Result: {:.4f}'),
    training.History()
])
trainer.run()

# ===========================================================================
# Keras
# ===========================================================================
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("Building KERAS network ...")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(ds['X_train'][:], ds['y_train'][:],
          batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(ds['X_test'][:], ds['y_test'][:]))

# ===========================================================================
# Final evaluation
# ===========================================================================
y_pred = f_pred(ds['X_test'][:]) >= 0.5
print('ODIN test accuracy:', accuracy_score(ds['y_test'][:], y_pred))
y_pred = model.predict(ds['X_test'][:]) >= 0.5
print('KERAS test accuracy:', accuracy_score(ds['y_test'][:], y_pred))
