from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,theano,gpu,cnmem=0.8'
import cPickle

import numpy as np

from odin import backend as K, nnet as N, utils, fuel as F, training
from odin.basic import has_roles, INITIAL_STATE
from sklearn.metrics import accuracy_score
import lasagne
import keras

max_features = 20000
embedding_size = 128
batch_size = 32
nb_epochs = 3
maxlen = 200
learning_rate = 0.0001

ds = F.load_imdb(nb_words=max_features, maxlen=maxlen)
print(ds)

X = K.placeholder(shape=(None,) + ds['X_train'].shape[1:], name='X', dtype='int32')
y = K.placeholder(shape=(None,), name='y', dtype='int32')


def rmsprop(loss, params, lr=0.001, rho=0.9, epsilon=1e-8):
    lr = K.variable(lr, name='lr')
    rho = K.variable(rho, name='rho')
    shapes = [K.get_shape(p) for p in params]

    grads = K.gradients(loss, params)
    accumulators = [K.variable(np.zeros(shape)) for shape in shapes]

    updates = []
    for p, g, a in zip(params, grads, accumulators):
        # update accumulator
        new_a = rho * a + (1. - rho) * K.square(g)
        updates.append((a, new_a))
        new_p = p - lr * g / (K.sqrt(new_a) + epsilon)
        # apply constraints
        updates.append((p, new_p))
    return updates


# ===========================================================================
# odin
# ===========================================================================
net_odin = N.Sequence([
    N.Embedding(input_size=max_features, output_size=embedding_size),
    N.Dimshuffle(pattern=(0, 'x', 1, 2)),
    N.Conv2D(8, (3, 3), stride=(1, 1), pad='same', activation=K.relu),
    N.Pool2D(pool_size=(2, 2), strides=None, mode='max'),
    N.Dimshuffle(pattern=(0, 2, 1, 3)),
    N.Flatten(outdim=3),
    # ====== LSTM ====== #
    N.Merge([
        N.Dense(64, activation=K.linear, name='ingate'), # input-gate
        N.Dense(64, activation=K.linear, name='forgetgate'), # forget-gate
        N.Dense(64, activation=K.linear, name='cellupdate'), # cell-update
        N.Dense(64, activation=K.linear, name='outgate') # output-gate
    ], merge_function=K.concatenate),
    N.LSTM(num_units=64)[:, -1],
    # N.Flatten(outdim=2),
    N.Dense(1, activation=K.sigmoid)
])
print('Building ODIN network ...')
y_odin = net_odin(X)

# ===========================================================================
# Lasagne
# ===========================================================================
net_lasagne = lasagne.layers.InputLayer(shape=(None,) + ds['X_train'].shape[1:])
net_lasagne.input_var = X
net_lasagne = lasagne.layers.EmbeddingLayer(net_lasagne,
                                            input_size=max_features,
                                            output_size=embedding_size)
net_lasagne = lasagne.layers.DimshuffleLayer(net_lasagne, pattern=(0, 'x', 1, 2))
net_lasagne = lasagne.layers.Conv2DLayer(net_lasagne, 8, (3, 3), stride=(1, 1),
                                         pad='same',
                                         nonlinearity=lasagne.nonlinearities.rectify)
net_lasagne = lasagne.layers.Pool2DLayer(net_lasagne, pool_size=(2, 2),
                                         stride=None, mode='max')
net_lasagne = lasagne.layers.DimshuffleLayer(net_lasagne, pattern=(0, 2, 1, 3))
net_lasagne = lasagne.layers.FlattenLayer(net_lasagne, outdim=3)
net_lasagne = lasagne.layers.LSTMLayer(net_lasagne, num_units=64,
                                       only_return_final=True)
# net_lasagne = lasagne.layers.FlattenLayer(net_lasagne, outdim=2)
net_lasagne = lasagne.layers.DenseLayer(net_lasagne, num_units=1,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

print('Building LASAGNE network ...')
y_lasagne = lasagne.layers.get_output(net_lasagne)

# ===========================================================================
# Keras
# ===========================================================================
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_features, embedding_size, input_length=maxlen))
model.add(keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, 1),
                              output_shape=(1, maxlen, embedding_size)))
model.add(keras.layers.Convolution2D(8, 3, 3, activation='relu',
                                     subsample=(1, 1), border_mode='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(keras.layers.Permute(dims=(2, 1, 3)))
model.add(keras.layers.Reshape(target_shape=(100, 512)))
model.add(keras.layers.LSTM(output_dim=64, return_sequences=False))
# model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(output_dim=1, activation='sigmoid'))

print('Compile KERAS network ...')
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=learning_rate),
              metrics=['accuracy'])
y_keras = model.outputs[0]
cost_train_keras = K.mean(keras.objectives.binary_crossentropy(y, y_keras.ravel()))
cost_score_keras = K.mean(lasagne.objectives.binary_accuracy(y_keras.ravel(), y))
parameters_keras = model.weights
updates_keras = lasagne.updates.rmsprop(cost_train_keras, parameters_keras,
                                    learning_rate=learning_rate)
print('Build training function KERAS ...')
f_train_keras = K.function([model.inputs[0], y], cost_train_keras, updates_keras)
print('Build scoring function KERAS ...')
f_score_keras = K.function([model.inputs[0], y], cost_score_keras)
print('Build predicting function KERAS ...')
f_pred_keras = K.function(model.inputs, y_keras)

# ===========================================================================
# Build training cost
# ===========================================================================
cost_train_odin = K.mean(K.binary_crossentropy(y_odin, y))
cost_score_odin = K.mean(K.binary_accuracy(y_odin, y))
parameters_odin = [p for p in net_odin.parameters if not has_roles(p, INITIAL_STATE)]
updates_odin = K.optimizers.RMSProp(lr=learning_rate)(cost_train_odin, parameters_odin)
print('Build training function ODIN ...')
f_train_odin = K.function([X, y], cost_train_odin, updates_odin)
print('Build scoring function ODIN ...')
f_score_odin = K.function([X, y], cost_score_odin)
print('Build predicting function ODIN ...')
f_pred_odin = K.function([X], y_odin)

cost_train_lasagne = K.mean(K.binary_crossentropy(y_lasagne, y))
cost_score_lasagne = K.mean(lasagne.objectives.binary_accuracy(y_lasagne, y))
parameters_lasagne = lasagne.layers.get_all_params(net_lasagne, trainable=True)
updates_lasagne = lasagne.updates.rmsprop(cost_train_lasagne, parameters_lasagne,
                                          learning_rate=learning_rate)
print('Build training function LASAGNE ...')
f_train_lasagne = K.function([X, y], cost_train_lasagne, updates_lasagne)
print('Build scoring function LASAGNE ...')
f_score_lasagne = K.function([X, y], cost_score_lasagne)
print('Build predicting function ODIN ...')
f_pred_lasagne = K.function([X], y_lasagne)

print('Params KERAS  :', [i.shape for i in model.get_weights()], len(model.get_weights()))
print('Params ODIN   :', [K.get_value(i).shape for i in parameters_odin], len(parameters_odin))
print('Params LASAGNE:', [i.get_value().shape for i in parameters_lasagne], len(parameters_lasagne))


# ===========================================================================
# Manually training
# ===========================================================================
def manually_train(f_train, f_pred):
    X_train = ds['X_train'][:]
    y_train = ds['y_train'][:]

    np.random.seed(1208251813)
    n = X_train.shape[0]
    for i in range(nb_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_tmp = X_train[idx]
        y_tmp = y_train[idx]
        start = 0
        prog = utils.Progbar(target=n)
        all_loss = []
        while start < n:
            end = start + batch_size
            loss = f_train(X_tmp[start:end], y_tmp[start:end])
            all_loss.append(loss)
            start = end
            prog.title = 'loss:%.4f' % loss
            prog.add(batch_size)
        # ====== validation ====== #
        pred = []
        for i in ds['X_test']:
            pred.append(f_pred(i))
        pred = (np.concatenate(pred, axis=0) > 0.5).astype('uint8')
        print("Mean loss:", np.mean(all_loss), ' Accuracy:', accuracy_score(ds['y_test'][:], pred))


# print("************ Manually train ODIN:")
# manually_train(f_train_odin, f_pred_odin)
# print("************ Manually train KERAS:")
# manually_train(f_train_keras, f_pred_keras)
# print("************ Manually train LASAGNE:")
# manually_train(f_train_lasagne, f_pred_lasagne)


# ===========================================================================
# Manually auto batch training
# ===========================================================================
def autobatch_train(f_train, f_pred):
    np.random.seed(1208251813)
    for i in range(nb_epochs):
        prog = utils.Progbar(target=ds['X_train'].shape[0])
        all_loss = []
        for x, y in zip(ds['X_train'].set_batch(batch_size, seed=1208251813, shuffle_level=1),
                        ds['y_train'].set_batch(batch_size, seed=1208251813, shuffle_level=1)):
            loss = f_train(x, y)
            all_loss.append(loss)
            prog.title = 'loss:%.4f' % loss
            prog.add(batch_size)
        # ====== validation ====== #
        pred = []
        for i in ds['X_test']:
            pred.append(f_pred(i))
        pred = (np.concatenate(pred, axis=0) > 0.5).astype('uint8')
        print("Mean loss:", np.mean(all_loss), ' Accuracy:', accuracy_score(ds['y_test'][:], pred))


# print("************ Autobatch train ODIN:")
# autobatch_train(f_train_odin, f_pred_odin)
# print("************ Autobatch train KERAS:")
# autobatch_train(f_train_keras, f_pred_keras)
# print("************ Autobatch train LASAGNE:")
# autobatch_train(f_train_lasagne, f_pred_lasagne)


# ===========================================================================
# Automatic Trainer
# ===========================================================================
def create_trainer(f_train, f_score):
    trainer = training.MainLoop(batch_size=batch_size, seed=12082518,
                                shuffle_level=2)
    trainer.set_task(f_train, (ds['X_train'], ds['y_train']),
                     epoch=nb_epochs, name='Train')
    trainer.set_subtask(f_score, (ds['X_test'], ds['y_test']),
                        freq=1.0, name='Valid')
    trainer.set_callback([
        training.ProgressMonitor('Train', 'Result:%.4f'),
        training.ProgressMonitor('Valid', 'Result:%.4f'),
        training.History()
    ])
    return trainer


# model.fit(ds['X_train'][:], ds['y_train'][:],
#           batch_size=batch_size, nb_epoch=nb_epochs,
#           validation_data=(ds['X_test'][:], ds['y_test'][:]))

print('Created trainer for ODIN!')
trainer_odin = create_trainer(f_train_odin, f_score_odin)
print('Created trainer for LASAGNE!')
trainer_lasagne = create_trainer(f_train_lasagne, f_score_lasagne)
print('Created trainer for KERAS!')
trainer_keras = create_trainer(f_train_keras, f_score_keras)

# ====== run ====== #
print('************ ODIN ************')
trainer_odin.run()
print('************ KERAS ************')
trainer_keras.run()
print('************ LASAGNE ************')
trainer_lasagne.run()

# ====== visual ====== #
# trainer_lasagne['History'].print_epoch('Valid')
# trainer_odin['History'].print_epoch('Valid')
