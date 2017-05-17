from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow,seed=12082518'

import numpy as np

from odin import fuel as F, nnet as N, backend as K, training, utils
MODEL_PATH = utils.get_modelpath(name='cifar10.ai', override=True)
REPORT_PATH = utils.get_logpath(name='cifar10.pdf', override=True)

# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.load_cifar10()
nb_labels = 10
print(ds)
X_train = ds['X_train'][:].astype('float32') / 255.
y_train = ds['y_train'][:]
X_test = ds['X_test'][:].astype('float32') / 255.
y_test = ds['y_test'][:]

# ===========================================================================
# Create network
# ===========================================================================
X = K.placeholder(shape=(None,) + X_train.shape[1:], name='X')
y = K.placeholder(shape=(None,), name='y_true', dtype='int32')

f = N.Sequence([
    N.Dimshuffle(pattern=(0, 2, 3, 1)),
    N.Conv(32, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Conv(32, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),

    N.Conv(64, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Conv(64, (3, 3), pad='same', stride=(1, 1), activation=K.relu),
    N.Pool(pool_size=(2, 2), strides=None, mode='max'),
    N.Dropout(level=0.25),

    N.Flatten(outdim=2),
    N.Dense(512, activation=K.relu),
    N.Dropout(level=0.5),
    N.Dense(nb_labels, activation=K.softmax)
], debug=True)
K.set_training(True); y_training = f(X)
K.set_training(False); y_scoring = f(X)

cost_train1 = K.mean(K.categorical_crossentropy(y_training, y), name='TrainCE')
cost_train2 = K.mean(K.categorical_accuracy(y_training, y), name='TrainACC')

cost_score1 = K.mean(K.categorical_crossentropy(y_scoring, y), name='ScoreCE')
cost_score2 = K.mean(K.categorical_accuracy(y_scoring, y), name='ScoreACC')
confusion = K.confusion_matrix(y_scoring, y, labels=nb_labels)

parameters = f.parameters
print('Parameters:', [p.name for p in parameters])

optz = K.optimizers.Adam(lr=0.001)
# ===========================================================================
# Create trainer
# ===========================================================================
print("Create trainer ...")
trainer, hist = training.standard_trainer(
    train_data=[X_train, y_train], valid_data=[X_test, y_test],
    cost_train=[cost_train1, cost_train2], cost_score=[cost_score1, cost_score2],
    confusion_matrix=confusion,
    parameters=parameters, optimizer=optz,
    batch_size=128, nb_epoch=5, valid_freq=1.,
    report_path=REPORT_PATH
)
trainer.run()
