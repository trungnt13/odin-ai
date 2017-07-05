from __future__ import print_function, division, absolute_import

from odin import backend as K, nnet as N
import tensorflow as tf


@N.ModelDescriptor
def cifar_cnn(X, nb_labels):
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
        N.Dense(nb_labels)
    ], debug=True)
    return f(X)
