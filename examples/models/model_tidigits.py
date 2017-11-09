from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K


@N.Model
def gender(X, f, **kwargs):
    nb_gender = kwargs.get('nb_gender', 4)
    if f is None:
        f = N.Sequence([
            N.Dimshuffle(pattern=(0, 1, 2, 'x')),
            N.Conv(num_filters=32, filter_size=3, strides=1, b_init=None, pad='valid'),
            N.BatchNorm(activation=K.relu),
            N.Pool(pool_size=2, mode='avg'),

            N.Conv(num_filters=64, filter_size=3, strides=1, b_init=None, pad='valid'),
            N.BatchNorm(activation=K.relu),
            N.Pool(pool_size=2, mode='avg'),

            N.Flatten(outdim=3),
            N.Dense(num_units=512, b_init=None),
            N.BatchNorm(axes=(0, 1)),
            N.AutoRNN(num_units=128, rnn_mode='gru', num_layers=2,
                      input_mode='linear', direction_mode='unidirectional'),

            N.Flatten(outdim=2),
            N.Dense(num_units=nb_gender, activation=K.softmax)
        ], debug=True)
    return f(X), f
