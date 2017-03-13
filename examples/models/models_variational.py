from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K


@N.ModelDescriptor
def convolutional_vae(X, saved_states, **kwargs):
    """ convolutional_vae
    Return
    ------
    [y_encoder, y_decoder]

    States
    ------
    [f_inference (encoder), f_generative (decoder)]
    """
    n = kwargs.get('n', 256)
    input_shape = K.get_shape(X)
    # ====== init ====== #
    if saved_states is None:
        # encoder
        f_inference = N.Sequence([
            N.Dimshuffle(pattern=(0, 1, 2, 'x')),
            N.Conv(num_filters=32, filter_size=5, strides=2, pad='same',
                   activation=K.relu),
            N.Conv(num_filters=64, filter_size=5, strides=2, pad='same',
                   activation=K.relu),
            N.Dropout(level=0.9),
            N.Flatten(outdim=2),
            N.Dense(num_units=n, activation=K.linear)
        ], debug=True, name="Encoder")

        f_generative = N.Sequence([
            f_inference.ops[-1].T,
            f_inference.ops[-2].T,
            N.TransposeConv(num_filters=32, filter_size=5,
                strides=2, pad='same', activation=K.relu),
            N.TransposeConv(num_filters=1, filter_size=5,
                strides=2, pad='same', activation=K.relu),
            N.Squeeze(axis=-1)
        ], debug=True)
    else:
        f_inference, f_generative = saved_states
    # ====== Perfrom ====== #
    y_encoder = f_inference(X)
    y_decoder = f_generative(y_encoder)
    exit()
    return [y_encoder, y_decoder], (f_inference, f_generative)


@N.ModelDescriptor
def feedforward_vae(X, X1, f):
    if f is None:
        f = N.Sequence([
            N.Dense(num_units=10, activation=K.softmax),
            N.Dropout(level=0.5)
        ])
    return f(X), f
