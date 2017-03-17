from __future__ import print_function, division, absolute_import

from odin.basic import add_shape
from odin import nnet as N, backend as K
from edward.models import Bernoulli, Normal


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
    d = kwargs.get('n', 10)
    # ====== init ====== #
    if saved_states is None:
        # Encoder
        f_inference = N.Sequence([
            N.Reshape(shape=(-1, 28, 28)),
            N.Dimshuffle(pattern=(0, 1, 2, 'x')),
            N.Conv(num_filters=32, filter_size=5, strides=2, pad='same',
                   b_init=None, activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.Conv(num_filters=64, filter_size=5, strides=2, pad='same',
                   b_init=None, activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.Conv(num_filters=128, filter_size=5, strides=1, pad='valid',
                   b_init=None, activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.Dropout(level=0.1),
            N.Flatten(outdim=2),
            N.Dense(num_units=d * 2, b_init=None, activation=K.linear), # *2 for mu and sigma
            N.BatchNorm(axes=0)
        ], debug=True, name="Encoder")
        # Decoder
        f_generative = N.Sequence([
            N.Dimshuffle(pattern=(0, 'x', 'x', 1)),
            N.TransposeConv(num_filters=128, filter_size=3,
                b_init=None, strides=1, pad='valid', activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.TransposeConv(num_filters=64, filter_size=5,
                b_init=None, strides=1, pad='valid', activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.TransposeConv(num_filters=32, filter_size=5,
                b_init=None, strides=2, pad='same', activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.elu),

            N.TransposeConv(num_filters=1, filter_size=5,
                b_init=None, strides=2, pad='same', activation=K.linear),
            N.BatchNorm(axes='auto', activation=K.linear),

            N.Flatten(outdim=2)
        ], debug=True, name="Decoder")
    else:
        f_inference, f_generative = saved_states
    # ====== Perfrom ====== #
    # Encoder
    y_encoder = f_inference(X)
    mu = add_shape(y_encoder[:, :n], (None, n))
    sigma = add_shape(K.softplus(y_encoder[:, n:]), (None, n))
    qz = Normal(mu=mu, sigma=sigma)
    # Decoder
    z = Normal(mu=K.zeros(shape=K.get_shape(mu, native=True)),
               sigma=K.ones(shape=K.get_shape(sigma, native=True)))
    logits = f_generative(z)
    x = K.add_shape(Bernoulli(logits=logits), K.get_shape(logits))
    return [z, qz, x], (f_inference, f_generative)


@N.ModelDescriptor
def feedforward_vae(X, X1, f):
    if f is None:
        f = N.Sequence([
            N.Dense(num_units=10, activation=K.softmax),
            N.Dropout(level=0.5)
        ])
    return f(X), f
