from __future__ import print_function, division, absolute_import

from odin.basic import add_shape
from odin import nnet as N, backend as K
import edward as ed
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
    n = kwargs.get('n', 10)
    batch_size = K.get_shape(X)[0]
    if batch_size is None:
        raise ValueError("You must specify batch_size dimension for the input placeholder.")
    # ====== init ====== #
    if saved_states is None:
        # Encoder
        f_inference = N.Sequence([
            N.Reshape(shape=(-1, 28, 28, 1)),
            N.Conv(num_filters=32, filter_size=5, strides=2, pad='same', b_init=None),
            N.BatchNorm(activation=K.elu),

            N.Conv(num_filters=64, filter_size=5, strides=2, pad='same', b_init=None),
            N.BatchNorm(activation=K.elu),

            N.Conv(num_filters=128, filter_size=5, b_init=None),
            N.BatchNorm(activation=K.elu),

            N.Dropout(level=0.1),
            N.Flatten(outdim=2),

            N.Dense(num_units=n * 2, b_init=None),
            N.BatchNorm(axes=0)
        ], debug=True, name='Encoder')
        # Decoder
        f_generative = N.Sequence([
            N.Dimshuffle(pattern=(0, 'x', 'x', 1)),
            N.TransposeConv(num_filters=128, filter_size=3, pad='valid', b_init=None),
            N.BatchNorm(activation=K.elu),

            N.TransposeConv(num_filters=64, filter_size=5, pad='valid', b_init=None),
            N.BatchNorm(activation=K.elu),

            N.TransposeConv(num_filters=32, filter_size=5, strides=2, pad='same', b_init=None),
            N.BatchNorm(activation=K.elu),

            N.TransposeConv(num_filters=1, filter_size=5, strides=2, pad='same', b_init=None),
            N.BatchNorm(activation=K.linear),

            N.Flatten(outdim=3)
        ], debug=True, name="Decoder")
    else:
        f_inference, f_generative = saved_states
    # ====== Perfrom ====== #
    # Encoder
    y_encoder = f_inference(K.cast(X, 'float32'))
    mu = y_encoder[:, :n]
    sigma = K.softplus(y_encoder[:, n:])
    qz = Normal(mu=mu, sigma=sigma, name='Normal_qz')
    # Decoder
    z = Normal(mu=K.zeros(shape=(batch_size, n)),
               sigma=K.ones(shape=(batch_size, n)), name="Normal_pz")
    logits = f_generative(z)
    X_reconstruct = Bernoulli(logits=logits)
    # inference
    params = f_inference.parameters + f_generative.parameters
    inference = ed.KLqp(latent_vars={z: qz}, data={X_reconstruct: X})
    inference.initialize()
    # ====== get cost for training ====== #
    # Bind p(x, z) and q(z | x) to the same placeholder for x.
    if K.is_training():
        import tensorflow as tf
        if True:
            optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
            updates = optimizer.apply_gradients(
                optimizer.compute_gradients(inference.loss, var_list=params))
            init = tf.global_variables_initializer()
            init.run()
            f_train = K.function(X, inference.loss, updates)
        else:
            optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
            inference.initialize(optimizer=optimizer, var_list=params)
            init = tf.global_variables_initializer()
            init.run()
            f_train = lambda x: inference.update(feed_dict={X: x})['loss']
    else:
        samples = K.sigmoid(logits)
    return (f_train if K.is_training() else samples, z), (f_inference, f_generative)


@N.ModelDescriptor
def feedforward_vae(X, X1, f):
    if f is None:
        f = N.Sequence([
            N.Dense(num_units=10, activation=K.softmax),
            N.Dropout(level=0.5)
        ])
    return f(X), f
