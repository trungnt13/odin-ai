from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K

# with slim.arg_scope([slim.conv2d_transpose],
#                     activation_fn=tf.nn.elu,
#                     normalizer_fn=slim.batch_norm,
#                     normalizer_params={'scale': True}):
#   net = tf.reshape(z, [M, 1, 1, d])
#   net = slim.conv2d_transpose(net, 128, 3, padding='VALID')
#   net = slim.conv2d_transpose(net, 64, 5, padding='VALID')
#   net = slim.conv2d_transpose(net, 32, 5, stride=2)
#   net = slim.conv2d_transpose(net, 1, 5, stride=2, activation_fn=None)
#   net = slim.flatten(net)

# with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                     activation_fn=tf.nn.elu,
#                     normalizer_fn=slim.batch_norm,
#                     normalizer_params={'scale': True}):
#   net = tf.reshape(x, [M, 28, 28, 1])
#   net = slim.conv2d(net, 32, 5, stride=2)
#   net = slim.conv2d(net, 64, 5, stride=2)
#   net = slim.conv2d(net, 128, 5, padding='VALID')
#   net = slim.dropout(net, 0.9)
#   net = slim.flatten(net)
#   params = slim.fully_connected(net, d * 2, activation_fn=None)


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
    else:
        f_inference = saved_states
    # ====== Perfrom ====== #
    y_encoder = f_inference(X)
    y_decoder = f_inference.T(y_encoder)
    return [y_encoder, y_decoder], f_inference


@N.ModelDescriptor
def feedforward_vae(X, X1, f):
    if f is None:
        f = N.Sequence([
            N.Dense(num_units=10, activation=K.softmax),
            N.Dropout(level=0.5)
        ])
    return f(X), f
