from __future__ import print_function, division, absolute_import

from odin import nnet as N, backend as K


@N.ModelDescriptor
def ladder1(X, y, states, **kwargs):
    noise = kwargs.get('noise', 0.3)
    # hyperparameters that denote the importance of each layer
    denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10]

    if states is None:
        #
        f_encoder = N.Sequence([
            N.Flatten(outdim=2),

            N.Dense(num_units=1024, b_init=None),
            N.BatchNorm(axes=0, noise_level=noise, noise_dims=None,
                activation=K.relu),

            N.Dense(num_units=512, b_init=None),
            N.BatchNorm(axes=0, noise_level=noise, noise_dims=None,
                activation=K.relu),

            N.Dense(num_units=256, b_init=None),
            N.BatchNorm(axes=0, noise_level=noise, noise_dims=None,
                activation=K.relu),

            N.Dense(num_units=128, b_init=None),
            N.BatchNorm(axes=0, noise_level=noise, noise_dims=None,
                activation=K.relu),

            N.Dense(num_units=10, activation=K.softmax),
        ], all_layers=True, debug=True, name='Encoder')
        #
        f_decoder = N.Sequence([
            N.Dense(num_units=128, b_init=None),
            N.BatchNorm(axes=0, activation=K.relu),

            N.Dense(num_units=256, b_init=None),
            N.BatchNorm(axes=0, activation=K.relu),

            N.Dense(num_units=512, b_init=None),
            N.BatchNorm(axes=0, activation=K.relu),

            N.Dense(num_units=1024, b_init=None),
            N.BatchNorm(axes=0, activation=K.relu),

            N.Reshape(shape=(-1, 28, 28)),
        ], all_layers=True, debug=True, name='Decoder')
    else:
        f_encoder, f_decoder = states
    y_encoder_clean = f_encoder(X, noise=-1)[2::2]
    y_encoder_corrp = f_encoder(X, noise=1)[2::2]
    print(len(y_encoder_clean), len(y_encoder_corrp))
    exit()
    return (None, None), [f_encoder, f_decoder]
