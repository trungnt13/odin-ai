from .base import Model
import tensorflow as tf

from odin import backend as K
from odin.nnet.base import Dense


def renorm_rms(x, target_rms=1.0, axis=1):
    """ scales the data such that RMS is 1.0
    """
    # scale = sqrt(x^t x / (D * target_rms^2)).
    D = tf.sqrt(tf.cast(tf.shape(x)[axis], 'float32'))
    x_rms = tf.sqrt(tf.reduce_sum(x * x, axis=0, keep_dims=True)) / D
    # x_rms[x_rms == 0] = 1.
    return target_rms * x / x_rms


class BNF_1024_MFCC39(Model):
    """ BNN """

    def __init__(self, **kwargs):
        super(BNF_1024_MFCC39, self).__init__(**kwargs)

    def get_input_info(self):
        w0 = self.get_loaded_param('w0')
        nb_ceps = w0.shape[1]
        return {'X': ((None, nb_ceps), 'float32')}

    def _apply(self, X):
        weights = (b0, w0, b1, w1, b2, w2, b3, w3, b4, w4) = \
            self.get_loaded_param(('b0', 'w0', 'b1', 'w1',
                                   'b2', 'w2', 'b3', 'w3',
                                   'b4', 'w4'))
        # ====== create ====== #
        layers = []
        nb_layers = len(weights) // 2
        for i in range(nb_layers):
            b = weights[i * 2].ravel()
            W = weights[i * 2 + 1].T
            num_units = b.shape[0]
            if i == nb_layers - 1:
                name = 'BNF'
                nonlinearity = K.linear
            else:
                name = "Layer%d" % (i + 1)
                nonlinearity = K.relu
            layers.append(
                Dense(num_units=num_units, W_init=W, b_init=b,
                      activation=nonlinearity, name=name))
        # ====== apply ====== #
        for l in layers[:-1]:
            X = l(X)
            X = renorm_rms(X, axis=1)
        return layers[-1](X)
