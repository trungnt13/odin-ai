from __future__ import division, absolute_import

from .base import NNOps

from odin.basic import WEIGHT, BIAS
from odin import backend as K


class Dropout(NNOps):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
        input tensor (any shape)
    level: float(0.-1.)
        probability dropout values in given tensor
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    rescale: bool
        whether rescale the outputs by dividing the retain probablity

    """

    def __init__(self, level=0.5,
                 noise_dims=None, noise_type='uniform',
                 rescale=True, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.level = level
        self.noise_dims = noise_dims
        self.noise_type = noise_type
        self.rescale = rescale

    def _apply(self, X, dropout=0):
        if dropout >= 0:
            training = K.is_training()
            if dropout > 0:
                K.set_training(True)
            X = K.apply_dropout(X, level=self.level,
                                noise_dims=self.noise_dims,
                                noise_type=self.noise_type,
                                rescale=self.rescale)
            K.set_training(training)
        return X

    def _transpose(self):
        return self


class Noise(NNOps):
    """
    Parameters
    ----------
    x: A tensor.
        input tensor (any shape)
    level : float or tensor scalar
        Standard deviation of added Gaussian noise, or range of
        uniform noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise

    """

    def __init__(self, level=0.075, noise_dims=None,
                 noise_type='gaussian', **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.level = level
        self.noise_dims = noise_dims
        self.noise_type = noise_type

    def _apply(self, X, noise=0):
        # possible mechanism to hard override the noise-state
        if noise >= 0:
            training = K.is_training()
            if noise > 0:
                K.set_training(True)
            X = K.apply_noise(X, level=self.level,
                              noise_dims=self.noise_dims,
                              noise_type=self.noise_type)
            K.set_training(training)
        return X

    def _transpose(self):
        return self


class GaussianDenoising(NNOps):
    """ Gaussian denoising function proposed in
    "Semi-Supervised Learning with Ladder Networks"
    """

    def __init__(self, activation=K.sigmoid, **kwargs):
        super(GaussianDenoising, self).__init__(**kwargs)
        self.activation = K.linear if activation is None else activation

    def _initialize(self):
        input_shape = self.input_shape
        shape = input_shape[1:]
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a1', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(1.), shape=shape, name='a2', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a3', roles=BIAS)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a4', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a5', roles=BIAS)

        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a6', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(1.), shape=shape, name='a7', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a8', roles=BIAS)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a9', roles=WEIGHT)
        self.config.create_params(
            K.init.constant(0.), shape=shape, name='a10', roles=BIAS)

    def _apply(self, u, mean, std, z_corr):
        mu = self.a1 * self.activation(self.a2 * u + self.a3) + self.a4 * u + self.a5
        v = self.a6 * self.activation(self.a7 * u + self.a8) + self.a9 * u + self.a10

        z_est = (z_corr - mu) * v + mu
        z_est_bn = (z_est - mean) / K.square(std)
        return z_est_bn

    def _transpose(self):
        raise NotImplementedError
