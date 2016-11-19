from __future__ import division, absolute_import

from .base import NNOps, NNConfig

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

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """

    def __init__(self, level=0.5,
                 noise_dims=None, noise_type='uniform',
                 rescale=True, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.level = level
        self.noise_dims = noise_dims
        self.noise_type = noise_type
        self.rescale = rescale

    def _initialize(self, x):
        return NNConfig()

    def _apply(self, x):
        return K.apply_dropout(x, level=self.level,
                               noise_dims=self.noise_dims,
                               noise_type=self.noise_type,
                               rescale=self.rescale)

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

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """

    def __init__(self, level=0.075, noise_dims=None,
                 noise_type='gaussian', **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.level = level
        self.noise_dims = noise_dims
        self.noise_type = noise_type

    def _initialize(self, x):
        return NNConfig()

    def _apply(self, x):
        return K.apply_noise(x, level=self.level,
                             noise_dims=self.noise_dims,
                             noise_type=self.noise_type)

    def _transpose(self):
        return self
