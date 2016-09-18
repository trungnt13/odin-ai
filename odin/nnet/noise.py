from __future__ import division, absolute_import

from .base import NNOps, NNConfig

from odin import backend as K
from odin.utils.decorators import autoinit


def _process_noise_dim(input_shape, dims):
    """
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Examples
    --------
    (None, 10, 10) with noise_dims=2
    => (None, 10, 1)
    """
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    # ====== get noise shape ====== #
    if dims is None:
        noise_shape = input_shape
    else:
        return tuple([1 if i in dims else j
                      for i, j in enumerate(input_shape)])
    return noise_shape


class Dropout(NNOps):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
    level: float(0.-1.)
        probability dropout values in given tensor
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    rng: `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """

    def __init__(self, level=0.5, noise_dims=None, rescale=True,
                 seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.level = level
        self.noise_dims = noise_dims
        self.rescale = rescale
        self.seed = seed

    def _initialize(self, x):
        return NNConfig()

    def _apply(self, x):
        if not hasattr(self, 'rng'):
            self.rng = K.rng(self.seed)
        return K.apply_dropout(x, level=self.level,
            noise_dims=self.noise_dims, rescale=self.rescale, seed=self.rng)

    def _transpose(self):
        return self


class Noise(NNOps):
    """
    Parameters
    ----------
    x: A tensor.
    sigma : float or tensor scalar
        Standard deviation of added Gaussian noise
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """

    def __init__(self, sigma=0.075, noise_dims=None,
                 noise_type='gaussian', seed=None, **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.sigma = sigma
        self.noise_dims = noise_dims
        self.noise_type = noise_type
        self.seed = seed

    def _initialize(self, x):
        return NNConfig()

    def _apply(self, x):
        if not hasattr(self, 'rng'):
            self.rng = K.rng(self.seed)
        return K.apply_noise(x, sigma=self.sigma, noise_dims=self.noise_dims,
                             noise_type=self.noise_type, seed=self.rng)

    def _transpose(self):
        return self
