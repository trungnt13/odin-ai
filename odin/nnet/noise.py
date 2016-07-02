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

    @autoinit
    def __init__(self, level=0.5, noise_dims=None, rescale=True, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rng = K.rng(seed=seed)

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def _apply(self, x):
        input_shape = K.shape(x)
        level = self.level
        noise_dims = self.noise_dims
        rescale = self.rescale
        rng = self.rng
        # ====== not a training variable NO dropout ====== #
        if not K.is_training(x):
            return x
        # ====== Dropout ====== #
        retain_prob = 1. - level
        shape = K.shape(x, none=False)
        if noise_dims is None:
            x = x * rng.binomial(shape=shape, p=retain_prob, dtype=x.dtype)
        else:
            noise_shape = _process_noise_dim(shape, noise_dims)
            # auto select broadcast shape
            broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
            if len(broadcast) > 0:
                x = x * K.addbroadcast(
                    rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype),
                    *broadcast)
            else:
                x = x * rng.binomial(shape=noise_shape, p=retain_prob, dtype=x.dtype)
        if rescale:
            x /= retain_prob
        if isinstance(input_shape, (tuple, list)):
            K.add_shape(x, input_shape)
        return x

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

    def __init__(self, sigma=0.075, noise_dims=None, noise_type='gaussian', seed=None):
        super(Noise, self).__init__()
        self.rng = K.rng(seed=seed)

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def _apply(self, x):
        input_shape = K.shape(x)
        noise_dims = self.noise_dims
        noise_type = self.noise_type.lower()
        sigma = self.sigma
        rng = self.rng
        # ====== not a training variable NO dropout ====== #
        if not K.is_training(x):
            return x
        # ====== applying noise ====== #
        shape = K.shape(x, none=False)
        noise_shape = (shape if noise_dims is None
                       else _process_noise_dim(shape, noise_dims))
        if 'normal' in noise_type or 'gaussian' in noise_type:
            noise = rng.normal(shape=noise_shape, mean=0.0, std=sigma, dtype=x.dtype)
        elif 'uniform' in noise_type:
            noise = rng.uniform(shape=noise_shape, low=-sigma, high=sigma, dtype=x.dtype)
            # no idea why uniform does not give any broadcastable dimensions
            if noise_dims is not None:
                broadcastable = [i for i, j in enumerate(noise_shape) if j == 1]
                if len(broadcastable) > 0:
                    noise = K.addbroadcast(noise, *broadcastable)
        x = x + noise
        if isinstance(input_shape, (tuple, list)):
            K.add_shape(x, input_shape)
        return x

    def _transpose(self):
        return self
