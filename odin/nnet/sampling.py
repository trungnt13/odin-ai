from __future__ import division, absolute_import


import numpy as np

from odin import backend as K
from odin.utils.decorators import autoinit

from .base import NNOps, NNConfig


class Pool(NNOps):
    """
    Parameters
    ----------
    pool_size : tuple of int or just an int.
        Factor by which to downscale (vertical ws, horizontal ws, ...).
    strides : tuple of two ints or theano vector of ints of size 2.
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of two ints or theano vector of ints of size 2.
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ws=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    mode : {'max', 'avg'}
        Operation executed on each window. `max` or `average`
    pool_func : 'auto' or callable
        if 'auto', auto select pool function based on number of input
        dimension (pool2D for 4D input, pool3D for 5D input)

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """

    @autoinit
    def __init__(self, pool_size=2, strides=None, pad='valid',
                 ignore_border=True, mode='max',
                 pool_func='auto', **kwargs):
        super(Pool, self).__init__(**kwargs)

    def _initialize(self, x):
        config = NNConfig(ndim=K.ndim(x))
        return config

    def _apply(self, x):
        if self.pool_func == 'auto':
            if self.ndim == 4:
                pool_func = K.pool2d
            elif self.ndim == 5:
                pool_func = K.pool3d
        else: # user sepecifed pool_func
            pool_func = self.pool_func
        return pool_func(x, pool_size=self.pool_size, strides=self.strides,
                         border_mode=self.pad, ignore_border=self.ignore_border,
                         mode=self.mode)

    def _transpose(self):
        raise NotImplementedError
