from __future__ import division, absolute_import


import numpy as np

from odin import backend as K
from odin.utils import as_tuple
from odin.utils.decorators import autoinit

from .base import NNOps, NNConfig


class Pool2D(NNOps):
    """
    Parameters
    ----------
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    """

    @autoinit
    def __init__(self, pool_size=(2, 2), ignore_border=True,
           strides=(1, 1), pad=(0, 0), mode='max', **kwargs):
        super(Pool2D, self).__init__(**kwargs)

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x)[1:])
        return config

    def _apply(self, x):
        return K.pool2d(x, pool_size=self.pool_size, strides=self.strides,
                        pad=self.pad, ignore_border=self.ignore_border,
                        mode=self.mode)

    def _transpose(self):
        raise NotImplementedError


class Pool3D(NNOps):
    """
    Parameters
    ----------
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    """

    @autoinit
    def __init__(self, pool_size=(2, 2, 2), ignore_border=True,
           strides=(1, 1, 1), pad=(0, 0, 0), mode='max', **kwargs):
        super(Pool3D, self).__init__(**kwargs)

    def _initialize(self, x):
        config = NNConfig(input_shape=K.get_shape(x)[1:])
        return config

    def _apply(self, x):
        return K.pool3d(x, pool_size=self.pool_size, strides=self.strides,
                        pad=self.pad, ignore_border=self.ignore_border,
                        mode=self.mode)

    def _transpose(self):
        raise NotImplementedError
