from __future__ import division, absolute_import


import numpy as np

from odin import backend as K
from odin.utils import as_tuple
from odin.utils.decorators import autoinit

from .base import NNOps, NNConfig


class Pool2D(NNOps):

    @autoinit
    def __init__(self, pool_size=(2, 2), ignore_border=True,
           strides=(1, 1), pad=(0, 0), mode='max', **kwargs):
        super(Pool2D, self).__init__(**kwargs)

    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)

        return K.pool2d(x, pool_size=self.pool_size, strides=self.strides,
                        pad=self.pad, ignore_border=self.ignore_border,
                        mode=self.mode)

    def _transpose(self):
        raise NotImplementedError


class Pool3D(NNOps):

    @autoinit
    def __init__(self, pool_size=(2, 2, 2), ignore_border=True,
           strides=(1, 1, 1), pad=(0, 0, 0), mode='max', **kwargs):
        super(Pool3D, self).__init__(**kwargs)

    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)

        return K.pool3d(x, pool_size=self.pool_size, strides=self.strides,
                        pad=self.pad, ignore_border=self.ignore_border,
                        mode=self.mode)

    def _transpose(self):
        raise NotImplementedError
