from __future__ import division, absolute_import, print_function

import numpy as np

from odin import backend as K
from odin.utils import is_number, is_string
from odin.utils.decorators import functionable

from .base import NNOps


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
        When True, (5,5) input with pool_size=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    mode : {'max', 'avg'}
        Operation executed on each window. `max` or `average`
    pool_func : 'auto' or callable
        if 'auto', auto select pool function based on number of input
        dimension (pool2D for 4D input, pool3D for 5D input)
    transpose_mode: 'nn', 'pad', 'pad_margin', 'repeat'
        One of the mode for `transposing` downsampling process,
        check `odin.nnet.sampling.Upsampling`.

    Note
    ----
    This pooling algorithm has non-deterministic behaviour on cuDNN
    """

    def __init__(self, pool_size=2, strides=None, pad='valid',
                 ignore_border=True, mode='max', pool_func='auto',
                 transpose_mode='nn', **kwargs):
        super(Pool, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.pad = pad
        self.ignore_border = ignore_border
        self.mode = mode
        self.transpose_mode = transpose_mode
        self.pool_func = functionable(pool_func) if callable(pool_func) \
            else pool_func

    def _apply(self, X):
        if self.pool_func == 'auto':
            if K.ndim(X) == 4:
                pool_func = K.pool2d
            elif K.ndim(X) == 5:
                pool_func = K.pool3d
            else:
                raise RuntimeError("Pooling unsupport for %d-D input." % K.ndim(X))
        else: # user sepecifed pool_func
            pool_func = self.pool_func
        return pool_func(X, pool_size=self.pool_size, strides=self.strides,
                         border_mode=self.pad, ignore_border=self.ignore_border,
                         mode=self.mode)

    def _transpose(self):
        ops = Upsample(size=self.pool_size, axes='auto',
            mode=self.transpose_mode, transpose_mode=self.mode,
            output_shape=lambda: self.input_shape,
            name=self.name + '_transpose')
        ops._transpose_ops = self
        return ops


class Upsample(NNOps):
    """ Upsampling

    Parameters
    ----------
    size: int
        upsampling size (new_size = input_size * size)
    axes: 'auto', int, list of int
        pass
    mode: str, int
        `repeat` is

    """

    def __init__(self, size=2, axes='auto', mode='nn', transpose_mode='max',
                 output_shape=None, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        self.size = size
        self.axes = axes
        self.mode = mode
        self.transpose_mode = transpose_mode
        self.output_shape = functionable(output_shape) \
            if callable(output_shape) else output_shape

    def _apply(self, X):
        axes = self.axes
        if is_string(axes) and axes.lower() == 'auto':
            if K.ndim(X) == 3:
                axes = (1,)
            elif K.ndim(X) == 4:
                axes = (1, 2)
            elif K.ndim(X) == 5:
                axes = (1, 2, 3)
        X = K.upsample(X, scale=self.size, axes=axes, method=self.mode)
        # ====== check output_shape ====== #
        output_shape = self.output_shape
        if output_shape is not None:
            if callable(output_shape):
                output_shape = output_shape()
            # do padding if necessary
            paddings = [[0, 0] if i is None or o is None or i >= o else
                        [K.cast(K.ceil((o - i) / 2), 'int32'),
                         K.cast(K.floor((o - i) / 2), 'int32')]
                        for i, o in zip(K.get_shape(X), output_shape)]
            if np.sum(paddings) > 0:
                X = K.pad(X, paddings=paddings, mode='constant')
            # do slice if necessary
            slices = [slice(K.cast(K.ceil((i - o) / 2), 'int32'),
                            K.cast(- K.floor((i - o) / 2), 'int32'), None)
                      if i > o else slice(None)
                      for i, o in zip(K.get_shape(X), output_shape)]
            if any(s != slice(None) for s in slices):
                X = X[slices]
            # add shape
            K.add_shape(X, tuple([i if is_number(i) else None
                                  for i in output_shape]))
        return X

    def _transpose(self):
        if self.axes != 'auto':
            raise RuntimeError("Do not support tranpose of Upsample with "
                               "axes=%s, the only support value is 'auto'."
                               % self.axes)
        ops = Pool(pool_size=self.size, strides=None, pad='valid',
            ignore_border=True, mode=self.transpose_mode,
            transpose_mode=self.mode, name=self.name + '_transpose')
        ops._transpose_ops = self
        return ops
