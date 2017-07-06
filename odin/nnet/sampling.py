from __future__ import division, absolute_import, print_function

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils import is_number, is_string, as_tuple
from odin.utils.decorators import functionable

from .base import NNOp, _nnops_initscope


def _preprocess_windows(window, ndims):
    if len(window) == ndims - 2:
        return window
    else:
        return as_tuple(window, N=(ndims - 2))


class Pool(NNOp):
    """
    Parameters
    ----------
    pool_size : tuple of int or just an int.
        Factor by which to downscale (vertical ws, horizontal ws, ...).
    strides : tuple of two ints or theano vector of ints of size 2.
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region.
        If stride is None, it is considered equal to pool_size (no overlap
        on pooling regions).
    pad : tuple of two ints or theano vector of ints of size 2.
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
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

    @_nnops_initscope
    def __init__(self, pool_size=2, strides=None, dilation=1,
                 pad='valid', mode='max', transpose_mode='nn', **kwargs):
        super(Pool, self).__init__(**kwargs)
        self.pool_size = as_tuple(pool_size, t=int)
        self.strides = self.pool_size if strides is None \
            else as_tuple(strides, t=int)
        self.dilation = (1,) if dilation is None else as_tuple(dilation, t=int)
        self.pad = pad.upper() if is_string(pad) else as_tuple(pad, t=int)
        self.mode = mode.upper()
        self.transpose_mode = transpose_mode

    def _apply(self, X):
        ndims = X.get_shape().ndims
        return tf.nn.pool(X,
            window_shape=_preprocess_windows(self.pool_size, ndims),
            strides=_preprocess_windows(self.strides, ndims),
            dilation_rate=_preprocess_windows(self.dilation, ndims),
            padding=self.pad, pooling_type=self.mode)

    def _transpose(self):
        ops = Upsample(size=self.strides, axes='auto',
            mode=self.transpose_mode, transpose_mode=self.mode,
            output_shape=self.input_shape_ref,
            name=self.name + '_transpose')
        ops._transpose_ops = self
        return ops


class Upsample(NNOp):
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

    @_nnops_initscope
    def __init__(self, size=2, axes='auto', mode='nn', transpose_mode='max',
                 output_shape=None, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        self.size = size
        self.axes = axes
        self.mode = mode
        self.transpose_mode = transpose_mode
        self.output_shape = output_shape

    def _apply(self, X):
        axes = self.axes
        ndims = X.get_shape().ndims
        if is_string(axes) and axes.lower() == 'auto':
            if ndims == 3:
                axes = (1,)
            elif ndims == 4:
                axes = (1, 2)
            elif ndims == 5:
                axes = (1, 2, 3)
        X = K.upsample(X, scale=self.size, axes=axes, method=self.mode)
        # ====== check output_shape ====== #
        output_shape = self.output_shape
        if output_shape is not None:
            if callable(output_shape):
                output_shape = output_shape()
            # do padding if necessary
            paddings = [[0, 0] if i is None or o is None or i >= o else
                        [tf.cast(tf.ceil((o - i) / 2), 'int32'),
                         tf.cast(tf.floor((o - i) / 2), 'int32')]
                        for i, o in zip(X.get_shape().as_list(), output_shape)]
            if not all(i == [0, 0] for i in paddings):
                X = tf.pad(X, paddings=paddings, mode='CONSTANT')
            # do slice if necessary
            slices = [slice(tf.cast(tf.floor((i - o) / 2), 'int32'),
                            tf.cast(-tf.ceil((i - o) / 2), 'int32'), None)
                      if i > o else slice(None)
                      for i, o in zip(X.get_shape().as_list(), output_shape)]
            if any(s is not slice(None) for s in slices):
                X = X[slices]
            K.set_shape(X, tuple([i if is_number(i) else None
                                  for i in output_shape]))
        return X

    def _transpose(self):
        if self.axes != 'auto':
            raise RuntimeError("Do not support tranpose of Upsample with "
                               "axes=%s, the only support value is 'auto'."
                               % self.axes)
        ops = Pool(pool_size=self.size, strides=None, pad='valid',
            mode=self.transpose_mode, transpose_mode=self.mode,
            name=self.name + '_transpose')
        ops._transpose_ops = self
        return ops
