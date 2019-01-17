from __future__ import division, absolute_import, print_function
from numbers import Number

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.nnet.base import NNOp
from odin.utils import is_number, is_string, as_tuple

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
  pool_func : 'auto' or call-able
      if 'auto', auto select pool function based on number of input
      dimension (pool2D for 4D input, pool3D for 5D input)
  transpose_mode: 'nn', 'pad', 'pad_margin', 'repeat'
      One of the mode for `transposing` downsampling process,
      check `odin.nnet.sampling.Upsampling`.

  Note
  ----
  This pooling algorithm has non-deterministic behaviour on cuDNN
  """

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
    ndims = X.shape.ndims
    return tf.nn.pool(X,
        window_shape=_preprocess_windows(self.pool_size, ndims),
        strides=_preprocess_windows(self.strides, ndims),
        dilation_rate=_preprocess_windows(self.dilation, ndims),
        padding=self.pad, pooling_type=self.mode)

  def _transpose(self):
    return Upsample(size=self.strides, axes='auto',
        mode=self.transpose_mode, transpose_mode=self.mode,
        desire_shape=self.input_shape)

class Upsample(NNOp):
  """ Upsampling

  Parameters
  ----------
  size: int
      upsampling size (new_size = input_size * size)
  axes: int, list of int
      the axes of tensor which the upsampling method will be applied
  mode: str, int
      'nn' for nearest neighbor (e.g. [1, 2] => [1, 1, 2, 2]),
      'pad' for padding within the tensor. 'pad_margin' do padding
      in the margin of the tensor. 'repeat' simple algorithm for
      repeating the element (e.g. [1, 2] => [1, 2, 1, 2])
  transpose_mode: {'max', 'avg'}
      pass
  desire_shape : shape tuple
      specific output shape, if the upsampled image is bigger, then crop
      the image, any axis specified with negative value or `None` will
      be kept unchanged
  """

  def __init__(self, size=2, axes='auto', mode='nn', transpose_mode='max',
               desire_shape=None, **kwargs):
    super(Upsample, self).__init__(**kwargs)
    self.size = size
    self.axes = axes
    self.mode = mode
    self.transpose_mode = transpose_mode
    self.desire_shape = desire_shape

  def _apply(self, X):
    axes = self.axes
    ndims = X.shape.ndims
    if is_string(axes) and axes.lower() == 'auto':
      if ndims == 3:
        axes = (1,)
      elif ndims == 4:
        axes = (1, 2)
      elif ndims == 5:
        axes = (1, 2, 3)
    X = K.upsample(X, scale=self.size, axes=axes, method=self.mode)
    # ====== check desire_shape ====== #
    desire_shape = self.desire_shape
    if desire_shape is not None:
      desire_shape = [None if i is None or i < 0 else int(i)
                      for i in desire_shape]
      # do padding if necessary
      paddings = [[0, 0] if i is None or o is None or i >= o else
                  [tf.cast(tf.ceil((o - i) / 2), 'int32'),
                   tf.cast(tf.floor((o - i) / 2), 'int32')]
                  for i, o in zip(X.shape.as_list(), desire_shape)]
      if not all(i == [0, 0] for i in paddings):
        X = tf.pad(X, paddings=paddings, mode='CONSTANT')
      # do slice if necessary
      slices = [slice(tf.cast(tf.floor((i - o) / 2), 'int32'),
                      tf.cast(-tf.ceil((i - o) / 2), 'int32'), None)
                if i is not None and o is not None and i > o else slice(None)
                for i, o in zip(X.shape.as_list(), desire_shape)]
      if any(s is not slice(None) for s in slices):
        X = X[slices]
      K.set_shape(X, tuple([i if is_number(i) else None
                            for i in desire_shape]))
    return X

  def _transpose(self):
    if self.axes != 'auto':
      raise RuntimeError("Do not support tranpose of Upsample with "
                         "axes=%s, the only support value is 'auto'."
                         % self.axes)
    return Pool(pool_size=self.size, strides=None, pad='valid',
        mode=self.transpose_mode, transpose_mode=self.mode)

class StatsPool(NNOp):
  """ Calculate mean and stddev of input Tensor along `axes`
  then merge the statistics based on given `output_mode`

  This class is for parsing lines like
    stats-layer name=tdnn1-stats config=mean+stddev(-99:3:9:99) input=tdnn1
    This adds statistics-pooling and statistics-extraction components.  An
    example string is 'mean(-99:3:9::99)', which means, compute the mean of
    data within a window of -99 to +99, with distinct means computed every 9
    frames (we round to get the appropriate one), and with the input extracted
    on multiples of 3 frames (so this will force the input to this layer to be
    evaluated every 3 frames).  Another example string is
    'mean+stddev(-99:3:9:99)', which will also cause the standard deviation to
    be computed.
    The dimension is worked out from the input. mean and stddev add a
    dimension of input_dim each to the output dimension. If counts is
    specified, an additional dimension is added to the output to store log
    counts.
  """
  _SUPPORT_OUTPUT_MODE = ('concat', 'sum', 'mean', 'max')

  def __init__(self, axes, output_mode='concat', **kwargs):
    super(StatsPool, self).__init__(**kwargs)
    self.axes = axes
    # ====== check output mode ====== #
    output_mode = str(output_mode).lower()
    if output_mode not in StatsPool._SUPPORT_OUTPUT_MODE:
      raise ValueError("No support for `output_mode=%s`, the following values "
                       "are accepted: %s" %
                       (output_mode, StatsPool._SUPPORT_OUTPUT_MODE))
    self.output_mode = output_mode

  def _apply(self, X):
    mean, var = tf.nn.moments(x=X, axes=self.axes, keep_dims=True)
    std = tf.sqrt(var)
    if self.output_mode == 'concat':
      X = tf.concat((mean, std), self.axes)
    elif self.output_mode == 'sum':
      X = mean + std
    elif self.output_mode == 'mean':
      X = (mean + std) / 2
    elif self.output_mode == 'max':
      X = tf.maximum(mean, std)
    return X
