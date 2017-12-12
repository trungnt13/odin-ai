from __future__ import print_function, division, absolute_import

from math import ceil
import numpy as np


# ===========================================================================
# Shape calculation for Pooling
# Contain code from theano: theano/tensor/signal/pool.py
# Copyright (c) 2008--2016, Theano Development Team
# ===========================================================================
def get_pool_output_shape(imgshape, ws, strides=None, pad=None):
  """
  Parameters
  ----------
  imgshape : tuple, list, or similar of integer or scalar Theano variable
      order: (samples, pool_dim1, pool_dim2, pool_dim3, ..., input_depth)
      (i.e tensorflow-NHWC format)
  ws : list or tuple of N ints
      Downsample factor over rows and column.
      ws indicates the pool region size.
  strides : list or tuple of N ints or None
      Stride size, which is the number of shifts over rows/cols/slices to get the
      next pool region. If stride is None, it is considered equal to ws
      (no overlap on pooling regions).
  pad : tuple of N ints or None
      For each downsampling dimension, this specifies the number of zeros to
      add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
      size of the top and bottom margins, pad_w specifies the size of the left and
      right margins. No padding is added if pad is None.

  """
  # convert tensorflow shape to theano shape
  imgshape = (imgshape[0], imgshape[-1]) + tuple(imgshape[1:-1])
  ndim = len(ws)
  # check valid pad (list or tuple of int)
  if isinstance(pad, str):
    if 'valid' in pad.lower():
      pad = (0,) * ndim
    elif 'same' in pad.lower():
      out_shape = tuple([int(ceil(float(i) / float(j)))
                         for i, j in zip(imgshape[-ndim:], strides)])
      return (imgshape[0],) + imgshape[2:-ndim] + out_shape + (imgshape[1],)

  def compute_out(v, downsample, stride):
    if downsample == stride:
      return v // stride
    else:
      out = (v - downsample) // stride + 1
      return np.maximum(out, 0)
  # ====== check input arguments ====== #
  if len(imgshape) < ndim:
    raise TypeError('imgshape must have at least {} dimensions'.format(ndim))
  if strides is None:
    strides = ws
  if pad is None:
    pad = (0,) * ndim
  patch_shape = tuple(imgshape[-ndim + i] + pad[i] * 2
                      for i in range(ndim))
  out_shape = [compute_out(patch_shape[i], ws[i], strides[i])
               for i in range(ndim)]
  rval = tuple(imgshape[:-ndim]) + tuple(out_shape)
  # convert theano shape to tensorflow shape
  rval = (rval[0],) + rval[2:] + (rval[1],)
  return rval


# ===========================================================================
# Shape calculation for Convolution
# Contain code from theano: theano/tensor/nnet/abstract_conv.py
# Copyright (c) 2008--2016, Theano Development Team
# ===========================================================================
def __get_conv_shape_1axis(image_shape, kernel_shape, border_mode,
                          subsample, dilation=1):
  if None in [image_shape, kernel_shape, border_mode,
              subsample, dilation]:
    return None
  # Implicit dilated kernel shape
  dil_kernel_shape = (kernel_shape - 1) * dilation + 1
  if isinstance(border_mode, str):
    border_mode = border_mode.lower()
  if border_mode == "half" or border_mode == "same":
    pad = dil_kernel_shape // 2
  elif border_mode == "full":
    pad = dil_kernel_shape - 1
  elif border_mode == "valid":
    pad = 0
  else:
    pad = border_mode
    if pad < 0:
      raise ValueError("border_mode must be >= 0")

  # In case of symbolic shape, we want to build the smallest graph
  # (image_shape + 2 * pad - dil_kernel_shape) // subsample + 1
  if pad == 0:
    out_shp = (image_shape - dil_kernel_shape)
  else:
    out_shp = (image_shape + 2 * pad - dil_kernel_shape)
  if subsample != 1:
    out_shp = out_shp // subsample
  out_shp = out_shp + 1

  # ====== get exact same border_mode for theano ====== #
  if (border_mode == 'half' or border_mode == 'same') and \
      kernel_shape % 2 == 0:
    out_shp = (image_shape + subsample - 1) // subsample
  return out_shp


def get_conv_output_shape(image_shape, kernel_shape,
                          border_mode, subsample,
                          filter_dilation=None):
  """
  This function compute the output shape of convolution operation.
  original code: abstract_conv.py (theano)

  Parameters
  ----------
  image_shape: tuple of int (symbolic or numeric) corresponding to the input
      order: (samples, conv_dim1, conv_dim2, conv_dim3, ..., input_depth)
      (i.e tensorflow-NHWC format)
  kernel_shape: tuple of int (symbolic or numeric) corresponding to the
      order: (kernel_dim1, kernel_dim2, kernel_dim3, ..., input_depth, out_depth)
      (i.e tensorflow-NHWC format)
  border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
      or numeric). If it is a string, it must be 'valid', 'half' or 'full'.
      If it is a tuple, its two (or three) elements respectively correspond
      to the padding on height and width (and possibly depth) axis.
  subsample: tuple of int (symbolic or numeric). Its or three elements
      espectively correspond to the subsampling on height and width (and
      possibly depth) axis.
  filter_dilation: tuple of int (symbolic or numeric). Its two elements
      correspond respectively to the dilation on height and width axis.

  Returns
  -------
  output_shape: tuple of int corresponding to the output image shape. Its
      four element must correspond respectively to: batch size, number of
      output channels, height and width of the image. None where undefined.

  """
  # ======  convert tensorflow shape to theano shape ====== #
  image_shape = (image_shape[0], image_shape[-1]) + tuple(image_shape[1:-1])
  kernel_shape = (kernel_shape[-1], kernel_shape[-2]) + tuple(kernel_shape[:-2])
  # ====== infer shape ====== #
  bsize, imshp = image_shape[0], image_shape[2:]
  nkern, kshp = kernel_shape[0], kernel_shape[2:]
  if filter_dilation is None:
    filter_dilation = np.ones(len(subsample), dtype='int')
  if isinstance(border_mode, tuple):
    out_shp = tuple(__get_conv_shape_1axis(
        imshp[i], kshp[i], border_mode[i],
        subsample[i], filter_dilation[i]) for i in range(len(subsample)))
  else:
    out_shp = tuple(__get_conv_shape_1axis(
        imshp[i], kshp[i], border_mode,
        subsample[i], filter_dilation[i]) for i in range(len(subsample)))
  # ====== convert theano to tensorflow shape ====== #
  return (bsize, ) + out_shp + (nkern,)
