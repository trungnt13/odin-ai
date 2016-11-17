from __future__ import print_function, division, absolute_import

import numpy as np


# ===========================================================================
# Shape calculation for Pooling
# Contain code from theano: theano/tensor/signal/pool.py
# Copyright (c) 2008--2016, Theano Development Team
# ===========================================================================
def get_pool_output_shape(imgshape, ws, ignore_border=False, stride=None, pad=None):
    """
    Parameters
    ----------
    imgshape : tuple, list, or similar of integer or scalar Theano variable
        The shape of a tensor of images. The last N elements are
        interpreted as the number of rows, and the number of cols.
    ws : list or tuple of N ints
        Downsample factor over rows and column.
        ws indicates the pool region size.
    ignore_border : bool
        If ws doesn't divide imgshape, do we include an extra row/col/slice
        of partial downsampling (False) or ignore it (True).
    stride : list or tuple of N ints or None
        Stride size, which is the number of shifts over rows/cols/slices to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of N ints or None
        For each downsampling dimension, this specifies the number of zeros to
        add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
        size of the top and bottom margins, pad_w specifies the size of the left and
        right margins. No padding is added if pad is None.

    """
    def compute_out(v, downsample, stride):
        if ignore_border:
            if downsample == stride:
                return v // stride
            else:
                out = (v - downsample) // stride + 1
                return np.maximum(out, 0)
        else:
            if stride >= downsample:
                return (v - 1) // stride + 1
            else:
                return max(0, (v - 1 - downsample + stride) // stride) + 1
    # ====== check input arguments ====== #
    ndim = len(ws)
    if len(imgshape) < ndim:
        raise TypeError('imgshape must have at least {} dimensions'.format(ndim))
    if stride is None:
        stride = ws
    if pad is None:
        pad = (0,) * ndim
    patch_shape = tuple(imgshape[-ndim + i] + pad[i] * 2 for i in range(ndim))
    out_shape = [compute_out(patch_shape[i], ws[i], stride[i]) for i in xrange(ndim)]
    rval = tuple(imgshape[:-ndim]) + tuple(out_shape)
    return rval


# ===========================================================================
# Shape calculation for Convolution
# Contain code from theano: theano/tensor/nnet/abstract_conv.py
# Copyright (c) 2008--2016, Theano Development Team
# ===========================================================================
def __get_conv_shape_1axis(image_shape, kernel_shape, border_mode,
                          subsample, dilation=1):
    """
    This function compute the output shape of convolution operation.
    original code: abstract_conv.py (theano)

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    border_mode: string or int. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    out_shp: int corresponding to the output image shape on the
        considered axis. None if undefined.

    """
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
        order: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3, ...)
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        order: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3, ...)
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
    return (bsize, ) + out_shp + (nkern,)
