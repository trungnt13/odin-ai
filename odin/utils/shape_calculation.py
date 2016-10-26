from __future__ import print_function, division, absolute_import

import numpy as np

from odin.utils import as_tuple


def _get_conv_shape_1axis(image_shape, kernel_shape, border_mode,
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
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. Its four (or five) elements must correspond respectively
        to: number of output channels, number of input channels, height and
        width (and possibly depth) of the kernel. None where undefined.
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
        out_shp = tuple(_get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode[i],
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    else:
        out_shp = tuple(_get_conv_shape_1axis(
            imshp[i], kshp[i], border_mode,
            subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    return (bsize, nkern) + out_shp
