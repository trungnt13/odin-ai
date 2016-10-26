from __future__ import print_function, division, absolute_import

import numpy as np

from odin.utils import as_tuple


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int or None
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


def conv_shape(image_shape, filter_shape, filter_dilation,
               stride, pad):
    if not isinstance(image_shape, (tuple, list)) or \
    not isinstance(filter_shape, (tuple, list)):
        return None
    # ====== important configuration ====== #
    n = len(image_shape) - 2
    num_filters = filter_shape[0]
    filter_size = filter_shape[2:]
    if pad == 'valid':
        pad = as_tuple(0, n)
    elif pad in ('full', 'same'):
        pad = pad
    else:
        pad = as_tuple(pad, n, int)
    filter_dilation = as_tuple(filter_dilation, n, int)
    batchsize = image_shape[0]
    pad = pad if isinstance(pad, tuple) else (pad,) * n
    # ====== calculate shape ====== #
    if filter_dilation == (1,) * n:
        output_shape = ((batchsize, num_filters) +
                        tuple(conv_output_length(input, filter, stride, p)
                              for input, filter, stride, p
                              in zip(image_shape[2:], filter_size,
                                     stride, pad)))
    else:
        output_shape = ((batchsize, num_filters) +
                       tuple(conv_output_length(input, (filter - 1) * dilate + 1, 1, 0)
                             for input, filter, dilate
                             in zip(image_shape[2:], filter_size,
                                    filter_dilation)))
    return output_shape
