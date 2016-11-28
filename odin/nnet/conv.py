from __future__ import division, absolute_import

from abc import abstractmethod

import numpy as np

from odin import backend as K
from odin.basic import PARAMETER, WEIGHT, BIAS
from odin.utils.decorators import autoinit
from odin.utils import as_tuple
from odin.utils.shape_calculation import get_conv_output_shape
from .base import NNOps, NNConfig


class Conv(NNOps):
    """ Convolutional Operator

    Performs a 2D or 3D convolution on its input and optionally adds a bias and
    applies an elementwise activation.

    Parameters
    ----------
    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        tuple specifying the size of the filters.

    stride : int or iterable of int
        specifying the stride of the convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 'valid')
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by theano or tensorflow.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W_init : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b_init : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untie_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    activation : callable or None
        The activation that is applied to the layer activations. If None
        is provided, the layer will be linear.

    dilation : int or iterable of int
        Specifying the dilation factor of the filters. A factor of
        :math:`x` corresponds to :math:`x - 1` zeros inserted between
        adjacent filter elements.

    **kwargs
        Any additional keyword arguments are passed to the `NNOps` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Note
    ----
    This Ops can be used for both 2D and 3D images (videos)
    """

    @autoinit
    def __init__(self, num_filters, filter_size, strides=1, pad='valid',
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 untie_biases=False,
                 activation=K.linear,
                 dilation=1, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.activation = K.linear if activation is None else activation

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # flip the input and hidden
        return TransposeConv(self)

    def _initialize(self, x):
        input_shape = K.get_shape(x)
        # ====== validate init arguments ====== #
        ndim = len(input_shape) - 2; self.ndim = ndim
        # padding
        if isinstance(self.pad, (tuple, list, int)):
            self.pad = as_tuple(self.pad, ndim, int)
        elif self.pad is None:
            self.pad = (0,) * ndim
        # strides
        if self.strides is None:
            self.strides = (0,) * ndim
        else:
            self.strides = as_tuple(self.strides, ndim, int)
        # dilation
        if self.dilation is None:
            self.dilation = (1,) * ndim
        else:
            self.dilation = as_tuple(self.dilation, ndim, int)
        # filter size
        self.filter_size = as_tuple(self.filter_size, ndim, int)
        # ====== create config ====== #
        config = NNConfig(input_shape=input_shape)
        # TF kernel shape: (kernel_dim1, kernel_dim2, ..., input_depth, out_depth)
        kernel_shape = self.filter_size + (input_shape[-1], self.num_filters)
        # weights
        config.create_params(self.W_init, shape=kernel_shape, name='W',
                             nnops=self, roles=WEIGHT)
        if self.b_init is not None:
            if self.untie_biases:
                output_shape = get_conv_output_shape(input_shape, kernel_shape,
                        border_mode=self.pad, subsample=self.strides,
                        filter_dilation=self.dilation)
                biases_shape = output_shape[1:]
            else:
                biases_shape = (self.num_filters,)
            config.create_params(self.b_init, shape=biases_shape, name='b',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        # store last input for deconvolution ops
        self._last_input = x
        conved = self.convolve(x)
        output_shape = K.get_shape(conved)
        if not hasattr(self, 'b'):
            conved = conved
        elif self.untie_biases:
            conved += K.expand_dims(self.b, 0)
        else:
            conved += K.dimshuffle(self.b, ('x',) * (self.ndim + 1) + (0,))
        activated = self.activation(conved)
        K.add_shape(activated, output_shape)
        # set shape for output
        return activated

    def convolve(self, x):
        if self.ndim == 2:
            conv_func = K.conv2d
        elif self.ndim == 3:
            conv_func = K.conv3d
        else:
            raise Exception('No support for %d-D input.' % self.ndim)
        conved = conv_func(x, kernel=self.W,
                           strides=self.strides,
                           border_mode=self.pad,
                           filter_dilation=self.dilation)
        return conved


class TransposeConv(NNOps):

    def __init__(self, conv):
        if not isinstance(conv, Conv):
            raise ValueError('TransposeConv Ops only accepts BaseConv as arguments.')
        super(TransposeConv, self).__init__(name=conv.name + '_transpose')
        self.conv = conv

    # ==================== abstract method ==================== #
    def _initialize(self, x):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        # check if original Ops is initialized
        if self.conv.configuration is None:
            raise Exception('Convolution ops:"%s" have not initialized.' % str(self.conv))
        output_shape = self.conv.input_shape
        config = NNConfig(output_shape=output_shape)
        # initialize parameters
        b_init = self.conv.b_init
        if b_init is not None:
            if self.conv.untie_biases:
                biases_shape = output_shape[1:]
            else:
                biases_shape = (output_shape[-1],)
            config.create_params(b_init, shape=biases_shape, name='b',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        if K.ndim(x) != self.conv.ndim + 2:
            raise ValueError('Input has %d dimensions, but this Ops require %d-D '
                             'tensor.' % (K.ndim(x), self.conv.ndim + 2))
        # ====== prepare the deconvolution ====== #
        stride = self.conv.strides
        border_mode = self.conv.pad
        W = self.conv.W
        dilation = self.conv.dilation
        # if Dilated Convolution, must transpose the Weights
        if self.conv.ndim == 2:
            deconv_func = K.deconv2d
        elif self.conv.ndim == 3:
            deconv_func = K.deconv3d
        else:
            raise Exception('No support for %d-D input in TransposedConv' %
                            self.conv.ndim)
        # theano require batch_dims is Constant or None, but tensorflow
        # require batch_dims is a native TensorVariable
        conved = deconv_func(x, kernel=W,
                            output_shape=K.get_shape(self.conv._last_input,
                                native=True if K.backend() == 'tensorflow' else False),
                            strides=stride,
                            border_mode=border_mode,
                            filter_dilation=dilation)
        if hasattr(self, 'b'):
            if self.conv.untie_biases:
                conved += K.expand_dims(self.b, 0)
            else:
                conved += K.dimshuffle(self.b, ('x',) * (self.conv.ndim + 1) + (0,))
        activated = self.conv.activation(conved)
        K.add_shape(activated, self.conv.input_shape)
        return activated

    def _transpose(self):
        return self.conv
