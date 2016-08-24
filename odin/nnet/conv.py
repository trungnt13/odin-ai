from __future__ import division, absolute_import

from abc import abstractmethod

import numpy as np

from odin import backend as K
from odin.roles import PARAMETER, WEIGHT, BIAS
from odin.utils.decorators import autoinit
from odin.utils import as_tuple
from .base import NNOps, NNConfig


def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int
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
    int
        The output size corresponding to the given convolution parameters.

    Raises
    ------
    RuntimeError
        When an invalid padding is specified, a `RuntimeError` is raised.
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


class BaseConv(NNOps):

    @autoinit
    def __init__(self, num_filters, filter_size, stride=1, pad=0,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 untie_biases=False,
                 activation=K.linear, n=None, **kwargs):
        super(BaseConv, self).__init__(**kwargs)
        self.activation = K.linear if activation is None else activation
        if n is not None:
            if self.pad not in ('valid', 'full', 'same', 'half'):
                self.pad = as_tuple(pad, n, int)
            if hasattr(self, 'stride'):
                self.stride = as_tuple(stride, n, int)
            self.filter_size = as_tuple(filter_size, n, int)

    # ==================== Utilities ==================== #
    def get_W_shape(self, input_shape):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        return (self.num_filters, input_shape[1],) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # flip the input and hidden
        return TransposeConv(self)

    def _initialize(self, x):
        input_shape = K.get_shape(x)
        # ====== validate init arguments ====== #
        if self.n is None:
            self.n = len(input_shape) - 2
        elif self.n != len(input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (self.n, input_shape, self.n + 2, self.n))
        if self.pad == 'valid':
            self.pad = 'valid' # as_tuple(0, self.n)
        elif self.pad not in ('full', 'same', 'half'):
            self.pad = as_tuple(self.pad, self.n, int)
        if hasattr(self, 'stride'):
            self.stride = as_tuple(self.stride, self.n, int)
        self.filter_size = as_tuple(self.filter_size, self.n, int)
        # ====== create config ====== #
        output_shape = self.get_output_shape_for(input_shape)
        self.output_shape = output_shape # assign output_shape
        config = NNConfig(input_shape=input_shape)
        # weights
        config.create_params(self.W_init, shape=self.get_W_shape(input_shape),
                             name='W', nnops=self, roles=WEIGHT)
        if self.b_init is not None:
            if self.untie_biases:
                biases_shape = (self.num_filters,) + output_shape[2:]
            else:
                biases_shape = (self.num_filters,)
            config.create_params(self.b_init, shape=biases_shape, name='b',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        # calculate projection
        conved = self.convolve(x)
        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + K.expand_dims(self.b, 0)
        else:
            activation = conved + K.dimshuffle(self.b, ('x', 0) + ('x',) * self.n)
        K.add_shape(activation, self.output_shape)
        activation = self.activation(activation)
        # set shape for output
        return activation

    def convolve(self, x):
        raise NotImplementedError


class TransposeConv(NNOps):

    def __init__(self, conv):
        if not isinstance(conv, BaseConv):
            raise ValueError('TransposeConv Ops only accepts BaseConv as arguments.')
        super(TransposeConv, self).__init__(name=conv.name + '_transpose')
        self.conv = conv
        if conv.configuration is None:
            raise Exception('Convolution ops:"%" have not initialized.' % str(conv))

    # ==================== abstract method ==================== #
    def _initialize(self, x):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        output_shape = self.conv.input_shape

        config = NNConfig(output_shape=output_shape)
        b_init = self.conv.b_init
        untie_biases = self.conv.untie_biases
        if b_init is not None:
            if untie_biases:
                biases_shape = output_shape[1:]
            else:
                biases_shape = (output_shape[1],)
            config.create_params(b_init, shape=biases_shape, name='b',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        # The AbstractConv2d_gradInputs op takes a kernel that was used for the
        # **convolution**. We therefore have to invert num_channels and
        # num_filters for W.
        output_shape = self.conv.input_shape
        if K.get_shape(x)[1:] != self.conv.output_shape[1:]:
            raise Exception('This Ops transpose convolved Variable from shape={}'
                            ' back to original shape={}, but given x has shape={}'
                            '.'.format(self.conv.output_shape[1:], output_shape[1:],
                                K.get_shape(x)[1:]))
        # ====== prepare the deconvolution ====== #
        W_shape = self.conv.get_W_shape(output_shape)
        stride = self.conv.stride
        border_mode = self.conv.pad
        W = self.conv.W
        # if Dilated Convolution, must transpose the Weights
        if hasattr(self.conv, 'dilation'):
            W = K.transpose(W, (1, 0, 2, 3))
            border_mode = 'valid'
        conved = K.deconv2d(x, W,
                            image_shape=output_shape,
                            filter_shape=W_shape,
                            strides=stride,
                            border_mode=border_mode,
                            flip_filters=False) # because default of Conv2D is True
        if self.b is not None:
            if self.conv.untie_biases:
                conved = K.add(conved, K.expand_dims(self.b, 0))
            else:
                conved = K.add(conved,
                               K.dimshuffle(self.b, ('x', 0) + ('x',) * self.conv.n))
        return self.conv.activation(conved)

    def _transpose(self):
        return self.conv


class Conv2D(BaseConv):
    """
    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise activation.

    Parameters
    ----------
    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
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
        integer values due to optimizations by Theano.

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

    **kwargs
        Any additional keyword arguments are passed to the `NNOps` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    # ==================== abstract method ==================== #
    def convolve(self, x):
        # by default we assume 'cross', consistent with corrmm.
        # conv_mode = 'conv' if self.flip_filters else 'cross'
        conved = K.conv2d(x,
            kernel=self.W,
            strides=self.stride,
            border_mode=self.pad,
            image_shape=self.input_shape,
            filter_shape=K.get_shape(self.W))
        return conved


class DilatedConv2D(BaseConv):
    """ 2D dilated convolution layer

    Performs a 2D convolution with dilated filters, then optionally adds a bias
    and applies an elementwise activation.

    Parameters
    ----------
    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    dilation : int or iterable of int
         An integer or a 2-element tuple specifying the dilation factor of the
         filters. A factor of :math:`x` corresponds to :math:`x - 1` zeros
         inserted between adjacent filter elements.

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
        ``(num_input_channels, num_filters, filter_rows, filter_columns)``.
        Note that the first two dimensions are swapped compared to a
        non-dilated convolution.
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

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Notes
    -----
    The dilated convolution is implemented as the backward pass of a
    convolution wrt. weights, passing the filters as the output gradient.
    It can be thought of as dilating the filters (by adding ``dilation - 1``
    zeros between adjacent filter elements) and cross-correlating them with the
    input. See [1]_ for more background.
    DilatedConv2D requires pad=0 / (0,0) / 'valid', but got %r.
    For a padded dilated convolution, add a PadLayer."

    References
    ----------
    .. [1] Fisher Yu, Vladlen Koltun (2016),
           Multi-Scale Context Aggregation by Dilated Convolutions. ICLR 2016.
           http://arxiv.org/abs/1511.07122, https://github.com/fyu/dilation
    """

    def __init__(self, num_filters, filter_size, dilation=(1, 1),
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 untie_biases=False,
                 activation=K.linear, **kwargs):
        super(DilatedConv2D, self).__init__(num_filters, filter_size, pad=0,
                                            W_init=W_init, b_init=b_init,
                                            untie_biases=untie_biases,
                                            activation=activation, n=2, **kwargs)
        self.dilation = as_tuple(dilation, 2, int)
        self.stride = (1, 1)

    def get_W_shape(self, input_shape):
        # first two sizes are swapped compared to a forward convolution
        return (input_shape[1], self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, (filter - 1) * dilate + 1, 1, 0)
                      for input, filter, dilate
                      in zip(input_shape[2:], self.filter_size,
                             self.dilation)))

    def convolve(self, input):
        # we perform a convolution backward pass wrt weights,
        # passing kernels as output gradient
        imshp = self.input_shape
        kshp = self.output_shape
        # and swapping channels and batchsize
        imshp = (imshp[1], imshp[0]) + imshp[2:]
        kshp = (kshp[1], kshp[0]) + kshp[2:]
        # Now only internal support for Theano (from Lasagne)
        from theano import tensor as T
        op = T.nnet.abstract_conv.AbstractConv2d_gradWeights(
            imshp=imshp, kshp=kshp,
            subsample=self.dilation, border_mode='valid',
            filter_flip=False)
        output_size = self.output_shape[2:]
        if any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(input.transpose(1, 0, 2, 3), self.W, output_size)
        return conved.transpose(1, 0, 2, 3)


class Conv3D(BaseConv):
    """
    3D convolutional layer

    Performs a 3D convolution on its input and optionally adds a bias and
    applies an elementwise activation.  This implementation uses
    ``theano.sandbox.cuda.dnn.dnn_conv3d`` directly.

    Parameters
    ----------
    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of three integers allows different symmetric
        padding per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        4D tensor.

    W_init : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape ``(num_filters,
        num_input_channels, filter_rows, filter_columns, filter_depth)``.
        See :func:`lasagne.utils.create_param` for more information.

    b_init : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untie_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns, output_depth)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    activation : callable or None
        The activation that is applied to the layer activations. If None
        is provided, the layer will be linear.

    **kwargs
        Any additional keyword arguments are passed to the `NNOps` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    # ==================== abstract method ==================== #
    def convolve(self, x):
        # by default we assume 'cross', consistent with corrmm.
        conved = K.conv3d(x, kernel=self.W,
                          strides=self.stride,
                          border_mode=self.pad,
                          image_shape=self.input_shape,
                          filter_shape=K.get_shape(self.W))
        return conved

    def _transpose(self):
        raise NotImplementedError
