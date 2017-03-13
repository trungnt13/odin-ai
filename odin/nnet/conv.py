from __future__ import division, absolute_import

import numpy as np

from odin import backend as K
from odin.basic import PARAMETER, WEIGHT, BIAS
from odin.utils import as_tuple
from odin.utils.shape_calculation import get_conv_output_shape, get_deconv_output_shape
from .base import NNOps, NNTransposeOps


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
    This Ops can be used for both 2D (images) and 3D (videos)
    """

    def __init__(self, num_filters, filter_size, strides=1, pad='valid',
                 W_init=K.init.glorot_uniform, b_init=K.init.constant(0),
                 untie_biases=False, activation=K.linear,
                 dilation=1, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.num_filters = int(num_filters)
        self.filter_size = filter_size
        self.strides = strides
        self.pad = pad
        self.W_init = W_init
        self.b_init = b_init
        self.untie_biases = bool(untie_biases)
        self.dilation = dilation
        self.activation = K.linear if activation is None else activation

    # ==================== abstract methods ==================== #
    @property
    def kernel_shape(self):
        # TF kernel shape: (kernel_dim1, kernel_dim2, ..., input_depth, out_depth)
        return self.filter_size + (self.input_shape[-1], self.num_filters)

    @property
    def output_shape(self):
        return get_conv_output_shape(self.input_shape, self.kernel_shape,
                border_mode=self.pad, subsample=self.strides,
                filter_dilation=self.dilation)

    def _transpose(self):
        return DeConv(self)

    def _initialize(self):
        # ====== validate init arguments ====== #
        ndim = len(self.input_shape) - 2; self.ndim = ndim
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
        # weights
        self.config.create_params(
            self.W_init, shape=self.kernel_shape, name='W', roles=WEIGHT)
        if self.b_init is not None:
            if self.untie_biases:
                biases_shape = self.output_shape[1:]
            else:
                biases_shape = (self.num_filters,)
            self.config.create_params(
                self.b_init, shape=biases_shape, name='b', roles=BIAS)

    def _apply(self, X):
        # ====== apply convolution ====== #
        self._last_input = X
        conved = self.convolve(X)
        output_shape = K.get_shape(conved)
        # ====== check output_shape match the estimated output_shape ====== #
        if len(output_shape) != len(self.output_shape) or \
        any(i != j for i, j in zip(output_shape, self.output_shape)
                if i is not None and j is not None):
            raise RuntimeError("The actual output_shape of this Convolution Ops "
                               "is %s, but the pre-estimated output_shape is: %s "
                               % (str(output_shape), str(self.output_shape)))
        # ====== apply bias ====== #
        if hasattr(self, 'b'):
            if self.untie_biases:
                conved += K.expand_dims(self.b, 0)
            else:
                conved += K.dimshuffle(self.b, ('x',) * (self.ndim + 1) + (0,))
        activated = self.activation(conved)
        K.add_shape(activated, output_shape)
        # set shape for output
        return activated

    def convolve(self, X):
        if self.ndim == 2:
            conv_func = K.conv2d
        elif self.ndim == 3:
            conv_func = K.conv3d
        else:
            raise Exception('No support for %d-D input.' % self.ndim)
        conved = conv_func(X, kernel=self.W,
                           strides=self.strides,
                           border_mode=self.pad,
                           filter_dilation=self.dilation)
        return conved


# ===========================================================================
# TransposeConv
# ===========================================================================
class TransposeConv(Conv):

    def __init__(self, num_filters, filter_size, strides=1, pad='valid',
                 W_init=K.init.glorot_uniform, b_init=K.init.constant(0),
                 untie_biases=False, activation=K.linear,
                 dilation=1, output_shape=None, **kwargs):
        super(TransposeConv, self).__init__(num_filters=num_filters,
                 filter_size=filter_size, strides=strides, pad=pad,
                 W_init=W_init, b_init=b_init,
                 untie_biases=untie_biases, activation=activation,
                 dilation=dilation, **kwargs)
        # explicit output shape
        self._output_shape = output_shape

    def _transpose(self):
        return DeConv(self)

    @property
    def kernel_shape(self):
        # TF kernel shape: (kernel_dim1, kernel_dim2, ..., input_depth, out_depth)
        # revert the output_channel in input_channel to keep the same kernel
        # shape as original Convolution ops
        return self.filter_size + (self.num_filters, self.input_shape[-1])

    @property
    def output_shape(self):
        if self._output_shape is None:
            return get_deconv_output_shape(self.input_shape, self.kernel_shape,
                    border_mode=self.pad, subsample=self.strides,
                    filter_dilation=self.dilation)
        return self._output_shape

    def convolve(self, X):
        if self.ndim == 2:
            deconv_func = K.deconv2d
        elif self.ndim == 3:
            deconv_func = K.deconv3d
        else:
            raise Exception('No support for %d-D input.' % self.ndim)
        # ====== prepare the deconvolution ====== #
        # theano require batch_dims is Constant or None, but tensorflow
        # require batch_dims is a native TensorVariable
        # output_shape = K.get_shape(self.T._last_input,
        #     native=True if K.backend() == 'tensorflow' else False)
        output_shape = self.output_shape
        deconved = deconv_func(X, kernel=self.W,
                               output_shape=output_shape,
                               strides=self.strides,
                               border_mode=self.pad,
                               filter_dilation=self.dilation)

        return deconved


# ===========================================================================
# Deconvolution
# ===========================================================================
class DeConv(NNTransposeOps):

    def __init__(self, ops):
        super(DeConv, self).__init__(ops)
        self._deconv = None

    @property
    def variables(self):
        v = super(DeConv, self).variables + self._deconv.variables
        v = list(set(v))
        return v

    @property
    def kernel_shape(self):
        return self.T.kernel_shape

    @property
    def output_shape(self):
        return self.T.input_shape

    # ==================== abstract method ==================== #
    def _initialize(self):
        super(DeConv, self)._initialize()
        ops = self.T
        if isinstance(ops, TransposeConv):
            self._deconv = Conv(num_filters=ops.input_shape[-1],
                    filter_size=ops.filter_size, strides=ops.strides, pad=ops.pad,
                    W_init=ops.W, b_init=ops.b_init,
                    untie_biases=ops.untie_biases, activation=ops.activation,
                    dilation=ops.dilation, name=self.name + '_deconv')
        elif isinstance(ops, Conv):
            self._deconv = TransposeConv(num_filters=ops.input_shape[-1],
                    filter_size=ops.filter_size, strides=ops.strides, pad=ops.pad,
                    W_init=ops.W, b_init=ops.b_init,
                    untie_biases=ops.untie_biases, activation=ops.activation,
                    dilation=ops.dilation, output_shape=ops.input_shape,
                    name=self.name + '_deconv')
        else:
            raise ValueError("Unsupport deconvolution for NNOps with type=%s"
                             % str(type(self.T)))

    def _apply(self, X):
        return self._deconv(X)
