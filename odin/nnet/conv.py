from __future__ import division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import backend as K
from odin.nnet.base import NNOp
from odin.autoconfig import randint
from odin.utils import as_tuple, is_string
from odin.backend.role import ConvKernel, Bias

# ===========================================================================
# Helper
# ===========================================================================
def __get_deconv_shape_1axis(n, kernel_shape, border_mode,
                             subsample, dilation=1):
  if None in [n, kernel_shape, border_mode,
              subsample, dilation]:
    return None
  if dilation != 1:
    raise ValueError("Deconvolution with dilation != 1 is not supported.")
  # ====== upsample ====== #
  if subsample != 1:
    n = n * subsample
  # ====== padding ====== #
  if isinstance(border_mode, str):
    border_mode = border_mode.lower()
  if border_mode == "half" or border_mode == "same":
    n = n
  elif border_mode == "full": # M + N - 1
    n += 1 - kernel_shape
  elif border_mode == "valid": # M - N + 1
    n += max(kernel_shape - subsample, 0)
  elif border_mode >= 0:
    n += max(kernel_shape - subsample, 0) - border_mode
  else:
    raise ValueError("border_mode must be >= 0")
  return n


def get_deconv_output_shape(image_shape, kernel_shape,
                            border_mode, subsample,
                            filter_dilation=None):
  """ Invert the process of calculating output shape for convolution
  (Deconvolution, or TransposedConvolution)

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
  outkern, kshp = kernel_shape[1], kernel_shape[2:]
  if filter_dilation is None:
    filter_dilation = np.ones(len(subsample), dtype='int')
  if isinstance(border_mode, tuple):
    out_shp = tuple(__get_deconv_shape_1axis(
        imshp[i], kshp[i], border_mode[i],
        subsample[i], filter_dilation[i]) for i in range(len(subsample)))
  else:
    out_shp = tuple(__get_deconv_shape_1axis(
        imshp[i], kshp[i], border_mode,
        subsample[i], filter_dilation[i]) for i in range(len(subsample)))
  # ====== convert theano to tensorflow shape ====== #
  return (bsize, ) + out_shp + (outkern,)


# ===========================================================================
# Ops
# ===========================================================================
class Conv(NNOp):
  """ Convolutional Operator

  Performs a 2D or 3D convolution on its input and optionally adds a bias and
  applies an elementwise activation.

  Parameters
  ----------
  num_filters : int
      The number of learnable convolutional filters this layer has.

  filter_size : int or iterable of int
      tuple specifying the size of the filters.

  strides : int or iterable of int
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
                  pad|                                      |pad
      inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
                  |________________|
                                 |_________________|
                                                |________________|
      ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).
      inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                     |________________|                dropped
                                    |_________________|
      Note that ``'full'`` and ``'same'`` can be faster than equivalent
      integer values due to optimizations by theano or tensorflow.

  untie_biases : bool (default: False)
      If ``False``, the layer will have a bias parameter for each channel,
      which is shared across all positions in this channel. As a result, the
      `b` attribute will be a vector (1D).

      If True, the layer will have separate bias parameters for each
      position in each channel. As a result, the `b` attribute will be a
      3D tensor.

  W_init : Theano shared variable, expression, numpy array or call-able
      Initial value, expression or initializer for the weights.
      These should be a 4D tensor with shape
      ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
      See :func:`lasagne.utils.create_param` for more information.

  b_init : Theano shared variable, expression, numpy array, call-able or ``None``
      Initial value, expression or initializer for the biases. If set to
      ``None``, the layer will have no biases. Otherwise, biases should be
      a 1D array with shape ``(num_filters,)`` if `untie_biases` is set to
      ``False``. If it is set to ``True``, its shape should be
      ``(num_filters, output_rows, output_columns)`` instead.
      See :func:`lasagne.utils.create_param` for more information.

  activation : call-able or None
      The activation that is applied to the layer activations. If None
      is provided, the layer will be linear.

  dilation : int or iterable of int
      Specifying the dilation factor of the filters. A factor of
      :math:`x` corresponds to :math:`x - 1` zeros inserted between
      adjacent filter elements.

  **kwargs
      Any additional keyword arguments are passed to the `NNOp` superclass.

  Note
  ----
  This Ops can be used for 1D, 2D (images) and 3D (videos).
  dim_ordering : tf-tensorflow (defaults)
      input shape: (samples, conv_dim1, conv_dim2, input_depth)
      kernel shape: (kernel_dim1, kernel_dim2, input_depth, out_depth)
  Only support float32 on CPU
  """

  def __init__(self, num_filters, filter_size, strides=1, pad='valid',
               W_init=init_ops.glorot_uniform_initializer(seed=randint()),
               b_init=init_ops.constant_initializer(0),
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

  def _transpose(self):
    return TransposeConv(num_filters=self.input_shape[-1],
                         filter_size=self.filter_size,
                         strides=self.strides, pad=self.pad,
                         W_init=self.get('W'),
                         b_init=None if self.b_init is None else 0.,
                         untie_biases=self.untie_biases,
                         activation=self.activation,
                         dilation=self.dilation,
                         output_shape=self.input_shape)

  def _initialize(self):
    # ====== validate init arguments ====== #
    self.ndim = len(self.input_shape) - 2
    # padding
    if isinstance(self.pad, (tuple, list, int)):
      self.pad = as_tuple(self.pad, self.ndim, int)
    elif self.pad is None:
      self.pad = (0,) * self.ndim
    elif is_string(self.pad):
      self.pad = self.pad.upper()
    # strides
    if self.strides is None:
      self.strides = (1,) * self.ndim
    else:
      self.strides = as_tuple(self.strides, self.ndim, int)
    # dilation
    if self.dilation is None:
      self.dilation = (1,) * self.ndim
    else:
      self.dilation = as_tuple(self.dilation, self.ndim, int)
    # filter size
    self.filter_size = as_tuple(self.filter_size, self.ndim, int)
    # ====== create config ====== #
    # weights
    self.get_variable_nnop(initializer=self.W_init,
        shape=self.kernel_shape, name='W', roles=ConvKernel)
    if self.b_init is not None:
      if self.untie_biases:
        biases_shape = self.output_shape[1:]
      else:
        biases_shape = (self.num_filters,)
      self.get_variable_nnop(initializer=self.b_init,
          shape=biases_shape, name='b', roles=Bias)

  def _apply(self, X):
    # ====== apply convolution ====== #
    conved = self.convolve(X)
    # ====== apply bias ====== #
    if 'b' in self.variable_info:
      if self.untie_biases:
        conved += tf.expand_dims(self.get('b'), axis=0)
      else:
        conved += K.dimshuffle(self.get('b'), ('x',) * (self.ndim + 1) + (0,))
    return self.activation(conved)

  def convolve(self, X):
    if self.ndim == 1:
      data_format = "NWC"
    elif self.ndim == 2:
      data_format = "NHWC"
    elif self.ndim == 3:
      data_format = "NDHWC"
    else:
      raise Exception('No support for %d-D input.' % self.ndim)
    # ====== perform normal convolution ====== #
    conved = tf.nn.convolution(input=X, filter=self.get('W'),
        padding=self.pad,
        strides=self.strides,
        dilation_rate=self.dilation,
        data_format=data_format)
    return conved

# ===========================================================================
# TransposeConv
# ===========================================================================
class TransposeConv(Conv):

  def __init__(self, num_filters, filter_size, strides=1, pad='valid',
               W_init=init_ops.glorot_uniform_initializer(seed=randint()),
               b_init=init_ops.constant_initializer(0),
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
    return Conv(num_filters=self.input_shape[-1], filter_size=self.filter_size,
                strides=self.strides, pad=self.pad,
                W_init=self.get('W'),
                b_init=None if self.b_init is None else 0.,
                untie_biases=self.untie_biases,
                activation=self.activation,
                dilation=self.dilation)

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
      deconv_func = tf.nn.conv2d_transpose
    elif self.ndim == 3:
      deconv_func = tf.nn.conv3d_transpose
    else:
      raise Exception('No support for %d-D input.' % self.ndim)
    # ====== prepare the deconvolution ====== #
    # theano require batch_dims is Constant or None, but tensorflow
    # require batch_dims is a native TensorVariable
    output_shape = self.output_shape
    _ = tuple(output_shape)
    native_shape = tf.shape(X)
    output_shape = [native_shape[i] if j is None else j
                    for i, j in enumerate(output_shape)]
    deconved = deconv_func(value=X, filter=self.get('W'),
        output_shape=output_shape, strides=(1,) + self.strides + (1,),
        padding=self.pad)
    return K.set_shape(deconved, _)
