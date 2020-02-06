from __future__ import absolute_import, division, print_function

from collections import Iterable

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import (Activation, BatchNormalization,
                                            Conv1D, Dense, Lambda, Layer)
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import variable_scope

__all__ = [
    'Identity',
    'Parallel',
    'BatchRenormalization',
    'Convolution1DTranspose',
    'Conv1DTranspose',
    'Deconvolution1D',
    'Deconv1D',
]


class ModuleList(Sequential):
  r""" Holds submodules in a list.
  :class:`~odin.networks.ModuleList` can be indexed like a regular Python list,
  but modules it contains are properly registered, and will be visible by all
  :class:`~keras.layers.Layer` methods.

  Arguments:
    modules (iterable, optional): an iterable of `Layer` to add
  """

  def __init__(self, modules=None, name=None):
    super().__init__(layers=modules, name=name)

  def modules(self):
    for l in self.layers:
      yield l

  def named_modules(self):
    for l in self.layers:
      yield l.name, l

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return self.__class__(self.layers.values()[idx])
    else:
      return self.layers[idx]

  def __setitem__(self, idx, module):
    idx = int(idx)
    return setattr(self, str(idx), module)

  def __delitem__(self, idx):
    if not self.layers:
      raise TypeError('There are no layers in the model.')
    ids = list(range(len(self._layers)))[idx]
    layers = self._layers[idx]
    if not isinstance(layers, (tuple, list)):
      layers = [layers]
      ids = [ids]

    self._layers = [l for i, l in enumerate(self._layers) if i not in ids]
    for layer in layers:
      self._layer_call_argspecs.pop(layer)
    # removed all layer
    if not self.layers:
      self.outputs = None
      self.inputs = None
      self.built = False
    # modifying the outputs and re-build in case of static graph
    elif self._is_graph_network:
      self.layers[-1]._outbound_nodes = []
      self.outputs = [self.layers[-1].output]
      self._init_graph_network(self.inputs, self.outputs, name=self.name)
      self.built = True

  def __len__(self):
    return len(self.layers)

  def __iter__(self):
    return iter(self.layers)

  def __iadd__(self, modules):
    return self.extend(modules)

  def insert(self, index, module):
    raise NotImplementedError()

  def append(self, module):
    r"""Appends a given layer to the end of the list.

    Arguments:
        module (keras.Layer): module to append
    """
    self.add(module)
    return self

  def extend(self, modules):
    r"""Appends layers from a Python iterable to the end of the list.

    Arguments:
        modules (iterable): iterable of modules to append
    """
    if not isinstance(modules, Iterable):
      raise TypeError("ModuleList.extend should be called with an "
                      "iterable, but got " + type(modules).__name__)
    for module in modules:
      self.add(module)
    return self


class BatchRenormalization(BatchNormalization):
  r""" Shortcut for batch renormalization

  References
  ----------
  [1] S. Ioffe, “Batch Renormalization: Towards Reducing Minibatch Dependence in
  Batch-Normalized Models,” arXiv:1702.03275 [cs], Feb. 2017.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super().__init__(name=name,
                     axis=axis,
                     momentum=momentum,
                     epsilon=epsilon,
                     center=center,
                     scale=scale,
                     beta_initializer=beta_initializer,
                     gamma_initializer=gamma_initializer,
                     moving_mean_initializer=moving_mean_initializer,
                     moving_variance_initializer=moving_variance_initializer,
                     beta_regularizer=beta_regularizer,
                     gamma_regularizer=gamma_regularizer,
                     beta_constraint=beta_constraint,
                     gamma_constraint=gamma_constraint,
                     renorm=True,
                     renorm_clipping=renorm_clipping,
                     renorm_momentum=renorm_momentum,
                     fused=fused,
                     trainable=trainable,
                     virtual_batch_size=virtual_batch_size,
                     adjustment=adjustment,
                     **kwargs)


class Identity(Layer):

  def __init__(self, name=None, **kwargs):
    super(Identity, self).__init__(name=name, **kwargs)
    self.supports_masking = True

  def call(self, inputs, training=None):
    return inputs

  def compute_output_shape(self, input_shape):
    return input_shape


class Parallel(Sequential):
  """ Similar design to keras `Sequential` but simultanously applying
  all the layer on the input and return all the results.

  This layer is important for implementing multitask learning.
  """

  def call(self, inputs, training=None, mask=None, **kwargs):  # pylint: disable=redefined-outer-name
    if self._is_graph_network:
      if not self.built:
        self._init_graph_network(self.inputs, self.outputs, name=self.name)
      return super(Parallel, self).call(inputs, training=training, mask=mask)

    outputs = []
    for layer in self.layers:
      # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
      # are the outputs of `layer` applied to `inputs`. At the end of each
      # iteration `inputs` is set to `outputs` to prepare for the next layer.
      kw = {}
      argspec = self._layer_call_argspecs[layer].args
      if 'mask' in argspec:
        kw['mask'] = mask
      if 'training' in argspec:
        kw['training'] = training
      # support custom keyword argument also
      for k, v in kwargs.items():
        if k in argspec:
          kw[k] = v

      o = layer(inputs, **kw)
      outputs.append(o)

    return tuple(outputs)

  def compute_output_shape(self, input_shape):
    shape = []
    for layer in self.layers:
      shape.append(layer.compute_output_shape(input_shape))
    return tuple(shape)

  def compute_mask(self, inputs, mask):
    outputs = self.call(inputs, mask=mask)
    return [o._keras_mask for o in outputs]


# ===========================================================================
# TransposeConv1D
# ===========================================================================
class Conv1DTranspose(Conv1D):
  r"""
  Arguments:
    output_padding: An integer, specifying the amount of padding along the
      feature dimension of the output tensor.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               output_padding=None,
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv1DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=keras.activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=keras.initializers.get(kernel_initializer),
        bias_initializer=keras.initializers.get(bias_initializer),
        kernel_regularizer=keras.regularizers.get(kernel_regularizer),
        bias_regularizer=keras.regularizers.get(bias_regularizer),
        activity_regularizer=keras.regularizers.get(activity_regularizer),
        kernel_constraint=keras.constraints.get(kernel_constraint),
        bias_constraint=keras.constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = keras.utils.conv_utils.normalize_tuple(
          self.output_padding, 1, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if len(input_shape) != 3:
      raise ValueError('Inputs should have rank 3. Received input shape: ' +
                       str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(name='kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True,
                                  dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  trainable=True,
                                  dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis = 2
    else:
      h_axis = 1

    height = inputs_shape[h_axis]
    kernel_w = self.kernel_size[0]
    stride_w = self.strides[0]

    if self.output_padding is None:
      out_pad_w = None
    else:
      out_pad_w = self.output_padding[0]

    # Infer the dynamic output shape:
    out_width = conv_utils.deconv_output_length(height,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[0])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_width)
      data_format = 'NCW'
      strides = [1, 1, self.strides[0]]
      dilations = [1, 1, self.dilation_rate[0]]
    else:
      output_shape = (batch_size, out_width, self.filters)
      data_format = 'NWC'
      strides = [1, self.strides[0], 1]
      dilations = [1, self.dilation_rate[0], 1]
    assert self.dilation_rate[0] == 1, "No support for dilation_rate > 1"

    output_shape_tensor = tf.stack(output_shape)
    # TODO: support dilations > 1 (atrous_conv2d_transpose)
    outputs = tf.nn.conv1d_transpose(inputs,
                                     self.kernel,
                                     output_shape_tensor,
                                     strides=strides,
                                     padding=self.padding.upper(),
                                     data_format=data_format,
                                     dilations=dilations)

    if not tf.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs,
                               self.bias,
                               data_format=conv_utils.convert_data_format(
                                   self.data_format, ndim=3))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis = 1, 2
    else:
      c_axis, h_axis = 2, 1

    kernel_w = self.kernel_size[0]
    stride_w = self.strides[0]

    if self.output_padding is None:
      out_pad_w = None
    else:
      out_pad_w = self.output_padding[0]

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[0])
    return tf.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv1DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


Convolution1DTranspose = Conv1DTranspose
Deconvolution1D = Deconv1D = Conv1DTranspose
