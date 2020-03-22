from __future__ import absolute_import, division, print_function

import dataclasses
import types
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers.convolutional import Conv as _Conv

from odin.backend.alias import (parse_activation, parse_constraint,
                                parse_initializer, parse_regularizer)
from odin.networks.util_layers import (Conv1DTranspose, ExpandDims, Identity,
                                       ReshapeMCMC)
from odin.utils import as_tuple

__all__ = [
    'SequentialNetwork',
    'DenseNetwork',
    'ConvNetwork',
    'AutoencoderConfig',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _as_arg_tuples(*args):
  ref = as_tuple(args[0], t=int)
  n = len(ref)
  return [ref] + [as_tuple(i, N=n) for i in args[1:]], n


def _store_arguments(d):
  self = d.pop('self')
  d.pop('__class__')
  self._init_arguments = dict(d)


def _rank_and_input_shape(rank, input_shape, start_layers):
  if rank is None and input_shape is None:
    raise ValueError(
        "rank or input_shape must be given so the convolution type "
        "can be determined.")
  if input_shape is not None:
    if len(start_layers) > 0:
      first = start_layers[0]
      if not hasattr(first, '_batch_input_shape'):
        first._batch_input_shape = (None,) + tuple(input_shape)
      input_shape = None
    elif rank is not None:
      if not isinstance(input_shape, tf.TensorShape):
        input_shape = tf.nest.flatten(input_shape)
      if rank != (len(input_shape) - 1):
        raise ValueError("rank=%d but given input_shape=%s (rank=%d)" %
                         (rank, str(input_shape), len(input_shape) - 1))
  if rank is None:
    rank = len(input_shape) - 1
  return rank, input_shape


_STORED_TRANSPOSE = {}


class SequentialNetwork(keras.Sequential):

  def __init__(self, start_layers=[], layers=None, end_layers=[], name=None):
    layers = [
        [] if l is None else list(l) for l in (start_layers, layers, end_layers)
    ]
    layers = tf.nest.flatten(layers)
    super().__init__(layers=None if len(layers) == 0 else layers, name=name)

  @property
  def init_arguments(self):
    return dict(self._init_arguments)

  def transpose(self, input_shape=None, tied_weights=False):
    r"""
    Arguments:
      input_shape : specific input shape for the transposed network
      tied_weights : Boolean. Tie the weight of the encoder and decoder.
    """
    raise NotImplementedError

  def get_config(self):
    return super().get_config()

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls.from_config(config, custom_objects)


# ===========================================================================
# Networks
# ===========================================================================
class DenseNetwork(SequentialNetwork):
  r""" Multi-layers neural network """

  def __init__(self,
               units=128,
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               flatten=False,
               batchnorm=True,
               input_dropout=0.,
               output_dropout=0.,
               layer_dropout=0.,
               input_shape=None,
               start_layers=[],
               end_layers=[],
               name=None):
    (units, activation, use_bias, kernel_initializer, bias_initializer,
     kernel_regularizer, bias_regularizer, activity_regularizer,
     kernel_constraint, bias_constraint,
     batchnorm, layer_dropout), nlayers = _as_arg_tuples(
         units, activation, use_bias, kernel_initializer, bias_initializer,
         kernel_regularizer, bias_regularizer, activity_regularizer,
         kernel_constraint, bias_constraint, batchnorm, layer_dropout)
    _store_arguments(locals())

    layers = []
    if flatten:
      layers.append(keras.layers.Flatten())
    if 0. < input_dropout < 1.:
      layers.append(keras.layers.Dropout(input_dropout))
    for i in range(nlayers):
      layers.append(
          keras.layers.Dense(\
            units[i],
            activation='linear',
            use_bias=(not batchnorm[i]) and use_bias[i],
            kernel_initializer=kernel_initializer[i],
            bias_initializer=bias_initializer[i],
            kernel_regularizer=kernel_regularizer[i],
            bias_regularizer=bias_regularizer[i],
            activity_regularizer=activity_regularizer[i],
            kernel_constraint=kernel_constraint[i],
            bias_constraint=bias_constraint[i],
            name="Layer%d" % i))
      if batchnorm[i]:
        layers.append(keras.layers.BatchNormalization())
      layers.append(keras.layers.Activation(activation[i]))
      if layer_dropout[i] > 0 and i != nlayers - 1:
        layers.append(keras.layers.Dropout(rate=layer_dropout[i]))
    if 0. < output_dropout < 1.:
      layers.append(keras.layers.Dropout(output_dropout))
    # matching input_shape and start_layers
    if input_shape is not None:
      if len(start_layers) > 0 and \
        not hasattr(start_layers[0], '_batch_input_shape'):
        start_layers[0]._batch_input_shape = (None,) + tuple(input_shape)
      else:
        layers = [keras.Input(shape=input_shape)] + layers
    super().__init__(start_layers=start_layers,
                     layers=layers,
                     end_layers=end_layers,
                     name=name)

  def transpose(self, input_shape=None, tied_weights=False):
    r""" Created a transposed network """
    if id(self) in _STORED_TRANSPOSE:
      return _STORED_TRANSPOSE[id(self)]
    args = self.init_arguments
    args['units'] = args['units'][::-1]
    args['input_shape'] = input_shape
    args['name'] = self.name + '_transpose'
    args['flatten'] = False
    if tied_weights:
      args['kernel_constraint'] = None
      args['kernel_regularizer'] = None
    del args['nlayers']
    # create the transposed network
    transpose_net = DenseNetwork(**args)
    _STORED_TRANSPOSE[id(self)] = transpose_net
    if tied_weights:
      weights = [w for w in self.weights if '/kernel' in w.name][::-1][:-1]
      layers = [
          l for l in transpose_net.layers if isinstance(l, keras.layers.Dense)
      ][1:]
      for w, l in zip(weights, layers):

        def build(self, input_shape):
          input_shape = tensor_shape.TensorShape(input_shape)
          last_dim = tensor_shape.dimension_value(input_shape[-1])
          self.input_spec = keras.layers.InputSpec(min_ndim=2,
                                                   axes={-1: last_dim})
          self.kernel = tf.transpose(self.tied_kernel)
          if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
          else:
            self.bias = None
          self.built = True

        l.tied_kernel = w
        l.build = types.MethodType(build, l)
    return transpose_net


class ConvNetwork(SequentialNetwork):
  r""" Multi-layers neural network """

  def __init__(self,
               filters,
               rank=2,
               kernel_size=3,
               strides=1,
               padding='same',
               dilation_rate=1,
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               batchnorm=True,
               input_dropout=0.,
               output_dropout=0.,
               layer_dropout=0.,
               input_shape=None,
               start_layers=[],
               end_layers=[],
               name=None):
    rank, input_shape = _rank_and_input_shape(rank, input_shape, start_layers)
    (filters, kernel_size, strides, padding, dilation_rate, activation,
     use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
     bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint,
     batchnorm, layer_dropout), nlayers = _as_arg_tuples(
         filters, kernel_size, strides, padding, dilation_rate, activation,
         use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
         bias_regularizer, activity_regularizer, kernel_constraint,
         bias_constraint, batchnorm, layer_dropout)
    _store_arguments(locals())

    layers = []
    if input_shape is not None:
      layers.append(keras.Input(shape=input_shape))
    if 0. < input_dropout < 1.:
      layers.append(keras.layers.Dropout(input_dropout))

    if rank == 3:
      layer_type = keras.layers.Conv3D
    elif rank == 2:
      layer_type = keras.layers.Conv2D
    elif rank == 1:
      layer_type = keras.layers.Conv1D

    for i in range(nlayers):
      layers.append(
          layer_type(\
            filters=filters[i],
            kernel_size=kernel_size[i],
            strides=strides[i],
            padding=padding[i],
            dilation_rate=dilation_rate[i],
            activation='linear',
            use_bias=(not batchnorm[i]) and use_bias[i],
            kernel_initializer=kernel_initializer[i],
            bias_initializer=bias_initializer[i],
            kernel_regularizer=kernel_regularizer[i],
            bias_regularizer=bias_regularizer[i],
            activity_regularizer=activity_regularizer[i],
            kernel_constraint=kernel_constraint[i],
            bias_constraint=bias_constraint[i],
            name="Layer%d" % i))
      if batchnorm[i]:
        layers.append(keras.layers.BatchNormalization())
      layers.append(keras.layers.Activation(activation[i]))
      if layer_dropout[i] > 0 and i != nlayers - 1:
        layers.append(keras.layers.Dropout(rate=layer_dropout[i]))
    if 0. < output_dropout < 1.:
      layers.append(keras.layers.Dropout(output_dropout))
    super().__init__(start_layers=start_layers,
                     layers=layers,
                     end_layers=end_layers,
                     name=name)

  def transpose(self, input_shape=None, tied_weights=False):
    if tied_weights:
      raise NotImplementedError(
          "No support for tied_weights in ConvNetwork.transpose")
    if id(self) in _STORED_TRANSPOSE:
      return _STORED_TRANSPOSE[id(self)]
    args = {
        k: v[::-1] if isinstance(v, tuple) else v
        for k, v in self.init_arguments.items()
    }
    rank = args['rank']
    # input_shape: infer based on output of ConvNetwork
    start_layers = []
    if hasattr(self, 'output_shape'):
      if input_shape is None:
        start_layers.append(keras.Input(input_shape=self.output_shape[1:]))
      else:
        input_shape = as_tuple(input_shape)
        shape = [
            l.output_shape[1:]
            for l in self.layers[::-1]
            if isinstance(l, _Conv)
        ][0]  # last convolution layer
        start_layers = [keras.layers.Flatten(input_shape=input_shape)]
        if input_shape != shape:
          if np.prod(input_shape) != np.prod(shape):
            start_layers.append(
                keras.layers.Dense(units=int(np.prod(shape)),
                                   use_bias=False,
                                   activation='linear'))
          start_layers.append(keras.layers.Reshape(shape))
    # create the transposed network
    transposed = DeconvNetwork(
        filters=args['filters'],
        rank=args['rank'],
        kernel_size=args['kernel_size'],
        strides=args['strides'],
        padding=args['padding'],
        dilation_rate=args['dilation_rate'],
        activation=args['activation'],
        use_bias=args['use_bias'],
        kernel_initializer=args['kernel_initializer'],
        bias_initializer=args['bias_initializer'],
        kernel_regularizer=args['kernel_regularizer'],
        bias_regularizer=args['bias_regularizer'],
        activity_regularizer=args['activity_regularizer'],
        kernel_constraint=args['kernel_constraint'],
        bias_constraint=args['bias_constraint'],
        batchnorm=args['batchnorm'],
        input_dropout=args['input_dropout'],
        output_dropout=args['output_dropout'],
        layer_dropout=args['layer_dropout'],
        start_layers=start_layers,
        end_layers=[])
    _STORED_TRANSPOSE[id(self)] = transposed
    return transposed


class DeconvNetwork(SequentialNetwork):
  r""" Multi-layers neural network """

  def __init__(self,
               filters,
               rank=2,
               kernel_size=3,
               strides=1,
               padding='same',
               output_padding=None,
               dilation_rate=1,
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               batchnorm=True,
               input_dropout=0.,
               output_dropout=0.,
               layer_dropout=0.,
               input_shape=None,
               start_layers=[],
               end_layers=[],
               name=None):
    rank, input_shape = _rank_and_input_shape(rank, input_shape, start_layers)
    (filters, kernel_size, strides, padding, output_padding, dilation_rate,
     activation, use_bias, kernel_initializer, bias_initializer,
     kernel_regularizer, bias_regularizer, activity_regularizer,
     kernel_constraint, bias_constraint,
     batchnorm, layer_dropout), nlayers = _as_arg_tuples(
         filters, kernel_size, strides, padding, output_padding, dilation_rate,
         activation, use_bias, kernel_initializer, bias_initializer,
         kernel_regularizer, bias_regularizer, activity_regularizer,
         kernel_constraint, bias_constraint, batchnorm, layer_dropout)
    _store_arguments(locals())

    layers = []
    if input_shape is not None:
      layers.append(keras.Input(shape=input_shape))
    if 0. < input_dropout < 1.:
      layers.append(keras.layers.Dropout(input_dropout))

    if rank == 3:
      raise NotImplementedError
    elif rank == 2:
      layer_type = keras.layers.Conv2DTranspose
    elif rank == 1:
      layer_type = Conv1DTranspose

    for i in range(nlayers):
      layers.append(
          layer_type(\
            filters=filters[i],
            kernel_size=kernel_size[i],
            strides=strides[i],
            padding=padding[i],
            output_padding=output_padding[i],
            dilation_rate=dilation_rate[i],
            activation='linear',
            use_bias=(not batchnorm[i]) and use_bias[i],
            kernel_initializer=kernel_initializer[i],
            bias_initializer=bias_initializer[i],
            kernel_regularizer=kernel_regularizer[i],
            bias_regularizer=bias_regularizer[i],
            activity_regularizer=activity_regularizer[i],
            kernel_constraint=kernel_constraint[i],
            bias_constraint=bias_constraint[i],
            name="Layer%d" % i))
      if batchnorm[i]:
        layers.append(keras.layers.BatchNormalization())
      layers.append(keras.layers.Activation(activation[i]))
      if layer_dropout[i] > 0 and i != nlayers - 1:
        layers.append(keras.layers.Dropout(rate=layer_dropout[i]))
    if 0. < output_dropout < 1.:
      layers.append(keras.layers.Dropout(output_dropout))
    super().__init__(start_layers=start_layers,
                     layers=layers,
                     end_layers=end_layers,
                     name=name)


# ===========================================================================
# Serializable configuration
# ===========================================================================
@dataclasses.dataclass(init=True,
                       repr=True,
                       eq=True,
                       order=False,
                       unsafe_hash=False,
                       frozen=True)
class AutoencoderConfig(dict):
  r""" A dataclass for storing the autoencoder networks (encoder and decoder)
  configuration

  Arguments:
    hidden_dim : An Integer, number of hidden units for each hidden layers
    nlayers : An Integer, number of hidden layers
    activation : a String, alias of activation function
    input_dropout : A Scalar [0., 1.], dropout rate, if 0., turn-off dropout.
      this rate is applied for input layer.
     - encoder_dropout : for the encoder output
     - latent_dropout : for the decoder input (right after the latent)
     - decoder_dropout : for the decoder output
     - layer_dropout : for each hidden layer
    batchnorm : A Boolean, batch normalization
    linear_decoder : A Boolean, if `True`, use an `Identity` (i.e. Linear)
      decoder
    pyramid : A Boolean, if `True`, use pyramid structure where the number of
      hidden units decrease as the depth increase
    use_conv : A Boolean, if `True`, use convolutional encoder and decoder
    kernel_size : An Integer, kernel size for convolution network
    strides : An Integer, stride step for convoltion
    conv_proj : An Integer, number of hidden units for the `Dense` projection
      layer right after convolutional network.
    sample_ndim : an Integer, if True, use stochastic encoder and decoder for
      variational autoencoder, i.e. intepret the first (or more) dimension(s)
      as sample shape from MCMC sampling.
  """

  hidden_dim: int = 64
  nlayers: int = 2
  activation: str = 'relu'
  input_dropout: float = 0.3
  encoder_dropout: float = 0.
  latent_dropout: float = 0.
  decoder_dropout: float = 0.
  layer_dropout: float = 0.
  batchnorm: bool = True
  linear_decoder: bool = False
  pyramid: bool = False
  use_conv: bool = False
  kernel_size: int = 5
  strides: int = 2
  conv_proj: int = 128
  sample_ndim: int = 0

  def keys(self):
    for i in dataclasses.fields(self):
      yield i.name

  def values(self):
    for i in dataclasses.fields(self):
      yield i.default

  def __iter__(self):
    for i in dataclasses.fields(self):
      yield i.name, i.default

  def __len__(self):
    return len(dataclasses.fields(self))

  def __getitem__(self, key):
    return getattr(self, key)

  def copy(self, **kwargs):
    obj = deepcopy(self)
    return dataclasses.replace(obj, **kwargs)

  def create_network(self, input_shape, latent_shape=None, name=None):
    r"""
    Arguments:
      input_shape : a tuple of Integer. Shape of input without the batch
         dimensions.
      latent_shape : a tuple of Integer. Shape of latent without the batch
         dimensions.
      name : a String (optional).

    Returns:
      encoder : keras.Sequential
      decoder : keras.Sequential or None (in case latent_shape is None)
    """
    encoder_name = None if name is None else "%s_%s" % (name, "encoder")
    decoder_name = None if name is None else "%s_%s" % (name, "decoder")
    encoder = None
    decoder = None
    ### prepare the shape
    input_shape = tf.nest.flatten(input_shape)
    input_ndim = len(input_shape)
    if latent_shape is not None:
      latent_shape = tf.nest.flatten(latent_shape)
    ### network config
    if self.pyramid:
      units = [int(self.hidden_dim / 2**i) for i in range(1, self.nlayers + 1)]
    else:
      units = [self.hidden_dim] * self.nlayers
    ### convolution network
    if self.use_conv:
      assert input_ndim in (1, 2, 3), \
        "Only support 2-D, 3-D or 4-D inputs, but given: %s" % str(input_shape)
      start_layers = []
      # reshape to 3-D
      if input_ndim == 1:
        start_layers.append(ExpandDims(axis=-1))
        rank = 1
        n_channels = 1
      else:
        rank = input_ndim - 1
        n_channels = input_shape[-1]
      # create the encoder
      encoder = ConvNetwork(units[::-1],
                            rank=rank,
                            kernel_size=self.kernel_size,
                            strides=self.strides,
                            padding='same',
                            dilation_rate=1,
                            activation=self.activation,
                            use_bias=True,
                            batchnorm=self.batchnorm,
                            input_dropout=self.input_dropout,
                            output_dropout=self.encoder_dropout,
                            layer_dropout=self.layer_dropout,
                            start_layers=start_layers,
                            input_shape=input_shape,
                            end_layers=[
                                keras.layers.Flatten(),
                                keras.layers.Dense(self.conv_proj,
                                                   activation=self.activation,
                                                   use_bias=True)
                            ],
                            name=encoder_name)
      # get the last convolution shape
      eshape = encoder.layers[-3].output_shape[1:]
      # Create the decoder
      if not self.linear_decoder and latent_shape is not None:
        decoder = DeconvNetwork(
            units[1:] + [n_channels],
            rank=rank,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
            dilation_rate=1,
            activation=self.activation,
            use_bias=True,
            batchnorm=self.batchnorm,
            input_dropout=self.latent_dropout,
            output_dropout=self.decoder_dropout,
            layer_dropout=self.layer_dropout,
            start_layers=[
                keras.layers.Dense(self.conv_proj,
                                   activation=self.activation,
                                   use_bias=True,
                                   input_shape=latent_shape),
                keras.layers.Dense(np.prod(eshape),
                                   activation=self.activation,
                                   use_bias=True),
                keras.layers.Reshape(eshape),
            ],
            end_layers=[keras.layers.Reshape(input_shape)],
            name=decoder_name)
    ### dense network
    else:
      encoder = DenseNetwork(units=units,
                             activation=self.activation,
                             flatten=True if input_ndim > 1 else False,
                             use_bias=True,
                             batchnorm=self.batchnorm,
                             input_dropout=self.input_dropout,
                             output_dropout=self.encoder_dropout,
                             layer_dropout=self.layer_dropout,
                             input_shape=input_shape,
                             name=encoder_name)
      if not self.linear_decoder and latent_shape is not None:
        decoder = DenseNetwork(units=units[::-1],
                               activation=self.activation,
                               use_bias=True,
                               batchnorm=self.batchnorm,
                               input_dropout=self.latent_dropout,
                               output_dropout=self.decoder_dropout,
                               layer_dropout=self.layer_dropout,
                               input_shape=latent_shape,
                               name=decoder_name)
    # ====== linear decoder ====== #
    if latent_shape is not None:
      if self.linear_decoder:
        decoder = Identity(name=decoder_name, input_shape=latent_shape)
      elif self.sample_ndim > 0:
        decoder = ReshapeMCMC(decoder,
                              sample_ndim=self.sample_ndim,
                              name=decoder_name)
    # ====== return ====== #
    return encoder, decoder
