from __future__ import absolute_import, division, print_function

import dataclasses
import inspect
import types
from copy import deepcopy
from numbers import Number

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.layers.convolutional import Conv as _Conv
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util import tf_inspect

from odin.backend.alias import (parse_activation, parse_constraint,
                                parse_initializer, parse_regularizer)
from odin.backend.keras_helpers import layer2text
from odin.networks.util_layers import (Conv1DTranspose, ExpandDims, Identity,
                                       ReshapeMCMC)
from odin.utils import as_tuple

__all__ = [
    'SequentialNetwork',
    'dense_network',
    'conv_network',
    'deconv_network',
    'NetworkConfig',
]


# ===========================================================================
# Helpers
# ===========================================================================
def _shape(shape):
  if shape is not None:
    if not (tf.is_tensor(shape) or isinstance(shape, tf.TensorShape) or
            isinstance(shape, np.ndarray)):
      shape = tf.nest.flatten(shape)
  return shape


def _as_arg_tuples(*args):
  ref = as_tuple(args[0], t=int)
  n = len(ref)
  return [ref] + [as_tuple(i, N=n) for i in args[1:]], n


def _infer_rank_and_input_shape(rank, input_shape):
  if rank is None and input_shape is None:
    raise ValueError(
        "rank or input_shape must be given so the convolution type "
        "can be determined.")
  if rank is not None:
    if input_shape is not None:
      input_shape = _shape(input_shape)
      if rank != (len(input_shape) - 1):
        raise ValueError("rank=%d but given input_shape=%s (rank=%d)" %
                         (rank, str(input_shape), len(input_shape) - 1))
  else:
    rank = len(input_shape) - 1
  return rank, input_shape


# ===========================================================================
# Base classes
# ===========================================================================
class SequentialNetwork(keras.Sequential):

  def __init__(self, layers=None, name=None):
    super().__init__(layers=None if layers is None else layers, name=name)

  def __repr__(self):
    return self.__str__()

  def __str__(self):
    return layer2text(self)


# ===========================================================================
# Networks
# ===========================================================================
def dense_network(units,
                  activation='relu',
                  use_bias=True,
                  kernel_initializer='glorot_uniform',
                  bias_initializer='zeros',
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  flatten_inputs=True,
                  batchnorm=True,
                  input_dropout=0.,
                  dropout=0.,
                  input_shape=None):
  r""" Multi-layers dense feed-forward neural network """
  (units, activation, use_bias, kernel_initializer, bias_initializer,
   kernel_regularizer, bias_regularizer, activity_regularizer,
   kernel_constraint, bias_constraint, batchnorm,
   dropout), nlayers = _as_arg_tuples(units, activation, use_bias,
                                      kernel_initializer, bias_initializer,
                                      kernel_regularizer, bias_regularizer,
                                      activity_regularizer, kernel_constraint,
                                      bias_constraint, batchnorm, dropout)
  layers = []
  if input_shape is not None:
    layers.append(keras.layers.InputLayer(input_shape=input_shape))
  if flatten_inputs:
    layers.append(keras.layers.Flatten())
  if input_dropout > 0:
    layers.append(keras.layers.Dropout(rate=input_dropout))
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
    if dropout[i] > 0:
      layers.append(keras.layers.Dropout(rate=dropout[i]))
  return layers


def conv_network(units,
                 rank=2,
                 kernel=3,
                 strides=1,
                 padding='same',
                 dilation=1,
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
                 dropout=0.,
                 projection=False,
                 input_shape=None,
                 name=None):
  r""" Multi-layers convolutional neural network

  Arguments:
    projection : {True, False, an Integer}.
      If True, flatten the output into 2-D.
      If an Integer, use a `Dense` layer with linear activation to project
      the output in to 2-D
  """
  rank, input_shape = _infer_rank_and_input_shape(rank, input_shape)
  (units, kernel, strides, padding, dilation, activation, use_bias,
   kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer,
   activity_regularizer, kernel_constraint, bias_constraint, batchnorm,
   dropout), nlayers = _as_arg_tuples(units, kernel, strides, padding, dilation,
                                      activation, use_bias, kernel_initializer,
                                      bias_initializer, kernel_regularizer,
                                      bias_regularizer, activity_regularizer,
                                      kernel_constraint, bias_constraint,
                                      batchnorm, dropout)

  layers = []
  if input_shape is not None:
    layers.append(keras.layers.InputLayer(input_shape=input_shape))
  if 0. < input_dropout < 1.:
    layers.append(keras.layers.Dropout(rate=input_dropout))

  if rank == 3:
    layer_type = keras.layers.Conv3D
  elif rank == 2:
    layer_type = keras.layers.Conv2D
  elif rank == 1:
    layer_type = keras.layers.Conv1D

  for i in range(nlayers):
    layers.append(
        layer_type(\
          filters=units[i],
          kernel_size=kernel[i],
          strides=strides[i],
          padding=padding[i],
          dilation_rate=dilation[i],
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
    if dropout[i] > 0:
      layers.append(keras.layers.Dropout(rate=dropout[i]))
  # projection
  if isinstance(projection, bool):
    if projection:
      layers.append(keras.layers.Flatten())
  elif isinstance(projection, Number):
    layers.append(keras.layers.Flatten())
    layers.append(
        keras.layers.Dense(int(projection), activation='linear', use_bias=True))
  return layers


def deconv_network(units,
                   rank=2,
                   kernel=3,
                   strides=1,
                   padding='same',
                   output_padding=None,
                   dilation=1,
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
                   dropout=0.,
                   projection=None,
                   input_shape=None):
  r""" Multi-layers transposed convolutional neural network """
  rank, input_shape = _infer_rank_and_input_shape(rank, input_shape)
  (units, kernel, strides, padding, output_padding, dilation, activation,
   use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
   bias_regularizer, activity_regularizer, kernel_constraint,
   bias_constraint, batchnorm, dropout), nlayers = _as_arg_tuples(
       units, kernel, strides, padding, output_padding, dilation, activation,
       use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
       bias_regularizer, activity_regularizer, kernel_constraint,
       bias_constraint, batchnorm, dropout)
  #
  layers = []
  if input_shape is not None:
    layers.append(keras.layers.InputLayer(input_shape=input_shape))
  if 0. < input_dropout < 1.:
    layers.append(keras.layers.Dropout(input_dropout))
  #
  if rank == 3:
    raise NotImplementedError
  elif rank == 2:
    layer_type = keras.layers.Conv2DTranspose
  elif rank == 1:
    layer_type = Conv1DTranspose

  for i in range(nlayers):
    layers.append(
        layer_type(\
          filters=units[i],
          kernel_size=kernel[i],
          strides=strides[i],
          padding=padding[i],
          output_padding=output_padding[i],
          dilation_rate=dilation[i],
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
    if dropout[i] > 0:
      layers.append(keras.layers.Dropout(rate=dropout[i]))
  # projection
  if isinstance(projection, bool):
    if projection:
      layers.append(keras.layers.Flatten())
  elif isinstance(projection, Number):
    layers.append(keras.layers.Flatten())
    layers.append(
        keras.layers.Dense(int(projection), activation='linear', use_bias=True))
  return layers


# ===========================================================================
# Serializable configuration
# ===========================================================================
@dataclasses.dataclass(init=True,
                       repr=True,
                       eq=True,
                       order=False,
                       unsafe_hash=False,
                       frozen=True)
class NetworkConfig(dict):
  r""" A dataclass for storing the autoencoder networks (encoder and decoder)
  configuration. Number of layers is determined by length of `units`

  Arguments:
    units : a list of Integer. Number of hidden units for each hidden layers
    kernel : a list of Integer, kernel size for convolution network
    strides : a list of Integer, stride step for convoltion
    activation : a String, alias of activation function
    input_dropout : A Scalar [0., 1.], dropout rate, if 0., turn-off dropout.
      this rate is applied for input layer.
    dropout : a list of Scalar [0., 1.], dropout rate between two hidden layers
    batchnorm : a Boolean, batch normalization
    linear_decoder : a Boolean, if `True`, use an `Identity` (i.e. Linear)
      decoder
    network : {'conv', 'deconv', 'dense'}.
      type of `Layer` for the network
    flatten_inputs: a Boolean. Flatten the inputs to 2D in case of `Dense`
      network
    projection : An Integer, number of hidden units for the `Dense`
      linear projection layer right after convolutional network.

  """

  units: int = 64
  kernel: int = 3
  strides: int = 1
  padding: str = 'same'
  dilation: int = 1
  activation: str = 'relu'
  use_bias: bool = True
  kernel_initializer: str = 'glorot_uniform'
  bias_initializer: str = 'zeros'
  kernel_regularizer: str = None
  bias_regularizer: str = None
  activity_regularizer: str = None
  kernel_constraint: str = None
  bias_constraint: str = None
  batchnorm: bool = False
  input_dropout: float = 0.
  dropout: float = 0.
  linear_decoder: bool = False
  network: str = 'dense'
  flatten_inputs: bool = True
  projection: int = None
  input_shape: tuple = None

  def __post_init__(self):
    network_types = ('deconv', 'conv', 'dense', 'lstm', 'gru', 'rnn')
    assert self.network in network_types, \
      "Given network '%s', only support: %s" % (self.network, network_types)

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

  ################ Create the networks
  def create_autoencoder(self, input_shape, latent_shape, name=None):
    r""" Create both encoder and decoder at once """
    encoder_name = None if name is None else "%s_%s" % (name, "encoder")
    decoder_name = None if name is None else "%s_%s" % (name, "decoder")
    encoder = self.create_network(input_shape=input_shape, name=encoder_name)
    decoder = self.create_decoder(encoder=encoder,
                                  latent_shape=latent_shape,
                                  name=decoder_name)
    return encoder, decoder

  def create_decoder(self,
                     encoder,
                     latent_shape,
                     n_parameterization=1,
                     name=None) -> SequentialNetwork:
    r"""
    Arguments:
      latent_shape : a tuple of Integer. Shape of latent without the batch
         dimensions.
      n_parameterization : number of parameters in case the output of decoder
        parameterize a distribution
      name : a String (optional).

    Returns:
      decoder : keras.Sequential
    """
    if name is None:
      name = "Decoder"
    latent_shape = _shape(latent_shape)
    input_shape = encoder.input_shape[1:]
    n_channels = input_shape[-1]
    rank = 1 if len(input_shape) == 2 else 2
    # ====== linear decoder ====== #
    if self.linear_decoder:
      return Identity(name=name, input_shape=latent_shape)
    ### convolution network
    if self.network == 'conv':
      # get the last convolution shape
      eshape = encoder.layers[-3].output_shape[1:]
      start_layers = [keras.layers.InputLayer(input_shape=latent_shape)]
      if self.projection is not None and not isinstance(self.projection, bool):
        start_layers += [
            keras.layers.Dense(int(self.projection),
                               activation='linear',
                               use_bias=True),
            keras.layers.Dense(np.prod(eshape),
                               activation=self.activation,
                               use_bias=True),
            keras.layers.Reshape(eshape),
        ]
      decoder = deconv_network(
          tf.nest.flatten(self.units)[::-1][1:] +
          [n_channels * int(n_parameterization)],
          rank=rank,
          kernel=tf.nest.flatten(self.kernel)[::-1],
          strides=tf.nest.flatten(self.strides)[::-1],
          padding=tf.nest.flatten(self.padding)[::-1],
          dilation=tf.nest.flatten(self.dilation)[::-1],
          activation=tf.nest.flatten(self.activation)[::-1],
          use_bias=tf.nest.flatten(self.use_bias)[::-1],
          batchnorm=tf.nest.flatten(self.batchnorm)[::-1],
          input_dropout=self.input_dropout,
          dropout=tf.nest.flatten(self.dropout)[::-1],
          kernel_initializer=self.kernel_initializer,
          bias_initializer=self.bias_initializer,
          kernel_regularizer=self.kernel_regularizer,
          bias_regularizer=self.bias_regularizer,
          activity_regularizer=self.activity_regularizer,
          kernel_constraint=self.kernel_constraint,
          bias_constraint=self.bias_constraint,
      )
      decoder = start_layers + decoder
      decoder.append(keras.layers.Reshape(input_shape))
    ### dense network
    elif self.network == 'dense':
      decoder = dense_network(
          tf.nest.flatten(self.units)[::-1],
          activation=tf.nest.flatten(self.activation)[::-1],
          use_bias=tf.nest.flatten(self.use_bias)[::-1],
          batchnorm=tf.nest.flatten(self.batchnorm)[::-1],
          input_dropout=self.input_dropout,
          dropout=tf.nest.flatten(self.dropout)[::-1],
          kernel_initializer=self.kernel_initializer,
          bias_initializer=self.bias_initializer,
          kernel_regularizer=self.kernel_regularizer,
          bias_regularizer=self.bias_regularizer,
          activity_regularizer=self.activity_regularizer,
          kernel_constraint=self.kernel_constraint,
          bias_constraint=self.bias_constraint,
          flatten_inputs=self.flatten_inputs,
          input_shape=latent_shape,
      )
    ### deconv
    else:
      raise NotImplementedError("'%s' network doesn't support decoding." %
                                self.network)
    decoder = SequentialNetwork(decoder, name=name)
    decoder.copy = types.MethodType(
        lambda s, name=None: self.create_decoder(
            encoder=encoder,
            latent_shape=latent_shape,
            n_parameterization=n_parameterization,
            name=name,
        ),
        decoder,
    )
    return decoder

  def create_network(self, input_shape=None, name=None) -> SequentialNetwork:
    r"""
    Arguments:
      input_shape : a tuple of Integer. Shape of input without the batch
         dimensions.
      name : a String (optional).

    Returns:
      encoder : keras.Sequential
    """
    if name is None:
      name = "Encoder"
    ### prepare the shape
    input_shape = _shape(
        self.input_shape if input_shape is None else input_shape)
    ### convolution network
    if self.network == 'conv':
      # create the encoder
      network = conv_network(self.units,
                             kernel=self.kernel,
                             strides=self.strides,
                             padding=self.padding,
                             dilation=self.dilation,
                             activation=self.activation,
                             use_bias=self.use_bias,
                             batchnorm=self.batchnorm,
                             input_dropout=self.input_dropout,
                             dropout=self.dropout,
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer,
                             kernel_regularizer=self.kernel_regularizer,
                             bias_regularizer=self.bias_regularizer,
                             activity_regularizer=self.activity_regularizer,
                             kernel_constraint=self.kernel_constraint,
                             bias_constraint=self.bias_constraint,
                             projection=self.projection,
                             input_shape=input_shape)
    ### dense network
    elif self.network == 'dense':
      network = dense_network(self.units,
                              activation=self.activation,
                              use_bias=self.use_bias,
                              batchnorm=self.batchnorm,
                              input_dropout=self.input_dropout,
                              dropout=self.dropout,
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer,
                              kernel_regularizer=self.kernel_regularizer,
                              bias_regularizer=self.bias_regularizer,
                              activity_regularizer=self.activity_regularizer,
                              kernel_constraint=self.kernel_constraint,
                              bias_constraint=self.bias_constraint,
                              flatten_inputs=self.flatten_inputs,
                              input_shape=input_shape)
    ### deconv
    elif self.network == 'deconv':
      network = deconv_network(self.units,
                               kernel=self.kernel,
                               strides=self.strides,
                               padding=self.padding,
                               dilation=self.dilation,
                               activation=self.activation,
                               use_bias=self.use_bias,
                               batchnorm=self.batchnorm,
                               input_dropout=self.input_dropout,
                               dropout=self.dropout,
                               kernel_initializer=self.kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               kernel_regularizer=self.kernel_regularizer,
                               bias_regularizer=self.bias_regularizer,
                               activity_regularizer=self.activity_regularizer,
                               kernel_constraint=self.kernel_constraint,
                               bias_constraint=self.bias_constraint,
                               projection=self.projection,
                               input_shape=input_shape)
    ### others
    else:
      raise NotImplementedError("No implementation for network of type: '%s'" %
                                self.network)
    # ====== return ====== #
    network = SequentialNetwork(network, name=name)
    network.copy = types.MethodType(
        lambda s, name=None: self.create_network(input_shape=input_shape,
                                                 name=name),
        network,
    )
    return network
