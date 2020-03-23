from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

from odin.bay.vi.utils import permute_dims
from odin.networks import AutoencoderConfig


class VariationalAutoencoder(keras.Model):

  def __init__(self,
               encoder: layers.Layer = None,
               decoder: layers.Layer = None,
               config: AutoencoderConfig = None,
               input_shape=None,
               **kwargs):
    if input_shape is not None:
      kwargs['input_shape'] = input_shape
    super().__init__(**kwargs)
    ### already got the encoder and decoder
    if config is None:
      if encoder is None and decoder is None:
        raise ValueError(
            "Either provide both encoder, decoder or only the config.")
    ### create the network from config
    else:
      assert isinstance(config, AutoencoderConfig), \
        "config must be instance of AutoencoderConfig but given: %s" % \
          str(type(config))
      assert input_shape is not None, \
        "Input shape must be provided in case AutoencoderConfig is specified."
      encoder, decoder = config.create_network(input_shape=input_shape)
    exit()

  def sample(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    raise NotImplementedError

  def encode(self, inputs, training=None, n_mcmc=None, **kwargs):
    r""" Encoding inputs to latent codes """
    raise NotImplementedError

  def decode(self, latents, training=None):
    r""" Decoding latent codes to reconstructed inputs """
    raise NotImplementedError
