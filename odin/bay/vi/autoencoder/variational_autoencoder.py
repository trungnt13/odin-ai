from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

from odin.bay.vi.utils import permute_dims
from odin.networks import DenseNetwork


class VariationalAutoencoder(keras.Model):

  def __init__(self, encoder, decoder):
    super().__init__()

  def sample(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    raise NotImplementedError

  def encode(self, inputs, training=None, n_mcmc=None, **kwargs):
    r""" Encoding inputs to latent codes """
    raise NotImplementedError

  def decode(self, latents, training=None):
    r""" Decoding latent codes to reconstructed inputs """
    raise NotImplementedError
