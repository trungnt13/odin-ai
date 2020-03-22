import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.networks import DenseNetwork



class BetaVAE(VariationalAutoencoder):

  def __init__(self, encoder, decoder, beta=1.0):
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
