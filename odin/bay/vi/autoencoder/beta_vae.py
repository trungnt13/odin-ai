import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class BetaVAE(VariationalAutoencoder):

  def __init__(self,
               encoder=None,
               decoder=None,
               config=None,
               beta=1.0,
               name='BetaVAE'):
    super().__init__(encoder=encoder, decoder=decoder, config=config, name=name)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')

  def sample(self, sample_shape=(), seed=1):
    r""" Sampling from prior distribution """
    raise NotImplementedError

  def encode(self, inputs, training=None, n_mcmc=None, **kwargs):
    r""" Encoding inputs to latent codes """
    raise NotImplementedError

  def decode(self, latents, training=None):
    r""" Decoding latent codes to reconstructed inputs """
    raise NotImplementedError
