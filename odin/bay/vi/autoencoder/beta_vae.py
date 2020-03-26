import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class BetaVAE(VariationalAutoencoder):

  def __init__(self,
               encoder=None,
               decoder=None,
               config=None,
               beta=1.0,
               name='BetaVAE',
               **kwargs):
    super().__init__(encoder=encoder,
                     decoder=decoder,
                     config=config,
                     name=name,
                     **kwargs)
    self.beta = tf.convert_to_tensor(beta, dtype=self.dtype, name='beta')
