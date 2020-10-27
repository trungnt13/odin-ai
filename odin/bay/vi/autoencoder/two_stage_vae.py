from odin.bay.vi.autoencoder.variational_autoencoder import (
    RVmeta, VariationalAutoencoder)

from tensorflow_probability.python.experimental import nn
nn.Sequential

class delayedVAE(VariationalAutoencoder):

  def __init__(self, llk_burn_in='auto', name='DelayedVAE', **kwargs):
    super().__init__(name=name, **kwargs)


class twostageVAE(VariationalAutoencoder):
  pass
