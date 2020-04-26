from odin.bay.random_variable import RandomVariable as RV
from odin.bay.vi.autoencoder.beta_vae import BetaVAE

class MultitaskVAE(BetaVAE):

  def __init__(self,
               labels=RV(10, 'onehot', projection=True, name="Label"),
               alpha=10,
               **kwargs):
    super().__init__(**kwargs)

  @property
  def is_semi_supervised(self):
    return True
