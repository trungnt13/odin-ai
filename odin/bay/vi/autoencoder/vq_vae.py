import tensorflow as tf
from tensorflow.python.training import moving_averages

from odin.bay.vi.autoencoder import BetaVAE


class VQVAE(BetaVAE):
  r"""

  Reference:
    Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. "Neural Discrete
      Representation Learning". In _Conference on Neural Information Processing
      Systems_, 2017. https://arxiv.org/abs/1711.00937
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
