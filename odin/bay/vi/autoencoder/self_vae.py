import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class AdaptiveVAE(VariationalAutoencoder):
  r"""

  Reference:
    Locatello, F., et al. 2020. "Weakly-Supervised Disentanglement Without
      Compromises". arXiv:2002.02886 [cs, stat].
  """
  pass


class GroupVAE(VariationalAutoencoder):
  r"""
  Reference:
    Hosoya, H., 2019. "Group-based Learning of Disentangled Representations
      with Generalizability for Novel Contents", in: Proceedings of the
      Twenty-Eighth International Joint Conference on Artificial Intelligence.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)


class MultiLevelVAE(VariationalAutoencoder):
  r"""
  Reference:
    Bouchacourt, D., Tomioka, R., Nowozin, S., 2017. "Multi-Level Variational
      Autoencoder: Learning Disentangled Representations from Grouped
      Observations". arXiv:1705.08841 [cs, stat].
    Code: https://github.com/ananyahjha93/multi-level-vae/blob/master/training.py
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
