import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class AdaptiveVAE(VariationalAutoencoder):
  r"""

  Arguments:
    base_method : {'g', 'ml'}. Base method for adapting the self-supervised
      objective:
      - 'group' for group VAE
      - 'multilevel' for multi-level VAE

  Reference:
    Locatello, F., et al. 2020. "Weakly-Supervised Disentanglement Without
      Compromises". arXiv:2002.02886 [cs, stat].
  """

  def __init__(self, base_method="group"):
    super().__init__()


class WeaklySupervisedVAE(VariationalAutoencoder):
  r"""

  Arguments:
    strategy : {'restricted', 'match', 'rank'}. Strategy for weak supervised
      objective
      - 'restricted' labelling
      - 'match' paring
      - 'rank' pairing

  Reference:
    Shu, R., Chen, Y., Kumar, A., Ermon, S., Poole, B., 2019.
      "Weakly Supervised Disentanglement with Guarantees".
      arXiv:1910.09772 [cs, stat].
    https://github.com/google-research/google-research/tree/master/weak_disentangle
  """

  def __init__(self, strategy="rank"):
    super().__init__()


class GroupVAE(VariationalAutoencoder):
  r"""
  Reference:
    Hosoya, H., 2019. "Group-based Learning of Disentangled Representations
      with Generalizability for Novel Contents", in: Proceedings of the
      Twenty-Eighth International Joint Conference on Artificial Intelligence.
    https://github.com/HaruoHosoya/gvae
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
