import tensorflow as tf

from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder


class SequentialVAE(VariationalAutoencoder):
  r"""

  References:
    Yingzhen Li and Stephan Mandt. "Disentangled Sequential Autoencoder".
      In _International Conference on Machine Learning_, 2018.
      https://arxiv.org/abs/1803.02991
    Fraccaro, M., Sønderby, S.K., Paquet, U., Winther, O., 2016.
      "Sequential Neural Models with Stochastic Layers".
      arXiv:1605.07571 [cs, stat]. (https://github.com/google/vae-seq)
    Zhao, S., Song, J., Ermon, S., 2017. "Towards Deeper Understanding
      of Variational Autoencoding Models". arXiv:1702.08658 [cs, stat].
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)


class SequentialAttentionVAE(VariationalAutoencoder):
  r"""
  Reference:
    Deng, Y., Kim, Y., Chiu, J., Guo, D., Rush, A.M., 2018.
      "Latent Alignment and Variational Attention".
      arXiv:1807.03756 [cs, stat].
    Bahuleyan, H., Mou, L., Vechtomova, O., Poupart, P., 2017.
      "Variational Attention for Sequence-to-Sequence Models".
      arXiv:1712.08207 [cs].
    https://github.com/HareeshBahuleyan/tf-var-attention
    https://github.com/harvardnlp/var-attn/
  """


class VariationalRNN(VariationalAutoencoder):
  r"""

  Reference:
    Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A.C., Bengio, Y.,
      2015. "A Recurrent Latent Variable Model for Sequential Data",
      Advances in Neural Information Processing Systems 28.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
