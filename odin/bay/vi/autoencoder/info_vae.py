import inspect
from functools import partial
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from odin.bay.random_variable import RVmeta
from odin.bay.vi.autoencoder.beta_vae import betaVAE
from odin.bay.vi.autoencoder.variational_autoencoder import TensorTypes
from odin.bay.vi.losses import maximum_mean_discrepancy
from odin.bay.vi.utils import permute_dims
from odin.utils import as_tuple
from tensorflow import Tensor
from tensorflow_probability.python.distributions import (Distribution,
                                                         OneHotCategorical)
from typing_extensions import Literal

class infoVAE(betaVAE):
  r""" For MNIST, the authors used scaling coefficient `lambda(gamma)=1000`,
  and information preference `alpha=0`.

  Increase `np` (number of prior samples) in `divergence_kw` to reduce the
  variance of MMD estimation.

  Arguments:
    alpha : a Scalar. Equal to `1 - beta`
      Higher value of alpha places lower weight on the KL-divergence
    gamma : a Scalar. This is the value of lambda in the paper
      Higher value of gamma place more weight on the Info-divergence (i.e. MMD)
    divergence : a Callable.
      Divergences families, for now only support 'mmd'
      i.e. maximum-mean discrepancy.

  Reference:
    Zhao, S., Song, J., Ermon, S., et al. "infoVAE: Balancing Learning and
      Inference in Variational Autoencoders".
    Shengjia Zhao. "A Tutorial on Information Maximizing Variational
      Autoencoders (infoVAE)".
      https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders
  """

  def __init__(self,
               alpha: float = 0.0,
               gamma: float = 100.0,
               divergence: Callable[[Distribution, Distribution],
                                    Tensor] = partial(maximum_mean_discrepancy,
                                                      kernel='gaussian',
                                                      q_sample_shape=None,
                                                      p_sample_shape=100),
               **kwargs):
    super().__init__(beta=1 - alpha, **kwargs)
    self.gamma = tf.convert_to_tensor(gamma, dtype=self.dtype, name='gamma')
    # select right divergence
    assert callable(divergence), \
      f"divergence must be callable, but given: {type(divergence)}"
    self.divergence = divergence

  @property
  def alpha(self):
    return 1 - self.beta

  def elbo_components(self, inputs, training=None, mask=None):
    llk, kl = super().elbo_components(inputs, mask=mask, training=training)
    px_z, qz_x = self.last_outputs
    # repeat for each latent
    for z, qz in zip(self.latents, as_tuple(qz_x)):
      # div(qZ||pZ)
      info_div = (self.gamma - self.beta) * self.divergence(
          qz, qz.KL_divergence.prior)
      kl[f'div_{z.name}'] = info_div
    return llk, kl


class InfoNCEVAE(betaVAE):
  r""" Mutual information bound based on Noise-Contrastive Estimation

  Reference:
    Tschannen, M., Djolonga, J., Rubenstein, P.K., Gelly, S., Lucic, M., 2019.
      "On Mutual Information Maximization for Representation Learning".
      arXiv:1907.13625 [cs, stat].
    https://github.com/google-research/google-research/tree/master/mutual_information_representation_learning
  """


class IFVAE(betaVAE):
  r""" Adversarial information factorized VAE

  Reference:
    Creswell, A., Mohamied, Y., Sengupta, B., Bharath, A.A., 2018.
      "Adversarial Information Factorization". arXiv:1711.05175 [cs].
  """
  pass


class InfoMaxVAE(betaVAE):
  r"""
  Reference:
    Rezaabad, A.L., Vishwanath, S., 2020. "Learning Representations by
      Maximizing Mutual Information in Variational Autoencoders".
      arXiv:1912.13361 [cs, stat].
    Hjelm, R.D., Fedorov, A., et al. 2019. "Learning Deep Representations by
      Mutual Information Estimation and Maximization". ICLR'19.
  """
  pass
