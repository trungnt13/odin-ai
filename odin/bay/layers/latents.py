from __future__ import absolute_import, division, print_function

from typing import Type

import tensorflow as tf
from tensorflow_probability.python.distributions import (Independent, LogNormal,
                                                         MultivariateNormalDiag,
                                                         Normal)

from odin.bay.layers.continuous import (LogNormalLayer, MultivariateNormalLayer,
                                        NormalLayer)
from odin.bay.layers.dense import DenseDistribution, MixtureDensityNetwork

__all__ = [
    'DiagonalGaussianLatent',
    'IndependentGaussianLatent',
    'MixtureDiagonalGaussianLatent',
    'MixtureIndependentGaussianLatent',
]


class DiagonalGaussianLatent(DenseDistribution):
  r"""  """

  def __init__(self, units, **kwargs):
    super().__init__(units,
                     posterior=MultivariateNormalLayer,
                     posterior_kwargs=dict(covariance='diag',
                                           scale_activation='softplus1'),
                     prior=MultivariateNormalDiag(loc=tf.zeros(shape=units),
                                                  scale_identity_multiplier=1.),
                     **kwargs)


class IndependentGaussianLatent(DenseDistribution):

  def __init__(self, units, **kwargs):
    super().__init__(units,
                     posterior=NormalLayer,
                     posterior_kwargs=dict(scale_activation='softplus1'),
                     prior=Independent(
                         Normal(loc=tf.zeros(shape=units),
                                scale=tf.ones(shape=units)), 1),
                     **kwargs)


class MixtureIndependentGaussianLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, **kwargs):
    kwargs['covariance'] = 'none'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, **kwargs)
    self.set_prior()


class MixtureDiagonalGaussianLatent(MixtureDensityNetwork):

  def __init__(self, units, n_components=8, **kwargs):
    kwargs['covariance'] = 'diag'
    kwargs['n_components'] = int(n_components)
    super().__init__(units, **kwargs)
    self.set_prior()
