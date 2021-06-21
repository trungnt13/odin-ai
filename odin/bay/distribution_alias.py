from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
from typing import Type, Tuple
from odin.bay import distributions as obd
from odin.bay import layers as obl
from odin.utils.python_utils import multikeysdict, partialclass
from six import string_types
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import distribution_layer as tfl
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.layers import DistributionLambda

# TODO: better specifying complex distribution
# mapping from alias to
_dist_mapping = multikeysdict({
    ('normal', 'gaussian'): (obl.NormalLayer, tfd.Normal),
    'mvndiag': (partialclass(obl.MultivariateNormalLayer,
                             covariance='diag'), tfd.MultivariateNormalDiag),
    'mvntril': (partialclass(obl.MultivariateNormalLayer,
                             covariance='tril'), tfd.MultivariateNormalTriL),
    'mvnfull': (partialclass(obl.MultivariateNormalLayer, covariance='full'),
                tfd.MultivariateNormalFullCovariance),
    'lognormal': (obl.LogNormalLayer, tfd.LogNormal),
    # ====== Mixture of Gaussian ====== #
    ('mdn', 'gmm'): (obl.MixtureGaussianLayer, tfd.MixtureSameFamily),
    ('mdndiag', 'gmmdiag'): (partialclass(obl.MixtureGaussianLayer,
                                          covariance='diag'),
                             tfd.MixtureSameFamily),
    ('mdntril', 'gmmtril'): (partialclass(obl.MixtureGaussianLayer,
                                          covariance='tril'),
                             tfd.MixtureSameFamily),
    # ====== Mixture of Logistic ====== #
    ('mixqlogist',): (obl.MixtureQLogisticLayer, tfd.Logistic),
    # ====== NegativeBinomial ====== #
    ('nb', 'negativebinomial'):
        (obl.NegativeBinomialLayer, tfd.NegativeBinomial),
    'nbshare': (partialclass(obl.NegativeBinomialLayer,
                             dispersion='share'), obd.NegativeBinomial),
    'nbsingle': (partialclass(obl.NegativeBinomialLayer,
                              dispersion='single'), obd.NegativeBinomial),
    'nbfull': (partialclass(obl.NegativeBinomialLayer,
                            dispersion='full'), obd.NegativeBinomial),
    'zinb': (obl.ZINegativeBinomialLayer, tfd.NegativeBinomial),
    'zinbshare': (partialclass(obl.ZINegativeBinomialLayer,
                               dispersion='share'), obd.NegativeBinomial),
    'zinbsingle': (partialclass(obl.ZINegativeBinomialLayer,
                                dispersion='single'), obd.NegativeBinomial),
    'zinbfull': (partialclass(obl.ZINegativeBinomialLayer,
                              dispersion='full'), obd.NegativeBinomial),
    # ====== NegativeBinomialDisp ====== #
    ('nbd', 'negativebinomialdisp'):
        (obl.NegativeBinomialDispLayer, obd.NegativeBinomialDisp),
    'nbdshare': (partialclass(obl.NegativeBinomialDispLayer,
                              dispersion='share'), obd.NegativeBinomialDisp),
    'nbdsingle': (partialclass(obl.NegativeBinomialDispLayer,
                               dispersion='single'), obd.NegativeBinomialDisp),
    'nbdfull': (partialclass(obl.NegativeBinomialDispLayer,
                             dispersion='full'), obd.NegativeBinomialDisp),
    'zinbd': (obl.ZINegativeBinomialDispLayer, obd.NegativeBinomialDisp),
    'zinbdshare': (partialclass(obl.ZINegativeBinomialDispLayer,
                                dispersion='share'), obd.NegativeBinomialDisp),
    'zinbdsingle': (partialclass(obl.ZINegativeBinomialDispLayer,
                                 dispersion='single'), obd.NegativeBinomialDisp
                   ),
    'zinbdfull': (partialclass(obl.ZINegativeBinomialDispLayer,
                               dispersion='full'), obd.NegativeBinomialDisp),
    # ====== mixture of NegativeBinomial ====== #
    ('mixnb', 'mnb', 'mixmass'):
        (obl.MixtureNegativeBinomialLayer, obd.NegativeBinomial),
    'mixnbd': (partialclass(obl.MixtureNegativeBinomialLayer,
                            alternative=True,
                            mean_activation='softplus',
                            disp_activation='softplus',
                            zero_inflated=False), obd.NegativeBinomial),
    'mixzinb': (partialclass(obl.MixtureNegativeBinomialLayer,
                             alternative=False,
                             mean_activation='softplus',
                             disp_activation='linear',
                             zero_inflated=True), obd.NegativeBinomial),
    'mixzinbd': (partialclass(obl.MixtureNegativeBinomialLayer,
                              alternative=True,
                              mean_activation='softplus',
                              disp_activation='softplus',
                              zero_inflated=True), obd.NegativeBinomial),
    # ====== Poisson ====== #
    ('pois', 'poisson'): (obl.PoissonLayer, tfd.Poisson),
    ('zip', 'zipois', 'zipoisson', 'zeroinflatedpoisson'):
        (obl.ZIPoissonLayer, tfd.Poisson),
    # ====== Dirichlet and beta ====== #
    'dirichlet': (obl.DirichletLayer, tfd.Dirichlet),
    'deterministic': (obl.DeterministicLayer, tfd.Deterministic),
    'vdeterministic': (obl.VectorDeterministicLayer, tfd.VectorDeterministic),
    'beta': (obl.BetaLayer, tfd.Beta),
    # ====== Discrete distribution ====== #
    'onehot': (obl.OneHotCategoricalLayer, tfd.OneHotCategorical),
    ('cat', 'categorical', 'discrete'): (obl.CategoricalLayer, tfd.Categorical),
    # binomial and multinomial (aka cat)
    'binomial': (obl.BinomialLayer, tfd.Binomial),
    # 'betabinomial': (obl.BetaBinomialLayer, tfd.Binomial),
    'multinomial': (obl.MultinomialLayer, tfd.Multinomial),
    ('dirimultinomial', 'dirichletmultinomial'):
        (obl.DirichletMultinomialLayer, tfd.DirichletMultinomial),
    # ====== Gumbel ====== #
    'bernoulli': (obl.BernoulliLayer, tfd.Bernoulli),
    'cbernoulli': (obl.ContinuousBernoulliLayer, tfd.ContinuousBernoulli),
    ('zibernoulli', 'zeroinflatedbernoulli'):
        (obl.ZIBernoulliLayer, tfd.Bernoulli),
    ('relaxedbern', 'relaxedsigmoid', 'relaxedbernoulli'):
        (obl.RelaxedBernoulliLayer, tfd.Bernoulli),
    ('relaxedsoftmax', 'relaxedonehot'):
        (obl.RelaxedOneHotCategoricalLayer, tfd.OneHotCategorical),
})


def parse_distribution(alias: str
                       ) -> Tuple[Type[DistributionLambda],
                                  Type[Distribution]]:
  r""" Parse a string alias to appropriate class of `DistributionLambda`
  and `Distribution`.

  Returns:
    layer : `tensorflow_probability.python.layers.DistributionLambda`
    dist : `tensorflow_probability.python.distributions.Distribution`
  """
  if isinstance(alias, string_types):
    alias = alias.lower()
    if alias not in _dist_mapping:
      raise ValueError("Cannot find distribution with alias: '%s', "
                       "all available distributions: %s" %
                       (alias, ', '.join(list(_dist_mapping.keys()))))
    layer, dist = _dist_mapping[alias]
    return layer, dist
  if not inspect.isclass(alias):
    alias = type(alias)
  if issubclass(alias, tfd.Distribution):
    for i, j in _dist_mapping.values():
      if issubclass(j, alias):
        return i, alias
  elif issubclass(alias, tfl.DistributionLambda):
    for i, j in _dist_mapping.values():
      if issubclass(i, alias):
        return alias, j
  raise ValueError("Cannot find distribution with alias: %s" % str(alias))
