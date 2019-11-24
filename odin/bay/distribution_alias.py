from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
from six import string_types
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import distribution_layer as tfl

from odin.bay import distribution_layers as obl
from odin.bay import distributions as obd
from odin.utils.python_utils import multikeysdict, partialclass

# mapping from alias to
_dist_mapping = multikeysdict({
    ('bern', 'bernoulli'): (obl.BernoulliLayer, tfd.Bernoulli),
    ('zibernoulli', 'zeroinflatedbernoulli'):
        (obl.ZIBernoulliLayer, tfd.Bernoulli),
    ('normal', 'gaussian'): (obl.NormalLayer, tfd.Normal),
    'lognormal': (obl.LogNormalLayer, tfd.LogNormal),
    ('nb', 'negativebinomial'):
        (obl.NegativeBinomialLayer, tfd.NegativeBinomial),
    ('zinb', 'zinegativebinomial', 'zeroinflatednegativebinomial'):
        (obl.ZINegativeBinomialLayer, tfd.NegativeBinomial),
    ('nbd', 'negativebinomialdisp'):
        (obl.NegativeBinomialDispLayer, obd.NegativeBinomialDisp),
    'nbshare': (partialclass(obl.NegativeBinomialDispLayer,
                             dispersion='share'), obd.NegativeBinomialDisp),
    'nbsingle': (partialclass(obl.NegativeBinomialDispLayer,
                              dispersion='single'), obd.NegativeBinomialDisp),
    'nbfull': (partialclass(obl.NegativeBinomialDispLayer,
                            dispersion='full'), obd.NegativeBinomialDisp),
    ('zinbd', 'zinegativebinomialdisp', 'zeroinflatednegativebinomialdisp'):
        (obl.ZINegativeBinomialDispLayer, obd.NegativeBinomialDisp),
    'zinbshare': (partialclass(obl.ZINegativeBinomialDispLayer,
                               dispersion='share'), obd.NegativeBinomialDisp),
    'zinbsingle': (partialclass(obl.ZINegativeBinomialDispLayer,
                                dispersion='single'), obd.NegativeBinomialDisp),
    'zinbfull': (partialclass(obl.ZINegativeBinomialDispLayer,
                              dispersion='full'), obd.NegativeBinomialDisp),
    ('pois', 'poisson'): (obl.PoissonLayer, tfd.Poisson),
    ('zipois', 'zipoisson', 'zeroinflatedpoisson'):
        (obl.ZIPoissonLayer, tfd.Poisson),
    'dirichlet': (obl.DirichletLayer, tfd.Dirichlet),
    'onehot': (obl.OneHotCategoricalLayer, tfd.OneHotCategorical),
    ('cat', 'categorical', 'discrete'): (obl.CategoricalLayer, tfd.Categorical),
    'deterministic': (obl.DeterministicLayer, tfd.Deterministic),
    'vdeterministic': (obl.VectorDeterministicLayer, tfd.VectorDeterministic),
    'beta': (obl.BetaLayer, tfd.Beta),
})


def parse_distribution(alias):
  """
  Return
  ------
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
  if not inspect.isclass(alias):
    alias = type(alias)
  if issubclass(alias, tfd.Distribution):
    for i, j in _dist_mapping.values():
      if j is alias:
        return i, j
  elif issubclass(alias, tfl.DistributionLambda):
    for i, j in _dist_mapping.values():
      if i is alias:
        return i, j
  return layer, dist
