from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import distribution_layer as tfl

from odin.bay import distribution_layers as obl
from odin.bay import distributions as obd
from odin.utils.python_utils import multikeysdict

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
    ('zinbd', 'zinegativebinomialdisp', 'zeroinflatednegativebinomialdisp'):
        (obl.ZINegativeBinomialDispLayer, obd.NegativeBinomialDisp),
    ('pois', 'poisson'): (obl.PoissonLayer, tfd.Poisson),
    ('zipois', 'zipoisson', 'zeroinflatedpoisson'):
        (obl.ZIPoissonLayer, tfd.Poisson),
    'dirichlet': (obl.DirichletLayer, tfd.Dirichlet),
    'onehot': (obl.OneHotCategoricalLayer, tfd.OneHotCategorical),
    'deterministic': (obl.DeterministicLayer, tfd.Deterministic),
    'vdeterministic': (obl.VectorDeterministicLayer, tfd.VectorDeterministic),
    # 'beta': (obl),
})


def parse_distribution(alias):
  """
  Return
  ------
  layer : `tensorflow_probability.python.layers.DistributionLambda`
  dist : `tensorflow_probability.python.distributions.Distribution`
  """
  alias = str(alias).lower()
  if alias not in _dist_mapping:
    raise ValueError("Cannot find distribution with alias: '%s', "
                     "all available distributions: %s" %
                     (alias, ', '.join(list(_dist_mapping.keys()))))
  layer, dist = _dist_mapping[alias]
  return layer, dist
