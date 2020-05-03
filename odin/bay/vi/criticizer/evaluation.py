from __future__ import absolute_import, division, print_function

import inspect
from numbers import Number

import tensorflow as tf

from odin.bay import distributions as tfd
from odin.bay.vi.criticizer._base_metrics import CriticizerMetrics
from odin.bay.vi.criticizer._base_plot import CriticizerPlot

__all__ = ['Criticizer']


class Criticizer(CriticizerMetrics, CriticizerPlot):
  r""" Probabilistic criticizer for variational model

  Basic progress of evaluating probabilistic model given groundtruth factors:
    - `sample_batch`
    - `conditioning` on known factors

  Arguments:
    vae : `odin.bay.vi.VariationalAutoencoder`.
    random_state : a Scalar. Random seed to ensure reproducibility.

  Attributes:
    pass
  """

  def copy(self, random_state=None):
    r""" Shallow copy of Criticizer and all its sampled data """
    crt = Criticizer(
        self._vae,
        random_state=self.randint if random_state is None else random_state)
    for name in dir(self):
      if '_' == name[0] and '__' != name[:2] and name != '_rand':
        attr = getattr(self, name)
        if not inspect.ismethod(attr):
          setattr(crt, name, getattr(self, name))
    return crt

  def _check_factors(self, factors):
    if factors is None:
      factors = list(range(self.n_factors))
    else:
      try:
        factors = [
            int(i) if isinstance(i, Number) else self.index(i)
            for i in tf.nest.flatten(factors)
        ]
      except ValueError:
        raise ValueError("Cannot find factors: %s, from list of factors: %s" %
                         (str(factors), self.factors_name))
    return factors

  def summary(self,
              n_samples=10000,
              n_neighbors=3,
              n_components=2,
              save_path=None,
              verbose=True):
    r""" Create a report of all quantitative metrics

    Arguments:
      save_path : a String (optional). Path to an YAML file for saving the
        scores
    """
    scores = {}
    for i, s in enumerate(
        self.cal_dcd_scores(n_samples=n_samples, n_components=n_components)):
      scores['dcd_%d' % i] = s
    for i, s in enumerate(self.cal_dcmi_scores(n_neighbors=n_neighbors)):
      scores['dcmi_%d' % i] = s
    for i, s in enumerate(self.cal_dcc_scores()):
      scores['dcc_%d' % i] = s
    for i, s in enumerate(self.cal_dci_scores()):
      scores['dci_%d' % i] = s
    #
    scores.update(
        {i.lower(): j for i, j in self.cal_clustering_scores().items()})
    #
    betavae = self.cal_betavae_score(n_samples=n_samples, verbose=verbose)
    scores['betavae'] = betavae
    #
    factorvae = self.cal_factorvae_score(n_samples=n_samples, verbose=verbose)
    scores['factorvae'] = factorvae
    #
    scores['rds_spearman'] = self.cal_relative_disentanglement_strength(
        method='spearman')
    scores['rds_pearson'] = self.cal_relative_disentanglement_strength(
        method='pearson')
    scores['rds_lasso'] = self.cal_relative_disentanglement_strength(
        method='lasso')
    #
    tc = self.cal_total_correlation()
    scores['tc_train'] = tc[0]
    scores['tc_test'] = tc[1]
    #
    scores['sap'] = self.cal_separated_attr_predictability()
    #
    mig = self.cal_mutual_info_gap()
    scores['mig_train'] = mig[0]
    scores['mig_test'] = mig[1]
    #
    if save_path is not None:
      with open(save_path, 'w') as f:
        for k, v in sorted(scores.items(), key=lambda x: x[0]):
          f.write("%s: %g\n" % (k, v))
    return scores

  def __str__(self):
    text = [str(self._vae)]
    text.append(" Factor name: %s" % ', '.join(self.factors_name))
    for name, data in [("Inputs", self.inputs), ("Factors", self.factors),
                       ("Original Factors", self.original_factors),
                       ("Representations", self.representations),
                       ("Reconstructions", self.reconstructions)]:
      text.append(" " + name)
      for d, x in zip(('train', 'test'), data):
        x = d + ' : ' + ', '.join([
            re.sub(r", dtype=[a-z]+\d*\)",
                   ")", str(i).replace("tfp.distributions.", "")) \
              if isinstance(i, tfd.Distribution) else str(i.shape)
            for i in tf.nest.flatten(x)
        ])
        text.append("  " + x)
    return "\n".join(text)

  def __repr__(self):
    return self.__str__()
