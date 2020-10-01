from collections import OrderedDict

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso

from odin import search
from odin.bay import distributions as tfd
from odin.bay.distributions import CombinedDistribution
from odin.bay.vi import losses, metrics
from odin.bay.vi.criticizer._criticizer_base import CriticizerBase


class CriticizerMetrics(CriticizerBase):

  ############## Helpers
  def _latent_codes(self, mean=True):
    if mean:
      return [i.mean().numpy() for i in self.representations]
    return [
        i.sample(sample_shape=(), seed=self.randint).numpy()
        for i in self.representations
    ]

  ############## Metrics
  def cal_dcmi_scores(self, mean=True, n_neighbors=3):
    r""" The same method is used for D.C.I scores, however, this metrics use
    mutual information matrix (estimated by nearest neighbor method)
    instead of importance matrix

    Return:
      tuple of 2 scalars:
        - disentanglement score of mutual information
        - completeness score of mutual information
    """
    train, test = self.create_mutualinfo_matrix(mean=mean,
                                                n_neighbors=n_neighbors)
    d = (metrics.disentanglement_score(train) +
         metrics.disentanglement_score(test)) / 2.
    c = (metrics.completeness_score(train) +
         metrics.completeness_score(test)) / 2.
    return d, c

  def cal_mutual_info_gap(self, mean=True):
    r"""
    Arguments:
      mean : a Boolean, if True use the mean of latent distribution for
        calculating the mutual information gap

    Return:
      a dictionary : {'mig': score}

    Reference:
      Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. Isolating Sources of
        Disentanglement in Variational Autoencoders. arXiv:1802.04942 [cs, stat].
    """
    mig = []
    for z, f in zip(self._latent_codes(mean), self.factors):
      mig.append(metrics.mutual_info_gap(z, f))
    return dict(mig=np.mean(mig))

  def cal_dci_scores(self, mean=True):
    r""" Disentanglement, Completeness, Informativeness

    Return:
      a dictionary:
        - 'disentanglement': The degree to which a representation factorises
          or disentangles the underlying factors of variation
        - 'completeness': The degree to which each underlying factor is
          captured by a single code variable.
        - 'informativeness': test accuracy of a factor recognizer trained
          on train data

    References:
      Based on "A Framework for the Quantitative Evaluation of Disentangled
      Representations" (https://openreview.net/forum?id=By-7dz-AZ).
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    d, c, i = metrics.dci_scores(z_train,
                                 f_train,
                                 z_test,
                                 f_test,
                                 random_state=self.randint)
    return dict(disentanglement=d, completeness=c, informativeness=i)

  def cal_total_correlation(self):
    r""" Estimation of total correlation based on fitted Gaussian

    Return:
      tuple of 2 scalars: total correlation estimation for train and test set
    """
    # memory complexity O(n*n*d), better do it on CPU
    with tf.device('/CPU:0'):
      return dict(tc=np.mean([
          losses.total_correlation(qz.sample(seed=self.randint), qz).numpy()
          for qz in self.representations
      ]))

  def cal_dcd_scores(self, n_samples=1000, lognorm=True, n_components=2):
    r""" Same as D.C.I but use density matrix instead of importance matrix
    """
    # smaller is better
    train, test = self.create_divergence_matrix(n_samples=n_samples,
                                                lognorm=lognorm,
                                                n_components=n_components,
                                                normalize_per_code=True,
                                                decode=False)
    # diag = np.diagflat(np.diag(density_mat))
    # higher is better
    train = 1. - train
    test = 1 - test
    d = (metrics.disentanglement_score(train) +
         metrics.disentanglement_score(test)) / 2.
    c = (metrics.completeness_score(train) +
         metrics.completeness_score(test)) / 2.
    return d, c

  def cal_dcc_scores(self, method='spearman', mean=True):
    r""" Same as D.C.I but use correlation matrix instead of importance matrix
    """
    train, test = self.create_correlation_matrix(mean=mean,
                                                 method=method,
                                                 decode=False)
    train = np.abs(train)
    test = np.abs(test)
    d = (metrics.disentanglement_score(train) +
         metrics.disentanglement_score(test)) / 2.
    c = (metrics.completeness_score(train) +
         metrics.completeness_score(test)) / 2.
    return d, c

  def cal_relative_disentanglement_strength(self, method='spearman', mean=True):
    r""" Relative strength for both axes of correlation matrix.
    Basically, is the mean of normalized maximum correlation per code, and
    per factor.

    Arguments:
      method : {'spearman', 'pearson', 'lasso', 'avg'}
          spearman - rank or monotonic correlation
          pearson - linear correlation
          lasso - lasso regression

    Return:
      a scalar - higher is better
    """
    corr_matrix = self.create_correlation_matrix(mean=mean, method=method)
    return dict(rds=metrics.relative_strength(corr_matrix))

  def cal_relative_mutual_strength(self, n_neighbors=3, mean=True):
    r""" Relative strength for both axes of mutual information matrix.
    Basically, is the mean of normalized maximum mutual information per code,
    and per factor.

    Return:
      a scalar - higher is better
    """
    matrix = self.create_mutualinfo_matrix(n_neighbors=n_neighbors)
    return dict(rms=metrics.relative_strength(matrix))

  ############## Downstream scores
  def cal_separated_attr_predictability(self, mean=True):
    r"""
    Return:
      a Scalar : single score representing SAP value for test set.

    Reference:
      Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. Variational Inference of
        Disentangled Latent Concepts from Unlabeled Observations.
        arXiv:1711.00848 [cs, stat].
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    sap = metrics.separated_attr_predictability(z_train,
                                                f_train,
                                                z_test,
                                                f_test,
                                                continuous_factors=False,
                                                random_state=self.randint)
    return dict(sap=sap)

  def cal_betavae_score(self, n_samples=10000, verbose=True):
    r""" The Beta-VAE score train a logistic regression to detect the invariant
    factor based on the absolute difference in the representations.

    Returns:
      tuple of 2 scalars: accuracy for train and test data
    """
    return dict(betavae=metrics.beta_vae_score(self.representations_full,
                                               self.factors_full,
                                               n_samples=n_samples,
                                               use_mean=True,
                                               random_state=self.randint,
                                               verbose=verbose))

  def cal_factorvae_score(self, n_samples=10000, verbose=True):
    r""" factorVAE based score

    Returns:
      tuple of 2 scalars: accuracy for train and test data
    """
    return dict(factorvae=metrics.factor_vae_score(self.representations_full,
                                                   self.factors_full,
                                                   n_samples=n_samples,
                                                   use_mean=True,
                                                   random_state=self.randint,
                                                   verbose=verbose))

  def cal_clustering_scores(self, algorithm='both'):
    r""" Calculating the unsupervised clustering Scores:

    - ASW: silhouette_score (higher is better, best is 1, worst is -1)
    - ARI: adjusted_rand_score (range [0, 1] - higher is better)
    - NMI: normalized_mutual_info_score (range [0, 1] - higher is better)
    - UCA: unsupervised_clustering_accuracy (range [0, 1] - higher is better)

    Return:
      dict(ASW=asw_score, ARI=ari_score, NMI=nmi_score, UCA=uca_score)
    """
    z_train, z_test = self.representations_mean
    f_train, f_test = self.factors
    return metrics.unsupervised_clustering_scores(
        representations=np.concatenate([z_train, z_test], axis=0),
        factors=np.concatenate([f_train, f_test], axis=0),
        algorithm=algorithm,
        random_state=self.randint)

  ##############  Posterior predictive check (PPC)
  def posterior_predictive_check(n_samples=100):
    r""" PPC - "simulating replicated data under the fitted model and then
    comparing these to the observed data"

    In other word, using posterior predictive to "look for systematic
    discrepancies between real and simulated data"

    Reference:
      Gelman and Hill, 2007, p. 158. "Data Analysis Using Regression and
        Multilevel/Hierarchical Models".
      Gelman et al. 2004, p. 169. "Bayesian Data Analysis".
      Clivio, O., Boyeau, P., Lopez, R., et al. (2019.) "Should we zero-inflate
        scVI?" https://yoseflab.github.io/2019/06/25/ZeroInflation/
    """
    # TODO
