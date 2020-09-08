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

  ############## Matrices
  def create_correlation_matrix(self,
                                method='spearman',
                                mean=True,
                                decode=False):
    r""" Correlation matrix of `latent codes` (row) and `groundtruth factors`
    (column).

    Arguments:
      mean : a Boolean. Using mean as the statistics, otherwise, sampling.
      method : {'spearman', 'pearson', 'lasso', 'avg'}
        spearman - rank or monotonic correlation
        pearson - linear correlation
        lasso - lasso regression
        avg - compute all known method then taking average
      decode : a Boolean. If True, reorganize the row of correlation matrix
        for the best match between code-factor (i.e. the largest diagonal sum).
        Note: the decoding is performed on train matrix, then applied to test
        matrix

    Returns:
      train, test : correlation matrices `[n_codes, n_factors]`
        for both training and testing data.
        All entries are in `[0, 1]`.
      (optional) OrderedDict mapping from decoded factor index to
        latent code index.
    """
    method = str(method).strip().lower()
    if method in ('avg', 'avr', 'average'):
      method = 'average'
    all_corr = ['spearman', 'lasso', 'pearson', 'average']
    assert isinstance(mean, bool), "mean is boolean but given: %s" % mean
    assert method in all_corr, \
      "Support %s correlation but given method='%s'" % (str(all_corr), method)
    # special average mode
    if method == 'average':
      mat = [
          self.create_correlation_matrix(mean=mean, method=corr, decode=False)
          for corr in ['spearman', 'pearson']
      ]
      n = len(all_corr) - 1
      train = sum(i[0] for i in mat) / n
      test = sum(i[1] for i in mat) / n
    else:
      # start form correlation matrix
      z_train, z_test = self._latent_codes(mean)
      f_train, f_test = self.factors

      # helper function
      def fn_corr(x1, x2):
        if method == 'lasso':
          model = Lasso(random_state=self.randint, alpha=0.1)
          model.fit(x1, x2)
          # coef_ is [n_target, n_features], so we need transpose here
          corr_mat = np.transpose(np.absolute(model.coef_))
        else:
          corr_mat = np.empty(shape=(self.n_representations, self.n_factors),
                              dtype=np.float64)
          for code in range(self.n_representations):
            for fact in range(self.n_factors):
              x, y = x1[:, code], x2[:, fact]
              if method == 'spearman':
                corr = sp.stats.spearmanr(x, y, nan_policy="omit")[0]
              elif method == 'pearson':
                corr = sp.stats.pearsonr(x, y)[0]
              elif method == 'lasso':
                pass
              corr_mat[code, fact] = corr
        return corr_mat

      train, test = fn_corr(z_train, f_train), fn_corr(z_test, f_test)
    ## decoding and return
    if decode:
      ids = search.diagonal_linear_assignment(train.T)
      train = train[ids, :]
      test = test[ids, :]
      return train, test, OrderedDict(zip(range(self.n_factors), ids))
    return train, test

  def create_mutualinfo_matrix(self, n_neighbors=3, mean=True):
    r""" Mututal information estimation using k-Nearest Neighbor

    Return:
      matrix `[num_latents, num_factors]`, estimated mutual information between
        each representation and each factors
    """
    mi = []
    # iterate over train and test data
    for z, f in zip(self._latent_codes(mean), self.factors):
      mi.append(metrics.mutual_info_estimate(z, f, n_neighbors=n_neighbors))
    train, test = mi
    return train, test

  def create_importance_matrix(self,
                               algo=GradientBoostingClassifier,
                               mean=True):
    r""" Using ensemble algorithm to estimate the feature importance of each
    pair of (representation, factor)

    Return:
      a matrix of shape `[n_codes, n_factors]`
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    importance_matrix, _, _ = \
      metrics.representative_importance_matrix(
        z_train, f_train, z_test, f_test,
        random_state=self.randint, algo=algo)
    return importance_matrix

  def create_divergence_matrix(self,
                               n_samples=1000,
                               lognorm=True,
                               n_components=2,
                               normalize_per_code=True,
                               decode=False):
    r""" Using GMM fitted on the factors to estimate the divergence to each
    latent code.

    It means calculating the divergence: `DKL(q(z|x)||p(y))`, where:
      - q(z|x) is latent code of Gaussian distribution
      - p(y) is factor of Gaussian mixture model with `n_components`

    The calculation is repeated for each pair of (code, factor). This method is
    recommended for factors that are continuous values.

    Return:
      a matrix of shape `[n_codes, n_factors]`
    """
    n_samples = int(n_samples)
    n_codes = self.n_codes
    n_factors = self.n_factors
    matrices = []
    for qZ, y in zip(self.representations, self.original_factors):
      ### normalizing the factors
      if lognorm:
        y = np.log1p(y)
      # standardizing for each factor
      y = (y - np.mean(y, axis=0, keepdims=True)) / (
          np.std(y, axis=0, keepdims=True) + 1e-10)
      ### train the Gaussian mixture on the factors
      f_gmm = []
      for fidx, (f, fname) in enumerate(zip(y.T, self.factor_names)):
        gmm = tfd.GaussianMixture.init(f[:, np.newaxis],
                                       n_components=n_components,
                                       covariance_type='diag',
                                       batch_shape=None,
                                       dtype=tf.float64,
                                       name=fname)
        f_gmm.append(gmm)
      ### the code Gaussian
      z_gau = []
      for mean, stddev, code_name in zip(tf.transpose(qZ.mean()),
                                         tf.transpose(qZ.stddev()),
                                         self.code_names):
        mean = tf.cast(mean, tf.float64)
        stddev = tf.cast(stddev, tf.float64)
        z_gau.append(
            tfd.Independent(tfd.Normal(loc=mean, scale=stddev, name=code_name),
                            reinterpreted_batch_ndims=1))
      ### calculate the KL divergence
      density_matrix = np.empty(shape=(n_codes, n_factors), dtype=np.float64)
      for zidx, gau in enumerate(z_gau):
        for fidx, gmm in enumerate(f_gmm):
          # non-analytic KL(q=gau||p=gmm)
          samples = gau.sample(n_samples)
          with tf.device("/CPU:0"):
            qllk = gau.log_prob(samples)
            pllk = tf.reduce_sum(tf.reshape(
                gmm.log_prob(tf.reshape(samples, (-1, 1))), (n_samples, -1)),
                                 axis=1)
            kl = tf.reduce_mean(qllk - pllk)
          density_matrix[zidx, fidx] = kl.numpy()
      if bool(normalize_per_code):
        density_matrix = density_matrix / np.sum(
            density_matrix, axis=1, keepdims=True)
      matrices.append(density_matrix)
    ### decoding and return
    train, test = matrices
    if decode:
      ids = search.diagonal_linear_assignment(train.T)
      train = train[ids]
      test = test[ids]
      return train, test, ids
    return train, test

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
    r""" FactorVAE based score

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
