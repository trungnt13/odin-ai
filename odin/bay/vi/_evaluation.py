from __future__ import absolute_import, division, print_function

import re
import warnings
from collections import OrderedDict
from numbers import Number

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from tqdm import tqdm

from odin import search
from odin.bay import distributions as tfd
from odin.bay.distributions.utils import concat_distribution
from odin.bay.vi import metrics, utils
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.data_utils import Factor
from odin.ml import dimension_reduce
from odin.utils import as_tuple


class _Criticizer(object):

  def __init__(self, vae: VariationalAutoencoder, random_state=1):
    super().__init__()
    assert isinstance(vae, VariationalAutoencoder), \
      "vae must be instance of odin.bay.vi.VariationalAutoencoder, given: %s" \
        % str(type(vae))
    self._vae = vae
    if isinstance(random_state, Number):
      random_state = np.random.RandomState(seed=random_state)
    # main arguments
    self._inputs = None
    self._factors = None
    self._original_factors = None
    self._factors_name = None
    self._representations = None
    self._reconstructions = None
    # others
    self._rand = random_state
    self._is_list_inputs = None

  def assert_sampled(self):
    if self._inputs is None or \
      self._factors is None or \
        self._representations is None:
      raise RuntimeError("Call the `sample_batch` method to sample "
                         "mini-batch of ground-truth data and learned "
                         "representations.")

  @property
  def inputs(self):
    self.assert_sampled()
    return self._inputs

  @property
  def is_list_inputs(self):
    self.assert_sampled()
    return self._is_list_inputs

  @property
  def representations(self):
    r""" Return the learned representations `distribution`
    (i.e. the latent code) for training and testing """
    self.assert_sampled()
    return self._representations

  @property
  def representations_mean(self):
    r""" Return the mean of learned representations distribution
    (i.e. the latent code) for training and testing """
    self.assert_sampled()
    return [z.mean().numpy() for z in self.representations]

  @property
  def representations_variance(self):
    r""" Return the variance of learned representations distribution
    (i.e. the latent code) for training and testing """
    self.assert_sampled()
    return [z.variance().numpy() for z in self.representations]

  def representations_sample(self, n=()):
    r""" Return the mean of learned representations distribution
    (i.e. the latent code) for training and testing """
    self.assert_sampled()
    return [
        z.sample(sample_shape=n, seed=self.randint).numpy()
        for z in self.representations
    ]

  @property
  def reconstructions(self):
    r""" Return the reconstructed distributions of inputs for training and
    testing """
    self.assert_sampled()
    return self._reconstructions

  @property
  def reconstructions_mean(self):
    r""" Return the mean of reconstructed distributions of inputs for
    training and testing """
    self.assert_sampled()
    return [[j.mean().numpy() for j in i] for i in self._reconstructions]

  @property
  def reconstructions_variance(self):
    r""" Return the variance of reconstructed distributions of inputs for
    training and testing """
    self.assert_sampled()
    return [[j.variance().numpy() for j in i] for i in self._reconstructions]

  def reconstructions_sample(self, n=()):
    r""" Return the mean of reconstructed distributions of inputs for
    training and testing """
    self.assert_sampled()
    return [[j.sample(sample_shape=n, seed=self.randint).numpy()
             for j in i]
            for i in self._reconstructions]

  @property
  def original_factors(self):
    r""" Return the training and testing original factors, i.e. the factors
    before discretizing """
    self.assert_sampled()
    # the original factors is the same for all samples set
    return self._original_factors

  @property
  def n_factors(self):
    return self.factors[0].shape[1]

  @property
  def n_representations(self):
    r""" return the number of latent codes """
    return self.representations[0].event_shape[0]

  @property
  def n_codes(self):
    r""" same as `n_representations`, return the number of latent codes """
    return self.n_representations

  @property
  def n_train(self):
    r""" Return number of samples for training """
    return self.factors[0].shape[0]

  @property
  def n_test(self):
    r""" Return number of samples for testing """
    return self.factors[1].shape[0]

  @property
  def factors(self):
    r""" Return the target variable (i.e. the factors of variation) for
    training and testing """
    self.assert_sampled()
    return self._factors

  @property
  def factors_name(self):
    self.assert_sampled()
    # the dataset is unchanged, always at 0-th index
    return np.array(self._factors_name)

  @property
  def codes_name(self):
    return np.array(["Code#%d" % i for i in range(self.n_representations)])

  @property
  def random_state(self):
    return self._rand

  @property
  def randint(self):
    return self._rand.randint(1e8)

  ############## proxy to VAE methods
  def index(self, factors_name):
    r""" Return the column index of given factors_name within the
    factor matrix """
    return self._factors_name.index(str(factors_name))

  def encode(self, inputs, first_latent=True):
    r""" Encode inputs to latent codes

    Arguments:
      inputs : a single Tensor or list of Tensor
      first_latent : a Boolean, indicator for returning  only the first latent
        (in case of multiple latents returned)

    Returns:
      `tensorflow_probability.Distribution`, q(z|x) the latent distribution
    """
    if self._is_list_inputs is not None:
      if self._is_list_inputs:
        inputs = tf.nest.flatten(inputs)
      elif isinstance(inputs, (tuple, list)):
        inputs = inputs[0]
    latents = self._vae.encode(inputs, training=False, n_mcmc=1)
    # only support single returned latent variable now
    for z in tf.nest.flatten(latents):
      assert isinstance(z, tfd.Distribution), \
        "The latent code return from `vae.encode` must be instance of " + \
          "tensorflow_probability.Distribution, but returned: %s" % \
            str(z)
    if isinstance(latents, (tuple, list)) and first_latent:
      return latents[0]
    return latents

  def decode(self, latents):
    r""" Decode the latents into reconstruction distribution """
    outputs = self._vae.decode(latents, training=False)
    for o in tf.nest.flatten(outputs):
      assert isinstance(o, tfd.Distribution), \
        "vae decode method must return reconstruction distribution, but " + \
          "returned: %s" % str(o)
    return outputs

  ############## Experiment setup
  def traversing(self,
                 indices=None,
                 min_val=-1.,
                 max_val=1.,
                 num=10,
                 n_samples=2,
                 mode='linear'):
    r"""

    Arguments:
      indices : a list of Integer or None. The indices of latent code for
        traversing. If None, all latent codes are used.

    Return:
      numpy.ndarray : traversed latent codes for training and testing,
        the shape is `[len(indices) * n_samples * num, n_representations]`
    """
    self.assert_sampled()
    num = int(num)
    n_samples = int(n_samples)
    assert num > 1 and n_samples > 0, "num > 1 and n_samples > 0"
    # ====== indices ====== #
    if indices is None:
      indices = list(range(self.n_representations))
    else:
      indices = [int(i) for i in tf.nest.flatten(indices)]
      assert all(i < self.n_factors for i in indices), \
        "There are %d factors, but the factor indices are: %s" % \
          (self.n_factors, str(indices))
    indices = np.array(indices)
    # ====== check the mode ====== #
    all_mode = ('quantile', 'linear')
    mode = str(mode).strip().lower()
    assert mode in all_mode, \
      "Only support %s, but given mode='%s'" % (str(all_mode), mode)

    # ====== helpers ====== #
    def _traverse(z):
      sampled_indices = self._rand.choice(z.shape[0],
                                          size=int(n_samples),
                                          replace=False)
      Zs = []
      for i in sampled_indices:
        n = len(indices) * num
        z_i = np.repeat(np.expand_dims(z[i], 0), n, axis=0)
        for j, idx in enumerate(indices):
          start = j * num
          end = (j + 1) * num
          # linear
          if mode == 'linear':
            z_i[start:end, idx] = np.linspace(min_val, max_val, num)
          # Gaussian quantile
          elif mode == 'quantile':
            base_code = z_i[0, idx]
            print(base_code)
            exit()
          # Gaussian linear
          elif mode == '':
            raise NotImplementedError
        Zs.append(z_i)
      Zs = np.concatenate(Zs, axis=0)
      return Zs, sampled_indices

    # ====== traverse through latent space ====== #
    z_train, z_test = self.representations_mean
    z_train, train_ids = _traverse(z_train)
    z_test, test_ids = _traverse(z_test)
    return z_train, z_test

  def conditioning(self, known={}, logical_not=False):
    r""" Conditioning the sampled dataset on known factors

    Arguments:
      known : a mapping from index or name of factor to a callable, the
        callable must return a list of boolean indices, which indicates
        the samples to be selected
      logical_not : a Boolean, if True applying the opposed conditioning
        of the known factors

    Return:
      a new Criticizer conditioned on the known factors
    """
    self.assert_sampled()
    known = {
        int(k) if isinstance(k, Number) else self.index(str(k)): v
        for k, v in dict(known).items()
    }
    assert len(known) > 0 and all(callable(v) for v in known.values()), \
      "'known' factors must be mapping from factor index to callable " + \
        "but given: %s" % str(known)
    # start conditioning
    x_train, x_test = self.inputs
    f_train, f_test = self.factors
    train_ids = np.full(shape=f_train.shape[0], fill_value=True, dtype=np.bool)
    test_ids = np.full(shape=f_test.shape[0], fill_value=True, dtype=np.bool)
    for f_idx, fn_filter in known.items():
      train_ids = np.logical_and(train_ids, fn_filter(f_train[:, f_idx]))
      test_ids = np.logical_and(test_ids, fn_filter(f_test[:, f_idx]))
    # opposing the conditions
    if logical_not:
      train_ids = np.logical_not(train_ids)
      test_ids = np.logical_not(test_ids)
    # add new samples set to stack
    o_train, o_test = self.original_factors
    x_train = [x[train_ids] for x in x_train]
    x_test = [x[test_ids] for x in x_test]
    z_train = self.encode(x_train, first_latent=False)
    z_test = self.encode(x_test, first_latent=False)
    r_train = self.decode(z_train)
    r_test = self.decode(z_test)
    # create a new critizer
    crt = self.copy()
    crt._representations = (\
      z_train[0] if isinstance(z_train, (tuple, list)) else z_train,
      z_test[0] if isinstance(z_test, (tuple, list)) else z_test)
    crt._inputs = (x_train, x_test)
    crt._reconstructions = (r_train, r_test)
    crt._factors = (f_train[train_ids], f_test[test_ids])
    crt._original_factors = (o_train[train_ids], o_test[test_ids])
    return crt

  def sample_batch(self,
                   inputs,
                   factors,
                   discretizing=False,
                   n_bins=5,
                   strategy='quantile',
                   factors_name=None,
                   train_percent=0.8,
                   n_samples=1000,
                   batch_size=32,
                   verbose=False):
    r"""
    Arguments:
      inputs : list of `ndarray`. Inputs to the model
      factors : a `ndarray`. Groundtruth factor
      discretizing : if True, turn continuous factors into discrete
      n_bins : int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.
      strategy : {'uniform', 'quantile', 'kmeans', 'gmm'}, (default='quantile')
        Strategy used to define the widths of the bins.
        uniform - All bins in each feature have identical widths.
        quantile - All bins in each feature have the same number of points.
        kmeans - Values in each bin have the same nearest center of a 1D
          k-means cluster.
        gmm - using the components (in sorted order of mean) of Gaussian
          mixture to label.
      factors_name :
      train_percent :
      n_samples :
      batch_size :

    Returns:
      `Criticizer` with sampled data
    """
    assert len(factors.shape) == 2, "factors must be a matrix"
    is_list_inputs = isinstance(inputs, (tuple, list))
    inputs = tf.nest.flatten(inputs)
    # ====== split train test ====== #
    ids = self.random_state.permutation(factors.shape[0])
    split = int(train_percent * factors.shape[0])
    train_ids, test_ids = ids[:split], ids[split:]
    train_inputs = [i[train_ids] for i in inputs]
    test_inputs = [i[test_ids] for i in inputs]
    n_samples = as_tuple(n_samples, t=int, N=2)
    # ====== create discretized factors ====== #
    f_original = (factors[train_ids], factors[test_ids])
    if discretizing:
      if verbose:
        print("Discretizing factors:", int(n_bins), '-', strategy)
      factors = utils.discretizing(factors,
                                   n_bins=int(n_bins),
                                   strategy=strategy)
    train_factors = Factor(factors[train_ids],
                           factors_name=factors_name,
                           random_state=self.randint)
    test_factors = Factor(factors[test_ids],
                          factors_name=factors_name,
                          random_state=self.randint)

    # ====== sampling ====== #
    def sampling(inputs_, factors_, nsamples, title):
      Xs = [list() for _ in range(len(inputs))]  # inputs
      Ys = []  # factors
      Zs = []  # latents
      Os = []  # outputs
      indices = []
      n = 0
      if verbose:
        prog = tqdm(desc='Sampling %s' % title, total=nsamples)
      while n < nsamples:
        batch = min(batch_size, nsamples - n, factors_.shape[0])
        if verbose:
          prog.update(int(batch))
        # factors
        y, ids = factors_.sample_factors(num=batch, return_indices=True)
        indices.append(ids)
        Ys.append(y)
        # inputs
        inps = []
        for x, i in zip(Xs, inputs_):
          i = i[ids, :]
          x.append(i)
          inps.append(i)
        # latents representation
        z = self.encode(inps, first_latent=False)
        Os.append(tf.nest.flatten(self.decode(z)))
        Zs.append(z[0] if isinstance(z, (tuple, list)) else z)
        # update the couter
        n += len(y)
      # aggregate all data
      Xs = [np.concatenate(x, axis=0) for x in Xs]
      Ys = np.concatenate(Ys, axis=0)
      Zs = concat_distribution(Zs, name="Latents")
      Os = [
          concat_distribution([j[i]
                               for j in Os], name="Output%d" % i)
          for i in range(len(Os[0]))
      ]
      return Xs, Ys, Zs, Os, np.concatenate(indices, axis=0)

    # perform sampling
    train = sampling(inputs_=train_inputs,
                     factors_=train_factors,
                     nsamples=n_samples[0],
                     title="Train")
    test = sampling(inputs_=test_inputs,
                    factors_=test_factors,
                    nsamples=n_samples[1],
                    title="Test ")
    ids_train = train[4]
    ids_test = test[4]
    # assign the variables
    self._is_list_inputs = is_list_inputs
    self._inputs = (train[0], test[0])
    self._factors = (train[1], test[1])
    self._factors_name = train_factors.factors_name
    self._representations = (train[2], test[2])
    self._reconstructions = (train[3], test[3])
    self._original_factors = (f_original[0][ids_train], f_original[1][ids_test])
    return self

  ############## Helpers
  def _latent_codes(self, mean=True):
    if mean:
      return [i.mean().numpy() for i in self.representations]
    return [
        i.sample(sample_shape=(), seed=self.randint).numpy()
        for i in self.representations
    ]

  ############## Metrics
  def cal_mutual_info_est(self, mean=True, n_neighbors=3):
    r""" Mututal information estimation using k-Nearest Neighbor """
    mi = []
    for z, f in zip(self._latent_codes(mean), self.factors):
      mi.append(metrics.mutual_info_estimate(z, f, n_neighbors=n_neighbors))
    return tuple(mi)

  def cal_mutual_info_gap(self, mean=True):
    r"""
    Arguments:
      mean : a Boolean, if True use the mean of latent distribution for
        calculating the mutual information gap

    Reference:
      Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2019. Isolating Sources of
        Disentanglement in Variational Autoencoders. arXiv:1802.04942 [cs, stat].
    """
    mig = []
    for z, f in zip(self._latent_codes(mean), self.factors):
      mig.append(metrics.mutual_info_gap(z, f))
    return tuple(mig)

  def cal_dci_scores(self, mean=True):
    r""" Disentanglement, Completeness, Informativeness

    References:
      Based on "A Framework for the Quantitative Evaluation of Disentangled
      Representations" (https://openreview.net/forum?id=By-7dz-AZ).
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    return metrics.dci_scores(z_train,
                              f_train,
                              z_test,
                              f_test,
                              random_state=self.randint)

  def cal_total_correlation(self):
    r""" Total correlation based on fitted Gaussian """
    samples = [qz.sample(seed=self.randint) for qz in self.representations]
    return tuple([
        utils.total_correlation(z, qz).numpy()
        for z, qz in zip(samples, self.representations)
    ])

  def cal_importance_matrix(self, mean=True, algo=GradientBoostingClassifier):
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

  def cal_density_matrix(self,
                         n_samples=1000,
                         lognorm=False,
                         n_components=2,
                         normalize_by_code=True,
                         decode=False):
    r""" """
    n_samples = int(n_samples)
    n_codes = self.n_codes
    n_factors = self.n_factors
    qZ = self.representations[0]
    y = self.original_factors[0]
    if lognorm:
      y = np.log1p(y)
    ### train the Gaussian mixture on the factors
    f_gmm = []
    for fidx, (f, fname) in enumerate(zip(y.T, self.factors_name)):
      gmm = tfd.GaussianMixture.init(f[:, np.newaxis],
                                     n_components=n_components,
                                     covariance_type='diag',
                                     batch_shape=None,
                                     name=fname)
      f_gmm.append(gmm)
    ### the code Gaussian
    dist_type = type(qZ)
    if isinstance(qZ, tfd.Independent):
      dist_type = type(qZ.distribution)
    support_type = (tfd.MultivariateNormalDiag, tfd.Normal)
    if dist_type not in support_type:
      raise RuntimeError(
          "No support posterior distribution: %s, the support distributions are: %s"
          % (str(dist_type), str(support_type)))
    z_gau = []
    for mean, stddev, code_name in zip(tf.transpose(qZ.mean()),
                                       tf.transpose(qZ.stddev()),
                                       self.codes_name):
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
        qllk = gau.log_prob(samples)
        pllk = tf.reduce_sum(tf.reshape(
            gmm.log_prob(tf.reshape(samples, (-1, 1))), (n_samples, -1)),
                             axis=1)
        kl = tf.reduce_mean(qllk - pllk)
        density_matrix[zidx, fidx] = kl.numpy()
    if bool(normalize_by_code):
      density_matrix = density_matrix / \
        np.sum(density_matrix, axis=1, keepdims=True)
    ### decoding
    if decode:
      ids = search.diagonal_beam_search(density_matrix.T)
      density_matrix = density_matrix[ids]
      return density_matrix, ids
    return density_matrix

  def cal_disentangled_density(self,
                               n_samples=1000,
                               lognorm=False,
                               n_components=2):
    r""" Higher is better, this estimate the disparity between diagonal
    components and off-diagonal components """
    density_mat, ids = self.cal_density_matrix(n_samples=n_samples,
                                               lognorm=lognorm,
                                               n_components=n_components,
                                               normalize_by_code=True,
                                               decode=True)
    diag = np.diagflat(np.diag(density_mat))
    off_diag = density_mat - diag
    return np.mean(diag) - np.mean(off_diag)

  def cal_relative_disentanglement_strength(self, mean=True, method='spearman'):
    r""" Relative strength for both axes of correlation matrix """
    corr_matrix = self.cal_correlation_matrix(mean=mean, method=method)
    return metrics.relative_strength(corr_matrix)

  def cal_correlation_matrix(self, mean=True, method='spearman', decode=False):
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
          self.cal_correlation_matrix(mean=mean, method=corr, decode=False)
          for corr in all_corr[:-1]
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
      ids = search.diagonal_beam_search(train.T)
      train = train[ids, :]
      test = test[ids, :]
      return train, test, OrderedDict(zip(range(self.n_factors), ids))
    return train, test

  ############## Downstream scores
  def cal_separated_attr_predictability(self, mean=True):
    r"""
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
    return sap

  def cal_betavae_score(self, mean=True, n_samples=1000, verbose=False):
    r""" BetaVAE based score

    Returns:
      accuracy for train and test data
    """
    z_train, z_test = self.representations
    f_train, f_test = self.factors
    score_train = metrics.beta_vae_score(z_train,
                                         f_train,
                                         n_samples=n_samples,
                                         use_mean=mean,
                                         random_state=self.randint,
                                         verbose=verbose)
    score_test = metrics.beta_vae_score(z_test,
                                        f_test,
                                        n_samples=n_samples,
                                        use_mean=mean,
                                        random_state=self.randint,
                                        verbose=verbose)
    return score_train, score_test

  def cal_factorvae_score(self, mean=True, n_samples=1000, verbose=False):
    r""" FactorVAE based score

    Returns:
      accuracy for train and test data
    """
    z_train, z_test = self.representations
    f_train, f_test = self.factors
    score_train = metrics.factor_vae_score(z_train,
                                           f_train,
                                           n_samples=n_samples,
                                           use_mean=mean,
                                           random_state=self.randint,
                                           verbose=verbose)
    score_test = metrics.factor_vae_score(z_test,
                                          f_test,
                                          n_samples=n_samples,
                                          use_mean=mean,
                                          random_state=self.randint,
                                          verbose=verbose)
    return score_train, score_test

  ############## Methods for summarizing
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
