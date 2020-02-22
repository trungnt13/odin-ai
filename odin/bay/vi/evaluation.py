from __future__ import absolute_import, division, print_function

import inspect
import re
import warnings
from numbers import Number

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from odin import visual as vs
from odin.bay.distributions.utils import concat_distribution
from odin.bay.vi import metrics, utils
from odin.bay.vi.data_utils import Factor
from odin.bay.vi.variational_autoencoder import VariationalAutoencoder
from odin.ml import dimension_reduce
from odin.utils import as_tuple


class Criticizer(vs.Visualizer):

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
    self._factor_name = None
    self._representations = None
    self._reconstructions = None
    # others
    self._rand = random_state
    self._is_list_inputs = None

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

  def _assert_sampled(self):
    if self._inputs is None or \
      self._factors is None or \
        self._representations is None:
      raise RuntimeError("Call the `sample_batch` method to sample "
                         "mini-batch of ground-truth data and learned "
                         "representations.")

  @property
  def inputs(self):
    self._assert_sampled()
    return self._inputs

  @property
  def is_list_inputs(self):
    self._assert_sampled()
    return self._is_list_inputs

  @property
  def representations(self):
    r""" Return the learned representations distribution (i.e. the latent code)
    for training and testing """
    self._assert_sampled()
    return self._representations

  @property
  def representations_mean(self):
    r""" Return the mean of learned representations distribution
    (i.e. the latent code) for training and testing """
    self._assert_sampled()
    return [z.mean().numpy() for z in self.representations]

  @property
  def representations_variance(self):
    r""" Return the variance of learned representations distribution
    (i.e. the latent code) for training and testing """
    self._assert_sampled()
    return [z.variance().numpy() for z in self.representations]

  def representations_sample(self, n=()):
    r""" Return the mean of learned representations distribution
    (i.e. the latent code) for training and testing """
    self._assert_sampled()
    return [
        z.sample(sample_shape=n, seed=self.randint).numpy()
        for z in self.representations
    ]

  @property
  def reconstructions(self):
    r""" Return the reconstructed distributions of inputs for training and
    testing """
    self._assert_sampled()
    return self._reconstructions

  @property
  def reconstructions_mean(self):
    r""" Return the mean of reconstructed distributions of inputs for
    training and testing """
    self._assert_sampled()
    return [[j.mean().numpy() for j in i] for i in self._reconstructions]

  @property
  def reconstructions_variance(self):
    r""" Return the variance of reconstructed distributions of inputs for
    training and testing """
    self._assert_sampled()
    return [[j.variance().numpy() for j in i] for i in self._reconstructions]

  def reconstructions_sample(self, n=()):
    r""" Return the mean of reconstructed distributions of inputs for
    training and testing """
    self._assert_sampled()
    return [[j.sample(sample_shape=n, seed=self.randint).numpy()
             for j in i]
            for i in self._reconstructions]

  @property
  def original_factors(self):
    r""" Return the training and testing original factors, i.e. the factors
    before discretizing """
    self._assert_sampled()
    # the original factors is the same for all samples set
    return self._original_factors

  @property
  def n_factors(self):
    return self.factors[0].shape[1]

  @property
  def n_representations(self):
    return self.representations[0].event_shape[0]

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
    self._assert_sampled()
    return self._factors

  @property
  def factor_name(self):
    self._assert_sampled()
    # the dataset is unchanged, always at 0-th index
    return np.array(self._factor_name)

  @property
  def code_name(self):
    return np.array(["Code#%d" % i for i in range(self.n_representations)])

  @property
  def random_state(self):
    return self._rand

  @property
  def randint(self):
    return self._rand.randint(1e8)

  ############## proxy to VAE methdos
  def index(self, factor_name):
    r""" Return the column index of given factor_name within the
    factor matrix """
    return self._factor_name.index(str(factor_name))

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
    self._assert_sampled()
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
    self._assert_sampled()
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
                   factor_name=None,
                   train_percent=0.8,
                   n_samples=1000,
                   batch_size=32):
    r"""
    Arguments:
      inputs :
      factors :
      discretizing :
      strategy :
      factor_name :
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
      factors = utils.discretizing(factors,
                                   n_bins=int(n_bins),
                                   strategy=strategy)
    train_factors = Factor(factors[train_ids],
                           factor_name=factor_name,
                           random_state=self.randint)
    test_factors = Factor(factors[test_ids],
                          factor_name=factor_name,
                          random_state=self.randint)

    # ====== sampling ====== #
    def sampling(inputs_, factors_, nsamples):
      Xs = [list() for _ in range(len(inputs))]  # inputs
      Ys = []  # factors
      Zs = []  # latents
      Os = []  # outputs
      indices = []
      n = 0
      while n < nsamples:
        batch = min(batch_size, nsamples - n, factors_.shape[0])
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
                     nsamples=n_samples[0])
    test = sampling(inputs_=test_inputs,
                    factors_=test_factors,
                    nsamples=n_samples[1])
    ids_train = train[4]
    ids_test = test[4]
    # assign the variables
    self._is_list_inputs = is_list_inputs
    self._inputs = (train[0], test[0])
    self._factors = (train[1], test[1])
    self._factor_name = train_factors.factor_name
    self._representations = (train[2], test[2])
    self._reconstructions = (train[3], test[3])
    self._original_factors = (f_original[0][ids_train], f_original[1][ids_test])
    return self

  ############## Metrics
  def _latent_codes(self, mean=True):
    if mean:
      return [i.mean().numpy() for i in self.representations]
    return [
        i.sample(sample_shape=(), seed=self.randint).numpy()
        for i in self.representations
    ]

  def mutual_info_estimate(self, mean=True, n_neighbors=3):
    mi = []
    for z, f in zip(self._latent_codes(mean), self.factors):
      mi.append(metrics.mutual_info_estimate(z, f, n_neighbors=n_neighbors))
    return tuple(mi)

  def mutual_info_gap(self, mean=True):
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

  def separated_attr_predictability(self, mean=True):
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

  def dci_scores(self, mean=True):
    r""" Disentanglement, Completeness, Informativeness """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    return metrics.dci_scores(z_train,
                              f_train,
                              z_test,
                              f_test,
                              random_state=self.randint)

  def total_correlation(self):
    r""" Total correlation based on fitted Gaussian """
    samples = [qz.sample(seed=self.randint) for qz in self.representations]
    return tuple([
        utils.total_correlation(z, qz).numpy()
        for z, qz in zip(samples, self.representations)
    ])

  def importance_matrix(self, mean=True, algo=GradientBoostingClassifier):
    r""" Using ensemble algorithm to estimate the feature importance of each
    pair of (representation, factor)

    Return:
      a matrix of shape `[n_codes, n_factors]`
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    importance_matrix, _, _ = \
      metrics.representation_importance_matrix(
        z_train, f_train, z_test, f_test,
        random_state=self.randint, algo=algo)
    return importance_matrix

  def correlation_matrix(self, mean=True, corr_type='spearman'):
    r"""
    Returns:
      correlation matrices `[n_codes, n_factors]` for both training and
        testing data
    """
    corr_type = str(corr_type).strip().lower()
    all_corr = {'spearman', 'lasso', 'pearson'}
    assert corr_type in all_corr, \
      "Support %s correlation but given corr_type='%s'" % \
        (str(all_corr), corr_type)
    # start form correlation matrix
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors

    # helper function
    def fn_corr(x1, x2):
      if corr_type == 'lasso':
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
            if corr_type == 'spearman':
              corr = sp.stats.spearmanr(x, y, nan_policy="omit")[0]
            elif corr_type == 'pearson':
              corr = sp.stats.pearsonr(x, y)[0]
            elif corr_type == 'lasso':
              pass
            corr_mat[code, fact] = corr
      return corr_mat

    return fn_corr(z_train, f_train), fn_corr(z_test, f_test)

  ############## Ploting
  def _check_factors(self, factors):
    if factors is None:
      factors = list(range(self.n_factors))
    else:
      factors = [
          int(i) if isinstance(i, Number) else self.index(i)
          for i in tf.nest.flatten(factors)
      ]
    return factors

  def plot_histogram(self, factors=None, bins=120, original_factors=True):
    r"""
    orginal_factors : optional original factors before discretized by
      `Criticizer`
    """
    self._assert_sampled()
    from matplotlib import pyplot as plt
    # prepare the data
    factors = self._check_factors(factors)
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0)[:, factors]
    X = [i for i in F.T] + [i for i in Z.T]
    labels = self.factor_name[factors].tolist() + \
      self.code_name.tolist()
    # create the figure
    ncol = int(np.ceil(np.sqrt(len(X)))) + 1
    nrow = int(np.ceil(len(X) / ncol))
    fig = vs.plot_figure(nrow=18, ncol=25, dpi=80)
    for i, (x, lab) in enumerate(zip(X, labels)):
      vs.plot_histogram(x,
                        ax=(nrow, ncol, i + 1),
                        bins=int(bins),
                        title=lab,
                        alpha=0.8,
                        color='blue',
                        fontsize=16)
    plt.tight_layout()
    self.add_figure(
        "histogram_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_code_factor_matrix(self, factors=None, original_factors=True):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    from matplotlib import pyplot as plt
    factors = self._check_factors(factors)
    Z = np.concatenate(self.representations_mean, axis=0)
    F = np.concatenate(
        self.original_factors if original_factors else self.factors,
        axis=0)[:, factors]
    labels = self.factor_name[factors]
    code_name = self.code_name
    # create the figure
    nrow = Z.shape[1]  # number representations
    ncol = F.shape[1]  # number factors
    fig, subplots = plt.subplots(nrows=nrow,
                                 ncols=ncol,
                                 sharex=True,
                                 sharey=True,
                                 squeeze=True,
                                 gridspec_kw=dict(wspace=0.01, hspace=0.01),
                                 figsize=(ncol * 3, nrow * 3),
                                 dpi=80)
    for code_idx, row in enumerate(subplots):
      for fact_idx, ax in enumerate(row):
        z = Z[:, code_idx]
        zname = code_name[code_idx]
        f = F[:, fact_idx]
        fname = labels[fact_idx]
        mean = np.mean(f)
        ax.scatter(f,
                   z,
                   s=6,
                   alpha=0.5,
                   c=['r' if i > mean else 'b' for i in f],
                   linewidths=0.)
        ax.grid(False)
        if fact_idx == 0:
          ax.set_ylabel(zname, fontsize=16)
        if code_idx == 0:
          ax.set_title(fname, fontsize=16)
    self.add_figure(
        "code_factor_%s" % ("original" if original_factors else "discretized"),
        fig)
    return self

  def plot_uncertainty_statistics(self, factors=None):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    factors = self._check_factors(factors)
    zmean = np.concatenate(self.representations_mean, axis=0)
    zstd = np.sqrt(np.concatenate(self.representations_variance, axis=0))
    labels = self.factor_name[factors]
    factors = np.concatenate(self.original_factors, axis=0)[:, factors]
    X = np.arange(zmean.shape[0])
    # create the figure
    nrow = self.n_representations
    ncol = len(labels)
    fig = vs.plot_figure(nrow=nrow * 4, ncol=ncol * 4, dpi=80)
    plot = 1
    for row, (code, mean, std) in enumerate(zip(self.code_name, zmean.T,
                                                zstd.T)):
      # prepare the code
      ids = np.argsort(mean)
      mean, std = mean[ids], std[ids]
      # show the factors
      for col, (name, y) in enumerate(zip(labels, factors.T)):
        axes = []
        # the variance
        ax = vs.plot_subplot(nrow, ncol, plot)
        ax.plot(mean, color='g', linestyle='--')
        ax.fill_between(X, mean - 2 * std, mean + 2 * std, alpha=0.2, color='b')
        if col == 0:
          ax.set_ylabel(code)
        if row == 0:
          ax.set_title(name)
        axes.append(ax)
        # factor
        y = y[ids]
        ax = ax.twinx()
        vs.plot_scatter_heatmap(x=X,
                                y=y,
                                val=y,
                                size=12,
                                colormap='bwr',
                                alpha=0.5)
        axes.append(ax)
        # update plot index
        for ax in axes:
          ax.tick_params(axis='both',
                         which='both',
                         top=False,
                         bottom=False,
                         left=False,
                         right=False,
                         labeltop=False,
                         labelleft=False,
                         labelright=False,
                         labelbottom=False)
        plot += 1
    fig.tight_layout()
    self.add_figure("uncertainty_stats", fig)
    return self

  def plot_uncertainty_scatter(self, factors=None, n_samples=2, algo='umap'):
    r"""
    factors : list of Integer or String. The index or name of factors taken
      into account for analyzing.
    """
    factors = self._check_factors(factors)
    # this all include tarin and test data separatedly
    z_mean = np.concatenate(self.representations_mean)
    z_var = np.concatenate(
        [np.mean(var, axis=1) for var in self.representations_variance])
    z_samples = [
        z for z in np.concatenate(self.representations_sample(int(n_samples)),
                                  axis=1)
    ]
    F = np.concatenate(self.original_factors, axis=0)[:, factors]
    labels = self.factor_name[factors]
    # preprocessing
    inputs = tuple([z_mean] + z_samples)
    Z = dimension_reduce(*inputs,
                         algo=algo,
                         n_components=2,
                         return_model=False,
                         random_state=self.randint)
    V = utils.discretizing(z_var[:, np.newaxis], n_bins=10).ravel()
    # the figure
    nrow = 3
    ncol = int(np.ceil(len(labels) / nrow))
    fig = vs.plot_figure(nrow=nrow * 4, ncol=ncol * 4, dpi=80)
    for idx, (name, y) in enumerate(zip(labels, F.T)):
      ax = vs.plot_subplot(nrow, ncol, idx + 1)
      for i, x in enumerate(Z):
        kw = dict(val=y,
                  colormap="coolwarm",
                  ax=ax,
                  x=x,
                  grid=False,
                  legend_enable=False,
                  fontsize=12)
        if i == 0:
          vs.plot_scatter_heatmap(size=V,
                                  size_range=(8., 64.),
                                  alpha=0.5,
                                  colorbar=True,
                                  colorbar_horizontal=True,
                                  title=name,
                                  **kw)
        else:
          vs.plot_scatter_heatmap(size=4, marker='x', alpha=0.4, **kw)
    fig.tight_layout()
    self.add_figure("uncertainty_scatter_%s" % algo, fig)
    return self

  ############## Methods for summarizing
  def __str__(self):
    text = [str(self._vae)]
    text.append(" Factor name: %s" % ', '.join(self.factor_name))
    for name, data in [
        ("Inputs", self.inputs),
        ("Factors", self.factors),
        ("Original Factors", self.original_factors),
        ("Representations", self.representations),
        ("Reconstructions", self.reconstructions),
    ]:
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
