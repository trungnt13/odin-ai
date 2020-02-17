from __future__ import absolute_import, division, print_function

import warnings
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow_probability import distributions as tfd
from tqdm import tqdm

from odin.bay.distributions.utils import concat_distribution
from odin.bay.vi import metrics, utils
from odin.bay.vi.data_utils import Factor
from odin.bay.vi.variational_autoencoder import VariationalAutoencoder


class Criticizer():

  def __init__(self, vae: VariationalAutoencoder):
    super().__init__()
    assert isinstance(vae, VariationalAutoencoder)
    self._vae = vae
    self._reset()

  def _assert_generated(self):
    if self._inputs is None or \
      self._representations is None or \
        self._factors is None:
      raise RuntimeError(
          "Call generate_batch to sample mini-batch of ground-truth data.")

  def _reset(self):
    self._representations = None
    self._factors = None
    self._inputs = None
    self._rand = None
    self._n_samples = None

  @property
  def inputs(self):
    self._assert_generated()
    return self._inputs

  @property
  def n_samples(self):
    self._assert_generated()
    return self._n_samples

  @property
  def representations(self):
    self._assert_generated()
    return self._representations

  @property
  def factors(self):
    self._assert_generated()
    return self._factors

  ############## Experiment setup
  @contextmanager
  def latent_traversal(self):
    pass

  @contextmanager
  def sampling_batch(self,
                     inputs,
                     factors,
                     discretizing=False,
                     n_bins=5,
                     strategy='quantile',
                     train_percent=0.8,
                     n_samples=1000,
                     batch_size=16,
                     seed=1):
    if discretizing:
      factors = utils.discretizing(factors,
                                   n_bins=int(n_bins),
                                   strategy=strategy)
    assert len(factors.shape) == 2, "factors must be a matrix"
    rand = np.random.RandomState(seed=seed)
    self._rand = rand
    is_list_inputs = isinstance(inputs, (tuple, list))
    inputs = tf.nest.flatten(inputs)
    # ====== split train test ====== #
    ids = rand.permutation(factors.shape[0])
    split = int(train_percent * factors.shape[0])
    train_ids, test_ids = ids[:split], ids[split:]
    train_inputs = [i[train_ids] for i in inputs]
    test_inputs = [i[test_ids] for i in inputs]
    train_factors = Factor(factors[train_ids], random_state=rand.randint(1e8))
    test_factors = Factor(factors[test_ids], random_state=rand.randint(1e8))

    # ====== sampling ====== #
    def sampling(inputs_, factors_):
      Xs, Ys = [list() for _ in range(len(inputs))], []
      Zs = []
      n = 0
      while n < n_samples:
        batch = min(batch_size, n_samples - n, factors_.shape[0])
        # factors
        y, ids = factors_.sample_factors(num=batch, return_indices=True)
        Ys.append(y)
        # inputs
        inps = []
        for x, i in zip(Xs, inputs_):
          i = i[ids, :]
          x.append(i)
          inps.append(i)
        # latents representation
        z = self._vae.encode(inps if is_list_inputs else inps[0],
                             training=False,
                             n_mcmc=1)
        if isinstance(z, (tuple, list)):
          z = z[0]
        assert isinstance(z, tfd.Distribution), \
          "The latent code return from `vae.encode` must be instance of " + \
            "tensorflow_probability.Distribution, but returned: %s" % str(type(z))
        Zs.append(z)
        # update the couter
        n += len(y)
      # aggregate all data
      Xs = [np.concatenate(x, axis=0) for x in Xs]
      Ys = np.concatenate(Ys, axis=0)
      Zs = concat_distribution(Zs, name="Latents")
      return Xs, Ys, Zs

    # assign the variables
    train = sampling(inputs_=train_inputs, factors_=train_factors)
    test = sampling(inputs_=test_inputs, factors_=test_factors)
    self._representations = (train[2], test[2])
    self._inputs = (train[0], test[0]) if is_list_inputs else \
      (train[0][0], test[0][0])
    self._factors = (train[1], test[1])
    self._n_samples = int(n_samples)
    yield self
    self._reset()

  ############## Metrics
  def _latent_codes(self, mean=True):
    if mean:
      return [i.mean().numpy() for i in self.representations]
    return [
        i.sample(sample_shape=(), seed=self._rand.randint(1e8)).numpy()
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
    sap = metrics.separated_attr_predictability(
        z_train,
        f_train,
        z_test,
        f_test,
        continuous_factors=False,
        random_state=self._rand.randint(1e8))
    return sap

  def importance_matrix(self, mean=True, algo=GradientBoostingClassifier):
    r""" Using ensemble algorithm to estimate the feature importance of each
    pair of (representation, factor)

    Return:
      matrix `[num_latents, num_factors]`
    """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    importance_matrix, _, _ = \
      metrics.representation_importance_matrix(
        z_train, f_train, z_test, f_test,
        random_state=self._rand.randint(1e8), algo=algo)
    return importance_matrix

  def dci_scores(self, mean=True):
    r""" Disentanglement, Completeness, Informativeness """
    z_train, z_test = self._latent_codes(mean)
    f_train, f_test = self.factors
    return metrics.dci_scores(z_train,
                              f_train,
                              z_test,
                              f_test,
                              random_state=self._rand.randint(1e8))

  def total_correlation(self):
    r""" Total correlation based on fitted Gaussian """
    samples = [
        qz.sample(seed=self._rand.randint(1e8)) for qz in self.representations
    ]
    return tuple([
        utils.total_correlation(z, qz).numpy()
        for z, qz in zip(samples, self.representations)
    ])
