import re
import warnings
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from odin.bay import distributions as tfd
from odin.bay.distributions import CombinedDistribution
from odin.bay.distributions.utils import concat_distribution
from odin.bay.vi import utils
from odin.bay.vi.autoencoder.variational_autoencoder import \
    VariationalAutoencoder
from odin.bay.vi.data_utils import Factor
from odin.stats import is_discrete
from odin.utils import as_tuple


def prepare_inputs_factors(inputs, latents, factors, verbose):
  if inputs is None:
    if latents is None:
      raise ValueError("Either inputs or latents must be provided")
    assert factors is not None, \
      "If latents is provided directly, factors must not be None."
    latents = tf.nest.flatten(latents)
    assert all(isinstance(z, tfd.Distribution) for z in latents), \
      ("All latents must be instance of Distribution but given: "
       f"{[type(z).__name__ for z in latents]}")
  ### inputs is a tensorflow Dataset, convert everything to numpy
  elif isinstance(inputs, tf.data.Dataset):
    struct = tf.data.experimental.get_structure(inputs)
    if isinstance(struct, dict):
      struct = struct['inputs']
    struct = tf.nest.flatten(struct)
    n_inputs = len(struct)
    if verbose:
      inputs = tqdm(inputs, desc="Reading data")
    if factors is None:  # include factors
      assert n_inputs >= 2, \
        "factors are not included in the dataset: %s" % str(inputs)
      x, y = [list() for _ in range((n_inputs - 1))], []
      for data in inputs:
        if isinstance(data, dict):  # this is an ad-hoc hack
          data = data['inputs']
        for i, j in enumerate(data[:-1]):
          x[i].append(j)
        y.append(data[-1])
      inputs = [tf.concat(i, axis=0).numpy() for i in x]
      if n_inputs == 2:
        inputs = inputs[0]
      factors = tf.concat(y, axis=0).numpy()
    else:  # factors separated
      x = [list() for _ in range(n_inputs)]
      for data in inputs:
        for i, j in enumerate(tf.nest.flatten(data)):
          x[i].append(j)
      inputs = [tf.concat(i, axis=0).numpy() for i in x]
      if n_inputs == 1:
        inputs = inputs[0]
      if isinstance(factors, tf.data.Dataset):
        if verbose:
          factors = tqdm(factors, desc="Reading factors")
        factors = tf.concat([i for i in factors], axis=0)
    # end the progress
    if isinstance(inputs, tqdm):
      inputs.clear()
      inputs.close()
  # post-processing
  else:
    inputs = tf.nest.flatten(inputs)
  assert len(factors.shape) == 2, "factors must be a matrix"
  return inputs, latents, factors


class CriticizerBase(object):

  def __init__(self,
               vae: VariationalAutoencoder,
               latent_indices=slice(None),
               random_state=1):
    super().__init__()
    assert isinstance(vae, VariationalAutoencoder), \
      "vae must be instance of odin.bay.vi.VariationalAutoencoder, given: %s" \
        % str(type(vae))
    self._vae = vae
    if latent_indices is None:
      latent_indices = slice(None)
    self._latent_indices = latent_indices
    if isinstance(random_state, Number):
      random_state = np.random.RandomState(seed=random_state)
    # main arguments
    self._inputs = None
    self._factors = None
    self._original_factors = None
    self._factor_names = None
    self._representations = None
    self._reconstructions = None
    # others
    self._rand = random_state
    self._is_multi_latents = 0

  @property
  def is_multi_latents(self):
    return self._is_multi_latents

  @property
  def is_sampled(self):
    if self._factors is None or self._representations is None:
      return False
    return True

  def assert_sampled(self):
    if not self.is_sampled:
      raise RuntimeError("Call the `sample_batch` method to sample mini-batch "
                         "of ground-truth data and learned representations.")

  @property
  def inputs(self):
    self.assert_sampled()
    return self._inputs

  @property
  def representations(self):
    r""" Return the learned representations `Distribution`
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
    r""" Return the reconstructed `Distributions` of inputs for training and
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
  def factor_names(self):
    self.assert_sampled()
    # the dataset is unchanged, always at 0-th index
    return np.array(self._factor_names)

  @property
  def code_names(self):
    return np.array([f"Z{i}" for i in range(self.n_representations)])

  @property
  def random_state(self):
    return self._rand

  @property
  def randint(self):
    return self._rand.randint(1e8)

  ############## proxy to VAE methods
  def index(self, factor_name):
    r""" Return the column index of given factor_names within the
    factor matrix """
    return self._factor_names.index(str(factor_name))

  def encode(self, inputs, mask=None, sample_shape=()):
    r""" Encode inputs to latent codes

    Arguments:
      inputs : a single Tensor or list of Tensor

    Returns:
      `tensorflow_probability.Distribution`, q(z|x) the latent distribution
    """
    inputs = tf.nest.flatten(inputs)[:len(self._vae.encoder.inputs)]
    latents = self._vae.encode(inputs[0] if len(inputs) == 1 else inputs,
                               training=False,
                               mask=mask,
                               sample_shape=sample_shape)
    # only support single returned latent variable now
    for z in tf.nest.flatten(latents):
      assert isinstance(z, tfd.Distribution), \
        "The latent code return from `vae.encode` must be instance of " + \
          "tensorflow_probability.Distribution, but returned: %s" % \
            str(z)
    return latents

  def decode(self, latents, mask=None, sample_shape=()):
    r""" Decode the latents into reconstruction distribution """
    outputs = self._vae.decode(latents,
                               training=False,
                               mask=mask,
                               sample_shape=sample_shape)
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

  def conditioning(self, known={}, logical_not=False, n_samples=None):
    r""" Conditioning the sampled dataset on known factors

    Arguments:
      known : a mapping from index or name of factor to a callable, the
        callable must return a list of boolean indices, which indicates
        the samples to be selected
      logical_not : a Boolean, if True applying the opposed conditioning
        of the known factors
      n_samples : an Integer (Optional), maximum number of selected samples.

    Return:
      a new `Criticizer` with the conditioned data and representations

    Example:
    ```
    # conditioning on: (1st-factor > 2) and (2nd-factor == 3)
    conditioning({1: lambda x: x > 2, 2: lambda x: x==3})
    ```
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
    # select n_samples
    if n_samples is not None:
      n_samples = int(n_samples)
      ratio = n_samples / (len(train_ids) + len(test_ids))
      train_ids = train_ids[:int(ratio * len(train_ids))]
      test_ids = test_ids[:int(ratio * len(test_ids))]
    # opposing the conditions
    if logical_not:
      train_ids = np.logical_not(train_ids)
      test_ids = np.logical_not(test_ids)
    # add new samples set to stack
    o_train, o_test = self.original_factors
    x_train = [x[train_ids] for x in x_train]
    x_test = [x[test_ids] for x in x_test]
    # convert boolean indices to integer
    z_train = self.encode(x_train)
    z_test = self.encode(x_test)
    r_train = self.decode(z_train)
    r_test = self.decode(z_test)
    if isinstance(z_train, (tuple, list)):
      z_train = z_train[self._latent_indices]
      z_test = z_test[self._latent_indices]
    if self.is_multi_latents:
      z_train = CombinedDistribution(z_train, name="LatentsTrain")
      z_test = CombinedDistribution(z_test, name="LatentsTest")
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
                   inputs=None,
                   latents=None,
                   factors=None,
                   n_bins=5,
                   strategy='quantile',
                   factor_names=None,
                   train_percent=0.8,
                   n_samples=[2000, 1000],
                   batch_size=64,
                   verbose=True):
    r""" Sample a batch of training and testing for evaluation of VAE

    Arguments:
      inputs : list of `ndarray` or `tensorflow.data.Dataset`.
        Inputs to the model, note all data will be loaded in-memory
      latents : list of `Distribution`
        distribution of learned representation
      factors : a `ndarray` or `tensorflow.data.Dataset`.
        a matrix of groundtruth factors, note all data will be loaded in-memory
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
      factor_names :
      train_percent :
      n_samples :
      batch_size :

    Returns:
      `Criticizer` with sampled data
    """
    inputs, latents, factors = prepare_inputs_factors(inputs,
                                                      latents,
                                                      factors,
                                                      verbose=verbose)
    n_samples = as_tuple(n_samples, t=int, N=2)
    n_inputs = factors.shape[0]
    # ====== split train test ====== #
    if inputs is None:
      latents = latents[self._latent_indices]
      split = int(n_inputs * train_percent)
      train_ids = slice(None, split)
      test_ids = slice(split, None)
      train_latents = [z[train_ids] for z in latents]
      test_latents = [z[test_ids] for z in latents]
      if len(latents) == 1:
        train_latents = train_latents[0]
        test_latents = test_latents[0]
      else:
        self._is_multi_latents = len(latents)
        train_latents = CombinedDistribution(train_latents, name="Latents")
        test_latents = CombinedDistribution(test_latents, name="Latents")
    else:
      ids = self.random_state.permutation(n_inputs)
      split = int(train_percent * n_inputs)
      train_ids, test_ids = ids[:split], ids[split:]
      train_inputs = [i[train_ids] for i in inputs]
      test_inputs = [i[test_ids] for i in inputs]
    # ====== create discretized factors ====== #
    f_original = (factors[train_ids], factors[test_ids])
    if not is_discrete(factors):
      if verbose:
        print(f"Discretizing factors: {n_bins} - {strategy}")
      factors = utils.discretizing(factors,
                                   n_bins=int(n_bins),
                                   strategy=strategy)
    train_factors = Factor(factors[train_ids],
                           factor_names=factor_names,
                           random_state=self.randint)
    test_factors = Factor(factors[test_ids],
                          factor_names=factor_names,
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
        z = self.encode(inps, sample_shape=())
        o = tf.nest.flatten(self.decode(z))
        if isinstance(z, (tuple, list)):
          z = z[self._latent_indices]
          if len(z) == 1:
            z = z[0]
          else:
            self._is_multi_latents = len(z)
        Os.append(o)
        Zs.append(z)
        # update the counter
        n += len(y)
      # end progress
      if verbose:
        prog.clear()
        prog.close()
      # aggregate all data
      Xs = [np.concatenate(x, axis=0) for x in Xs]
      Ys = np.concatenate(Ys, axis=0)
      if self.is_multi_latents:
        Zs = CombinedDistribution(
            [
                concat_distribution(
                    [z[zi] for z in Zs],
                    name="Latents%d" % zi,
                ) for zi in range(self.is_multi_latents)
            ],
            name="Latents",
        )
      else:
        Zs = concat_distribution(Zs, name="Latents")
      Os = [
          concat_distribution(
              [j[i] for j in Os],
              name="Output%d" % i,
          ) for i in range(len(Os[0]))
      ]
      return Xs, Ys, Zs, Os, np.concatenate(indices, axis=0)

    # perform sampling
    if inputs is not None:
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
      self._inputs = (train[0], test[0])
      self._factors = (train[1], test[1])
      self._representations = (train[2], test[2])
      self._reconstructions = (train[3], test[3])
      self._original_factors = (f_original[0][ids_train],
                                f_original[1][ids_test])
    else:
      self._inputs = (None, None)
      self._factors = (train_factors.factors, test_factors.factors)
      self._representations = (train_latents, test_latents)
      self._reconstructions = (None, None)
      self._original_factors = (f_original[0], f_original[1])
    self._factor_names = train_factors.factor_names
    return self
