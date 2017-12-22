# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'
"""
import time
import random
import threading

import numpy as np
import tensorflow as tf

from sklearn.base import DensityMixin, BaseEstimator, TransformerMixin

from odin import backend as K
from odin.fuel import Data, DataDescriptor, Feeder
from odin.utils import (MPI, batching, ctext, cpu_count, Progbar,
                        is_number, as_tuple, uuid, is_string)
from odin.config import EPS, get_ngpu


# minimum batch size that will be optimal to transfer
# the data to GPU for calculation (tested on Titan X)
# NOTE: tensorflow has a lagging effect, it will be
#       slower than numpy if you evaluate the
#       expression for first time.
MINIMUM_GPU_BLOCK = 8000 * 120 * 4 # bytes
STANDARD_BATCH_SIZE = 8 * 1024 * 1024 # 8 Megabytes


# ===========================================================================
# Helper
# ===========================================================================
def zeroStat(post):
  """ Shape: (1, nmix)
  # sum over all samples
  # Zero-order statistics over all samples and dimension for
  each components
  """
  # ====== tensorflow tensor or variable ====== #
  if K.is_tensor(post, inc_distribution=True, inc_variable=True):
    y = tf.reduce_sum(post, axis=0, keep_dims=True,
                      name="zero_stat")
  # ====== numpy array ====== #
  else:
    y = np.sum(post, axis=0, keepdims=True) # (1, M)
  return y


def firstStat(X, post):
  """ Shape: (feat_dim, nmix)
  First-order statistics over all samples for each components
  """
  # ====== tensorflow tensor or variable ====== #
  if K.is_tensor(X, inc_distribution=True, inc_variable=True):
    y = tf.matmul(tf.transpose(X), post, name='first_stat')
  # ====== numpy array ====== #
  else:
    y = np.dot(X.T, post)
  return y


def secondStat(X, post):
  """ Shape: (feat_dim, nmix)
  Second-order statistics over all samples for each components
  """
  # ====== tensorflow tensor or variable ====== #
  if K.is_tensor(X, inc_distribution=True, inc_variable=True):
    y = tf.matmul(tf.transpose(tf.pow(X, 2)), post, name="second_stat")
  # ====== numpy array ====== #
  else:
    y = np.dot((X ** 2).T, post)
  return y


def logsumexp(X, axis):
  """
  Compute log(sum(exp(x),dim)) while avoiding numerical underflow
  """
  # ====== tensorflow tensor or variable ====== #
  if K.is_tensor(X, inc_distribution=True, inc_variable=True):
    xmax = tf.reduce_max(X, axis=axis, keep_dims=True)
    y = tf.add_n(inputs=[
        xmax,
        tf.log(tf.reduce_sum(input_tensor=tf.exp(X - xmax),
                             axis=axis,
                             keep_dims=True))],
        name='llk')
  # ====== numpy array ====== #
  else:
    xmax = np.max(X, axis=axis, keepdims=True)
    y = xmax + np.log(np.sum(a=np.exp(X - xmax),
                             axis=axis,
                             keepdims=True))
  return y


def _split_jobs(nb_samples, gpu_factor, ncpu, device):
  # number of GPU
  ngpu = get_ngpu()
  if ngpu == 0:
    device = 'cpu'
  # jobs split based on both number of CPU and GPU
  if device == 'mix':
    njob = ncpu + 1 + ngpu * gpu_factor
  elif device == 'cpu':
    njob = ncpu + 1
  elif device == 'gpu':
    njob = ngpu + 1
  jobs = np.linspace(start=0, stop=nb_samples,
                     num=njob, dtype='int32')
  jobs = list(zip(jobs, jobs[1:]))
  # use both GPU and CPU
  if device == 'mix':
    jobs_gpu = [(jobs[i * gpu_factor][0],
                 jobs[i * gpu_factor + gpu_factor - 1][1])
                for i in range(ngpu)]
    jobs_cpu = [jobs[i] for i in range(ngpu * gpu_factor, len(jobs))]
  elif device == 'gpu': # only GPU
    jobs_gpu = jobs
    jobs_cpu = []
  elif device == 'cpu': # only CPU
    jobs_gpu = []
    jobs_cpu = jobs
  return jobs_cpu, jobs_gpu


def _create_batch(start, end, batch_size,
                  downsample, stochastic,
                  seed, curr_nmix, curr_niter):
  """ Return:
  (start, end, nframe, is_batch_selected)
  """
  # stochastic downsasmple, seed change every iter and mixup
  if stochastic:
    random.seed(seed + curr_nmix + curr_niter)
  else: # deterministic
    random.seed(seed)
  all_batches = list(batching(n=end - start, batch_size=batch_size))
  random.shuffle(all_batches)
  # iterate over batches
  for batch_id, (s, e) in enumerate(all_batches):
    selected = False
    n = e - s
    s += start
    e += start
    # first batch always selected,
    # downsample by randomly ignore a batch
    if batch_id == 0 or \
    downsample == 1 or \
    (downsample > 1 and random.random() <= 1. / downsample):
      selected = True
    yield s, e, n, selected


def _select_device(X, gpu):
  if is_string(gpu) and gpu.lower() == 'auto':
    n = np.prod(X.shape) * 4 # assume always float32
    if n >= MINIMUM_GPU_BLOCK:
      gpu = True
    else:
      gpu = False
  gpu &= (get_ngpu() > 0)
  return gpu


# ===========================================================================
# Main GMM
# ===========================================================================
class _ExpectationResults(object):
  """ ExpectationResult """

  def __init__(self, nb_samples, name):
    super(_ExpectationResults, self).__init__()
    # thread lock
    self.lock = threading.Lock()
    # progress bar
    self.prog = Progbar(target=nb_samples, print_report=True,
                        print_summary=False, name=name)
    # Z, F, S, L, nfr
    self.stats = [0., 0., 0., 0., 0]

  def update(self, res):
    # thread-safe udpate
    self.lock.acquire()
    try:
      # returned number of processed samples
      if is_number(res):
        self.prog.add(res)
      # return the statistics, end of process
      else:
        for i, r in enumerate(res):
          self.stats[i] += r
    finally:
      self.lock.release()


class GMM(DensityMixin, BaseEstimator, TransformerMixin):
  """ The following symbol is used:
  N: number of samples (frames)
  D: number of features dimension
  M: current number of mixture

  Parameters
  ----------
  batch_size : {int, 'auto'}
      if 'auto', used `8 Megabytes` block for batch size.
  covariance_type : {'full', 'tied', 'diag', 'spherical'},
          defaults to 'full'.
      String describing the type of covariance parameters to use.
      Must be one of::
          'full' (each component has its own general covariance matrix),
          'tied' (all components share the same general covariance matrix),
          'diag' (each component has its own diagonal covariance matrix),
          'spherical' (each component has its own single variance).
  device : {'gpu', 'cpu', 'mix'}
      which devices using for the EM
      'gpu' (only run on tensorflow implementation using GPU)
      'cpu' (only run on numpy implemetation using CPU)
      'mix' (using both GPU and CPU)
      * It is suggested to use mix of GPU and CPU if you have
        more than 24 cores CPU, and keep the `gpu_factor` high-enough.
  gpu_factor : int (> 0)
      how much jobs GPU will handle more than CPU
      (i.e. `njob_gpu = gpu_factor * njob_cpu`)
  ncpu : int
      number of processes for parallel calculating Expectation
  stochastic_downsample : bool
      if True, a subset of data is selected differently after
      each iteration => the training is stochastic.
      if False, a deterministic selection of data is performed
      each iteration => the training is deterministic.
  seed : int
      random seed for reproducible
  dtype : {str, numpy.dtype}
      desire dtype for mean, std, weights and input matrices
  """

  def __init__(self, nmix, nmix_start=1, niter=16, batch_size='auto',
               covariance_type='diag',
               downsample=2, stochastic_downsample=True, seed=5218,
               device='gpu', gpu_factor=80, ncpu=None,
               dtype='float32', name=None):
    super(GMM, self).__init__()
    # start from 1 mixture, then split and up
    self._nmix = 2**int(np.round(np.log2(nmix)))
    self._curr_nmix = np.clip(int(nmix_start), 1, self._nmix)
    self._feat_dim = None
    self._niter = int(niter)
    self.batch_size = batch_size
    # ====== downsample ====== #
    self.downsample = int(downsample)
    self.stochastic_downsample = bool(stochastic_downsample)
    self.seed = int(seed)
    # ====== different mode ====== #
    self.covariance_type = str(covariance_type)
    # ====== multi-processing ====== #
    # cpu
    if ncpu is None:
      ncpu = cpu_count() - 1
    self.ncpu = int(ncpu)
    # gpu
    self.gpu_factor = int(gpu_factor)
    # device
    device = str(device).lower()
    if device not in ('mix', 'gpu', 'cpu'):
      raise ValueError("`device` can only be one of following option±: "
                       "'mix', 'gpu', 'cpu'.")
    self.device = device
    # ====== state variable ====== #
    self._llk = [] # store history of LLK
    # ====== name ====== #
    self.dtype = dtype
    if name is None:
      name = uuid(length=8)
      self._name = 'GMM_%s' % name
    else:
      self._name = str(name)

  def __getstate__(self):
    # 'means', 'variances', 'weights'
    # self.mu, self.sigma, self.w
    if not self.is_initialized:
      raise RuntimeError("GMM hasn't been initialized, nothing to save")
    pass

  def __setstate__(self, states):
    pass

  def __str__(self):
    if not self.is_initialized:
      return '<"%s" nmix:%d initialized:False>' % (self.name, self._nmix)
    s = '<"%s" nmix:%s ndim:%s mean:%s std:%s w:%s>' %\
        (ctext(self.name, 'yellow'),
            ctext(self._nmix, 'cyan'),
            ctext(self._feat_dim, 'cyan'),
            ctext(self.mu.shape, 'cyan'),
            ctext(self.sigma.shape, 'cyan'),
            ctext(self.w.shape, 'cyan'))
    return s

  # ==================== properties ==================== #
  @property
  def name(self):
    return self._name

  @property
  def is_initialized(self):
    return self._feat_dim is not None

  @property
  def is_fitted(self):
    return self._curr_nmix == self._nmix

  @property
  def nmix(self):
    return self._nmix

  @property
  def feat_dim(self):
    if not self.is_initialized:
      raise RuntimeError("GMM has not been initialized on data.")
    return self._feat_dim

  # ==================== initialization ==================== #
  def _resfresh_cpu_posterior(self):
    """ Refresh cached value for CPu computations. """
    expressions = {}
    precision = 1 / (self.sigma + EPS)
    C = np.sum((self.mu ** 2) * precision, axis=0, keepdims=True) + \
        np.sum(np.log(self.sigma + EPS), axis=0, keepdims=True) - \
        2 * np.log(self.w)
    mu_precision = self.mu * precision
    expressions['precision'] = precision
    expressions['mu_precision'] = mu_precision
    expressions['C'] = C
    self.__expressions_cpu = expressions

  def _refresh_gpu_posterior(self):
    """ Call this function when you update the mixture
    components.
    Unlike CPU computation, tensorflow graph on need to
    renew it placeholder which represent: mu, sigma, weight
    when GMM mixup.
    """
    expressions = {}
    # ====== proper scope ====== #
    if self._curr_nmix < self.nmix:
      scope = self.name + str(self._curr_nmix)
    else:
      scope = self.name
    # ====== build the graph ====== #
    with tf.variable_scope(scope):
      mu = tf.placeholder(shape=(self.feat_dim, self._curr_nmix),
                          dtype=self.dtype,
                          name='GMM_mu')
      sigma = tf.placeholder(shape=(self.feat_dim, self._curr_nmix),
                            dtype=self.dtype,
                            name='GMM_sigma')
      w = tf.placeholder(shape=(1, self._curr_nmix),
                        dtype=self.dtype,
                        name='GMM_weight')
      expressions['mu'] = mu
      expressions['sigma'] = sigma
      expressions['w'] = w
      # ====== log probability ====== #
      # (feat_dim, nmix)
      precision = 1 / (sigma + EPS)
      C = tf.reduce_sum((mu ** 2) * precision,
                        axis=0, keep_dims=True) + \
          tf.reduce_sum(tf.log(sigma + EPS),
                        axis=0, keep_dims=True) - \
          2 * tf.log(w)
      D = tf.matmul(self.X_ ** 2, precision) - \
          2 * tf.matmul(self.X_, mu * precision) + \
          self.feat_dim * np.log(2 * np.pi)
      # (batch_size, nmix)
      logprob = tf.multiply(x=-0.5, y=C + D,
                            name='logprob')
      expressions['logprob'] = logprob # (batch_size, nmix)
      # ====== posterior and likelihood ====== #
      llk = logsumexp(logprob, axis=1) # (batch_size, 1)
      post = tf.exp(logprob - llk, name='postprob') # (batch_size, nmix)
      expressions['llk'] = llk
      expressions['post'] = post
      # ====== expectation ====== #
      expressions['zero'] = zeroStat(post)
      expressions['first'] = firstStat(self.X_, post)
      expressions['second'] = secondStat(self.X_, post)
      expressions['L'] = tf.reduce_sum(llk, axis=None, name='sum_llk')
    self.__expressions_gpu = expressions

  def _initialize(self, X):
    # ====== get input info ====== #
    if hasattr(X, 'ndim'):
      ndim = X.ndim
    elif hasattr(X, 'get_shape'):
      ndim = len(X.get_shape().as_list())
    else:
      raise ValueError("Cannot number of dimension from input.")

    if hasattr(X, 'shape'):
      feat_dim = X.shape[1]
    elif hasattr(X, 'get_shape'):
      feat_dim = X.get_shape().as_list()[1]
    else:
      raise ValueError("Cannot get feature dimension from input.")
    # ====== already init ====== #
    if self.is_initialized:
      # validate the inputs
      if ndim != 2 or feat_dim != self._feat_dim:
        raise RuntimeError("Input must be 2-D matrix with the 1st "
            "dimension equal to: %d" % feat_dim)
      return
    # ====== create input placeholder ====== #
    self._feat_dim = int(feat_dim)
    self._feat_const = self.feat_dim * np.log(2 * np.pi)
    if is_string(self.batch_size):
      self.batch_size = int(STANDARD_BATCH_SIZE / self.feat_dim * 4)
    # [batch_size, feat_dim]
    self.X_ = tf.placeholder(shape=(None, self.feat_dim),
                             dtype='float32',
                             name='GMM_input')
    # ====== init ====== #
    # (D, M)
    self.mu = np.zeros((feat_dim, self._curr_nmix), dtype=self.dtype)
    # (D, M)
    self.sigma = np.ones((feat_dim, self._curr_nmix), dtype=self.dtype)
    # (1, M)
    self.w = np.ones((1, self._curr_nmix), dtype=self.dtype)
    # init posterior
    self._resfresh_cpu_posterior()
    self._refresh_gpu_posterior()

  # ==================== sklearn ==================== #
  def fit(self, X, y=None):
    """
    NOTE
    ----
    from 1-8 components, python multi-threading is fastest
    from 16-32 components, python multi-processing is fastest
    from > 64 components, GPU scales much much better.
    """
    if not isinstance(X, (Data, np.ndarray)):
      raise ValueError("`X` must be numpy.ndarray or instance of odin.fuel.Data.")
    if isinstance(X, Feeder):
      raise ValueError("No support for fitting GMM on odin.fuel.Feeder")
    # ====== check input ====== #
    self._initialize(X)
    if X.shape[0] < self.batch_size:
      raise RuntimeError("Input has shape %s, not enough data points for a "
                         "single batch of size: %d." %
                         (str(X.shape), self.batch_size))
    is_indices = True if isinstance(X, DataDescriptor) else False
    # ====== divide the batches ====== #
    nb_samples = len(X.indices) if is_indices else X.shape[0]

    # ====== mapping method ====== #
    def map_expectation(start_end_dev_bs):
      (start, end), device, batch_size = start_end_dev_bs
      # Z, F, S, L, nfr
      results = [0., 0., 0., 0., 0]
      for s, e, n, selected in _create_batch(start, end, batch_size,
                                  downsample=self.downsample,
                                  stochastic=self.stochastic_downsample,
                                  seed=self.seed,
                                  curr_nmix=self._curr_nmix,
                                  curr_niter=self._curr_niter):
        # downsample by randomly ignore a batch
        if selected:
          x = X[s:e]
          for i, res in enumerate(self.expectation(x, gpu=device)):
            results[i] += res
          results[-1] += x.shape[0]
        yield n
      yield tuple(results)

    def thread_expectation(results, start_end, device):
      if device == 'cpu':
        device = False
        batch_size = self.batch_size
      elif device == 'gpu':
        device = True
        batch_size = max(int(MINIMUM_GPU_BLOCK / self.feat_dim / 4),
                         self.batch_size)
      for res in map_expectation((start_end, device, batch_size)):
        results.update(res)

    # ====== start GMM ====== #
    # supports 16384 components, modify for more components
    niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10, 16, 16]
    niter[int(np.log2(self._nmix))] = self._niter
    # run the algorithm
    while True:
      # spliting the jobs
      jobs_cpu, jobs_gpu = _split_jobs(nb_samples=nb_samples,
                  gpu_factor=self.gpu_factor, ncpu=self.ncpu,
                  device='cpu' if self._curr_nmix <= 32 else self.device)
      # fitting the mixtures
      idx = int(np.log2(self._curr_nmix))
      print(ctext('Estimating the GMM for %s components in %d iter ...' %
                  (self._curr_nmix, niter[idx]),
                  color='cyan'))
      for self._curr_niter in range(niter[idx]):
        # ====== init ====== #
        start_time = time.time()
        # Z, F, S, L, nfr
        results = _ExpectationResults(nb_samples=nb_samples,
            name="[GMM] cmix:%d nmix:%d iter:%d" %
                       (self._curr_nmix, self.nmix, self._curr_niter))
        # ====== run multiprocessing ====== #
        # create CPU processes
        mpi = []
        cpu_threads = []
        if len(jobs_cpu) > 0:
          if 0 <= self._curr_nmix < 16:
            cpu_threads = [threading.Thread(target=thread_expectation,
                                            args=(results, j, 'cpu'))
                           for j in jobs_cpu]
          elif 16 <= self._curr_nmix:
            mpi = MPI(jobs=[(j, False, self.batch_size)
                            for j in jobs_cpu],
                      func=map_expectation,
                      ncpu=self.ncpu, batch=1, hwm=2**25,
                      backend='python')
        # create GPU threads
        gpu_threads = [threading.Thread(target=thread_expectation,
                                        args=(results, j, 'gpu'))
                       for j in jobs_gpu]
        # start gpu and cpu threads
        for t in gpu_threads + cpu_threads:
          t.start()
        # start the cpu processes
        for res in mpi:
          results.update(res)
        # finish all threads
        for t in gpu_threads + cpu_threads:
          t.join()
        # ====== Maximization ====== #
        Z, F, S, L, nfr = results.stats
        self.maximization(Z, F, S)
        # print Log-likelihood
        self._llk.append(L / nfr)
        print("#iter:", ctext('%.2d' % (self._curr_niter + 1), 'yellow'),
              "llk:", ctext('%.4f' % self._llk[-1], 'yellow'),
              "%.2f(s)" % (time.time() - start_time))
        # release memory
        del Z, F, S
      # update the mixtures
      if self._curr_nmix < self._nmix:
        self.gmm_mixup()
      else:
        break
    return self

  def score(self, X, y=None, gpu='auto'):
    """ Compute the log-likelihood of each example to
    the Mixture of Components.
    """
    post = self.logprob(X, gpu=gpu)  # (batch_size, nmix)
    return logsumexp(post, axis=1) # (batch_size, 1)

  def transform(self, X, gpu='auto'):
    """ Compute centered statistics given X and fitted mixtures


    NOTE
    ----
    For more option check `GMM.expectation`
    """
    Z, F = self.expectation(X, gpu=gpu,
                            zero=True, first=True,
                            second=False, llk=False)
    # this equal to: .ravel()[:, np.newaxis]
    F_hat = np.reshape(F - self.mu * Z,
                       newshape=(self._feat_dim * self._curr_nmix, 1))
    return Z, F_hat

  # ==================== math helper ==================== #
  def logprob(self, X, gpu='auto'):
    """ Shape: [batch_size, nmix]
    the log probability of each observations to each components
    given the GMM.
    """
    self._initialize(X)
    gpu = _select_device(X, gpu)
    if gpu:
      feed_dict = {self.X_: X}
      feed_dict[self.__expressions_gpu['mu']] = self.mu
      feed_dict[self.__expressions_gpu['sigma']] = self.sigma
      feed_dict[self.__expressions_gpu['w']] = self.w
      return K.eval(x=self.__expressions_gpu['logprob'],
                    feed_dict=feed_dict)
    # ====== run on numpy ====== #
    # (feat_dim, nmix)
    precision = self.__expressions_cpu['precision']
    mu_precision = self.__expressions_cpu['mu_precision']
    C = self.__expressions_cpu['C']
    X_2 = X ** 2
    D = np.dot(X_2, precision) - \
        2 * np.dot(X, mu_precision) + \
        self._feat_const
    # (batch_size, nmix)
    logprob = -0.5 * (C + D)
    return logprob

  def postprob(self, X, gpu='auto'):
    """ Shape: (batch_size, nmix)
    The posterior probability of mixtures for each frame
    """
    self._initialize(X)
    gpu = _select_device(X, gpu)
    if gpu:
      feed_dict = {self.X_: X}
      feed_dict[self.__expressions_gpu['mu']] = self.mu
      feed_dict[self.__expressions_gpu['sigma']] = self.sigma
      feed_dict[self.__expressions_gpu['w']] = self.w
      return K.eval(x=self.__expressions_gpu['post'],
                    feed_dict=feed_dict)
    # ====== run on numpy ====== #
    # (feat_dim, nmix)
    precision = self.__expressions_cpu['precision']
    mu_precision = self.__expressions_cpu['mu_precision']
    C = self.__expressions_cpu['C']
    X_2 = X ** 2
    D = np.dot(X_2, precision) - \
        2 * np.dot(X, mu_precision) + \
        self._feat_const
    # (batch_size, nmix)
    logprob = -0.5 * (C + D)
    # ====== posterior and likelihood ====== #
    llk = logsumexp(logprob, axis=1) # (batch_size, 1)
    post = np.exp(logprob - llk) # (batch_size, nmix)
    return post

  def llk(self, X, gpu='auto'):
    """ Shape: (batch_size, 1)
    The log-likelihood value of each frame to all components
    """
    self._initialize(X)
    gpu = _select_device(X, gpu)
    if gpu:
      feed_dict = {self.X_: X}
      feed_dict[self.__expressions_gpu['mu']] = self.mu
      feed_dict[self.__expressions_gpu['sigma']] = self.sigma
      feed_dict[self.__expressions_gpu['w']] = self.w
      return K.eval(x=self.__expressions_gpu['llk'],
                    feed_dict=feed_dict)
    # ====== run on numpy ====== #
    # (feat_dim, nmix)
    precision = self.__expressions_cpu['precision']
    mu_precision = self.__expressions_cpu['mu_precision']
    C = self.__expressions_cpu['C']
    X_2 = X ** 2
    D = np.dot(X_2, precision) - \
        2 * np.dot(X, mu_precision) + \
        self._feat_const
    # (batch_size, nmix)
    logprob = -0.5 * (C + D)
    # ====== posterior and likelihood ====== #
    llk = logsumexp(logprob, axis=1) # (batch_size, 1)
    return llk

  def expectation(self, X, gpu='auto',
                  zero=True, first=True, second=True,
                  llk=True):
    """
    Parameters
    ----------
    X : numpy.ndarray [batch_size, feat_dim]
        input array, with feature dimension is the final dimension
    gpu : {True, False, 'auto'}
        if True, always perform calculation on GPU
        if False, always on CPU
        if 'auto', based on shape of `X`, select optimial method
        (numpy on CPU, or tensorflow on GPU)
    zero : bool (default: True)
        if True, return zero-order statistics
    first : bool (default: True)
        if True, return first-order statistics
    second : bool (default: True)
        if True, return second-order statistics
    llk : bool (default: True)
        if True, return log-likelihood

    Return
    ------
    The order of return value:
    zero  (optional) : ndarray [1, nmix]
    first (optional) : ndarray [feat_dim, nmix]
    second(optional) : ndarray [feat_dim, nmix]
    llk   (optional) : scalar ()
    """
    self._initialize(X)
    gpu = _select_device(X, gpu)
    # ====== run on GPU ====== #
    if gpu:
      Z, F, S, L = [self.__expressions_gpu[name]
                    for name in ('zero', 'first', 'second', 'L')]
      feed_dict = {self.X_: X}
      feed_dict[self.__expressions_gpu['mu']] = self.mu
      feed_dict[self.__expressions_gpu['sigma']] = self.sigma
      feed_dict[self.__expressions_gpu['w']] = self.w
      outputs = [i for i, j in zip((Z, F, S, L),
                                   (zero, first, second, llk))
                 if j]
      results = K.eval(x=outputs, feed_dict=feed_dict)
    # ====== run on numpy ====== #
    else:
      results = []
      # (feat_dim, nmix)
      precision = self.__expressions_cpu['precision']
      mu_precision = self.__expressions_cpu['mu_precision']
      C = self.__expressions_cpu['C']
      X_2 = X ** 2
      D = np.dot(X_2, precision) - \
          2 * np.dot(X, mu_precision) + \
          self._feat_const
      # (batch_size, nmix)
      logprob = -0.5 * (C + D)
      # ====== posterior and likelihood ====== #
      LLK = logsumexp(logprob, axis=1) # (batch_size, 1)
      post = np.exp(logprob - LLK) # (batch_size, nmix)
      # ====== expectation ====== #
      if zero:
        Z = zeroStat(post)
        results.append(Z)
      if first:
        F = firstStat(X, post)
        results.append(F)
      if second:
        S = np.dot(X_2.T, post) # dont calculate X**2 again
        results.append(S)
      if llk:
        L = np.sum(LLK, axis=None)
        results.append(L)
    # ====== return ====== #
    return results if len(results) > 1 else results[0]

  def maximization(self, Z, F, S, floor_const=None):
    """
    Parameters
    ----------
    Z : numpy.ndarray (1, nmix)
        zero statistics
    F : numpy.ndarray (feat_dim, nmix)
        first-order statistics
    S : numpy.ndarray (feat_dim, nmix)
        second-order statistics
    floor_const : {None, small float}
        numerical stablize the sigma (e.g. 1e-3)
    """
    # TheReduce
    iN = 1. / (Z + EPS)
    self.w = Z / Z.sum()
    self.mu = F * iN
    self.sigma = S * iN - self.mu * self.mu
    # applying variance floors
    if floor_const is not None:
      vFloor = self.sigma.dot(self.w.T) * floor_const
      self.sigma = self.sigma.clip(vFloor)
    # refresh cpu cached value
    self._resfresh_cpu_posterior()

  def gmm_mixup(self):
    if self._curr_nmix >= self._nmix:
      return
    # ====== double up the components ====== #
    ndim, nmix = self.sigma.shape
    sig_max, arg_max = self.sigma.max(0), self.sigma.argmax(0)
    eps = np.zeros((ndim, nmix), dtype='f')
    eps[arg_max, np.arange(nmix)] = np.sqrt(sig_max)
    perturb = 0.55 * eps
    self.mu = np.c_[self.mu - perturb, self.mu + perturb]
    self.sigma = np.c_[self.sigma, self.sigma]
    self.w = 0.5 * np.c_[self.w, self.w]
    # update information
    self._curr_nmix = min(2 * self._curr_nmix, self.nmix)
    self._refresh_gpu_posterior()
    self._resfresh_cpu_posterior()
