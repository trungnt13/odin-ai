# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'
"""
import os
import time
import random
import threading
from collections import OrderedDict, defaultdict, Mapping

import numpy as np
from scipy import linalg
import tensorflow as tf

from sklearn.base import DensityMixin, BaseEstimator, TransformerMixin

from odin import backend as K
from odin.fuel import Data, DataDescriptor, Feeder
from odin.utils import (MPI, batching, ctext, cpu_count, Progbar,
                        is_number, as_tuple, uuid, is_string,
                        wprint, segment_list, defaultdictkey)
from odin.config import EPS, get_ngpu


# minimum batch size that will be optimal to transfer
# the data to GPU for calculation (tested on Titan X)
# NOTE: tensorflow has a lagging effect, it will be
#       slower than numpy if you evaluate the
#       expression for first time.
MINIMUM_GPU_BLOCK = 8000 * 120 * 4 # bytes

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


def _split_jobs(nb_samples, ncpu, device, gpu_factor):
  """ Return: jobs_cpu, jobs_gpu"""
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
  else:
    raise ValueError("Unknown device: '%s'" % device)
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

# ===========================================================================
# Main GMM
# ===========================================================================
class _ExpectationResults(object):
  """ ExpectationResult """

  def __init__(self, nb_samples, nb_results, name, print_progress):
    super(_ExpectationResults, self).__init__()
    # thread lock
    self.lock = threading.Lock()
    # progress bar
    self.prog = Progbar(target=nb_samples, print_report=True,
                        print_summary=False, name=name)
    # GMM: Z, F, S, L, nframes
    # I-vector: LU, RU, llk, nframes
    self.stats = [0. for i in range(int(nb_results))]
    self.print_progress = bool(print_progress)

  def update(self, res):
    """
    integer (or a number): number of processed samples (update the progress bar)
    otherwise: list of results (update the statistics)
    """
    # thread-safe udpate
    self.lock.acquire()
    try:
      # returned number of processed samples
      if is_number(res) and self.print_progress:
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
      if 'auto', used `1 Megabytes` block for batch size.
  covariance_type : {'full', 'tied', 'diag', 'spherical'},
          defaults to 'full'.
      String describing the type of covariance parameters to use.
      Must be one of::
          'full' (each component has its own general covariance matrix),
          'tied' (all components share the same general covariance matrix),
          'diag' (each component has its own diagonal covariance matrix),
          'spherical' (each component has its own single variance).
  device : {'cpu', 'gpu', 'mix'}
      'gpu' - run the computaiton on GPU
      'cpu' - use multiprocessing for multiple cores
      'mix' - use both GPU and multi-processing
      * It is suggested to use mix of GPU and CPU if you have
        more than 24 cores CPU, otherwise, 'gpu' gives the best
        performance
  ncpu : int
      number of processes for parallel calculating Expectation
  gpu_factor : int
      how much jobs GPU will handle more than CPU
      (i.e. `njob_gpu = gpu_factor * njob_cpu`)
  stochastic_downsample : bool
      if True, a subset of data is selected differently after
      each iteration => the training is stochastic.
      if False, a deterministic selection of data is performed
      each iteration => the training is deterministic.
  seed : int
      random seed for reproducible
  dtype : {str, numpy.dtype}
      desire dtype for mean, std, weights and input matrices

  Attributes
  ----------
  mu : (feat_dim, nmix)
    mean vector for each component
  sigma : (feat_dim, nmix)
    standard deviation for each component
  w : (1, nmix)
    weights of each component

  """

  STANDARD_BATCH_SIZE = 1.2 * 1024 * 1024 # 8 Megabytes

  def __init__(self, nmix, nmix_start=1, niter=16, batch_size='auto',
               covariance_type='diag',
               downsample=2, stochastic_downsample=True,
               device='gpu', ncpu=None, gpu_factor=80,
               dtype='float32', seed=5218, name=None):
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
    self.gpu_factor = int(gpu_factor)
    # cpu
    if ncpu is None:
      ncpu = cpu_count() - 1
    self.ncpu = int(ncpu)
    # device
    device = str(device).lower()
    if device not in ('mix', 'gpu', 'cpu'):
      raise ValueError("`device` can only be one of following option: "
                       "'mix', 'gpu', 'cpu'.")
    self.device = device
    # ====== state variable ====== #
    # store history of {nmix -> [llk_1, llk_2] ...}
    self._llk_hist = defaultdict(list)
    # ====== name ====== #
    self._dtype = np.dtype(dtype)
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
    return (self.mu, self.sigma, self.w,
            self._nmix, self._curr_nmix, self._feat_dim,
            self._niter, self.batch_size,
            self.downsample, self.stochastic_downsample, self.seed,
            self.covariance_type, self._llk_hist,
            self.ncpu, self.device,
            self._dtype, self._name)

  def __setstate__(self, states):
    (self.mu, self.sigma, self.w,
     self._nmix, self._curr_nmix, self._feat_dim,
     self._niter, self.batch_size,
     self.downsample, self.stochastic_downsample, self.seed,
     self.covariance_type, self._llk_hist,
     self.ncpu, self.device,
     self._dtype, self._name) = states
    # basic constants
    self._feat_const = self.feat_dim * np.log(2 * np.pi)
    self.X_ = tf.placeholder(shape=(None, self.feat_dim),
                             dtype='float32',
                             name='GMM_input')
    # init posterior
    self._resfresh_cpu_posterior()
    self._refresh_gpu_posterior()
    # ====== warning no GPU ====== #
    if self.device == 'gpu' and get_ngpu() == 0:
      wprint("Enabled GPU device, but no GPU found!")

  def __str__(self):
    if not self.is_initialized:
      return '<"%s" nmix:%d initialized:False>' % (self.name, self._nmix)
    s = '<"%s" nmix:%s ndim:%s mean:%s std:%s w:%s bs:%s>' %\
        (ctext(self.name, 'yellow'),
            ctext(self._nmix, 'cyan'),
            ctext(self._feat_dim, 'cyan'),
            ctext(self.mu.shape, 'cyan'),
            ctext(self.sigma.shape, 'cyan'),
            ctext(self.w.shape, 'cyan'),
            ctext(self.batch_size, 'cyan'),
          )
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

  @property
  def history(self):
    """ Return the history of fitting this GMM in following format:
      `[(current_nmix, current_niter, llk), ...]`
    """
    return tuple(self._llk_hist)

  @property
  def dtype(self):
    return self._dtype

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
                            dtype=self._dtype,
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

  def initialize(self, X):
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
    # const for specific dimension
    self._feat_const = self.feat_dim * np.log(2 * np.pi)
    if is_string(self.batch_size):
      self.batch_size = int(GMM.STANDARD_BATCH_SIZE /
       (self.feat_dim * self.dtype.itemsize))
    # [batch_size, feat_dim]
    self.X_ = tf.placeholder(shape=(None, self.feat_dim),
                             dtype='float32',
                             name='GMM_input')
    # ====== init ====== #
    # (D, M)
    self.mu = np.zeros((feat_dim, self._curr_nmix), dtype=self._dtype)
    # (D, M)
    self.sigma = np.ones((feat_dim, self._curr_nmix), dtype=self._dtype)
    # (1, M)
    self.w = np.ones((1, self._curr_nmix), dtype=self._dtype)
    # init posterior
    self._resfresh_cpu_posterior()
    self._refresh_gpu_posterior()
    # ====== warning no GPU ====== #
    if self.device == 'gpu' and get_ngpu() == 0:
      wprint("Enabled GPU device, but no GPU found!")

  # ==================== sklearn ==================== #
  def fit(self, X, y=None):
    """
    NOTE
    ----
    from 1, 2, 4 components, python multi-threading is fastest
    from 8, 16 components, python multi-processing is fastest
    from > 32 components, GPU scales much much better.
    """
    if not isinstance(X, (Data, np.ndarray)):
      raise ValueError("`X` must be numpy.ndarray or instance of odin.fuel.Data.")
    if isinstance(X, Feeder):
      raise ValueError("No support for fitting GMM on odin.fuel.Feeder")
    # ====== start GMM ====== #
    # supports 16384 components, modify for more components
    niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10, 16, 16]
    niter[int(np.log2(self._nmix))] = self._niter
    # run the algorithm
    while True:
      # fitting the mixtures
      curr_nmix = self._curr_nmix
      last_niter = len(self._llk_hist[curr_nmix])
      idx = int(np.log2(curr_nmix))
      curr_niter = niter[idx] - last_niter
      if curr_niter > 0:
        for i in range(curr_niter):
          self.expectation_maximization(X, print_progress=True)
      # update the mixtures
      if curr_nmix < self._nmix:
        self.gmm_mixup()
      else:
        break
    return self

  def score(self, X, y=None):
    """ Compute the log-likelihood of each example to
    the Mixture of Components.
    """
    post = self.logprob(X)  # (batch_size, nmix)
    return logsumexp(post, axis=1) # (batch_size, 1)

  def transform(self, X, device=None):
    """ Compute centered statistics given X and fitted mixtures

    Return
    ------
    zero-th statistics: [1, nmix]
      e.g. the assignment score each samples to each components, hence,
      `#frames = Z.sum()`
    first statistics: [1, feat_dim * nmix]
      dot-product of each sample and the posteriors.

    NOTE
    ----
    For more option check `GMM.expectation`
    """
    if device is None:
      device = self.device
    Z, F = self._fast_expectation(X, zero=True, first=True,
                                  second=False, llk=False,
                                  on_gpu=device != 'cpu')
    # this equal to: .ravel()[np.newaxis, :]
    F_hat = np.reshape(F - self.mu * Z,
                       newshape=(1, self.feat_dim * self._curr_nmix),
                       order='F')
    return Z, F_hat

  def transform_to_disk(self, X, indices, pathZ=None, pathF=None,
                        dtype='float32', device='cpu', override=True):
    """ Same as `transform`, however, save the transformed statistics
    to file using `odin.fuel.MmapData`

    Return
    ------
    zero-th statistics: [1, nmix]
      e.g. the assignment score each samples to each components, hence,
      `#frames = Z.sum()`
    first statistics: [1, feat_dim * nmix]
      dot-product of each sample and the posteriors.

    Note
    ----
    If your data contain many very long utterances, it is suggested to use
    `device='gpu'`, otherwise, 'cpu' is mostly significant faster.
    """
    from odin.fuel import MmapData
    if device is None:
      device = self.device
    on_gpu = True if device != 'cpu' and get_ngpu() > 0 else False
    name_list = []
    prog = Progbar(target=len(indices),
                   print_report=True, print_summary=True,
                   name="Saving zero-th and first order statistics")
    # ====== init data files ====== #
    if pathZ is not None:
      if os.path.exists(pathZ):
        if not override:
          raise RuntimeError("Path at %s exist and cannot be overrided" % pathZ)
        os.remove(pathZ)
      z_dat = MmapData(path=pathZ, dtype=dtype, shape=(0, self.nmix))
    else:
      z_dat = None
    if pathF is not None:
      if os.path.exists(pathF):
        if not override:
          raise RuntimeError("Path at %s exist and cannot be overrided" % pathF)
        os.remove(pathF)
      f_dat = MmapData(path=pathF, dtype=dtype, shape=(0, self.nmix * self.feat_dim))
    else:
      f_dat = None
    # ====== running on GPU ====== #
    if on_gpu:
      for name, (start, end) in indices:
        res = self._fast_expectation(X[start:end],
                                     zero=z_dat is not None or f_dat is not None,
                                     first=f_dat is not None,
                                     second=False, llk=False, on_gpu=True)
        if z_dat is not None:
          z_dat.append(res[0])
        if f_dat is not None:
          Z, F = res
          f_dat.append(np.reshape(F - self.mu * Z,
                                  newshape=(1, self._feat_dim * self._curr_nmix),
                                  order='F'))
        name_list.append(name)
        prog.add(1)
    # ====== run on CPU ====== #
    else:
      if isinstance(indices, Mapping):
        indices = list(indices.items())

      def map_func(j):
        Z, F = [], []
        for name, (start, end) in j:
          res = self._fast_expectation(X[start:end],
                                       zero=z_dat is not None or f_dat is not None,
                                       first=f_dat is not None,
                                       second=False, llk=False, on_gpu=True)
          Z.append(res[0])
          if f_dat is not None:
            z, f = res
            F.append(np.reshape(f - self.mu * z,
                                newshape=(1, self._feat_dim * self._curr_nmix),
                                order='F'))
        # concatenate into single large matrix
        if len(Z) > 0:
          Z = np.concatenate(Z, axis=0)
        if len(F) > 0:
          F = np.concatenate(F, axis=0)
        return Z, F, len(j)
      # run the MPI task
      mpi = MPI(jobs=indices, func=map_func, ncpu=self.ncpu,
                batch=max(1, self.batch_size // self.ncpu))
      for Z, F, n in mpi:
        prog.add(n)
        if z_dat is not None:
          z_dat.append(Z)
        if f_dat is not None:
          f_dat.append(F)
    # ====== flush and return ====== #
    if z_dat is not None:
      z_dat.flush()
      z_dat.close()
    if f_dat is not None:
      f_dat.flush()
      f_dat.close()
    return name_list

  # ==================== math helper ==================== #
  def logprob(self, X):
    """ Shape: [batch_size, nmix]
    the log probability of each observations to each components
    given the GMM.
    """
    self.initialize(X)
    if self.device != 'cpu':
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
    self.initialize(X)
    if self.device != 'cpu':
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
    self.initialize(X)
    if self.device != 'cpu':
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

  def _fast_expectation(self, X, zero=True, first=True, second=True,
                        llk=True, on_gpu=False):
    # ====== run on GPU ====== #
    if on_gpu:
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

  def expectation(self, X, zero=True, first=True, second=True,
                  llk=True, device=None, print_progress=True):
    """
    Parameters
    ----------
    X : numpy.ndarray [batch_size, feat_dim]
        input array, with feature dimension is the final dimension
    zero : bool (default: True)
        if True, return zero-order statistics
    first : bool (default: True)
        if True, return first-order statistics
    second : bool (default: True)
        if True, return second-order statistics
    llk : bool (default: True)
        if True, return the mean log-likelihood
    device : {None, 'cpu', 'gpu', 'mix'}
        None - keep the orginal device specified in init
        'gpu' - run the computaiton on GPU
        'cpu' - use multiprocessing for multiple cores
        'mix' - use both GPU and multi-processing
    print_progress : bool (default: True)
        if fitting required multiple batches, print the
        progress bar.

    Return
    ------
    The order of return value:
    zero  (optional) : ndarray [1, nmix]
    first (optional) : ndarray [feat_dim, nmix]
    second(optional) : ndarray [feat_dim, nmix]
    llk   (optional) : scalar ()
    """
    self.initialize(X)
    nb_samples = X.shape[0]
    # ====== pick device ====== #
    device = self.device if device is None else str(device).lower()
    if device not in ('gpu', 'cpu', 'mix'):
      raise ValueError("`device` can only be of the following:"
                       "'gpu', 'cpu', and 'mix'.")
    # ====== only 1 batch ====== #
    if nb_samples <= self.batch_size:
      results = self._fast_expectation(X, zero, first, second, llk,
                                       on_gpu=self.device != 'cpu')
      if llk:
        if isinstance(results, (tuple, list)):
          results = tuple(results[:-1]) + (np.array(results[-1] / nb_samples),)
        else:
          results = np.array(results / nb_samples)
      return results
    # ====== mapping method ====== #
    curr_niter = len(self._llk_hist[self._curr_nmix])
    curr_nmix = self._curr_nmix

    def map_expectation(start_end_gpu_bs):
      (start, end), on_gpu, batch_size = start_end_gpu_bs
      # Z, F, S, L, nfr
      results = [0., 0., 0., 0., 0]
      for s, e, n, selected in _create_batch(start, end, batch_size,
                                  downsample=self.downsample,
                                  stochastic=self.stochastic_downsample,
                                  seed=self.seed,
                                  curr_nmix=curr_nmix,
                                  curr_niter=curr_niter):
        # downsample by randomly ignore a batch
        if selected:
          x = X[s:e]
          # update expectation
          for i, res in enumerate(self._fast_expectation(x, on_gpu=on_gpu)):
            results[i] += res
          results[-1] += x.shape[0]
        yield n
      yield tuple(results)

    def thread_expectation(results, start_end):
      batch_size = max(int(MINIMUM_GPU_BLOCK / self.feat_dim / 4),
                       self.batch_size)
      for res in map_expectation((start_end, True, batch_size)):
        results.update(res)
    # ====== split the jobs ====== #
    jobs_cpu, jobs_gpu = _split_jobs(nb_samples=nb_samples,
                                     ncpu=self.ncpu, device=device,
                                     gpu_factor=self.gpu_factor)
    # Z, F, S, L, nfr
    results = _ExpectationResults(nb_samples=nb_samples, nb_results=5,
        name="[GMM] cmix:%d nmix:%d iter:%d" %
                   (curr_nmix, self.nmix, curr_niter),
        print_progress=print_progress)
    # ====== run multiprocessing ====== #
    mpi = []
    if len(jobs_cpu) > 0:
      # create CPU processes
      mpi = MPI(jobs=[(j, False, self.batch_size)
                      for j in jobs_cpu],
                func=map_expectation,
                ncpu=self.ncpu, batch=1, hwm=2**25,
                backend='python')
    # create GPU threads
    gpu_threads = [threading.Thread(target=thread_expectation,
                                    args=(results, j))
                   for j in jobs_gpu]
    # start gpu and cpu threads
    for t in gpu_threads:
      t.start()
    # start the cpu processes
    for res in mpi:
      results.update(res)
    # finish all threads
    for t in gpu_threads:
      t.join()
    # ====== summary ====== #
    Z, F, S, L, nfr = results.stats
    L = L / nfr
    results = []
    if zero:
      results.append(Z)
    if first:
      results.append(F)
    if second:
      results.append(S)
    if llk:
      results.append(L)
    return results[0] if len(results) == 1 else results

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
    return self

  def expectation_maximization(self, X, device=None, print_progress=True):
    self.initialize(X)
    curr_nmix = self._curr_nmix
    curr_niter = len(self._llk_hist[curr_nmix]) + 1
    # ====== Expectation ====== #
    start_time = time.time()
    Z, F, S, L = self.expectation(X, device=device, print_progress=print_progress)
    time_Estep = time.time() - start_time
    # ====== maximization ====== #
    start_time = time.time()
    self.maximization(Z, F, S)
    time_Mstep = time.time() - start_time
    # store history
    self._llk_hist[self._curr_nmix].append(L)
    # print log
    if print_progress:
      print("#mix:%s #iter:%s llk:%s Estep:%s(s) Mstep:%s(s)" %
        (ctext('%.2d' % curr_nmix, 'cyan'),
         ctext('%.2d' % curr_niter, 'yellow'),
         ctext('%.4f' % L, 'yellow'),
         ctext('%.2f' % time_Estep, 'yellow'),
         ctext('%.4f' % time_Mstep, 'yellow'),
        ))
    # release memory
    del Z, F, S
    return self

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
    # update current number of mixture information
    self._curr_nmix = min(2 * self._curr_nmix, self.nmix)
    self._refresh_gpu_posterior()
    self._resfresh_cpu_posterior()


# ===========================================================================
# Ivector
# ===========================================================================
class _CenteredStatsIter(object):
  """_CenteredStatsIter"""

  def __init__(self, gmm, batch_size,
               X=None, Z=None, F=None, indices=None,
               start=None, end=None):
    super(_CenteredStatsIter, self).__init__()
    self.gmm = gmm
    self.batch_size = batch_size
    self.X = X
    self.Z = Z
    self.F = F
    self.indices = indices
    # start and end only for provided Z and F
    self.start = 0 if start is None else start
    self.end = Z.shape[0] if end is None and Z is not None else end
    # ====== infer length (i.e. number of files) ====== #
    if self.indices is not None:
      self._length = len(self.indices)
    elif self.X is not None:
      self._length = len(as_tuple(self.X))
    elif Z is not None and F is not None:
      self._length = self.end - self.start
    else:
      raise RuntimeError("No data is given.")

  def __len__(self):
    return self._length

  def __iter__(self):
    get_stats = None
    # ====== given indices ====== #
    if self.indices is not None:
      get_stats = (self.gmm.transform(self.X[start:end], device='cpu')
                   for name, (start, end) in self.indices)
    # ====== given list of utterances ====== #
    elif self.X is not None:
      get_stats = (self.gmm.transform(x, device='cpu')
                   for x in as_tuple(self.X))
    # ====== given raw data ====== #
    if get_stats is not None:
      Z, F = [], []
      for y in get_stats:
        Z.append(y[0])
        F.append(y[1])
        if len(Z) >= self.batch_size:
          Z = np.concatenate(Z, axis=0)
          F = np.concatenate(F, axis=0)
          yield Z, F
          Z, F = [], []
      # final batch
      if len(Z) > 0:
        Z = np.concatenate(Z, axis=0)
        F = np.concatenate(F, axis=0)
        yield Z, F
    # ====== given Z and F ====== #
    elif self.Z is not None and self.F is not None:
      for start, end in batching(n=self.end - self.start,
                                 batch_size=self.batch_size):
        start += self.start
        end += self.start
        yield self.Z[start:end, :], self.F[start:end, :]
    # ====== exception ====== #
    else:
      raise RuntimeError("No data for iteration.")

  def __del__(self):
    self.gmm = None
    self.X = None
    self.Z = None
    self.F = None
    self.indices = None

class Ivector(DensityMixin, BaseEstimator, TransformerMixin):
  STANDARD_BATCH_SIZE = 25 * 1024 * 1024 # 20 Megabytes

  def __init__(self, tv_dim, gmm, niter=16,
               batch_size='auto', dtype='float32',
               device='gpu', ncpu=1, gpu_factor=3,
               seed=5218, name=None):
    super(Ivector, self).__init__()
    # ====== init ====== #
    self.niter = niter
    self._tv_dim = tv_dim
    self._t2_dim = tv_dim * (tv_dim + 1) // 2
    self._feat_dim = gmm.feat_dim
    self._nmix = gmm.nmix
    self._gmm = gmm
    # ====== others ====== #
    self._llk_hist = []
    if name is None:
      name = uuid(length=8)
      self._name = 'Ivector_%s' % name
    else:
      self._name = str(name)
    # ====== training ====== #
    self._dtype = np.dtype(dtype)
    if is_string(batch_size):
      batch_size = int(Ivector.STANDARD_BATCH_SIZE /
       ((self.feat_dim * self.nmix * self.dtype.itemsize) +
        (self.nmix * self.dtype.itemsize)))
    self.batch_size = batch_size
    # ====== select device ====== #
    device = str(device).lower()
    if device not in ('cpu', 'gpu', 'mix'):
      raise ValueError("`device` must be one of the following: 'cpu', 'gpu', or 'mix'")
    if device in ('gpu', 'mix') and get_ngpu() == 0:
      wprint("Using GPU device but NO GPU detected, "
             "tensorflow will switch to slower CPU computation!")
    self.device = device
    # cpu
    if ncpu is None:
      ncpu = cpu_count() // 2
    self.ncpu = int(ncpu)
    self.gpu_factor = int(gpu_factor)
    # ====== load ubm ====== #
    self.Im = np.eye(self.tv_dim, dtype=self.dtype)
    self.Sigma = np.array(
        gmm.sigma.reshape((1, self.feat_dim * self.nmix), order='F'),
        dtype=self.dtype)
    np.random.seed(seed)
    self.Tm = (np.random.randn(self.tv_dim, self.feat_dim * self.nmix) *
               self.Sigma.sum() * 0.001).astype(self.dtype)
    self.T_invS_Tt = np.empty((self.nmix, self.t2_dim), dtype=self.dtype)
    # ====== cache, 10% faster here ====== #
    self._itril = np.tril_indices(tv_dim)
    self._mix_idx = [np.arange(self.feat_dim) + mix * self.feat_dim
                     for mix in range(self.nmix)]
    self._Ex_Exx_llk = defaultdictkey(
        lambda nfiles: (np.empty((nfiles, self.tv_dim), dtype=self.dtype),
                        np.empty((nfiles, self.t2_dim), dtype=self.dtype),
                        np.empty((nfiles, 1), dtype=self.dtype)))
    # ====== calculate stats first ====== #
    self._refresh_T_statistics()
    self._refresh_gpu()

  # ==================== properties ==================== #
  @property
  def feat_dim(self):
    return self._feat_dim

  @property
  def tv_dim(self):
    return self._tv_dim

  @property
  def t2_dim(self):
    return self._t2_dim

  @property
  def nmix(self):
    return self._nmix

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  @property
  def gmm(self):
    return self._gmm

  # ==================== i-vec ==================== #
  def _refresh_T_statistics(self):
    """ depend on: Tm and Sigma """
    # (self.tv_dim, self.feat_dim * self.nmix)
    self.T_invS = self.Tm / self.Sigma
    # (self.nmix, self.tv_dim * (self.tv_dim + 1) / 2)
    T_invS2 = self.Tm / np.sqrt(self.Sigma)
    # update each row for each mixture
    for mix, idx in enumerate(self._mix_idx):
      tmp = T_invS2[:, idx].dot(T_invS2[:, idx].T)
      self.T_invS_Tt[mix] = tmp[self._itril]

  def _refresh_gpu(self):
    if hasattr(self, '_gpu_inputs') and hasattr(self, '_gpu_outputs'):
      return
    with tf.variable_scope(self.name):
      Z = K.placeholder(shape=(None, self.nmix),
                        dtype=self.dtype,
                        name='ZeroStats')
      F = K.placeholder(shape=(None, self.nmix * self.feat_dim),
                        dtype=self.dtype,
                        name='ZeroStats')
      Tm = K.placeholder(shape=self.Tm.shape,
                         dtype=self.dtype,
                         name='Tmatrix')
      T_invS_Tt = K.placeholder(shape=self.T_invS_Tt.shape,
                                dtype=self.dtype,
                                name='T_invS_Tt')
      Sigma = tf.constant(value=self.Sigma,
                          dtype=self.dtype,
                          name='Sigma')
      Im = tf.eye(self.tv_dim, dtype=self.dtype, name='Im')
      # ====== start the calculation ====== #
      T_invS = Tm / Sigma
      L1B1 = tf.concat((
          tf.matmul(Z, T_invS_Tt, name='L1'), # (nfiles, t2_dim)
          tf.matmul(F, tf.transpose(T_invS), name='B1')), # (nfiles, tv_dim)
      axis=-1)

      # ====== repeat for each utterance (file) ====== #
      def map_fn(L1B1):
        L1 = L1B1[:self.t2_dim]
        B1 = L1B1[self.t2_dim:]
        L = tf.scatter_nd(indices=[(i, j) for i, j in zip(*self._itril)],
                          updates=L1,
                          shape=(self.tv_dim, self.tv_dim),
                          name='L')
        L = L + tf.transpose(K.tril(L, k=-1)) + Im
        Cxx = tf.linalg.inv(L)
        B = tf.expand_dims(B1, axis=-1)
        this_Ex = tf.matmul(Cxx, B)
        Ex = tf.transpose(this_Ex)
        llk = -0.5 * tf.matmul(Ex, B - this_Ex) + tf.matmul(Ex, B)
        Exx = tf.gather_nd(params=(Cxx + tf.matmul(this_Ex, Ex)),
                           indices=[(i, j) for i, j in zip(*self._itril)])
        return tf.concat(
            (tf.squeeze(llk, axis=0), tf.squeeze(Ex, axis=0), Exx), axis=0)
      # ====== compute ====== #
      llk_Ex_Exx = tf.map_fn(fn=map_fn, elems=L1B1, dtype=self.dtype,
                             back_prop=False)
      llk = llk_Ex_Exx[:, 0]
      Ex = llk_Ex_Exx[:, 1:1 + self.tv_dim]
      Exx = llk_Ex_Exx[:, 1 + self.tv_dim:]
      RU = tf.matmul(tf.transpose(Ex), F)
      LU = tf.matmul(tf.transpose(Z), Exx)
      llk = tf.reduce_sum(llk)
      # ====== assign inputs outputs for GPU function ====== #
      self._gpu_inputs = [Z, F, Tm, T_invS_Tt]
      self._gpu_outputs = [LU, RU, llk]

  def _fast_expectation(self, Z, F, on_gpu):
    nframes = np.ceil(Z.sum())
    nfiles = F.shape[0]
    # ====== GPU ====== #
    if on_gpu:
      LU, RU, llk = K.eval(self._gpu_outputs,
        feed_dict={i: j for i, j in zip(self._gpu_inputs,
                                        (Z, F, self.Tm, self.T_invS_Tt))}
      )
      return LU, RU, llk, nframes
    # ====== CPU ====== #
    # (nfiles, tv_dim * (tv_dim + 1) / 2)
    L1 = np.dot(Z, self.T_invS_Tt)
    # (nfiles, tv_dim)
    B1 = np.dot(F, self.T_invS.T)
    Ex, Exx, llk = self._Ex_Exx_llk[nfiles]
    for ix in range(nfiles):
      L = np.zeros((self.tv_dim, self.tv_dim))
      L[self._itril] = L1[ix]
      L = L + np.tril(L, k=-1).T + self.Im
      Cxx = linalg.inv(L)
      B = B1[ix][:, np.newaxis]
      this_Ex = np.dot(Cxx, B)
      this_ExT = this_Ex.T
      Ex[ix] = this_ExT
      llk[ix] = -0.5 * this_ExT.dot(B - this_Ex) + this_ExT.dot(B)
      Exx[ix] = (Cxx + this_Ex.dot(this_ExT))[self._itril]
    # (tdim, nmix * feat_dim)
    RU = np.dot(Ex.T, F)
    # (nmix, tdim * (tdim + 1) / 2)
    LU = np.dot(Z.T, Exx)
    return LU, RU, llk.sum(), nframes

  def expectation(self, Z, F, device=None, print_progress=True):
    if device is None:
      device = self.device
    nfiles = Z.shape[0]
    # ====== single batch ====== #
    if nfiles < self.batch_size:
      results = self._fast_expectation(Z=Z, F=F,
                    on_gpu=True if device in ('gpu', 'mix') else False)
    # ====== multiple batches ====== #
    else:
      def map_expectation(start_end_onGPU):
        (start, end), on_gpu = start_end_onGPU
        tmp = [0., 0., 0., 0.] # LU, RU, llk, nframes
        for s, e in batching(n=end - start, batch_size=self.batch_size):
          s += start
          e += start
          results.update(e - s)
          # update local results
          for i, r in enumerate(self._fast_expectation(Z=Z[s:e], F=F[s:e],
                                                       on_gpu=on_gpu)):
            tmp[i] += r
        # update actual results
        results.update(tmp)
      # ====== prepare the jobs ====== #
      jobs_cpu, jobs_gpu = _split_jobs(nb_samples=nfiles, ncpu=self.ncpu,
                                       device=device,
                                       gpu_factor=self.gpu_factor)
      jobs_cpu = [(j, False) for j in jobs_cpu]
      jobs_gpu = [(j, True) for j in jobs_gpu]
      # LU, RU, llk, nframes
      results = _ExpectationResults(nb_samples=nfiles, nb_results=4,
          name="[Ivector] Tdim:%d nmix:%d feat_dim:%d iter:%d" %
                     (self.tv_dim, self.nmix, self.feat_dim,
                      len(self._llk_hist) + 1),
          print_progress=print_progress)
      # ====== create gpu thread ====== #
      threads = [threading.Thread(target=map_expectation,
                                  args=(j,))
                 for j in jobs_gpu + jobs_cpu]
      # start gpu and threads
      for t in threads:
        t.start()
      # finish all threads
      for t in threads:
        t.join()
    # return
    return results.stats

  def maximization(self, LU, RU, nframes=None,
                   min_div_est=True, orthogonalize=True):
    # ML re-estimation of the total subspace matrix or the factor loading
    # matrix
    for mix, idx in enumerate(self._mix_idx):
      Lu = np.zeros((self.tv_dim, self.tv_dim), dtype=self.dtype)
      Lu[self._itril] = LU[mix, :]
      Lu += np.tril(Lu, -1).T
      self.Tm[:, idx] = linalg.solve(Lu, RU[:, idx])
    # min_div_est
    if min_div_est:
      if nframes is None:
        raise ValueError("`nframes` must be specified if `min_div_est=True`")
      Lu = np.zeros((self.tv_dim, self.tv_dim))
      Lu[self._itril] = LU.sum(0) / nframes
      Lu += np.tril(Lu, -1).T
      self.Tm = np.dot(linalg.cholesky(Lu), self.Tm)
    # orthogonalize the columns
    if orthogonalize:
      U_, s_, V_ = linalg.svd(self.Tm, full_matrices=False)
      self.Tm = np.diag(s_).dot(V_)
    # refresh stats
    self._refresh_T_statistics()
    return self

  def expectation_maximization(self, Z, F, device=None, print_progress=True):
    nfiles = Z.shape[0]
    # ====== Expectation ====== #
    start_time = time.time()
    LU, RU, LLK, nframes = self.expectation(Z=Z, F=F, device=device,
                                            print_progress=print_progress)
    time_Estep = time.time() - start_time
    # ====== maximization ====== #
    start_time = time.time()
    self.maximization(LU, RU, nframes,
                      min_div_est=True, orthogonalize=True)
    time_Mstep = time.time() - start_time
    # store history
    LLK = LLK / nfiles
    self._llk_hist.append(LLK)
    # print log
    if print_progress:
      print("#iter:%s llk:%s Estep:%s(s) Mstep:%s(s)" %
        (ctext('%.2d' % len(self._llk_hist), 'yellow'),
         ctext('%.4f' % LLK, 'yellow'),
         ctext('%.2f' % time_Estep, 'yellow'),
         ctext('%.4f' % time_Mstep, 'yellow'),
        ))
    return self

  # ==================== sklearn ==================== #
  def transform(self, Z, F):
    L = np.zeros((self.tv_dim, self.tv_dim))
    L[self._itril] = Z.dot(self.T_iS_Tt)
    L += np.tril(L, -1).T + self.Im
    Cxx = linalg.inv(L)
    B = np.dot(self.T_invS, F)
    Ex = np.dot(Cxx, B)
    return Ex

  def fit(self, Z, F):
    print('Re-estimating the total subspace with {} factors ...'.format(self.tv_dim))
    # LU, RU, LLK, nframes
    for iter in range(self.niter):
      self.expectation_maximization(Z, F)
