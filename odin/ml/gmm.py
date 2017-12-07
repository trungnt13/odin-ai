# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'
"""
import time
import random

import numpy as np

from sklearn.base import DensityMixin, BaseEstimator, TransformerMixin

from odin.fuel import Data, DataDescriptor, Feeder
from odin.utils import MPI, batching, ctext, cpu_count, Progbar, is_number

EPS = np.finfo(float).eps


# ===========================================================================
# Helper
# ===========================================================================
def zeroStat(post):
    # sum over all samples
    return np.sum(post, axis=0, keepdims=True) # (1, M)


def firstStat(X, post):
    return np.dot(X.T, post) # (D, M)


def secondStat(X, post):
    return np.dot((X ** 2).T, post) # (D, M)


def logsumexp(x, axis):
    xmax = x.max(axis=axis, keepdims=True)
    y = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
    return y


# ===========================================================================
# Main GMM
# ===========================================================================
class GMM(DensityMixin, BaseEstimator, TransformerMixin):
    """ The following symbol is used:
    N: number of samples (frames)
    D: number of features dimension
    M: current number of mixture

    Parameters
    ----------
    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).
    init_algo: {'split', 'kmean'}
        'split'
        'kmean'
    device: {'gpu', 'cpu', 'mix'}
        which devices using for the EM
        'gpu' (only run on tensorflow implementation using GPU)
        'cpu' (only run on numpy implemetation using CPU)
        'mix' (using both GPU and CPU)
    downsample: int
        downsampling factor
    stochastic_downsample: bool
        if True, a subset of data is selected differently after
        each iteration => the training is stochastic.
        if False, a deterministic selection of data is performed
        each iteration => the training is deterministic.
    ncpu: int
        number of processes for parallel calculating Expectation
    seed: int
        random seed for reproducible
    """

    def __init__(self, nmix, niter=16, batch_size=2056,
                 covariance_type='diag', init_algo='split',
                 downsample=4, stochastic_downsample=True, seed=5218,
                 device='mix', ncpu=1):
        super(GMM, self).__init__()
        self._nmix = 2**int(np.round(np.log2(nmix)))
        self._niter = int(niter)
        self.batch_size = int(batch_size)
        # ====== downsample ====== #
        self.downsample = int(downsample)
        self.stochastic_downsample = bool(stochastic_downsample)
        self.seed = int(seed)
        # ====== different mode ====== #
        self.init_algo = str(init_algo)
        self.covariance_type = str(covariance_type)
        # ====== multi-processing ====== #
        self.device = str(device)
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = int(ncpu)
        # ====== state variable ====== #
        self._is_initialized = False
        self._is_fitted = False
        self._llk = [] # store history of LLK
        # 'means', 'variances', 'weights'
        # self.mu, self.sigma, self.w
        # self.C_ = self.compute_C()

    def __str__(self):
        if self._initialize:
            mu, std, w = self.mu.shape, self.sigma.shape, self.w.shape
        else:
            mu, std, w = None, None, None
        s = '<%s init:%s fitted:%s mu:%s std:%s weight:%s>' %\
            (ctext('GMM:%d' % self._nmix, 'cyan'),
                self._is_initialized, self._is_fitted,
                mu, std, w)
        return s

    # ==================== properties ==================== #
    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def niter(self):
        if self.init_algo == 'split':
            n = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10]
            n[int(np.log2(self._nmix))] = self._niter
        else:
            raise NotImplementedError
        return n

    @property
    def nmix(self):
        return self._nmix

    @property
    def ndim(self):
        if not self.is_initialized:
            raise RuntimeError("GMM has not been fitted on data.")
        return self._ndim

    # ==================== sklearn ==================== #
    def _initialize(self, ndim):
        if self._is_initialized:
            return
        self._ndim = ndim
        self._is_initialized = True
        # ====== init ====== #
        # (D, M)
        self.mu = np.zeros((ndim, 1), dtype='f4')
        # (D, M)
        self.sigma = np.ones((ndim, 1), dtype='f4')
        # (1, M)
        self.w = np.ones((1, 1), dtype='f4')
        # (1, M)
        self.C_ = self.compute_C()

    def fit(self, X, y=None):
        if not isinstance(X, (Data, np.ndarray)):
            raise ValueError("`X` must be numpy.ndarray or instance of odin.fuel.Data.")
        if isinstance(X, Feeder):
            raise ValueError("No support for fitting GMM on odin.fuel.Feeder")
        # ====== check input ====== #
        self._initialize(ndim=X.shape[1])
        assert (X.ndim == 2 or X.ndim == [2]) and X.shape[1] == self._ndim
        is_indices = True if isinstance(X, DataDescriptor) else False
        # ====== divide the batches ====== #
        nb_samples = len(X.indices) if is_indices else X.shape[0]
        jobs = np.linspace(start=0, stop=nb_samples,
                           num=self.ncpu + 1, dtype='int32')
        jobs = list(zip(jobs, jobs[1:]))
        curr_nmix = 1
        curr_niter = 0

        # ====== mapping method ====== #
        def _map_func(start_end):
            start, end = start_end
            Z, F, S, L, nfr = 0., 0., 0., 0., 0
            # stochastic downsasmple, seed change every iter and mixup
            if self.stochastic_downsample:
                random.seed(self.seed + curr_nmix + curr_niter)
            else: # deterministic
                random.seed(self.seed)
            # iterate over batches
            for s, e in batching(n=end - start, batch_size=self.batch_size):
                # downsample by randomly ignore a batch
                if self.downsample == 1 or \
                (self.downsample > 1 and random.random() <= 1. / self.downsample):
                    s += start; e += start
                    x = X[s:e]
                    res_Z, res_F, res_S, res_L = self.expectation(x)
                    Z += res_Z
                    F += res_F
                    S += res_S
                    L += res_L
                    nfr += x.shape[0]
                yield e - s
            yield Z, F, S, L, nfr
        # ====== start GMM ====== #
        # supports 4096 components, modify for more components
        niter = self.niter
        while curr_nmix <= self._nmix:
            print(ctext(
                'Re-estimating the GMM for {} components ...'.format(curr_nmix),
                'cyan'))
            for curr_niter in range(niter[int(np.log2(curr_nmix))]):
                start_time = time.time()
                # New C_ value
                self.C_ = self.compute_C()
                # Expectation
                mpi = MPI(jobs=jobs, func=_map_func,
                          ncpu=min(len(jobs), self.ncpu),
                          batch=1, hwm=2**25, backend='python')
                prog = Progbar(target=nb_samples,
                        print_report=True, print_summary=False,
                        name="[GMM] nmix:%d  max_nmix:%d" % (curr_nmix, self.nmix))
                Z, F, S, L, nfr = 0., 0., 0., 0., 0
                for res in mpi:
                    # returned number of processed samples
                    if is_number(res):
                        prog.add(res)
                    # return the statistics, end of process
                    else:
                        res_Z, res_F, res_S, res_L, res_nfr = res
                        Z += res_Z
                        F += res_F
                        S += res_S
                        L += res_L
                        nfr += res_nfr
                # Maximization
                self.maximization(Z, F, S)
                # print Log-likelihood
                self._llk.append(L / nfr)
                print("#iter:", ctext('%.2d' % (curr_niter + 1), 'yellow'),
                      "llk:", ctext('%.4f' % self._llk[-1], 'yellow'),
                      "%.2f(s)" % (time.time() - start_time))
                # release memory
                del Z, F, S
            # update the mixtures
            if curr_nmix < self._nmix:
                self.gmm_mixup()
            curr_nmix *= 2
        # set is_fit and return
        self._is_fitted = True
        return self

    def score(self, X, y=None):
        # compute_llk
        if not self.is_initialized or not self.is_fitted:
            raise RuntimeError("GMM has not been fitted on data.")
        post = self.lgmmprob(X) # (N, M)
        return logsumexp(post, axis=1)

    def transform(self, X):
        if not self.is_initialized or not self.is_fitted:
            raise RuntimeError("GMM has not been fitted on data.")
        post = self.postprob(X)[0]
        Z = zeroStat(post) # (1, M)
        F = firstStat(X, post) # (D, M)
        F_hat = np.reshape(F - self.mu * Z,
                           newshape=(self._ndim * self._nmix, 1))
        return Z, F_hat

    # ==================== math helper ==================== #
    def lgmmprob(self, X):
        precision = 1 / (self.sigma + EPS) # (D, M)
        D = np.dot(X ** 2, precision) - \
            2 * np.dot(X, self.mu * precision) + \
            self._ndim * np.log(2 * np.pi)
        return -0.5 * (self.C_ + D) # (N, M)

    def postprob(self, X):
        post = self.lgmmprob(X) # (N, M)
        llk = logsumexp(post, axis=1) # (N, 1)
        post = np.exp(post - llk)
        return post, llk

    def compute_C(self):
        precision = 1 / (self.sigma + EPS)
        log_det = np.sum(np.log(self.sigma + EPS), 0, keepdims=True)
        return np.sum((self.mu ** 2) * precision, 0, keepdims=True) + \
            log_det - \
            2 * np.log(self.w)

    def expectation(self, X):
        # The Map
        ndim = X.shape[1]
        if ndim != self._ndim:
            raise ValueError('Dimensionality of the data ({}) does not match '
                'the specified dimension ndim={}!'.format(ndim, self._ndim))
        post, llk = self.postprob(X)
        Z = zeroStat(post)
        F = firstStat(X, post)
        S = secondStat(X, post)
        L = llk.sum()
        return Z, F, S, L

    def maximization(self, Z, F, S, floor_const=None):
        # TheReduce
        iN = 1. / (Z + EPS)
        self.w = Z / Z.sum()
        self.mu = F * iN
        self.sigma = S * iN - self.mu * self.mu
        # applying variance floors
        if floor_const is not None: # example: floor_const=1e-3
            vFloor = self.sigma.dot(self.w.T) * floor_const
            self.sigma = self.sigma.clip(vFloor)

    def gmm_mixup(self):
        ndim, nmix = self.sigma.shape
        sig_max, arg_max = self.sigma.max(0), self.sigma.argmax(0)
        eps = np.zeros((ndim, nmix), dtype='f')
        eps[arg_max, np.arange(nmix)] = np.sqrt(sig_max)
        perturb = 0.55 * eps
        self.mu = np.c_[self.mu - perturb, self.mu + perturb]
        self.sigma = np.c_[self.sigma, self.sigma]
        self.w = 0.5 * np.c_[self.w, self.w]
