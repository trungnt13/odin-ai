# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'
"""
import time
import numpy as np

from sklearn.base import DensityMixin, BaseEstimator, TransformerMixin

from odin.utils import MPI, batching, ctext, cpu_count, Progbar, is_number

EPS = np.finfo(float).eps


# ===========================================================================
# Helper
# ===========================================================================
def zeroStat(post):
    return np.sum(post, axis=0, keepdims=True) # (1, M)


def firstStat(X, post):
    return np.dot(X.T, post) # (D, M)


def secondStat(X, post):
    return np.dot((X ** 2).T, post) # (D, M)


def compute_llk(post):
    return logsumexp(post, 1)


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def logsumexp(x, axis):
    xmax = x.max(axis=axis, keepdims=True)
    y = xmax + np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True))
    return y


def reduce_expectation_res(res):
    Z, F, S, L, nframes = res[0]
    for r in res[1:]:
        n, f, s, l, nfr = r
        Z += n
        F += f
        S += s
        L += l
        nframes += nfr
    return Z, F, S, L, nframes


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
    init_algo: {'iter', 'kmean'}
    device: {'gpu', 'cpu', 'mix'}
        which devices using for the EM
        'gpu' (only run on tensorflow implementation using GPU)
        'cpu' (only run on numpy implemetation using CPU)
        'mix' (using both GPU and CPU)
    """

    def __init__(self, nmix, niter=16,
                 covariance_type='diag', init_algo='gmm',
                 downsample=1,
                 batch_size=2056, device='mix', ncpu=1):
        super(GMM, self).__init__()
        self.nmix = 2**int(np.round(np.log2(nmix)))
        self.niter = int(niter)
        self.downsample = int(downsample)
        self.init_algo = str(init_algo)
        # ====== multi-processing ====== #
        self.batch_size = int(batch_size)
        self.device = str(device)
        if ncpu is None:
            ncpu = cpu_count() - 1
        self.ncpu = int(ncpu)
        # ====== state variable ====== #
        self._is_initialized = False
        self._is_fitted = False
        # 'means', 'variances', 'weights'
        # self.mu, self.sigma, self.w
        # self.C_ = self.compute_C()

    def __str__(self):
        if self._initialize:
            mu, std, w = self.mu.shape, self.sigma.shape, self.w.shape
        else:
            mu, std, w = None, None, None
        s = '<%s init:%s fitted:%s mu:%s std:%s weight:%s>' %\
            (ctext('GMM:%d' % self.nmix, 'cyan'),
                self._is_initialized, self._is_fitted,
                mu, std, w)
        return s

    # ==================== properties ==================== #
    def is_initialized(self):
        return self._is_initialized

    def is_fitted(self):
        return self._is_fitted

    # ==================== sklearn ==================== #
    def _initialize(self, ndim):
        if self._is_initialized:
            return
        self.ndim = ndim
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

    def fit(self, X, y=None, log=False):
        # ====== check input ====== #
        self._initialize(ndim=X.shape[1])
        assert X.ndim == 2 and X.shape[1] == self.ndim
        # ====== divide the batches ====== #
        nframes = X.shape[0]
        jobs = np.linspace(start=0, stop=nframes, num=self.ncpu + 1, dtype='int32')
        jobs = list(zip(jobs, jobs[1:]))

        def _map_func(start_end):
            start, end = start_end[0]
            Z, F, S, L = 0., 0., 0., 0.
            for s, e in batching(n=end - start, batch_size=self.batch_size):
                s += start; e += start
                x = X[s:e]
                res_Z, res_F, res_S, res_L = self.expectation(x)
                Z += res_Z
                F += res_F
                S += res_S
                L += res_L
                yield x.shape[0]
            yield Z, F, S, L

        # ====== start GMM ====== #
        # supports 4096 components, modify for more components
        niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10]
        niter[int(np.log2(self.nmix))] = self.niter
        mix = 1
        while mix <= self.nmix:
            print('Re-estimating the GMM hyperparameters for {} components ...'.format(mix))
            for iter in range(niter[int(np.log2(mix))]):
                self.C_ = self.compute_C()
                # Expectation
                mpi = MPI(jobs, func=_map_func, ncpu=self.ncpu, batch=1, hwm=2**25,
                          backend='python')
                prog = Progbar(target=nframes, print_report=True,
                               print_summary=True,
                               name="#Mixture:%d  iter#:%d" % (mix, iter + 1))
                Z, F, S, L = 0., 0., 0., 0.
                for res in mpi:
                    if is_number(res):
                        prog.add(res)
                    else:
                        res_Z, res_F, res_S, res_L = res
                        Z += res_Z
                        F += res_F
                        S += res_S
                        L += res_L
                # Maximization
                self.maximization(Z, F, S)
                llk = L / nframes
                print("llk:", llk)
                del res
            if mix < self.nmix:
                self.gmm_mixup()
            mix *= 2

    def score(self, X, y=None):
        pass

    def transform(self, X):
        post, llk = self.postprob(X)
        N = zeroStat(post)
        F = firstStat(X, post)
        F_hat = np.reshape(F - self.mu * N, (self.ndim * self.nmix, 1),
                           order='F')
        return N, F_hat

    # ==================== math helper ==================== #
    def lgmmprob(self, X):
        precision = 1 / (self.sigma + EPS) # (D, M)
        D = np.dot(X ** 2, precision) - \
            2 * np.dot(X, self.mu * precision) + \
            self.ndim * np.log(2 * np.pi)
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
        if ndim != self.ndim:
            raise ValueError('Dimensionality of the data ({}) does not match '
                'the specified dimension ndim={}!'.format(ndim, self.ndim))
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
