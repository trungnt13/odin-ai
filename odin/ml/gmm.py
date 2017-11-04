# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
"""
from __future__ import print_function, division, absolute_import

__version__ = '1.1'
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'

import time
import numpy as np
import multiprocessing as mp

EPS = np.finfo(float).eps


def unwrap_expectation(args):
    return GMM.expectation(*args)


def logsumexp(x, dim):
    xmax = x.max(axis=dim, keepdims=True)
    y = xmax + np.log(np.sum(np.exp(x - xmax), axis=dim, keepdims=True))
    return y


class GmmUtils:

    def __init__(self):
        pass

    def postprob(self, data):
        post = self.lgmmprob(data)
        llk = logsumexp(post, 0)
        post = np.exp(post - llk)
        return post, llk

    def compute_C(self):
        precision = 1 / self.sigma
        log_det = np.sum(np.log(self.sigma), 0, keepdims=True)
        return np.sum(self.mu * self.mu * precision, 0, keepdims=True) + \
            log_det - 2 * np.log(self.w)

    def lgmmprob(self, data):
        precision = 1 / self.sigma
        D = precision.T.dot(data * data) - \
            2 * (self.mu * precision).T.dot(data) + \
            self.ndim * np.log(2 * np.pi)
        return -0.5 * (self.C_.T + D)

    @staticmethod
    def compute_zeroStat(post):
        return np.sum(post, 1, keepdims=True).T

    @staticmethod
    def compute_firstStat(data, post):
        return data.dot(post.T)

    @staticmethod
    def compute_secondStat(data, post):
        return (data * data).dot(post.T)

    @staticmethod
    def compute_llk(post):
        return logsumexp(post, 1)


class GMM(GmmUtils):

    def __init__(self, ndim, nmix, ds_factor, final_niter, nworkers):
        # rounding up to the nearest power of 2
        self.nmix = int(np.power(2, np.ceil(np.log2(nmix))))
        self.final_iter = int(final_niter)
        self.ds_factor = int(ds_factor)
        self.nworkers = int(nworkers)
        self.ndim = int(ndim)
        self.mu = np.zeros((ndim, 1), dtype='f4')
        self.sigma = np.ones((ndim, 1), dtype='f4')
        self.w = np.ones((1, 1), dtype='f4')
        self.C_ = self.compute_C() # shape=(1, 1)

    def fit(self, data_list, gmmFilename=""):
        # binding of the main procedure gmm_em
        p = mp.Pool(processes=self.nworkers)
        print('\nInitializing the GMM hyperparameters ...\n')
        # supports 4096 components, modify for more components
        niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10]
        niter[int(np.log2(self.nmix))] = self.final_iter
        mix = 1
        while mix <= self.nmix:
            print('\nRe-estimating the GMM hyperparameters for {} components ...'.format(mix))
            for iter in range(niter[int(np.log2(mix))]):
                print('EM iter#: {} \t'.format(iter + 1), end=" ")
                self.C_ = self.compute_C()
                tic = time.time()
                res = self.expectation(data_list)
                N, F, S, L, nframes = GMM.reduce_expectation_res(res)
                self.maximization(N, F, S)
                print("[llk = {:.2f}]\t[elaps = {:.2f}s]".format(L / nframes, time.time() - tic))
                del res
            if mix < self.nmix:
                self.gmm_mixup()
            mix *= 2
        p.close()
        if gmmFilename:
            print('\nSaving GMM to file {}'.format(gmmFilename))
            self.save(gmmFilename)

    def load_data(self, datalist, p):
        # check the match between ndim and features dimensionality
        features_list = np.genfromtxt(datalist, dtype='str')
        nparts = self.nworkers * 10  # NOTE: this is set empirically
        split_f_list = np.array_split(features_list, nparts)
        data = p.map(read_data, split_f_list)
        return data

    def expectation(self, data_list):
        # The Map
        data = read_data(data_list, self.ds_factor)
        nfr, ndim = data.shape
        if ndim != self.ndim:
            raise ValueError('Dimensionality of the data ({}) does not match the specified dimension ndim={}!'.format(ndim, self.ndim))
        parts = 2500
        nbatch = int(nfr / parts + 0.99999)
        N, F, S, L = 0., 0., 0., 0.
        for batch in range(nbatch):  # Careful for the index
            start = batch * parts
            fin = min((batch + 1) * parts, nfr)
            data_b = data[start:fin, :]
            post, llk = self.postprob(data_b)
            N += GMM.compute_zeroStat(post)
            F += GMM.compute_firstStat(data_b, post)
            S += GMM.compute_secondStat(data_b, post)
            L += llk.sum()
        print(N.shape)
        print(F.shape)
        print(S.shape)
        print(L.shape)
        print(nfr)
        exit()
        return N, F, S, L, nfr

    @staticmethod
    def reduce_expectation_res(res):
        N, F, S, L, nframes = res[0]
        for r in res[1:]:
            n, f, s, l, nfr = r
            N += n
            F += f
            S += s
            L += l
            nframes += nfr
        return N, F, S, L, nframes

    def maximization(self, N, F, S):
        # TheReduce
        iN = 1. / (N + EPS)
        self.w = N / N.sum()
        self.mu = F * iN
        self.sigma = S * iN - self.mu * self.mu
#        self.apply_var_floors()

    def gmm_mixup(self):
        ndim, nmix = self.sigma.shape
        sig_max, arg_max = self.sigma.max(0), self.sigma.argmax(0)
        eps = np.zeros((ndim, nmix), dtype='f')
        eps[arg_max, np.arange(nmix)] = np.sqrt(sig_max)
        perturb = 0.55 * eps
        self.mu = np.c_[self.mu - perturb, self.mu + perturb]
        self.sigma = np.c_[self.sigma, self.sigma]
        self.w = 0.5 * np.c_[self.w, self.w]

    def apply_var_floors(self, floor_const=1e-3):
        vFloor = self.sigma.dot(self.w.T) * floor_const
        self.sigma = self.sigma.clip(vFloor)

    def load(self, gmmFilename):
        self.mu, self.sigma, self.w = h5read(gmmFilename, ['means', 'variances', 'weights'])
        self.C_ = self.compute_C()

    def save(self, gmmFilename):
        h5write(gmmFilename, [self.mu, self.sigma, self.w], ['means', 'variances', 'weights'])

    def compute_centered_stats(self, data):
        post = self.postprob(data)[0]
        N = GMM.compute_zeroStat(post)
        F = GMM.compute_firstStat(data, post)
        F_hat = np.reshape(F - self.mu * N, (self.ndim * self.nmix, 1), order='F')
        return N, F_hat

    def compute_log_lik(self, data):
        return self.postprob(data)[1]


def read_data(feature_files, ds_factor=2):
    data_list_temp = []
    for X in feature_files:
        data_list_temp.append(X[0::ds_factor, :])
    return np.row_stack(data_list_temp)
