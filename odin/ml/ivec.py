# -*- coding: utf-8 -*-
""""
This module contains tools T matrix training and i-vector extraction
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'
"""


import os
import time
import glob
import random
import string
import multiprocessing as mp
import numpy as np
from scipy.linalg import inv, svd, cholesky, solve

from sklearn.base import DensityMixin, BaseEstimator, TransformerMixin


def unwrap_expectation_tv(args):
    return TMatrix.expectation_tv(*args)


class Ivector(BaseEstimator, TransformerMixin):

    def __init__(self, tv_dim, ndim, nmix):
        self.tv_dim = tv_dim
        self.ndim = ndim
        self.nmix = nmix
        self.itril = np.tril_indices(tv_dim)
        self.Sigma = np.empty((self.ndim * self.nmix, 1))
        self.T_iS = None  # np.empty((self.tv_dim, self.ndim * self.nmix))
        self.T_iS_Tt = None  # np.empty((self.nmix, self.tv_dim * (self.tv_dim+1)/2))
        self.Tm = np.empty((self.tv_dim, self.ndim * self.nmix))
        self.Im = np.eye(self.tv_dim)

    def load_ubm(self, ubmFilename):
        sigma = h5read(ubmFilename, 'variances')[0]
        ndim, nmix = sigma.shape
        if self.ndim != ndim or self.nmix != nmix:
            raise ValueError('UBM nmix and ndim do not match what was specified!')
        self.Sigma = sigma.reshape((1, self.ndim * self.nmix), order='F')

    def tmat_init(self, T_mat_filename=""):
        if T_mat_filename:
            self.Tm = h5read(T_mat_filename, 'T')[0]
        else:
            np.random.seed(7)
            print('\n\nRandomly initializing T matrix ...\n')
            self.Tm = np.random.randn(self.tv_dim, self.ndim * self.nmix) * self.Sigma.sum() * 0.001

    def initialize(self, ubmFilename, T_mat_filename):
        self.load_ubm(ubmFilename)
        self.tmat_init(T_mat_filename)
        self.T_iS = self.Tm / self.Sigma
        self.T_iS_Tt = self.comp_T_invS_Tt()

    def comp_T_invS_Tt(self):
        T_invS2 = self.Tm / np.sqrt(self.Sigma)
        T_invS_Tt = np.zeros((self.nmix, self.tv_dim * (self.tv_dim + 1) // 2))
        for mix in range(self.nmix):
            idx = np.arange(self.ndim) + mix * self.ndim
            tmp = T_invS2[:, idx].dot(T_invS2[:, idx].T)
            T_invS_Tt[mix] = tmp[self.itril]
        return T_invS_Tt

    def extract(self, N, F):
        L = np.zeros((self.tv_dim, self.tv_dim))
        L[self.itril] = N.dot(self.T_iS_Tt)
        L += np.tril(L, -1).T + self.Im
        Cxx = inv(L)
        B = self.T_iS.dot(F)
        Ex = Cxx.dot(B)
        return Ex


class TMatrix(Ivector):

    def __init__(self, tv_dim, ndim, nmix, niter, nworkers):
        super().__init__(tv_dim, ndim, nmix)
        self.niter = niter
        self.nworkers = nworkers
        self.tmpdir = ""

    def train(self, dataList, ubmFilename, tmpdir="", tvFilename=""):
        self.tmpdir = tmpdir
        if self.tmpdir:
            mkdir_p(self.tmpdir)
        if type(dataList) == str:
            datafile = np.genfromtxt(dataList, dtype='str')
        else:
            datafile = dataList
        nparts = min(self.nworkers, len(datafile))
        data_split = np.array_split(datafile, nparts)
        nfiles = datafile.size
        self.load_ubm(ubmFilename)
        self.tmat_init()

        print('Re-estimating the total subspace with {} factors ...'.format(self.tv_dim))
        p = mp.Pool(self.nworkers)
        for iter in range(self.niter):
            print('EM iter#: {} \t'.format(iter + 1), end=" ")
            tic = time.time()
            res = p.map(self.expectation_tv, data_split)
            LU, RU, LLK, nframes = TMatrix.reduce_expectation_res(res, self.tmpdir)
            self.maximization_tv(LU, RU)
            self.min_div_est(LU, nframes)
            self.make_orthogonal()
            tac = time.time() - tic
            print('[llk = {0:.2f}] \t\t [elaps = {1:.2f}s]'.format(LLK / nfiles, tac))
        p.close()
        if tvFilename:
            print('\nSaving Tmatrix to file {}'.format(tvFilename))
            h5write(tvFilename, self.Tm, 'T')
        # return self.Tm

    def expectation_tv(self, data_list):
        N, F = read_data(data_list, self.nmix, self.ndim)
        nfiles = F.shape[0]
        nframes = N.sum()
        LU = np.zeros((self.nmix, self.tv_dim * (self.tv_dim + 1) // 2))
        RU = np.zeros((self.tv_dim, self.nmix * self.ndim))
        LLK = 0.
        T_invS = self.Tm / self.Sigma
        T_iS_Tt = self.comp_T_invS_Tt()
        parts = 2500  # modify this based on your HW resources (e.g., memory)
        nbatch = int(nfiles / parts + 0.99999)
        for batch in range(nbatch):
            start = batch * parts
            fin = min((batch + 1) * parts, nfiles)
            length = fin - start
            N1 = N[start:fin]
            F1 = F[start:fin]
            L1 = N1.dot(T_iS_Tt)
            B1 = F1.dot(T_invS.T)
            Ex = np.empty((length, self.tv_dim))
            Exx = np.empty((length, self.tv_dim * (self.tv_dim + 1) // 2))
            llk = np.zeros((length, 1))
            for ix in range(length):
                L = np.zeros((self.tv_dim, self.tv_dim))
                L[self.itril] = L1[ix]
                L += np.tril(L, -1).T + self.Im
                Cxx = inv(L)
                B = B1[ix][:, np.newaxis]
                this_Ex = Cxx.dot(B)
                llk[ix] = self.res_llk(this_Ex, B)
                Ex[ix] = this_Ex.T
                Exx[ix] = (Cxx + this_Ex.dot(this_Ex.T))[self.itril]
            RU += Ex.T.dot(F1)
            LU += N1.T.dot(Exx)
            LLK += llk.sum()
        self.Tm = None
        tmp_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        tmpfile = self.tmpdir + 'tmat_' + tmp_string + '.h5'
        h5write(tmpfile, LU, 'LU')
        return RU, LLK, nframes

    def res_llk(self, Ex, B):
        return -0.5 * Ex.T.dot(B - Ex) + Ex.T.dot(B)

    @staticmethod
    def reduce_expectation_res(res, tmpdir):
        tfiles = glob.glob(tmpdir + '/tmat_*')
        LU = 0.
        for f in tfiles:
            LU += h5read(f, 'LU')[0]
            os.remove(f)
        RU, LLK, nframes = res[0]
        for r in res[1:]:
            ru, llk, nfr = r
            RU += ru
            LLK += llk
            nframes += nfr
        return LU, RU, LLK, nframes

    def maximization_tv(self, LU, RU):
        # ML re-estimation of the total subspace matrix or the factor loading
        # matrix
        for mix in range(self.nmix):
            idx = np.arange(self.ndim) + mix * self.ndim
            Lu = np.zeros((self.tv_dim, self.tv_dim))
            Lu[self.itril] = LU[mix, :]
            Lu += np.tril(Lu, -1).T
            self.Tm[:, idx] = solve(Lu, RU[:, idx])

    def min_div_est(self, LU, nframes):
        Lu = np.zeros((self.tv_dim, self.tv_dim))
        Lu[self.itril] = LU.sum(0) / nframes
        Lu += np.tril(Lu, -1).T
        self.Tm = cholesky(Lu).dot(self.Tm)

    def make_orthogonal(self):
        # orthogonalize the columns
        U, s, V = svd(self.Tm, full_matrices=False)
        self.Tm = np.diag(s).dot(V)


def read_data(stat_list, nmix, ndim):
    nfiles = stat_list.size
    N = np.empty((nfiles, nmix), dtype='f')
    F = np.empty((nfiles, nmix * ndim), dtype='f')
    for i, stat_file in enumerate(stat_list):
        n, f = h5read(stat_file, ['N', 'F'])
        N[i], F[i] = n, f.T
    return N, F
