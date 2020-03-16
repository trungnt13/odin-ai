from __future__ import absolute_import, division, print_function

import os
import unittest
from tempfile import mkstemp

import numpy as np
import tensorflow as tf

from odin.bay import distributions as obd

np.random.seed(8)


class MixtureTest(unittest.TestCase):

  def test_gmm(self):
    nsamples = 3
    ndims = 5
    x = np.random.rand(nsamples, ndims).astype('float32')
    #
    for cov in ('tril', 'none', 'diag'):
      gmm = obd.GaussianMixture(
          loc=np.random.rand(nsamples, 2, ndims).astype('float32'),
          scale=np.random.rand(nsamples, 2,
                               obd.GaussianMixture.params_size(
                                   ndims, cov)).astype('float32'),
          logits=np.random.rand(nsamples, 2).astype('float32'),
          covariance_type=cov)
      print(gmm, gmm.sample().shape)
      gmm.log_prob(x)
    #
    for cov in ('tied', 'full', 'diag', 'spherical'):
      tfp_gmm, sk_gmm = obd.GaussianMixture.fit(x,
                                                n_components=2,
                                                covariance_type=cov,
                                                return_sklearn=True)
      print(cov, tfp_gmm, tfp_gmm.sample().shape)
      llk1 = tfp_gmm.log_prob(x).numpy()
      llk2 = sk_gmm.score_samples(x)
      assert np.all(
          np.isclose(llk1, llk2, rtol=1.e-3, atol=1.e-3, equal_nan=True))

  def test_trainable_gmm(self):
    X = 2 + 3 * np.random.randn(100000, 5)
    gmm = obd.GaussianMixture.init(X,
                                   n_components=3,
                                   covariance_type='tril',
                                   max_samples=64,
                                   trainable=True)
    gmm.fit(X, verbose=True, max_iter=1000)

if __name__ == '__main__':
  unittest.main()
