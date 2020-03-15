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
    x = np.random.rand(5, 12).astype('float32')
    for cov in ('tril', 'spherical', 'diag'):
      gmm = obd.GaussianMixture(loc=np.random.rand(5, 2, 12).astype('float32'),
                                scale=np.random.rand(
                                    5, 2,
                                    obd.GaussianMixture.params_size(
                                        12, cov)).astype('float32'),
                                logits=np.random.rand(5, 2).astype('float32'),
                                covariance_type=cov)
      print(gmm, gmm.sample().shape)
      gmm.log_prob(x)


if __name__ == '__main__':
  unittest.main()
