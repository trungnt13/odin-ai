from __future__ import absolute_import, division, print_function

import os
import unittest
from itertools import product
from tempfile import mkstemp

import numpy as np
import tensorflow as tf

from odin.bay import distributions as obd
from odin.bay.layers import MixtureDensityNetwork, MixtureMassNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(8)
tf.random.set_seed(1)


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

  def test_mixture_density_network(self):
    x = tf.random.uniform((8, 4), dtype='float32')
    it = [True, False]
    for i, (covariance, tie_mixtures, tie_loc, tie_scale) in enumerate(
        product(['none', 'diag', 'tril'], it, it, it)):
      print(f"#{i} MixtureDensityNetwork tie_mixtures:{tie_mixtures} "
            f"tie_loc:{tie_loc} tie_scale:{tie_scale} covariance:{covariance}")
      kw = dict(covariance=covariance,
                tie_mixtures=tie_mixtures,
                tie_loc=tie_loc,
                tie_scale=tie_scale)
      try:
        net = MixtureDensityNetwork(units=5, **kw)
        y = net(x)
      except ValueError as e:
        pass
      except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

  def test_mixture_mass_network(self):
    x = tf.random.uniform((8, 4), dtype='float32')
    it = [True, False]

    for i, (alternative, tie_mixtures, tie_mean, dispersion,
            inflation) in enumerate(
                product(it, it, it, ['full', 'share', 'single'],
                        ['full', 'share', 'single', None])):
      print(f"#{i} MixtureMassNetwork tie_mixtures:{tie_mixtures} "
            f"tie_mean:{tie_mean} disp:{dispersion} inflated:{inflation}")
      kw = dict(tie_mixtures=tie_mixtures,
                tie_mean=tie_mean,
                zero_inflated=inflation is not None,
                dispersion=dispersion,
                inflation='full' if inflation is None else inflation,
                alternative=alternative)
      try:
        net = MixtureMassNetwork(event_shape=(5,), **kw)
        y = net(x)
      except ValueError as e:
        pass
      except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
  unittest.main()
