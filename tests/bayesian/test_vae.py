from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from odin.bay import vi
from odin.bay.vi import autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(8)
tf.random.set_seed(8)


class VAETest(unittest.TestCase):

  def test_permute_dims(self):
    x = tf.reshape(tf.range(8), (4, 2))
    z = vi.permute_dims(x)
    w = tf.convert_to_tensor([[2, 5], [4, 7], [6, 3], [0, 1]], dtype=tf.int32)
    self.assertTrue(np.all(z.numpy() == w.numpy()))

    x = tf.random.uniform((128, 64), dtype=tf.float64)
    z = vi.permute_dims(x)
    self.assertTrue(np.any(x.numpy() != z.numpy()))
    self.assertTrue(np.all(np.any(i != j) for i, j in zip(x, z)))
    self.assertTrue(
        all(i == j for i, j in zip(sorted(x.numpy().ravel()),
                                   sorted(z.numpy().ravel()))))

  def test_all_models(self):
    all_vae = autoencoder.get_vae()
    for vae_cls in all_vae:
      for latents in [
          (autoencoder.RandomVariable(10, name="Latent1"),
           autoencoder.RandomVariable(10, name="Latent2")),
          autoencoder.RandomVariable(10, name="Latent1"),
      ]:
        for sample_shape in [
            (),
            2,
            (4, 2, 3),
        ]:
          vae_name = vae_cls.__name__
          print(vae_name, sample_shape)
          try:
            if isinstance(vae_cls, autoencoder.VariationalAutoencoder):
              vae = vae_cls
            else:
              vae = vae_cls(latents=latents)
            params = vae.trainable_variables
            if hasattr(vae, 'discriminator'):
              disc_params = set(
                  id(v) for v in vae.discriminator.trainable_variables)
              params = [i for i in params if id(i) not in disc_params]
            s = vae.sample_prior()
            px = vae.decode(s)
            x = vae.sample_data(5)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
              tape.watch(params)
              px, qz = vae(x, sample_shape=sample_shape)
              elbo, llk, div = vae.elbo(x, px, qz)
            grads = tape.gradient(elbo, params)
            for p, g in zip(params, grads):
              assert g is not None, \
                  "Gradient is None, param:%s shape:%s" % (p.name, p.shape)
              g = g.numpy()
              assert np.all(np.logical_not(np.isnan(g))), \
                              "NaN gradient param:%s shape:%s" % (p.name, p.shape)
              assert np.all(np.isfinite(g)), \
                  "Infinite gradient param:%s shape:%s" % (p.name, p.shape)
          except Exception as e:
            raise e


if __name__ == '__main__':
  unittest.main()
