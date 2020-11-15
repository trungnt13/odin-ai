from __future__ import absolute_import, division, print_function

import os
import unittest
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from odin.bay import vi
from odin.bay.vi import RVmeta, VariationalAutoencoder, factorVAE
from tensorflow.python import keras
from tqdm import tqdm

np.random.seed(8)
tf.random.set_seed(8)

# ===========================================================================
# Helpers
# ===========================================================================
tfpl = tfp.layers
tfd = tfp.distributions
fit_kw = dict(max_iter=25000, learning_rate=1e-3)
encoded_size = 16
base_depth = 32
input_shape = (None, 28, 28, 1)


def load_mnist():
  train, valid, test = tfds.load('binarized_mnist',
                                 split=['train', 'validation', 'test'])
  preprocess = lambda x: tf.cast(x['image'], tf.float32)
  return (train.map(preprocess).batch(32).repeat(-1),
          test.map(preprocess).batch(32).repeat(1))


def create_networks():
  # he_uniform is better for leaky_relu
  conv2D = partial(keras.layers.Conv2D,
                   padding='same',
                   kernel_initializer='he_uniform',
                   activation=tf.nn.leaky_relu)
  deconv2D = partial(keras.layers.Conv2DTranspose,
                     padding='same',
                     kernel_initializer='he_uniform',
                     activation=tf.nn.leaky_relu)
  encoder = keras.Sequential(
      [
          keras.layers.InputLayer(input_shape=input_shape[1:]),
          conv2D(base_depth, 5, strides=1, name='Encoder0'),
          conv2D(base_depth, 5, strides=2, name='Encoder1'),
          conv2D(2 * base_depth, 5, strides=1, name='Encoder2'),
          conv2D(2 * base_depth, 5, strides=2, name='Encoder3'),
          conv2D(
              4 * encoded_size, 7, strides=1, padding='valid', name='Encoder4'),
          keras.layers.Flatten(),
          keras.layers.Dense(
              tfpl.MultivariateNormalTriL.params_size(encoded_size),
              activation=None,
              name='Encoder5')
      ],
      name='encoder',
  )
  decoder = keras.Sequential(
      [
          keras.layers.InputLayer(input_shape=[encoded_size]),
          keras.layers.Reshape([1, 1, encoded_size]),
          deconv2D(
              2 * base_depth, 7, strides=1, padding='valid', name='Decoder0'),
          deconv2D(2 * base_depth, 5, strides=1, name='Decoder1'),
          deconv2D(2 * base_depth, 5, strides=2, name='Decoder2'),
          deconv2D(base_depth, 5, strides=1, name='Decoder3'),
          deconv2D(base_depth, 5, strides=2, name='Decoder4'),
          deconv2D(base_depth, 5, strides=1, name='Decoder5'),
          conv2D(1, 5, strides=1, activation=None, name='Decoder6'),
          keras.layers.Flatten()
      ],
      name='decoder',
  )
  latents = RVmeta(event_shape=(encoded_size,),
                   posterior='mvntril',
                   projection=False,
                   name="latents"),
  observation = RVmeta(event_shape=input_shape[1:],
                       posterior="bernoulli",
                       projection=False,
                       name="image"),
  return dict(encoder=encoder,
              decoder=decoder,
              observation=observation,
              latents=latents)


# ===========================================================================
# Tests
# ===========================================================================
class VAETest(unittest.TestCase):

  def test_traverse_dims(self):
    x = tf.reshape(tf.range(0, 12), (3, 4))
    y = traverse_dims(x, (1, 3), n_traverse_points=5, n_random_samples=2, seed=1)
    z = tf.convert_to_tensor(
        [[0, -2, 2, 3], [0, -1, 2, 3], [0, 0, 2, 3], [0, 1, 2, 3], [0, 2, 2, 3],
         [8, -2, 10, 11], [8, -1, 10, 11], [8, 0, 10, 11], [8, 1, 10, 11],
         [8, 2, 10, 11], [0, 1, 2, -2], [0, 1, 2, -1], [0, 1, 2, 0], [0, 1, 2, 1],
         [0, 1, 2, 2], [8, 9, 10, -2], [8, 9, 10, -1], [8, 9, 10, 0], [8, 9, 10, 1],
         [8, 9, 10, 2]],
        dtype=y.dtype)
    self.assertTrue(np.all(np.array_equal(y, z)))

  def test_permute_dims(self):
    tf.random.set_seed(1)
    x = tf.reshape(tf.range(8), (4, 2))
    z = vi.permute_dims(x)
    w = tf.convert_to_tensor([[0, 7], [2, 5], [4, 1], [6, 3]], dtype=tf.int32)
    self.assertTrue(np.all(z.numpy() == w.numpy()))

    x = tf.random.uniform((128, 64), dtype=tf.float64)
    z = vi.permute_dims(x)
    self.assertTrue(np.any(x.numpy() != z.numpy()))
    self.assertTrue(np.all(np.any(i != j) for i, j in zip(x, z)))
    self.assertTrue(
        all(i == j for i, j in zip(sorted(x.numpy().ravel()),
                                   sorted(z.numpy().ravel()))))

  def test_vanila_vae(self):
    train, test = load_mnist()
    # vae = VariationalAutoencoder(**create_networks(), analytic=True, name='vae')
    # vae.build(input_shape)
    # vae.fit(train, **fit_kw)

  def test_factor_vae(self):
    train, test = load_mnist()
    vae = factorVAE(**create_networks(), analytic=True, name='factorvae')
    vae.build(input_shape)
    mllk = tf.reduce_mean([vae.marginal_log_prob(x)[0] for x in test])
    print(mllk)
    # vae.fit(train, **fit_kw)

  # def test_all_models(self):
  #   all_vae = autoencoder.get_vae()
  #   for vae_cls in all_vae:
  #     for latents in [
  #         (autoencoder.RVmeta(10, name="Latent1"),
  #          autoencoder.RVmeta(10, name="Latent2")),
  #         autoencoder.RVmeta(10, name="Latent1"),
  #     ]:
  #       for sample_shape in [
  #           (),
  #           2,
  #           (4, 2, 3),
  #       ]:
  #         vae_name = vae_cls.__name__
  #         print(vae_name, sample_shape)
  #         try:
  #           if isinstance(vae_cls, autoencoder.VariationalAutoencoder):
  #             vae = vae_cls
  #           else:
  #             vae = vae_cls(latents=latents)
  #           params = vae.trainable_variables
  #           if hasattr(vae, 'discriminator'):
  #             disc_params = set(
  #                 id(v) for v in vae.discriminator.trainable_variables)
  #             params = [i for i in params if id(i) not in disc_params]
  #           s = vae.sample_prior()
  #           px = vae.decode(s)
  #           x = vae.sample_data(5)
  #           with tf.GradientTape(watch_accessed_variables=False) as tape:
  #             tape.watch(params)
  #             px, qz = vae(x, sample_shape=sample_shape)
  #             elbo = vae.elbo(x, px, qz, return_components=False)
  #           grads = tape.gradient(elbo, params)
  #           for p, g in zip(params, grads):
  #             assert g is not None, \
  #                 "Gradient is None, param:%s shape:%s" % (p.name, p.shape)
  #             g = g.numpy()
  #             assert np.all(np.logical_not(np.isnan(g))), \
  #                             "NaN gradient param:%s shape:%s" % (p.name, p.shape)
  #             assert np.all(np.isfinite(g)), \
  #                 "Infinite gradient param:%s shape:%s" % (p.name, p.shape)
  #         except Exception as e:
  #           raise e


if __name__ == '__main__':
  unittest.main()
