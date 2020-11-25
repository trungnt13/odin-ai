import os
from functools import partial
from itertools import product

import seaborn as sns
from matplotlib import pyplot as plt
from odin import visual as vs

sns.set()

import numpy as np
import tensorflow as tf
from odin.bay import distributions as D
from odin.bay import vi
from odin.training import get_current_trainer, get_output_dir, run_hydra
from odin.fuel import MNIST, FashionMNIST, dSprites
from tensorflow_probability.python import bijectors as bj

# ===========================================================================
# Config
# ===========================================================================
CONFIG = \
"""
vae:
"""
output_dir = '/tmp/vae_pp'

# network configuration
batch_size = 32
max_iter = 50000
encoder = vi.NetworkConfig([256, 256, 256], flatten_inputs=True, name='Encoder')
decoder = vi.NetworkConfig([256, 256, 256], flatten_inputs=True, name='Decoder')
encoded_size = 16
posteriors_info = [
    ('gaussian', 'mvndiag', 'mvntril'),
    (
        D.Sample(D.Normal(loc=0., scale=1.),
                 sample_shape=encoded_size,
                 name='independent'),
        D.MultivariateNormalDiag(loc=tf.zeros(encoded_size),
                                 scale_diag=tf.ones(encoded_size),
                                 name='mvndiag'),
        D.MultivariateNormalTriL(loc=tf.zeros(encoded_size),
                                 scale_tril=bj.FillScaleTriL()(tf.ones(
                                     encoded_size * (encoded_size + 1) // 2)),
                                 name='mvntril'),
        D.MixtureSameFamily(
            components_distribution=D.MultivariateNormalDiag(
                loc=tf.zeros([10, encoded_size]),
                scale_diag=tf.ones([10, encoded_size])),
            mixture_distribution=D.Categorical(logits=tf.fill([10], 1.0 / 10)),
            name='gmm10'),
        D.MixtureSameFamily(components_distribution=D.MultivariateNormalDiag(
            loc=tf.zeros([100, encoded_size]),
            scale_diag=tf.ones([100, encoded_size])),
                            mixture_distribution=D.Categorical(
                                logits=tf.fill([100], 1.0 / 100)),
                            name='gmm100'),
    ),
    ('identity', 'relu', 'softplus', 'softplus1'),
]


# ===========================================================================
# Main
# ===========================================================================
def callback(vae: vi.VariationalAutoencoder, x: np.ndarray, y: np.ndarray):
  trainer = get_current_trainer()
  px, qz = [], []
  X_i = []
  for x_i in tf.data.Dataset.from_tensor_slices(x).batch(64):
    _ = vae(x_i, training=False)
    px.append(_[0])
    qz.append(_[1])
    X_i.append(x_i)
  # llk
  llk_test = tf.reduce_mean(
      tf.concat([p.log_prob(x_i) for p, x_i in zip(px, X_i)], axis=0))
  # latents
  qz_mean = tf.reduce_mean(tf.concat([q.mean() for q in qz], axis=0), axis=0)
  qz_std = tf.reduce_mean(tf.concat([q.stddev() for q in qz], axis=0), axis=0)
  w = tf.reduce_sum(vae.decoder.trainable_variables[0], axis=1)
  # plot the latents and its weights
  fig = plt.figure(figsize=(6, 4), dpi=200)
  ax = plt.gca()
  l1 = ax.plot(qz_mean,
               label='mean',
               linewidth=1.0,
               linestyle='--',
               marker='o',
               markersize=4,
               color='r',
               alpha=0.5)
  l2 = ax.plot(qz_std,
               label='std',
               linewidth=1.0,
               linestyle='--',
               marker='o',
               markersize=4,
               color='g',
               alpha=0.5)
  ax1 = ax.twinx()
  l3 = ax1.plot(w,
                label='weight',
                linewidth=1.0,
                linestyle='--',
                marker='o',
                markersize=4,
                color='b',
                alpha=0.5)
  lines = l1 + l2 + l3
  labs = [l.get_label() for l in lines]
  ax.grid(True)
  ax.legend(lines, labs)
  img_qz = vs.plot_to_image(fig)
  # reconstruction
  img = px[10].mean().numpy()
  if img.shape[-1] == 1:
    img = np.squeeze(img, axis=-1)
  fig = plt.figure(figsize=(8, 8), dpi=120)
  vs.plot_images(img, grids=(8, 8))
  img_reconstructed = vs.plot_to_image(fig)
  # latents traverse
  # TODO
  return dict(llk_test=llk_test,
              qz_mean=qz_mean,
              qz_std=qz_std,
              w_decoder=w,
              reconstructed=img_reconstructed,
              latents=img_qz)


@run_hydra(output_dir=output_dir)
def main(cfg: dict):
  assert cfg.vae is not None, 'vae is not provided in the configuration'
  outdir = get_output_dir()
  # load dataset
  ds = MNIST()
  shape = ds.shape
  train = ds.create_dataset(partition='train',
                            batch_size=batch_size,
                            shuffle=2000,
                            drop_remainder=True)
  valid = ds.create_dataset(partition='valid', batch_size=batch_size)
  x_test, y_test = ds.numpy(partition='test', shuffle=None, inc_labels=True)
  y_test = ds.labels[np.argmax(y_test, axis=-1)]
  # create the model
  vae_class = vi.get_vae(cfg.vae)
  for i, (posterior, prior, activation) in enumerate(product(*posteriors_info)):
    name = f"{posterior}_{prior.name}_{activation}"
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
      os.makedirs(path)
    model_path = os.path.join(path, 'model')
    vae = vae_class(encoder=encoder.create_network(),
                    decoder=decoder.create_network(),
                    observation=vi.RVmeta(shape,
                                          'bernoulli',
                                          projection=True,
                                          name='Image'),
                    latents=vi.RVmeta(encoded_size,
                                      posterior,
                                      projection=True,
                                      prior=prior,
                                      kwargs=dict(scale_activation=activation),
                                      name='Latents'),
                    analytic=False,
                    path=model_path,
                    name=name)
    vae.build((None,) + shape)
    vae.load_weights()
    vae.fit(train=train,
            valid=valid,
            max_iter=max_iter,
            valid_freq=1000,
            compile_graph=True,
            skip_fitted=True,
            callback=partial(callback, vae=vae, x=x_test, y=y_test),
            logdir=path,
            track_gradients=True).save_weights()


if __name__ == "__main__":
  main(CONFIG)
