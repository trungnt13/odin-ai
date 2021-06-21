import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from odin.bay import TwoStageVAE, plot_latent_stats
from odin.fuel import MNIST
from odin.networks import get_networks
from odin import visual as vs
from tqdm import tqdm
from odin.ml import fast_tsne

ds = MNIST()
train = ds.create_dataset('train', batch_size=32)
valid = ds.create_dataset('valid', batch_size=36, label_percent=1.0,
                          drop_remainder=True)

vae = TwoStageVAE(**get_networks(ds.name))
vae.build(ds.full_shape)
if True:
  vae.load_weights('/tmp/twostagevae', verbose=True, raise_notfound=True)
else:
  vae.fit(train, learning_rate=1e-3, max_iter=300000)
  vae.save_weights('/tmp/twostagevae')
  exit()

Z = []
U = []
Z_hat = []
Y = []
for x, y in tqdm(valid):
  qz_x, qu_z, qz_u = vae.encode_two_stages(x)
  Z.append(qz_x.mean())
  U.append(qu_z.mean())
  Z_hat.append(qz_u.mean())
  Y.append(np.argmax(y, axis=-1))
Z = np.concatenate(Z, 0)[:5000]
U = np.concatenate(U, 0)[:5000]
Z_hat = np.concatenate(Z_hat, 0)[:5000]
Y = np.concatenate(Y, 0)[:5000]

plt.figure(figsize=(15, 5), dpi=150)
vs.plot_scatter(fast_tsne(Z), color=Y, grid=False, ax=(1, 3, 1))
vs.plot_scatter(fast_tsne(U), color=Y, grid=False, ax=(1, 3, 2))
vs.plot_scatter(fast_tsne(Z_hat), color=Y, grid=False, ax=(1, 3, 3))
plt.tight_layout()

ids = np.argsort(np.mean(qz_x.stddev(), 0))
ids_u = np.argsort(np.mean(qu_z.stddev(), 0))

plt.figure(figsize=(10, 10), dpi=200)
plot_latent_stats(mean=np.mean(qz_x.mean(), 0)[ids],
                  stddev=np.mean(qz_x.stddev(), 0)[ids],
                  ax=(3, 1, 1), name='q(z|x)')
plot_latent_stats(mean=np.mean(qu_z.mean(), 0)[ids_u],
                  stddev=np.mean(qu_z.stddev(), 0)[ids_u],
                  ax=(3, 1, 2), name='q(u|z)')
plot_latent_stats(mean=np.mean(qz_u.mean(), 0)[ids],
                  stddev=np.mean(qz_u.stddev(), 0)[ids],
                  ax=(3, 1, 3), name='q(z|u)')
plt.tight_layout()

vae.set_eval_stage(1)
px1, _ = vae(x)
llk1 = np.mean(tf.concat([vae(x)[0].log_prob(x) for x, _ in tqdm(valid)], 0))
print('Stage1:', llk1)

vae.set_eval_stage(2)
px2, _ = vae(x)
llk2 = np.mean(tf.concat([vae(x)[0].log_prob(x) for x, _ in tqdm(valid)], 0))
print('Stage2:', llk2)

images = np.squeeze(px1.mean().numpy(), -1)
plt.figure(figsize=(10, 10), dpi=150)
for i in range(36):
  img = images[i]
  plt.subplot(6, 6, i + 1)
  plt.imshow(img, cmap='Greys_r')
  plt.axis('off')
  plt.margins(0)
plt.tight_layout()

images = np.squeeze(px2.mean().numpy(), -1)
plt.figure(figsize=(10, 10), dpi=150)
for i in range(36):
  img = images[i]
  plt.subplot(6, 6, i + 1)
  plt.imshow(img, cmap='Greys_r')
  plt.axis('off')
  plt.margins(0)
plt.tight_layout()

images = np.squeeze(vae.sample_observation(36, two_stage=False).mean().numpy(),
                    -1)
plt.figure(figsize=(10, 10), dpi=150)
for i in range(36):
  img = images[i]
  plt.subplot(6, 6, i + 1)
  plt.imshow(img, cmap='Greys_r')
  plt.axis('off')
  plt.margins(0)
plt.tight_layout()

images = np.squeeze(vae.sample_observation(36, two_stage=True).mean().numpy(),
                    -1)
plt.figure(figsize=(10, 10), dpi=150)
for i in range(36):
  img = images[i]
  plt.subplot(6, 6, i + 1)
  plt.imshow(img, cmap='Greys_r')
  plt.axis('off')
  plt.margins(0)
plt.tight_layout()

vs.plot_save()
