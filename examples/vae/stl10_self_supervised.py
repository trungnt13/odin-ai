from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from odin.bay.vi import RVmeta as RV
from odin.bay.vi.autoencoder import betaVAE, multitaskVAE, SemifactorVAE
from odin.fuel import MNIST, STL10, LegoFaces

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)

ds = LegoFaces()
train = ds.create_dataset(partition='train', inc_labels=True)
train_l = ds.create_dataset(partition='train_labelled', inc_labels=True)
test = ds.create_dataset(partition='test', inc_labels=True)
train_u = ds.create_dataset(partition='train', inc_labels=False)
test_u = ds.create_dataset(partition='test', inc_labels=False)
save_path = f'/tmp/{ds.name}.w'

vae = multitaskVAE(encoder=ds.name,
                   alpha=10.,
                   outputs=RV(ds.shape,
                              'bernoulli',
                              projection=False,
                              name='Image'),
                   labels=RV(10, 'onehot', projection=True, name="Labels"),
                   path=save_path)
vae.fit(
    train_l,
    learning_rate=1e-4,
    max_iter=20000,
).fit(
    train_u,
    learning_rate=1e-4,
    max_iter=80000,
    earlystop_threshold=0.001,
    earlystop_patience=-1,
    compile_graph=True,
).save_weights(vae.save_path)

z = vae.sample_prior(64)
img = tf.nest.flatten(vae.decode(z))[0].mean().numpy()
fig = plt.figure(figsize=(8, 8))
for idx, i in enumerate(img):
  ax = plt.subplot(8, 8, idx + 1)
  if i.shape[-1] == 1:
    i = np.squeeze(i, axis=-1)
  ax.imshow(i)
  ax.axis('off')
fig.tight_layout()
fig.savefig(f'/tmp/{ds.name}_z.png', dpi=100)
vae.plot_learning_curves(f'/tmp/{ds.name}.png')
