from functools import partial

import numpy as np
import glob
import os
import shutil
from argparse import ArgumentParser
from typing import List, Tuple

from tensorflow_probability.python.distributions import Distribution, Bernoulli, \
  Independent
from tensorflow_probability.python.layers import DistributionLambda
from tqdm import tqdm

from odin.bay import VariationalAutoencoder, HierarchicalVAE, \
  BidirectionalLatents
from odin.bay.distributions import QuantizedLogistic
from odin.fuel import CIFAR10, Shapes3DSmall, ImageDataset
from tensorflow.keras.layers import *
from odin.networks import CenterAt0, get_optimizer_info
import tensorflow as tf
from tensorflow.keras import Sequential
from odin.bay.layers import MVNDiagLatents, NormalLatents
from odin.bay.random_variable import RVconf
from odin.utils import as_tuple, minibatch
from odin.networks import resnets as rn

# ===========================================================================
# Constants and helpers
# ===========================================================================
ROOT_PATH = os.path.expanduser('~/exp/hierarchical')
if not os.path.exists(ROOT_PATH):
  os.makedirs(ROOT_PATH)

IMAGE_SHAPE = [32, 32, 3]
BATCH_SIZE = 32
ZDIM = 256


def find(name: str, layers: List[Layer]) -> Tuple[int, Layer]:
  for idx, l in enumerate(layers):
    if l.name == name:
      return idx, l
  raise ValueError(f'Cannot find layer with name {name}')


# ===========================================================================
# Layers
# * Upsampling and pooling is not recommended
# * Residual make training more stable but promoting posterior collapse
# ===========================================================================
RES_KW = dict(filters_in=64, filters_out=64, expand_ratio=2.,
              batchnorm=False, activation=tf.nn.elu)
Skip = partial(rn.Skip, coef=1.0)
Tonv = partial(Conv2DTranspose,
               # filters=64,
               # kernel_size=3,
               # strides=1,
               padding='same',
               activation=tf.nn.elu,
               kernel_initializer='he_normal')
Conv = partial(Conv2D,
               # filters=64,
               # kernel_size=3,
               # strides=1,
               padding='same',
               activation=tf.nn.elu,
               kernel_initializer='he_normal')


def encoder() -> List[Layer]:
  return [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    AveragePooling2D(),  # (16, 16)
    Conv(32, 3, 1, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]


def decoder(zdim, n_params=1) -> List[Layer]:
  return [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]


# def encoder() -> List[Layer]:
#   return [
#     InputLayer(IMAGE_SHAPE),
#     CenterAt0(),
#     Conv(32, 3, 1, name='e1'),  # (32, 32)
#     Conv(64, 4, 2, name='e2'),  # (16, 16)
#     Conv(64, 3, 1, name='e3'),  # (16, 16)
#     Conv(64, 4, 2, name='e4'),  # (8, 8)
#     Conv(64, 3, 1, name='e5'),  # (8, 8)
#     Conv(64, 4, 2, name='e6'),  # (4, 4)
#     Conv(64, 3, 1, name='e7'),  # (4, 4)
#     Flatten(),
#     Dense(512)
#   ]
#
#
# def decoder(zdim, n_params=1) -> List[Layer]:
#   return [
#     InputLayer([zdim]),
#     Dense(512),
#     Reshape([4, 4, 32]),
#     Conv(64, 3, 1, name='d7'),  # (4, 4)
#     Conv(64, 3, 1, name='d6'),  # (4, 4)
#     UpSampling2D(),
#     Conv(64, 3, 1, name='d5'),  # (8, 8)
#     Conv(64, 3, 1, name='d4'),  # (8, 8)
#     UpSampling2D(),  # (16, 16)
#     Conv(64, 3, 1, name='d3'),  # (16, 16)
#     Conv(64, 3, 1, name='d2'),  # (16, 16)
#     UpSampling2D(),  # (32, 32)
#     Conv(32, 3, 1, name='d1'),  # (32, 32)
#     Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),  # (32, 32, 3)
#   ]
#

# Tonv(64, 4, 2, name='d7'),  # (4, 4)
# Conv(64, 3, 1, name='d6'),  # (4, 4)
# Tonv(64, 4, 2, name='d5'),  # (8, 8)
# Conv(64, 3, 1, name='d4'),  # (8, 8)
# Tonv(64, 4, 2, name='d3'),  # (16, 16)
# Conv(64, 3, 1, name='d2'),  # (16, 16)
# Tonv(32, 4, 2, name='d1'),  # (32, 32)


# def encoder() -> List[Layer]:
#   return [
#     InputLayer(IMAGE_SHAPE),
#     CenterAt0(),
#     Conv(name='e0'),  # (32, 32, 64)
#     AveragePooling2D(),  # (16, 16)
#     Skip(Conv(name='e1')),
#     AveragePooling2D(),  # (8, 8)
#     Skip(Conv(name='e2')),
#     AveragePooling2D(),  # (4, 4)
#     Skip(Conv(name='e3')),
#     AveragePooling2D(),  # (2, 2)
#     Skip(Conv(name='e4')),
#     Flatten(),
#     Dense(256)
#   ]
#
#
# def decoder(zdim, n_params) -> List[Layer]:
#   return [
#     InputLayer([zdim]),
#     Dense(256),
#     Reshape([2, 2, 64]),  # (2, 2, 64)
#     Skip(Conv(name='d4')),
#     UpSampling2D(),  # (4, 4)
#     Skip(Conv(name='d3')),
#     UpSampling2D(),  # (8, 8)
#     Skip(Conv(name='d2')),
#     UpSampling2D(),  # (16, 16)
#     Skip(Conv(name='d1')),
#     UpSampling2D(),  # (32, 32)
#     Skip(Conv(name='d0')),
#     Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),  # (32, 32, 3)
#   ]
#


def latents(zdim):
  return MVNDiagLatents(zdim)


def bi_latents(idx, filters, kernel, strides, e_layers, d_layers, name):
  i1, d = find(f'd{idx}', d_layers)
  i2, e = find(f'e{idx}', e_layers)
  d = BidirectionalLatents(d, encoder=e, filters=filters,
                           kernel_size=kernel, strides=strides,
                           name=name)
  d_layers[i1] = d


def observation(distribution: str):
  if distribution == 'qlogistic':
    n_params = 2
    obs = DistributionLambda(
      lambda params: QuantizedLogistic(
        *[
          # loc
          p if i == 0 else
          # Ensure scales are positive and do not collapse to near-zero
          tf.nn.softplus(p) + tf.cast(tf.exp(-7.), tf.float32)
          for i, p in enumerate(tf.split(params, 2, -1))],
        low=0,
        high=255,
        inputs_domain='sigmoid',
        reinterpreted_batch_ndims=3),
      convert_to_tensor_fn=Distribution.sample,
      name='image')
  elif distribution == 'bernoulli':
    n_params = 1
    obs = DistributionLambda(
      lambda params: Independent(Bernoulli(logits=params, dtype=tf.float32),
                                 len(IMAGE_SHAPE)),
      convert_to_tensor_fn=Distribution.sample,
      name='image')
  else:
    raise NotImplementedError
  return n_params, obs


def model_vae(zdim, dist):
  n_params, obs = observation(dist)
  return VariationalAutoencoder(
    encoder=Sequential(encoder(), 'Encoder'),
    decoder=Sequential(decoder(zdim, n_params), 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,64x8x4,------,32x8x4
def model_bvae1(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 32x3x1,d,32x3x1,64x3x1,d,64x3x1
# 64x3x1,u,64x3x1,32x3x1,u,32x3x1
# --------,64x8x4,--------,32x8x4
def model_bvae2(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    AveragePooling2D(),  # (16, 16)
    Conv(32, 3, 1, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    AveragePooling2D(),  # (8, 8)
    Conv(64, 3, 1, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Conv(64, 3, 1, name='d3'),  # (8, 8, 64)
    UpSampling2D(),  # (16, 16)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Conv(32, 3, 1, name='d1'),  # (16, 16, 32)
    UpSampling2D(),  # (32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),
  ]
  bi_latents(2, 64, 8, 4, e_layers, d_layers, 'latents1')
  bi_latents(0, 32, 8, 4, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,4x3x1,------,2x3x1
def model_bvae3(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 4, 3, 1, e_layers, d_layers, 'latents1')
  bi_latents(0, 2, 3, 1, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# 32x3x1,32x4x2,64x3x1,64x4x2
# 64x4x2,64x3x1,32x4x2,32x3x1
# ------,16x4x2,------,8x4x2
def model_bvae4(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv(32, 3, 1, name='e0'),  # (32, 32, 32)
    Conv(32, 4, 2, name='e1'),  # (16, 16, 32)
    Conv(64, 3, 1, name='e2'),  # (16, 16, 64)
    Conv(64, 4, 2, name='e3'),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]
  d_layers = [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Tonv(64, 4, 2, name='d3'),  # (16, 16, 64)
    Conv(64, 3, 1, name='d2'),  # (16, 16, 64)
    Tonv(32, 4, 2, name='d1'),  # (32, 32, 32)
    Conv(32, 3, 1, name='d0'),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1)
  ]
  bi_latents(2, 16, 4, 2, e_layers, d_layers, 'latents1')
  bi_latents(0, 8, 4, 2, e_layers, d_layers, 'latents2')
  return HierarchicalVAE(
    encoder=Sequential(e_layers, 'Encoder'),
    decoder=Sequential(d_layers, 'Decoder'),
    latents=latents(zdim),
    observation=obs)


def model_pvae(zdim, dist):
  n_params, obs = observation(dist)
  return HierarchicalVAE(
    encoder=Sequential(encoder(), 'Encoder'),
    decoder=Sequential(decoder(zdim, n_params), 'Decoder'),
    latents=latents(zdim),
    observation=obs)


# ===========================================================================
# Main
# ===========================================================================
def get_vae(args) -> Tuple[VariationalAutoencoder, ImageDataset]:
  dist = 'bernoulli' if args.ds == 'shapes3d' else 'qlogistic'
  model = globals()[f'model_{args.vae}'](args.zdim, dist)
  model: VariationalAutoencoder
  model.build([None] + IMAGE_SHAPE)
  ds = CIFAR10() if args.ds == 'cifar10' else Shapes3DSmall()
  return model, ds


def get_dir(args) -> str:
  path = f'{ROOT_PATH}/{args.vae}_{args.ds}_z{args.zdim}_i{args.it}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def train(args):
  model, ds = get_vae(args)
  print(model)
  save_dir = get_dir(args)
  model_path = f'{save_dir}/model'
  # check override
  files = (glob.glob(model_path + '*') +
           glob.glob(f'{save_dir}/events.out.tfevents*'))
  if len(files) > 0:
    if args.override:
      for f in files:
        print('Override:', f)
        os.remove(f)
    else:
      print('Found:', files)
      print('Skip training:', args)
      return
  train = ds.create_dataset('train', batch_size=BATCH_SIZE)
  valid = ds.create_dataset('valid', batch_size=64)

  best_llk = [-np.inf]

  def callback():
    llk = []
    for x in valid.take(200):
      px, qz = model(x)
      llk.append(px.log_prob(x))
    llk = tf.reduce_mean(tf.concat(llk, 0)).numpy()
    if llk > best_llk[0]:
      best_llk[0] = llk
      model.save_weights(model_path)
      model.trainer.print(f'best llk: {llk:.2f}')

  kw = get_optimizer_info(args.ds, batch_size=BATCH_SIZE * 2)
  kw['max_iter'] = args.it
  model.fit(train,
            callback=callback,
            valid_interval=120.,
            global_clipnorm=100.,
            logdir=save_dir,
            **kw)


def evaluate(args):
  model, ds = get_vae(args)
  save_dir = get_dir(args)
  model.load_weights(f'{save_dir}/model', raise_notfound=True, verbose=True)

  test = ds.create_dataset('test', batch_size=32)
  # === 1. marginalized llk
  n_mcmc = 100
  llk = []
  kl = []
  for x in tqdm(test.take(10)):
    qz = model.encode(x, training=False)
    batch_llk = []
    for s, e in minibatch(8, n_mcmc):
      n = e - s
      z = qz.sample(n)
      z = tf.reshape(z, (-1, z.shape[-1]))
      px = model.decode(z, training=False)
      # llk
      batch_llk.append(
        tf.reshape(px.log_prob(tf.tile(x, (n, 1, 1, 1))), (n, -1)))
      # kl
      exit()
    batch_llk = tf.concat(batch_llk, 0)
    llk.append(batch_llk)
  llk = tf.concat(llk, axis=-1)
  print(llk.shape)
  print('LLK:', tf.reduce_mean(tf.reduce_logsumexp(llk, 0)))


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('vae', type=str)
  parser.add_argument('ds', type=str)
  parser.add_argument('zdim', type=int)
  parser.add_argument('-it', type=int, default=100000)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--override', action='store_true')
  args = parser.parse_args()
  assert args.ds in ('shapes3d', 'cifar10')
  if args.eval:
    evaluate(args)
  else:
    train(args)
