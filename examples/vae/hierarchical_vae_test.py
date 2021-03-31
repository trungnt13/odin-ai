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

# ===========================================================================
# Constants
# ===========================================================================
ROOT_PATH = os.path.expanduser('~/exp/hierarchical')
if not os.path.exists(ROOT_PATH):
  os.makedirs(ROOT_PATH)

IMAGE_SHAPE = [32, 32, 3]
BATCH_SIZE = 32
ZDIM = 256
CONV_KW = dict(padding='same', activation=tf.nn.elu,
               kernel_initializer='he_normal')
Tonv2D = Conv2DTranspose


# ===========================================================================
# Layers
# DO NOT use softplus1
# ===========================================================================
def encoder() -> List[Layer]:
  return [
    InputLayer(IMAGE_SHAPE),
    CenterAt0(),
    Conv2D(32, 4, 1, name='e1', **CONV_KW),  # (32, 32, 32)
    Conv2D(32, 4, 2, name='e2', **CONV_KW),  # (16, 16, 32)
    Conv2D(64, 4, 1, name='e3', **CONV_KW),  # (16, 16, 64)
    Conv2D(64, 4, 2, name='e4', **CONV_KW),  # (8, 8, 64)
    Flatten(),
    Dense(512)
  ]


def decoder(zdim, n_params=1) -> List[Layer]:
  return [
    InputLayer([zdim]),
    Dense(512),
    Reshape([8, 8, 8]),
    Tonv2D(64, 4, 2, name='d4', **CONV_KW),  # (16, 16, 64)
    Conv2D(64, 4, 1, name='d3', **CONV_KW),  # (16, 16, 64)
    Tonv2D(32, 4, 2, name='d2', **CONV_KW),  # (32, 32, 32)
    Conv2D(32, 4, 1, name='d1', **CONV_KW),  # (32, 32, 32)
    Conv2D(IMAGE_SHAPE[-1] * n_params, 1, 1),  # (32, 32, 3)
  ]


def find(name: str, layers: List[Layer]) -> Tuple[int, Layer]:
  for idx, l in enumerate(layers):
    if l.name == name:
      return idx, l
  raise ValueError(f'Cannot find layer with name {name}')


def latents(zdim):
  return MVNDiagLatents(zdim)


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


def model_bvae(zdim, dist):
  n_params, obs = observation(dist)
  e_layers = encoder()
  d_layers = decoder(zdim, n_params)

  def bi_latents(idx, units, name):
    # latents 1
    i1, d = find(f'd{idx}', d_layers)
    i2, e = find(f'e{idx}', e_layers)
    d = BidirectionalLatents(d, encoder=e, filters=units, name=name)
    d_layers[i1] = d

  bi_latents(3, 64, 'latents1')
  bi_latents(1, 32, 'latents2')
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
  path = f'{ROOT_PATH}/{args.vae}_{args.ds}_z{args.zdim}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def train(args):
  model, ds = get_vae(args)
  print(model)
  save_dir = get_dir(args)
  model_path = f'{save_dir}/model'
  # check override
  files = glob.glob(model_path + '*')
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

  model.fit(train,
            callback=callback,
            valid_interval=120,
            logdir=save_dir,
            **get_optimizer_info(args.ds, batch_size=BATCH_SIZE * 2))


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
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--override', action='store_true')
  args = parser.parse_args()
  assert args.ds in ('shapes3d', 'cifar10')
  if args.eval:
    evaluate(args)
  else:
    train(args)
