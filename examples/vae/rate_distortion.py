import argparse
import glob
import itertools
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from tqdm import tqdm

from odin.fuel import MNIST
from odin.utils import MPI
from multiprocessing import cpu_count
from odin.bay import BetaVAE, MVNDiagLatents, DistributionDense, BetaGammaVAE
from odin.networks.image_networks import CenterAt0

# ===========================================================================
# Config
# ===========================================================================
save_path = '~/exp/rate_distortion'
save_path = os.path.expanduser(save_path)
if not os.path.exists(save_path):
  os.makedirs(save_path)
cache_path = os.path.join(save_path, 'cache')
if not os.path.exists(cache_path):
  os.makedirs(cache_path)
# [0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1., 2., 3., 5.]
beta = [0.001, 0.01, 0.1, 0.5, 1., 2.5, 5.]
gamma = [0.001, 0.01, 0.1, 0.5, 1., 2.5, 5.]
zdim = [2, 5, 10, 20, 35, 60, 80]
OVERRIDE = False


def get_path(b, g, z) -> str:
  path = f'{save_path}/{z}'
  if not os.path.exists(path):
    os.makedirs(path)
  path = f'{path}/{b}_{g}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def networks(zdim):
  return dict(
    encoder=keras.Sequential([
      Flatten(),
      CenterAt0(),
      Dense(512, activation=tf.nn.relu),
      Dense(512, activation=tf.nn.relu),
      Dense(512, activation=tf.nn.relu),
    ]),
    decoder=keras.Sequential([
      Dense(512, activation=tf.nn.relu),
      Dense(512, activation=tf.nn.relu),
      Dense(512, activation=tf.nn.relu),
    ]),
    latents=MVNDiagLatents(zdim),
    observation=DistributionDense(event_shape=(28, 28, 1),
                                  posterior='bernoulli',
                                  name='Image'))


# ===========================================================================
# Main
# ===========================================================================
def training(job):
  np.random.seed(1)
  tf.random.set_seed(1)
  b, g, z = job
  path = get_path(b, g, z)
  exist_files = glob.glob(f'{path}*')
  if OVERRIDE:
    for f in exist_files:
      if os.path.isdir(f):
        shutil.rmtree(f)
      else:
        os.remove(f)
      print('Remove:', f)
    os.makedirs(path)
  elif len(exist_files) > 1:
    print('Skip training:', job)
    return
  ds = MNIST()
  train = ds.create_dataset('train', batch_size=32)
  vae = BetaGammaVAE(beta=b, gamma=g, **networks(z))
  vae.build(ds.full_shape)
  vae.fit(train, learning_rate=1e-3, max_iter=80000, logdir=path)
  vae.save_weights(path, overwrite=True)


def evaluate(job):
  np.random.seed(1)
  tf.random.set_seed(1)
  b, g, z = job
  path = get_path(b, g, z)
  ds = MNIST()
  vae = BetaGammaVAE(beta=b, gamma=g, **networks(z))
  vae.build(ds.full_shape)
  vae.trainable = False
  try:
    vae.load_weights(path, verbose=False, raise_notfound=True)
  except FileNotFoundError:
    return None
  test = ds.create_dataset('test', batch_size=32)
  llk = []
  kl = []
  mean = []
  stddev = []
  for x in test:
    px_z, qz_x = vae(x, training=False)
    llk.append(px_z.log_prob(x))
    kl.append(qz_x.KL_divergence(analytic=False))
    mean.append(qz_x.mean().numpy())
    stddev.append(qz_x.stddev().numpy())
  llk = np.mean(np.concatenate(llk, 0))
  kl = np.mean(np.concatenate(kl, 0))
  mean = np.mean(np.concatenate(mean, 0), 0)
  stddev = np.mean(np.concatenate(stddev, 0), 0)
  # active units
  threshold = 1e-3
  au_mean = len(mean) - np.sum(np.abs(mean) <= threshold)
  au_std = len(stddev) - np.sum(np.abs(stddev - 1.0) <= threshold)
  return dict(beta=b, gamma=g, zdim=z,
              llk=llk, kl=kl,
              au_mean=au_mean, au_std=au_std)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', type=int)
  parser.add_argument('-ncpu', type=int, default=-1)
  parser.add_argument('--override', action='store_true')
  # === 1. prepare
  args = parser.parse_args()
  ncpu = args.ncpu
  if ncpu <= 0:
    ncpu = cpu_count() - 1
  jobs = list(itertools.product(beta, gamma, zdim))

  OVERRIDE = args.override
  # === 2. training
  if args.mode == 0:
    for _ in MPI(jobs, training, ncpu=ncpu):
      pass
  # === 3. evaluating
  elif args.mode == 1:
    path = os.path.join(cache_path, 'results')
    if os.path.exists(path) and OVERRIDE:
      os.remove(path)
    if os.path.exists(path):
      with open(path, 'rb') as f:
        df = pickle.load(f)
    else:
      df = []
      progress = tqdm(total=len(jobs), desc='Evaluating')
      for results in MPI(jobs, evaluate, ncpu=ncpu):
        progress.update(1)
        if results is None:
          continue
        df.append(results)
      progress.close()
      df = pd.DataFrame(df)
      with open(path, 'wb') as f:
        pickle.dump(df, f)
    print(df)
  else:
    raise NotImplementedError(f'No support mode={args.mode}')
