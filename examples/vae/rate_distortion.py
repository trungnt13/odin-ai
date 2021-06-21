import argparse
import glob
import itertools
import os
import pickle
import shutil
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten
from tqdm import tqdm

from odin.fuel import MNIST
from odin.utils import MPI
from multiprocessing import cpu_count
from odin.bay import MVNDiagLatents, DistributionDense, BetaGammaVAE
from odin.networks.image_networks import CenterAt0
from odin import visual as vs
import seaborn as sns
from dataclasses import dataclass

sns.set()

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
BETA = [0.001, 0.005, 0.01, 0.1, 0.5, 1., 2.5, 5., 10]
GAMMA = [0.001, 0.005, 0.01, 0.1, 0.5, 1., 2.5, 5., 10]
ZDIM = [2, 5, 10, 20, 35, 60, 80]
OVERWRITE = False


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


@dataclass()
class Job:
  beta: float = 1.0
  gamma: float = 1.0
  zdim: int = 10


def get_path(job: Job) -> str:
  path = f'{save_path}/{job.zdim}'
  if not os.path.exists(path):
    os.makedirs(path)
  path = f'{path}/{job.beta}_{job.gamma}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def get_cache_path(suffix: str = '') -> str:
  path = os.path.join(cache_path, f'results{suffix}')
  if os.path.exists(path) and OVERWRITE:
    print('Overwrite results at path:', path)
    os.remove(path)
  return path


def load_vae_eval(job: Job):
  np.random.seed(1)
  tf.random.set_seed(1)
  path = get_path(job)
  ds = MNIST()
  vae = BetaGammaVAE(beta=job.beta, gamma=job.gamma, **networks(job.zdim))
  vae.build(ds.full_shape)
  vae.trainable = False
  try:
    vae.load_weights(path, verbose=False, raise_notfound=True)
  except FileNotFoundError:
    return None, None
  return ds, vae


# ===========================================================================
# Main
# ===========================================================================
def training(job: Job):
  np.random.seed(1)
  tf.random.set_seed(1)
  path = get_path(job)
  exist_files = glob.glob(f'{path}*')
  if OVERWRITE:
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
  vae = BetaGammaVAE(beta=job.beta, gamma=job.gamma, **networks(job.zdim))
  vae.build(ds.full_shape)
  vae.fit(train, learning_rate=1e-3, max_iter=80000, logdir=path,
          skip_fitted=True)
  vae.save_weights(path, overwrite=True)


def evaluate_reconstruction(job: Job):
  ds, vae = load_vae_eval(job)
  vae: BetaGammaVAE
  if vae is None:
    return
  ## reconstruction
  test = ds.create_dataset('test', batch_size=32)
  for x in test.take(1):
    px, qz = vae(x, training=False)
  x = px.mean().numpy()
  n_images = x.shape[0]
  vmin = x.reshape((n_images, -1)).min(axis=1).reshape((n_images, 1, 1, 1))
  vmax = x.reshape((n_images, -1)).max(axis=1).reshape((n_images, 1, 1, 1))
  x = (x - vmin) / (vmax - vmin)
  x = np.squeeze(x[0].astype(np.float32), -1)
  reconstruction = x
  ## sample prior
  test = ds.create_dataset('test', label_percent=1.0, batch_size=32)
  z_map = defaultdict(list)
  for x, y in test.take(20):
    y = np.argmax(y, axis=-1)
    px, qz = vae(x, training=False)
    z = qz.mean().numpy()
    for i, j in zip(y, z):
      z_map[i].append(j[np.newaxis, :])
  z_map = {k: np.mean(np.concatenate(v, 0), 0)
           for k, v in z_map.items()}
  samples = vae.sample_observation(n=100).mean()
  ref = z_map[8]  # number 8
  distances = [np.linalg.norm(x - ref, 2)
               for x in vae.encode(samples).mean().numpy()]
  samples = np.squeeze(np.clip(samples[np.argmin(distances)], 0.0, 1.0), -1)
  ## return
  return dict(beta=job.beta, gamma=job.gamma, zdim=job.zdim,
              image=reconstruction,
              sample=samples)


def evaluate_balance(job: Job):
  ds, vae = load_vae_eval(job)
  if vae is None:
    return
  test = ds.create_dataset('test', batch_size=64)
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
  threshold = 0.1  # this threhold to be checked again
  au_mean = len(mean) - np.sum(np.abs(mean) <= threshold)
  au_std = len(stddev) - np.sum(np.abs(stddev - 1.0) <= threshold)
  print(sorted(stddev), job.zdim, au_std)
  return dict(beta=job.beta, gamma=job.gamma, zdim=job.zdim,
              llk=llk, kl=kl,
              au_mean=au_mean, au_std=au_std)


# ===========================================================================
# Plotting helper
# ===========================================================================
def plot(df: pd.DataFrame,
         x: str, y: str,
         hue: Optional[str] = None,
         size: Optional[str] = None,
         style: Optional[str] = None,
         title: Optional[str] = None,
         ax=None):
  if ax is None:
    ax = plt.gca()
  splot = sns.scatterplot(x=x, y=y, hue=hue, size=size, style=style,
                          data=df, sizes=(40, 250), ax=ax, alpha=0.95,
                          linewidth=0, palette='Spectral')
  splot.set(xscale="log")
  splot.set(yscale="log")
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
  if title is not None:
    ax.set_title(title)


def plot_images(df: pd.DataFrame, path: str, show_reconstruction: bool = True):
  for zdim, group1 in tqdm(df.groupby('zdim')):
    tmp = group1.groupby('gamma')
    n_row = len(tmp)
    n_col = max(len(g) for _, g in tmp)
    plt.figure(figsize=(n_col * 1.5, n_row * 1.5 + 1.5), dpi=150)
    count = 0
    for gamma, group2 in list(iter(tmp))[::-1]:  # reverse the row order
      for i, (beta, _, _, recon, sample) in enumerate(group2.values):
        img = recon if show_reconstruction else sample
        img[np.isnan(img)] = 1.
        plt.subplot(n_row, n_col, count + 1)
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
        if not args.no_anno:
          plt.title(f'b={beta} g={gamma}', fontsize=10)
        count += 1
    plt.suptitle(f'z={zdim}')
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.001])
  vs.plot_save(path, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', type=int)
  parser.add_argument('-ncpu', type=int, default=1)
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--no-anno', action='store_true')
  # === 1. prepare
  args = parser.parse_args()
  ncpu = args.ncpu
  if ncpu <= 0:
    ncpu = cpu_count() - 1
  jobs = [Job(beta=b, gamma=g, zdim=z)
          for b, g, z in itertools.product(BETA, GAMMA, ZDIM)]
  OVERWRITE = args.overwrite
  # === 2. training
  if args.mode == 0:
    for _ in MPI(jobs, training, ncpu=ncpu):
      pass
  # === 3. evaluating reconstruction and sampling
  elif args.mode == 2:
    path = get_cache_path(suffix='_reconstruction')
    if not os.path.exists(path):
      progress = tqdm(total=len(jobs), desc='Evaluating Reconstruction')
      df = []
      for results in MPI(jobs, evaluate_reconstruction, ncpu=ncpu):
        progress.update(1)
        if results is None:
          continue
        df.append(results)
      progress.close()
      df = pd.DataFrame(df)
      with open(path, 'wb') as f:
        pickle.dump(df, f)
    else:
      with open(path, 'rb') as f:
        df = pickle.load(f)
    ## plot the image
    plot_images(df, os.path.join(save_path, 'reconstruction.pdf'),
                show_reconstruction=True)
    plot_images(df, os.path.join(save_path, 'sampling.pdf'),
                show_reconstruction=False)
  # === 4. evaluating ELBO balancing
  elif args.mode == 1:
    path = get_cache_path()
    # get evaluating results
    if os.path.exists(path):
      with open(path, 'rb') as f:
        df = pickle.load(f)
    else:
      df = []
      progress = tqdm(total=len(jobs), desc='Evaluating Balance')
      for results in MPI(jobs, evaluate_balance, ncpu=ncpu):
        progress.update(1)
        if results is None:
          continue
        df.append(results)
      progress.close()
      df = pd.DataFrame(df)
      with open(path, 'wb') as f:
        pickle.dump(df, f)
    # add elbo
    df['elbo'] = df['llk'] - df['kl']
    # plotting: fix zdim, show llk and kl
    n_cols = 4
    n_rows = int(np.ceil(len(ZDIM) / n_cols))
    plt.figure(figsize=(n_cols * 6, n_rows * 5), dpi=200)
    for i, (zdim, group) in tqdm(enumerate(df.groupby('zdim'))):
      ax = plt.subplot(n_rows, n_cols, i + 1)
      plot(group, x='beta', y='gamma', hue='llk', size='kl',
           title=f'zdim={zdim}', ax=ax)
    plt.tight_layout()
    # fix zdim, show au and llk
    plt.figure(figsize=(n_cols * 6, n_rows * 5), dpi=200)
    for i, (zdim, group) in tqdm(enumerate(df.groupby('zdim'))):
      ax = plt.subplot(n_rows, n_cols, i + 1)
      plot(group, x='beta', y='gamma', hue='llk', size='au_std',
           title=f'zdim={zdim}', ax=ax)
    plt.tight_layout()
    # fix zdim, show au and elbo
    plt.figure(figsize=(n_cols * 6, n_rows * 5), dpi=200)
    for i, (zdim, group) in tqdm(enumerate(df.groupby('zdim'))):
      ax = plt.subplot(n_rows, n_cols, i + 1)
      plot(group, x='beta', y='gamma', hue='elbo', size='au_std',
           title=f'zdim={zdim}', ax=ax)
    plt.tight_layout()
    # save all figures
    vs.plot_save(os.path.join(save_path, 'rate_distortion.pdf'), verbose=True)
    # save score file
    score_path = os.path.join(save_path, 'results.txt')
    with open(score_path, 'w') as f:
      df.to_string(f, index=False)
      print('Saved score file:', score_path)
  else:
    raise NotImplementedError(f'No support mode={args.mode}')
