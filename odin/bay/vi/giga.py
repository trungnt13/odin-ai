from typing import Callable, Generator, Iterator, Union, Optional
import warnings

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import entropy as entropy1D
from tensorflow_probability.python import distributions as tfd

from odin.utils import one_hot
from odin.bay.vi import VAE, RVmeta, NetConf, beta10VAE, factorVAE
from odin.fuel import dSprites
from odin.networks import get_networks, get_optimizer_info
from tqdm import tqdm
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

np.random.seed(1)
tf.random.set_seed(1)


def estimate_Izx(fn_px_z: Callable[[tf.Tensor], tfd.Distribution],
                 pz: tfd.Distribution,
                 n_samples_z: int = 10000,
                 n_mcmc_x: int = 100,
                 batch_size: int = 32,
                 verbose: bool = True):
  log_px_z = []
  prog = tqdm(desc='I(Z;X)',
              total=n_samples_z * n_mcmc_x,
              unit='samples',
              disable=not verbose)
  for start in range(0, n_samples_z, batch_size):
    batch_z = min(n_samples_z - start, batch_size)
    z = pz.sample(batch_z)
    px_z = fn_px_z(z)
    batch_llk = []
    for start in range(0, n_mcmc_x, batch_size):
      batch_x = min(n_mcmc_x - start, batch_size)
      x = px_z.sample(batch_x)
      batch_llk.append(px_z.log_prob(x))
      prog.update(batch_z * batch_x)
    batch_llk = tf.concat(batch_llk, axis=0)
    log_px_z.append(batch_llk)
  ## finalize
  prog.clear()
  prog.close()
  log_px_z = tf.concat(log_px_z, axis=1)  # [n_mcmc_x, n_samples_z]
  ## calculate the MI
  log_px = tf.reduce_logsumexp(log_px_z, axis=1, keepdims=True) - \
    tf.math.log(tf.cast(n_samples_z, tf.float32))
  mi = tf.reduce_mean(log_px_z - log_px)
  return mi


def estimate_Izy(X_y: Union[tf.data.Dataset, Generator, Iterator],
                 fn_qz_x: Callable[[tf.Tensor], tfd.Distribution],
                 n_samples: int = 10000,
                 n_mcmc: int = 100,
                 batch_size: int = 32,
                 verbose: bool = True):
  ## process the data into mini batches
  if not isinstance(X_y, (tf.data.Dataset, Generator, Iterator)):
    X, y = X_y
    if not isinstance(X, tf.data.Dataset):
      X = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    if not isinstance(y, tf.data.Dataset):
      y = tf.data.Dataset.from_tensor_slices(y).batch(batch_size)
    X_y = tf.data.Dataset.zip((X, y))
  if isinstance(X_y, tf.data.Dataset):
    X_y = X_y.repeat(-1).shuffle(1000)
    X_y = iter(X_y)
  ## iterate the dataset until have enough n_samples
  count = 0
  log_qz_x = []
  qy = []
  prog = tqdm(desc='I(Z;Y)',
              total=n_samples * n_mcmc,
              unit='samples',
              disable=not verbose)
  while count < n_samples:
    ## step 1: sample q(x, y)
    try:
      X, y = next(X_y)
    except StopIteration:
      warnings.warn(f'Not enough data for {n_samples} samples.')
      break
    batch_x = min(X.shape[0], n_samples - count)
    X = X[:batch_x]
    y = y[:batch_x]
    qy.append(y)
    qz_x = fn_qz_x(X)
    ## step 2: sample q(z|x)
    batch_llk_qz = []
    for start in range(0, n_mcmc, batch_size):
      batch_z = min(n_mcmc - start, batch_size)
      z = qz_x.sample(batch_z)
      llk_qz = qz_x.log_prob(z)
      batch_llk_qz.append(llk_qz)
      ## update progress
      prog.update(batch_z * batch_x)
    ## step 4: aggregate the log-likelihoods
    batch_llk_qz = tf.concat(batch_llk_qz, axis=0)
    log_qz_x.append(batch_llk_qz)
    count += batch_x
  ## finalizing results
  prog.clear()
  prog.close()
  log_qz_x = tf.concat(log_qz_x, axis=1)  # [n_mcmc, n_samples]
  qy = tf.concat(qy, axis=0)
  ## Calculate I(Z; Y) - H(Z)
  I_zy = {}  # for each factor
  n_factors = qy.shape[1]
  for i in range(n_factors):
    y = np.asarray(qy[:, i], dtype=np.int32)
    I_zyi = {} # for each label of the factor
    labels = np.unique(y)
    for yk in labels:
      ids = (y == yk)
      K = np.sum(ids)
      log_qz_xk = tf.boolean_mask(log_qz_x, ids, axis=1)
      log_qz_xk = tf.reduce_logsumexp(log_qz_xk, axis=1) - tf.math.log(
          tf.constant(K, dtype=tf.float32))
      I_zyi[yk] = tf.reduce_mean(log_qz_xk, axis=0)
    # average among labels within a factor
    I_zyi = sum(v for v in I_zyi.values()) / len(labels)
    I_zy[i] = I_zyi
  # average among all factors
  I_zy = np.array(list(I_zy.values()))
  I_zy = np.mean(I_zy)
  ## giga
  return I_zy


def giga(X_y: Union[tf.data.Dataset, Generator, Iterator],
         fn_qz_x: Callable[[tf.Tensor], tfd.Distribution],
         fn_px_z: Callable[[tf.Tensor], tfd.Distribution],
         pz: Optional[tfd.Distribution] = None,
         n_samples: int = 10000,
         n_mcmc: int = 100,
         batch_size: int = 32,
         adjusted: bool = True,
         verbose: bool = True):
  C_mcmc = tf.math.log(tf.constant(n_mcmc, dtype=tf.float32))
  ## process the data into mini batches
  if not isinstance(X_y, (tf.data.Dataset, Generator, Iterator)):
    X, y = X_y
    if not isinstance(X, tf.data.Dataset):
      X = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)
    if not isinstance(y, tf.data.Dataset):
      y = tf.data.Dataset.from_tensor_slices(y).batch(batch_size)
    X_y = tf.data.Dataset.zip((X, y))
  if isinstance(X_y, tf.data.Dataset):
    X_y = X_y.repeat(-1).shuffle(1000)
    X_y = iter(X_y)
  ## iterate the dataset until have enough n_samples
  count = 0
  log_qz_x = []
  log_px_z = []
  log_pz = []
  qy = []
  prog = tqdm(desc='GIGA',
              total=n_samples * n_mcmc,
              unit='samples',
              disable=not verbose)
  while count < n_samples:
    ## step 1: sample q(x, y)
    try:
      X, y = next(X_y)
    except StopIteration:
      warnings.warn(f'Not enough data for {n_samples} samples.')
      break
    batch_x = min(X.shape[0], n_samples - count)
    X = X[:batch_x]
    y = y[:batch_x]
    qy.append(y)
    qz_x = fn_qz_x(X)
    # infer the prior of z
    if pz is not None:
      pass
    elif hasattr(qz_x, 'KL_divergence'):
      pz = qz_x.KL_divergence.prior
    else:
      pz = tfd.Normal(tf.zeros(qz_x.event_shape), 1., dtype=qz_x.dtype)
    ## step 2: sample q(z|x)
    batch_llk_px = []
    batch_llk_qz = []
    batch_llk_pz = []
    for start in range(0, n_mcmc, batch_size):
      batch_z = min(n_mcmc - start, batch_size)
      z = qz_x.sample(batch_z)
      llk_qz = qz_x.log_prob(z)
      batch_llk_qz.append(llk_qz)
      llk_pz = pz.log_prob(z)
      batch_llk_pz.append(llk_pz)
      z = tf.reshape(z, (-1, z.shape[-1]))
      ## step 3: calculate log(p(x|z))
      px_z = fn_px_z(z)
      llk_x = px_z.log_prob(px_z.sample())
      llk_x = tf.reshape(llk_x, (batch_z, -1))
      batch_llk_px.append(llk_x)
      ## update progress
      prog.update(batch_z * batch_x)
    ## step 4: aggregate the log-likelihoods
    batch_llk_qz = tf.concat(batch_llk_qz, axis=0)
    log_qz_x.append(batch_llk_qz)
    batch_llk_pz = tf.concat(batch_llk_pz, axis=0)
    log_pz.append(batch_llk_pz)
    batch_llk_px = tf.concat(batch_llk_px, axis=0)
    log_px_z.append(batch_llk_px)
    count += batch_x
  ## finalizing results
  prog.clear()
  prog.close()
  log_px_z = tf.concat(log_px_z, axis=1)  # [n_mcmc, n_samples]
  log_qz_x = tf.concat(log_qz_x, axis=1)  # [n_mcmc, n_samples]
  log_pz = tf.concat(log_pz, axis=1)  # [n_mcmc, n_samples]
  qy = tf.concat(qy, axis=0)
  n_factors = qy.shape[1]
  ## Calculate I(X; Z)
  log_pxz = log_px_z + log_pz - log_qz_x
  log_px = tf.reduce_logsumexp(log_pxz, axis=0, keepdims=True) - C_mcmc
  log_qx = tf.math.log(1. / n_samples)
  pxz = tf.math.exp(log_pxz - log_qx)
  I_xz = pxz * (log_px_z - log_px)  # [n_mcmc, n_samples]
  # entropy of x
  H_x = tf.reduce_mean(-pxz * log_px)
  # entropy of z
  H_z = tf.reduce_mean(-tf.math.exp(log_pz - log_qz_x) * log_pz)
  I_xz = tf.reduce_mean(I_xz, axis=0)  # [n_samples]
  # I_xz = I_xz / tf.math.sqrt(H_x * H_z)
  I_xz = tf.reduce_mean(I_xz)
  ## Calculate I(Z; Y) - H(Z)
  I_zy = {}  # for each factor
  for i in range(n_factors):
    y = np.asarray(qy[:, i], dtype=np.int32)
    I_zyi = {}
    labels = np.unique(y)
    for yk in labels:
      ids = (y == yk)
      K = np.sum(ids)
      log_qz_xk = tf.boolean_mask(log_qz_x, ids, axis=1)
      log_qz_xk = tf.reduce_logsumexp(log_qz_xk, axis=1) - tf.math.log(
          tf.constant(K, dtype=tf.float32))
      I_zyi[yk] = tf.reduce_mean(log_qz_xk, axis=0)
    # average among labels within a factor
    I_zyi = sum(v for v in I_zyi.values()) / len(labels)
    I_zy[i] = I_zyi
  # average among all factors
  H_y = np.array([entropy1D(qy[:, i]) for i in range(n_factors)])
  I_zy = np.array(list(I_zy.values()))
  I_zy = np.mean(I_zy / H_y)
  ## giga
  return I_xz + I_zy
