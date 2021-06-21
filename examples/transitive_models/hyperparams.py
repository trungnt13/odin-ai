import os
import shutil
from functools import partial
from itertools import product
import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from tensorflow_probability.python.distributions import Blockwise, Categorical
from tqdm import tqdm
from typing_extensions import Literal

from odin import visual as vs
from odin.backend.interpolation import Interpolation, linear
from odin.bay.vi import (VAE, Beta10VAE, ConditionalM2VAE, FactorVAE, MIVAE,
                         MultiheadVAE, RemafoVAE, SemafoVAE, SkiptaskVAE)
from odin.ml import DimReduce
from odin.utils import MPI, as_tuple, minibatch, get_all_files, ArgController

N_CPU = 2
OVERRIDE = False
configs = dict(
    vae=[
        SemafoVAE,
        RemafoVAE,
    ],
    py=[
        0.004,
        # 0.06,
        # 0.2,
        # 0.5,
        # 0.95,
    ],
    coef=[
        linear(vmin=0.1, vmax=0.01, length=20000, cyclical=False),
        # linear(vmin=0.2, vmax=0.02, length=20000, cyclical=True),
        # linear(vmin=0.2, vmax=0.02, length=20000, cyclical=False),
        # linear(vmin=0.5, vmax=0.1, length=20000, cyclical=True),
    ],
    ds=[
        # 'mnist',
        # 'fashionmnist',
        # 'shapes3d',
        # 'dsprites',
        'celeba'
    ],
)
outdir = '/home/trung/exp/hyperparams'
if not os.path.exists(outdir):
  os.makedirs(outdir)

# 1) Linear scaling is better
# 2) 0.2 is better than 0.5
# 3) Remafo vs Semafo ?


# ===========================================================================
# Training
# ===========================================================================
def _ymean(qy):
  if isinstance(qy, Blockwise):
    y = tf.stack([
        i.mode() if isinstance(i, Categorical) else i.mean()
        for i in qy.distributions.model
    ],
                 axis=-1)
  else:
    y = qy.mean()
  return y


def evaluate(vae,
             ds,
             expdir: str,
             title: str,
             batch_size: int = 32,
             seed: int = 1):
  from odin.bay.vi import Correlation
  rand = np.random.RandomState(seed=seed)
  if not os.path.exists(expdir):
    os.makedirs(expdir)
  tanh = True if ds.name.lower() == 'celeba' else False
  ## data for training semi-supervised
  # careful don't allow any data leakage!
  train = ds.create_dataset('train',
                            batch_size=batch_size,
                            label_percent=True,
                            shuffle=False,
                            normalize='tanh' if tanh else 'probs')
  data = [(vae.encode(x, training=False), y) \
    for x, y in tqdm(train, desc=title)]
  x_semi_train = tf.concat(
      [tf.concat([i.mean(), _ymean(j)], axis=1) for (i, j), _ in data],
      axis=0).numpy()
  y_semi_train = tf.concat([i for _, i in data], axis=0).numpy()
  # shuffle
  ids = rand.permutation(x_semi_train.shape[0])
  x_semi_train = x_semi_train[ids]
  y_semi_train = y_semi_train[ids]
  ## data for testing
  test = ds.create_dataset('test',
                           batch_size=batch_size,
                           label_percent=True,
                           shuffle=False,
                           normalize='tanh' if tanh else 'probs')
  prog = tqdm(test, desc=title)
  llk_x = []
  llk_y = []
  z = []
  y_true = []
  y_pred = []
  x_true = []
  x_pred = []
  x_org, x_rec = [], []
  for x, y in prog:
    px, (qz, qy) = vae(x, training=False)
    y_true.append(y)
    y_pred.append(_ymean(qy))
    z.append(qz.mean())
    llk_x.append(px.log_prob(x))
    llk_y.append(qy.log_prob(y))
    if rand.uniform() < 0.005 or len(x_org) < 2:
      x_org.append(x)
      x_rec.append(px.mean())
  ## llk
  llk_x = tf.reduce_mean(tf.concat(llk_x, axis=0)).numpy()
  llk_y = tf.reduce_mean(tf.concat(llk_y, axis=0)).numpy()
  ## the latents
  z = tf.concat(z, axis=0).numpy()
  y_true = tf.concat(y_true, axis=0).numpy()
  y_pred = tf.concat(y_pred, axis=0).numpy()
  x_semi_test = tf.concat([z, y_pred], axis=-1).numpy()
  # shuffle
  ids = rand.permutation(z.shape[0])
  z = z[ids]
  y_true = y_true[ids]
  y_pred = y_pred[ids]
  x_semi_test = x_semi_test[ids]
  ## saving reconstruction images
  x_org = tf.concat(x_org, axis=0).numpy()
  x_rec = tf.concat(x_rec, axis=0).numpy()
  ids = rand.permutation(x_org.shape[0])
  x_org = x_org[ids][:36]
  x_rec = x_rec[ids][:36]
  vmin = x_rec.reshape((36, -1)).min(axis=1).reshape((36, 1, 1, 1))
  vmax = x_rec.reshape((36, -1)).max(axis=1).reshape((36, 1, 1, 1))
  if tanh:
    x_org = (x_org + 1.) / 2.
  x_rec = (x_rec - vmin) / (vmax - vmin)
  if x_org.shape[-1] == 1:  # grayscale image
    x_org = np.squeeze(x_org, -1)
    x_rec = np.squeeze(x_rec, -1)
  else:  # color image
    x_org = np.transpose(x_org, (0, 3, 1, 2))
    x_rec = np.transpose(x_rec, (0, 3, 1, 2))
  plt.figure(figsize=(15, 8))
  ax = plt.subplot(1, 2, 1)
  vs.plot_images(x_org, grids=(6, 6), ax=ax, title='Original')
  ax = plt.subplot(1, 2, 2)
  vs.plot_images(x_rec, grids=(6, 6), ax=ax, title='Reconstructed')
  plt.tight_layout()
  ## prepare the labels
  if ds.name in ('mnist', 'fashionmnist', 'celeba'):
    true = np.argmax(y_true, axis=-1)
    pred = np.argmax(y_pred, axis=-1)
    y_semi_train = np.argmax(y_semi_train, axis=-1)
    y_semi_test = true
    labels_name = ds.labels
  else:  # shapes3d dsprites
    true = y_true[:, 2].astype(np.int32)
    pred = y_pred[:, 2].astype(np.int32)
    y_semi_train = y_semi_train[:, 2].astype(np.int32)
    y_semi_test = true
    if ds.name == 'shapes3d':
      labels_name = ['cube', 'cylinder', 'sphere', 'round']
    elif ds.name == 'dsprites':
      labels_name = ['square', 'ellipse', 'heart']
  plt.figure(figsize=(8, 8))
  vs.plot_confusion_matrix(cm=confusion_matrix(y_true=true, y_pred=pred),
                           labels=labels_name,
                           cbar=True,
                           fontsize=10,
                           title=title)
  labels = np.array([labels_name[i] for i in true])
  labels_pred = np.array([labels_name[i] for i in pred])
  ## save arrays for later inspectation
  np.savez_compressed(f'{expdir}/arrays',
                      x_train=x_semi_train,
                      y_train=y_semi_train,
                      x_test=x_semi_test,
                      y_test=y_semi_test,
                      zdim=z.shape[1],
                      labels=labels_name)
  print(f'Export arrays to "{expdir}/arrays.npz"')
  ## semi-supervised
  with open(f'{expdir}/results.txt', 'w') as f:
    print(f'Export results to "{expdir}/results.txt"')
    f.write(f'Steps: {vae.step.numpy()}\n')
    f.write(f'llk_x: {llk_x}\n')
    f.write(f'llk_y: {llk_y}\n')
    for p in [0.004, 0.06, 0.2, 0.99]:
      x_train, x_test, y_train, y_test = train_test_split(
          x_semi_train,
          y_semi_train,
          train_size=int(np.round(p * x_semi_train.shape[0])),
          random_state=1,
      )
      m = LogisticRegression(max_iter=3000, random_state=1)
      m.fit(x_train, y_train)
      # write the report
      f.write(f'{m.__class__.__name__} Number of labels: '
              f'{p} {x_train.shape[0]}/{x_test.shape[0]}')
      f.write('\nValidation:\n')
      f.write(classification_report(y_true=y_test, y_pred=m.predict(x_test)))
      f.write('\nTest:\n')
      f.write(
          classification_report(y_true=y_semi_test,
                                y_pred=m.predict(x_semi_test)))
      f.write('------------\n')
  ## scatter plot
  n_points = 4000
  # tsne plot
  tsne = DimReduce.TSNE(z[:n_points], n_components=2)
  kw = dict(x=tsne[:, 0], y=tsne[:, 1], grid=False, size=12.0, alpha=0.6)
  plt.figure(figsize=(8, 8))
  vs.plot_scatter(color=labels[:n_points], title=f'[True-tSNE]{title}', **kw)
  plt.figure(figsize=(8, 8))
  vs.plot_scatter(color=labels_pred[:n_points],
                  title=f'[Pred-tSNE]{title}',
                  **kw)
  # pca plot
  pca = DimReduce.PCA(z, n_components=2)
  kw = dict(x=pca[:, 0], y=pca[:, 1], grid=False, size=12.0, alpha=0.6)
  plt.figure(figsize=(8, 8))
  vs.plot_scatter(color=labels, title=f'[True-PCA]{title}', **kw)
  plt.figure(figsize=(8, 8))
  vs.plot_scatter(color=labels_pred, title=f'[Pred-PCA]{title}', **kw)
  ## factors plot
  corr = (Correlation.Spearman(z, y_true) + Correlation.Pearson(z, y_true)) / 2.
  best_z = np.argsort(np.abs(corr), axis=0)[-2:]
  style = dict(size=15.0, alpha=0.6, grid=False)
  for fi, (z1, z2) in enumerate(best_z.T):
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    vs.plot_scatter(x=z[:n_points, z1],
                    y=z[:n_points, z2],
                    val=y_true[:n_points, fi],
                    ax=ax,
                    title=ds.labels[fi],
                    **style)
    ax = plt.subplot(1, 2, 2)
    vs.plot_scatter(x=z[:n_points, z1],
                    y=z[:n_points, z2],
                    val=y_pred[:n_points, fi],
                    ax=ax,
                    title=ds.labels[fi],
                    **style)
    plt.tight_layout()
  ## save all plot
  vs.plot_save(f'{expdir}/analysis.pdf', dpi=180, verbose=True)


def run_task(args, evaluation=False):
  from odin.fuel import get_dataset
  from odin.networks import get_networks, get_optimizer_info
  from odin.bay.vi import DisentanglementGym, Correlation, DimReduce
  ######## arguments
  model: SemafoVAE = args['model']
  dsname: str = args['ds']
  py: float = args['py']
  coef: Interpolation = args['coef']
  model_name = model.__name__.lower()
  ######## prepare path
  logdir = f'{outdir}/{dsname}_{py}/{model_name}_{coef.name}'
  if OVERRIDE and os.path.exists(logdir) and not evaluation:
    shutil.rmtree(logdir)
    print(f'Override model at path: {logdir}')
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  modelpath = f'{logdir}/model'
  ######## dataset
  ds = get_dataset(dsname)
  tanh_norm = True if dsname == 'celeba' else False
  train = ds.create_dataset('train',
                            batch_size=32,
                            label_percent=py,
                            normalize='tanh' if tanh_norm else 'probs')
  valid = ds.create_dataset('valid',
                            batch_size=32,
                            label_percent=True,
                            normalize='tanh' if tanh_norm else 'probs').shuffle(
                                1000, seed=1, reshuffle_each_iteration=True)
  ######## model
  networks = get_networks(dsname,
                          centerize_image=False if tanh_norm else True,
                          is_semi_supervised=True)
  vae = model(alpha=1. / py, mi_coef=coef, **networks)
  vae.build((None,) + ds.shape)
  vae.load_weights(modelpath, verbose=True)

  ######## evaluation
  if evaluation:
    if vae.step.numpy() <= 1:
      return
    evaluate(vae,
             ds,
             expdir=f'{logdir}/analysis',
             title=f'{dsname}{py}_{model_name}_{coef.name}')
    return

  ######## training
  best_llk_x = []
  best_llk_y = []

  def callback():
    llk_x = []
    llk_y = []
    for x, y in valid.take(300):
      px, (qz, qy) = vae(x, training=False)
      llk_x.append(px.log_prob(x))
      llk_y.append(qy.log_prob(y))
    llk_x = tf.reduce_mean(tf.concat(llk_x, axis=0))
    llk_y = tf.reduce_mean(tf.concat(llk_y, axis=0))
    best_llk_x.append(llk_x)
    best_llk_y.append(llk_y)
    if llk_x >= np.max(best_llk_x):
      vae.save_weights(modelpath)
      vae.trainer.print(f'{model_name} {dsname} {py} '
                        f'best weights at iter#{vae.step.numpy()} '
                        f'llk_x={llk_x:.2f} llk_y={llk_y:.2f}')

  opt_info = get_optimizer_info(dsname)
  opt_info['max_iter'] += 20000
  vae.fit(
      train,
      logdir=logdir,
      on_valid_end=callback,
      valid_interval=60,
      logging_interval=2,
      nan_gradients_policy='stop',
      compile_graph=True,
      skip_fitted=True,
      **opt_info,
  )
  print(f'Trained {model_name} {dsname} {py} {vae.step.numpy()}(steps)')


# ===========================================================================
# main
# ===========================================================================
def main(mode: Literal['train', 'evaluate'], fn_filter=lambda job: True):
  jobs = product(*[[(k, i) for i in v] for k, v in configs.items()])
  jobs = [dict(j) for j in jobs]
  jobs = [j for j in jobs if fn_filter(j)]
  print(f'Start {mode} {len(jobs)} jobs ...')
  ######## training mode
  if mode == 'train':
    for _ in MPI(jobs, partial(run_task, evaluation=False), ncpu=N_CPU,
                 batch=1):
      pass
  ######## evaluation mode
  elif mode == 'evaluate':
    for j in jobs:
      if j['ds'] == 'celeba' and j['coef'].vmin == 0.1:
        # print(j)
        run_task(j, evaluation=True)
  ######## others
  else:
    raise NotImplementedError(f'No support for mode="{mode}"')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', help='Training mode', action='store_true')
  parser.add_argument('--eval', help='Evaluation mode', action='store_true')
  parser.add_argument('--override',
                      help='Override exist models',
                      action='store_true')
  parser.add_argument('-ds', default='')
  args = parser.parse_args()
  OVERRIDE = bool(args.override)
  ## create the filter
  if args.ds:
    ds = set([str(i).lower() for i in args.ds.split(',')])
    fn_filter = lambda job: job['ds'] in ds
  else:
    fn_filter = lambda job: True
  ## just print some debugging
  if not args.train and not args.eval:
    log = sorted(
        [(path.split('/')[-3:-1], path)
         for path in get_all_files(outdir, lambda path: 'log.txt' in path)],
        key=lambda x: x[0][0] + x[0][1])
    for (ds, model), path in log:
      print(ds, model)
      with open(path, 'r') as f:
        lines = [
            line[:-1].split('at ')[-1]
            for line in f.readlines()
            if 'best' in line
        ][-1:]
        for l in lines:
          print(' ', l)
  ## run train or evaluation tasks
  else:
    main(mode='evaluate' if args.eval else 'train', fn_filter=fn_filter)
