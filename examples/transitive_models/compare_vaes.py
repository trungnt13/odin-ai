import argparse
import os
import shutil
from typing import Tuple, List, Union, Sequence, Type

from scipy.special import logsumexp
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow_probability.python.distributions import Categorical, \
  Distribution, Normal, Independent
from odin.bay.distributions import Batchwise, Blockwise
from odin.bay.vi import Correlation
from tqdm import tqdm

from odin import visual as vs
from odin.bay.vi import MIVAE, VariationalAutoencoder
from odin.bay.vi.autoencoder.semafo_vae import SemafoBase
from odin.bay.vi.autoencoder import get_vae
from odin.fuel import ImageDataset, get_dataset
from odin.ml import DimReduce
from odin.networks import get_networks, get_optimizer_info
from odin.utils import as_tuple
import cloudpickle as pickle
from odin.utils import minibatch

sns.set()

# ===========================================================================
# Helpers
# ===========================================================================
ROOT = '/home/trung/exp/fastexp'
if not os.path.exists(ROOT):
  os.makedirs(ROOT)


def flatten(x: np.ndarray) -> np.ndarray:
  if len(x.shape) > 2:
    x = np.reshape(x, (x.shape[0], -1))
  else:
    x = np.asarray(x)
  return x


def get_ymean(py: Batchwise) -> np.ndarray:
  y_mean = []
  if isinstance(py, Batchwise):
    py = py.distributions
  else:
    py = [py]
  for p in py:
    p = _dist(p)  # remove the DeferredTensor
    if isinstance(p, Blockwise):
      y = [i.mode() if isinstance(i, Categorical) else i.mean()
           for i in p.distributions.model]
      y = tf.stack(y, axis=-1)
    else:
      y = p.mode() if isinstance(p, Categorical) else p.mean()
    y_mean.append(y)
  return tf.concat(y_mean, 0).numpy()


def _dist(p: Union[Distribution, Sequence[Distribution]]
          ) -> Union[Sequence[Distribution], Distribution]:
  """Convert DeferredTensor back to original Distribution"""
  if isinstance(p, (tuple, list)):
    return [_dist(i) for i in p]
  return (p.parameters['distribution']
          if 'deferred_tensor' in str(type(p)) else p)


def _prepare_images(x, normalize=False):
  n_images = x.shape[0]
  if normalize:
    vmin = x.reshape((n_images, -1)).min(axis=1).reshape((n_images, 1, 1, 1))
    vmax = x.reshape((n_images, -1)).max(axis=1).reshape((n_images, 1, 1, 1))
    x = (x - vmin) / (vmax - vmin)
  if x.shape[-1] == 1:  # grayscale image
    x = np.squeeze(x, -1)
  else:  # color image
    x = np.transpose(x, (0, 3, 1, 2))
  return x


def _call(vae: VariationalAutoencoder,
          *,
          ds: tf.data.Dataset,
          rand: np.random.RandomState,
          take_count: int = -1,
          n_images: int = 36,
          verbose: bool = True
          ) -> Tuple[float, float,
                     np.ndarray, np.ndarray, np.ndarray,
                     Distribution,
                     List[Distribution], List[Distribution]]:
  """
  Returns
  -------
    llk_x, llk_y, x_org, x_rec, y_true, y_pred, all_qz, all_pz
  """
  ds = ds.take(take_count)
  prog = tqdm(ds, disable=not verbose)
  llk_x, llk_y = [], []
  y_true, y_pred = [], []
  x_org, x_rec = [], []
  Q_zs = []
  P_zs = []
  for x, y in prog:
    P, Q = vae(x, training=False)
    P = as_tuple(P)
    Q, Q_prior = vae.get_latents(return_prior=True)
    Q = as_tuple(Q)
    Q_prior = as_tuple(Q_prior)
    y_true.append(y)
    px = P[0]
    # semi-supervised
    if len(P) > 1:
      py = P[-1]
      y_pred.append(_dist(py))
      if y.shape[1] == py.event_shape[0]:
        llk_y.append(py.log_prob(y))
    Q_zs.append(_dist(Q))
    P_zs.append(_dist(Q_prior))
    llk_x.append(px.log_prob(x))
    # for the reconstruction
    if rand.uniform() < 0.005 or len(x_org) < 2:
      x_org.append(x)
      x_rec.append(px.mean())
  # log-likelihood
  llk_x = tf.reduce_mean(tf.concat(llk_x, axis=0)).numpy()
  llk_y = tf.reduce_mean(tf.concat(llk_y, axis=0)).numpy() \
    if len(llk_y) > 0 else -np.inf
  # latents
  n_latents = len(Q_zs[0])
  all_qz = [Batchwise([z[i] for z in Q_zs]) for i in range(n_latents)]
  all_pz = [Batchwise([z[i] for z in P_zs])
            if len(P_zs[0][i].batch_shape) > 0 else
            P_zs[0][i]
            for i in range(n_latents)]
  # reconstruction
  x_org = tf.concat(x_org, axis=0).numpy()
  x_rec = tf.concat(x_rec, axis=0).numpy()
  ids = rand.permutation(x_org.shape[0])
  x_org = x_org[ids][:n_images]
  x_rec = x_rec[ids][:n_images]
  x_rec = _prepare_images(x_rec, normalize=True)
  x_org = _prepare_images(x_org, normalize=False)
  # labels
  y_true = tf.concat(y_true, axis=0).numpy()
  if len(y_pred) > 0:
    y_pred = Batchwise(y_pred, name='LabelsTest')
  return llk_x, llk_y, x_org, x_rec, y_true, y_pred, all_qz, all_pz


# ===========================================================================
# Plotting method
# ===========================================================================
def plot_latents_pairs(
    z: np.ndarray,
    f: np.ndarray,
    correlation: np.ndarray,
    labels: List[str],
    n_points: int = 1000,
    seed: int = 1):
  n_latents, n_factors = correlation.shape
  assert z.shape[1] == n_latents
  assert f.shape[1] == n_factors
  assert z.shape[0] == f.shape[0]
  rand = np.random.RandomState(seed=seed)
  ids = rand.permutation(z.shape[0])
  z = np.asarray(z)[ids][:n_points]
  f = np.asarray(f)[ids][:n_points]
  ## find the best latents for each labels
  f2z = {f_idx: z_idx
         for f_idx, z_idx in enumerate(np.argmax(correlation, axis=0))}
  ## special cases
  selected_labels = set(labels)
  n_pairs = len(selected_labels) * (len(selected_labels) - 1) // 2
  ## plotting each pairs
  ncol = 2
  nrow = n_pairs
  fig = plt.figure(figsize=(ncol * 3.5, nrow * 3))
  c = 1
  styles = dict(size=10,
                alpha=0.8,
                color='bwr',
                cbar=True,
                cbar_nticks=5,
                cbar_ticks_rotation=0,
                cbar_fontsize=8,
                fontsize=10,
                grid=False)
  for f1 in range(n_factors):
    for f2 in range(f1 + 1, n_factors):
      if (labels[f1] not in selected_labels or
          labels[f2] not in selected_labels):
        continue
      z1 = f2z[f1]
      z2 = f2z[f2]
      vs.plot_scatter(x=z[:, z1],
                      y=z[:, z2],
                      val=f[:, f1].astype(np.float32),
                      xlabel=f'Z{z1}',
                      ylabel=f'Z{z2}',
                      cbar_title=labels[f1],
                      ax=(nrow, ncol, c),
                      **styles)
      vs.plot_scatter(x=z[:, z1],
                      y=z[:, z2],
                      val=f[:, f2].astype(np.float32),
                      xlabel=f'Z{z1}',
                      ylabel=f'Z{z2}',
                      cbar_title=labels[f2],
                      ax=(nrow, ncol, c + 1),
                      **styles)
      c += 2
  plt.tight_layout()
  return fig


# ===========================================================================
# For evaluation
# ===========================================================================
def evaluate(vae: VariationalAutoencoder,
             ds: ImageDataset,
             expdir: str,
             title: str,
             batch_size: int = 64,
             take_count: int = -1,
             n_images: int = 36,
             seed: int = 1):
  n_rows = int(np.sqrt(n_images))
  is_semi = vae.is_semi_supervised()
  is_hierarchical = vae.is_hierarchical()
  ds_kw = dict(batch_size=batch_size, label_percent=1.0, shuffle=False)
  ## prepare
  rand = np.random.RandomState(seed=seed)
  if not os.path.exists(expdir):
    os.makedirs(expdir)
  ## data for training semi-supervised
  train = ds.create_dataset('train', **ds_kw)
  (llkx_train, llky_train,
   x_org_train, x_rec_train,
   y_true_train, y_pred_train,
   z_train, pz_train) = _call(vae, ds=train, rand=rand, take_count=take_count,
                              n_images=n_images, verbose=True)
  ## data for testing
  test = ds.create_dataset('test', **ds_kw)
  (llkx_test, llky_test,
   x_org_test, x_rec_test,
   y_true_test, y_pred_test,
   z_test, pz_test) = _call(vae, ds=test, rand=rand, take_count=take_count,
                            n_images=n_images, verbose=True)
  # === 0. plotting latent-factor pairs
  for idx, z in enumerate(z_test):
    z = z.mean()
    f = y_true_test
    corr_mat = Correlation.Spearman(z, f)  # [n_latents, n_factors]
    plot_latents_pairs(z, f, corr_mat, ds.labels)
    vs.plot_save(f'{expdir}/latent{idx}_factor.pdf', dpi=100, verbose=True)
  # === 0. latent traverse plot
  x_travs = x_org_test
  if x_travs.ndim == 3:  # grayscale image
    x_travs = np.expand_dims(x_travs, -1)
  else:  # color image
    x_travs = np.transpose(x_travs, (0, 2, 3, 1))
  x_travs = x_travs[rand.permutation(x_travs.shape[0])]
  n_visual_samples = 5
  n_traverse_points = 21
  n_top_latents = 10
  plt.figure(figsize=(8, 3 * n_visual_samples))
  for i in range(n_visual_samples):
    images = vae.sample_traverse(x_travs[i:i + 1],
                                 min_val=-np.min(z_test[0].mean()),
                                 max_val=np.max(z_test[0].mean()),
                                 n_top_latents=n_top_latents,
                                 n_traverse_points=n_traverse_points,
                                 mode='linear')
    images = as_tuple(images)[0]
    images = _prepare_images(images.mean().numpy(), normalize=True)
    vs.plot_images(images, grids=(n_top_latents, n_traverse_points),
                   ax=(n_visual_samples, 1, i + 1))
    if i == 0:
      plt.title('Latents traverse')
  plt.tight_layout()
  vs.plot_save(f'{expdir}/latents_traverse.pdf', dpi=180, verbose=True)
  # === 0. prior sampling plot
  images = as_tuple(vae.sample_observation(n=n_images, seed=seed))[0]
  images = _prepare_images(images.mean().numpy(), normalize=True)
  plt.figure(figsize=(5, 5))
  vs.plot_images(images, grids=(n_rows, n_rows), title='Sampled')
  # === 1. reconstruction plot
  plt.figure(figsize=(15, 15))
  vs.plot_images(x_org_train, grids=(n_rows, n_rows), ax=(2, 2, 1),
                 title='[Train]Original')
  vs.plot_images(x_rec_train, grids=(n_rows, n_rows), ax=(2, 2, 2),
                 title='[Train]Reconstructed')
  vs.plot_images(x_org_test, grids=(n_rows, n_rows), ax=(2, 2, 3),
                 title='[Test]Original')
  vs.plot_images(x_rec_test, grids=(n_rows, n_rows), ax=(2, 2, 4),
                 title='[Test]Reconstructed')
  plt.tight_layout()
  ## prepare the labels
  label_type = ds.label_type
  if label_type == 'categorical':
    labels_name = ds.labels
    true = np.argmax(y_true_test, axis=-1)
    labels_true = np.array([labels_name[i] for i in true])
    labels_pred = labels_true
    if is_semi:
      pred = np.argmax(y_pred_test.mean().numpy(), axis=-1)
      labels_pred = np.array([labels_name[i] for i in pred])
  elif label_type == 'factor':  # dsprites, shapes3d
    labels_name = ['cube', 'cylinder', 'sphere', 'round'] \
      if 'shapes3d' in ds.name else ['square', 'ellipse', 'heart']
    true = y_true_test[:, 2].astype('int32')
    labels_true = np.array([labels_name[i] for i in true])
    labels_pred = labels_true
    if is_semi:
      pred = get_ymean(y_pred_test)[:, 2].astype('int32')
      labels_pred = np.array([labels_name[i] for i in pred])
  else:  # CelebA
    raise NotImplementedError
  ## confusion matrix
  if is_semi:
    plt.figure(figsize=(8, 8))
    acc = accuracy_score(y_true=true, y_pred=pred)
    vs.plot_confusion_matrix(cm=confusion_matrix(y_true=true, y_pred=pred),
                             labels=labels_name,
                             cbar=True,
                             fontsize=10,
                             title=f'{title} Acc:{acc:.2f}')
  ## save arrays for later inspections
  with open(f'{expdir}/arrays', 'wb') as f:
    pickle.dump(dict(z_train=z_train,
                     y_pred_train=y_pred_train,
                     y_true_train=y_true_train,
                     z_test=z_test,
                     y_pred_test=y_pred_test,
                     y_true_test=y_true_test,
                     labels=labels_name,
                     ds=ds.name,
                     label_type=label_type), f)
  print(f'Exported arrays to "{expdir}/arrays"')
  ## semi-supervised
  z_mean_train = np.concatenate([z.mean().numpy().reshape(z.batch_shape[0], -1)
                                 for z in z_train], -1)
  z_mean_test = np.concatenate([z.mean().numpy().reshape(z.batch_shape[0], -1)
                                for z in z_test], -1)
  # === 2. scatter points latents plot
  n_points = 5000
  ids = rand.permutation(len(labels_true))[:n_points]
  Y_true = labels_true[ids]
  Y_pred = labels_pred[ids]
  # tsne plot
  n_latents = 0 if len(z_train) == 1 else len(z_train)
  for name, X in zip(
      ['all'] + [f'latents{i}' for i in range(n_latents)],
      [z_mean_test[ids]] + [z_test[i].mean().numpy()[ids]
                            for i in range(n_latents)]):
    print(f'Plot scatter points for {name}')
    X = X.reshape(X.shape[0], -1)  # flatten to 2D
    X = Pipeline([
      ('zscore', StandardScaler()),
      ('pca', PCA(min(X.shape[1], 512), random_state=seed))]).fit_transform(X)
    tsne = DimReduce.TSNE(X, n_components=2, framework='sklearn')
    kw = dict(x=tsne[:, 0], y=tsne[:, 1], grid=False, size=12.0, alpha=0.8)
    plt.figure(figsize=(12, 6))
    vs.plot_scatter(color=Y_true, title=f'[True]{title}-{name}', ax=(1, 2, 1),
                    **kw)
    vs.plot_scatter(color=Y_pred, title=f'[Pred]{title}-{name}', ax=(1, 2, 2),
                    **kw)
  ## save all plot
  vs.plot_save(f'{expdir}/analysis.pdf', dpi=180, verbose=True)

  # === 3. show the latents statistics
  n_latents = len(z_train)
  colors = sns.color_palette(n_colors=len(labels_true))
  styles = dict(grid=False, ticks_off=False, alpha=0.6,
                xlabel='mean', ylabel='stddev')

  # scatter between latents and labels (assume categorical distribution)
  def _show_latents_labels(Z, Y, title):
    plt.figure(figsize=(5 * n_latents, 5), dpi=150)
    for idx, z in enumerate(Z):
      if len(z.batch_shape) == 0:
        mean = np.repeat(np.expand_dims(z.mean(), 0), Y.shape[0], 0)
        stddev = z.sample(Y.shape[0]) - mean
      else:
        mean = flatten(z.mean())
        stddev = flatten(z.stddev())
      y = np.argmax(Y, axis=-1)
      data = [[], [], []]
      for y_i, c in zip(np.unique(y), colors):
        mask = (y == y_i)
        data[0].append(np.mean(mean[mask], 0))
        data[1].append(np.mean(stddev[mask], 0))
        data[2].append([labels_true[y_i]] * mean.shape[1])
      vs.plot_scatter(x=np.concatenate(data[0], 0),
                      y=np.concatenate(data[1], 0),
                      color=np.concatenate(data[2], 0),
                      ax=(1, n_latents, idx + 1),
                      size=15 if mean.shape[1] < 128 else 8,
                      title=f'[Test-{title}]#{idx} - {mean.shape[1]} (units)',
                      **styles)
    plt.tight_layout()

  # simple scatter mean-stddev each latents
  def _show_latents(Z, title):
    plt.figure(figsize=(3.5 * n_latents, 3.5), dpi=150)
    for idx, z in enumerate(Z):
      mean = flatten(z.mean())
      stddev = flatten(z.stddev())
      if mean.ndim == 2:
        mean = np.mean(mean, 0)
        stddev = np.mean(stddev, 0)
      vs.plot_scatter(x=mean, y=stddev,
                      ax=(1, n_latents, idx + 1),
                      size=15 if len(mean) < 128 else 8,
                      title=f'[Test-{title}]#{idx} - {len(mean)} (units)',
                      **styles)

  _show_latents_labels(z_test, y_true_test, 'post')
  _show_latents_labels(pz_test, y_true_test, 'prior')
  _show_latents(z_test, 'post')
  _show_latents(pz_test, 'prior')

  # KL statistics
  vs.plot_figure()
  for idx, (qz, pz) in enumerate(zip(z_test, pz_test)):
    kl = []
    qz = Normal(loc=qz.mean(), scale=qz.stddev(), name=f'posterior{idx}')
    pz = Normal(loc=pz.mean(), scale=pz.stddev(), name=f'prior{idx}')
    for s, e in minibatch(batch_size=8, n=100):
      z = qz.sample(e - s)
      # don't do this in GPU, it explodes!
      kl.append((qz.log_prob(z) - pz.log_prob(z)).numpy())
    kl = np.concatenate(kl, 0)  # (mcmc, batch, event)
    # per sample
    kl_samples = np.sum(kl, as_tuple(list(range(2, kl.ndim))))
    kl_samples = logsumexp(kl_samples, 0)
    plt.subplot(n_latents, 2, idx * 2 + 1)
    sns.histplot(kl_samples, bins=50)
    plt.title(f'Z#{idx} KL per sample (nats)')
    # per latent
    kl_latents = np.mean(flatten(logsumexp(kl, 0)), 0)
    plt.subplot(n_latents, 2, idx * 2 + 2)
    plt.plot(np.sort(kl_latents))
    plt.title(f'Z#{idx} KL per dim (nats)')
  plt.tight_layout()

  vs.plot_save(f'{expdir}/latents.pdf', dpi=180, verbose=True)


# ===========================================================================
# Main
# ===========================================================================
SEMI_SETTING = dict(
  mnist=100,
  fashionmnist=100,
  svhn=1000,
  cifar10=4000,
  dsprites=1000,
  shapes3dsmall=4000,
  shapes3d=4000,
  celeba=10000,
)

BS_SETTING = dict(
  mnist=100,
  fashionmnist=100,
  svhn=64,
  cifar10=64,
  dsprites=100,
  shapes3dsmall=64,
  shapes3d=64,
  celeba=64
)


def main(vae, ds, args, parser):
  n_labeled = SEMI_SETTING[ds]
  vae = get_vae(vae)
  ds = get_dataset(ds)
  batch_size = BS_SETTING[ds.name]
  assert isinstance(ds,
                    ImageDataset), f'Only support image dataset but given {ds}'
  vae: Type[VariationalAutoencoder]
  ds: ImageDataset
  is_semi = vae.is_semi_supervised()
  ## skip unsupervised system, if there are semi-supervised modifications
  if not is_semi:
    for key in ('alpha', 'coef', 'ratio'):
      if parser.get_default(key) != getattr(args, key):
        print('Skip semi-supervised training for:', args)
        return
  ## prepare the arguments
  kw = {}
  ## path
  name = f'{ds.name}_{vae.__name__.lower()}'
  path = f'{ROOT}/{name}'
  anno = []
  if args.zdim > 0:
    anno.append(f'z{int(args.zdim)}')
  if is_semi:
    anno.append(f'a{args.alpha:g}')
    anno.append(f'r{args.ratio:g}')
    kw['alpha'] = args.alpha
  if issubclass(vae, (SemafoBase, MIVAE)):
    anno.append(f'c{args.coef:g}')
    kw['mi_coef'] = args.coef
  if len(anno) > 0:
    path += f"_{'_'.join(anno)}"
  if args.override and os.path.exists(path):
    shutil.rmtree(path)
    print('Override:', path)
  if not os.path.exists(path):
    os.makedirs(path)
  print(path)
  ## data
  train = ds.create_dataset('train',
                            batch_size=batch_size,
                            label_percent=n_labeled if is_semi else False,
                            oversample_ratio=args.ratio)
  valid = ds.create_dataset('valid', label_percent=1.0,
                            batch_size=batch_size // 2)
  ## create model
  vae = vae(**kw,
            **get_networks(ds.name,
                           zdim=int(args.zdim),
                           is_semi_supervised=is_semi))
  vae.build((None,) + ds.shape)
  print(vae)
  vae.load_weights(f'{path}/model', verbose=True)
  best_llk = []

  ## training
  def callback():
    llk = []
    y_true = []
    y_pred = []
    for x, y in tqdm(valid.take(500)):
      P, Q = vae(x, training=False)
      P = as_tuple(P)
      llk.append(P[0].log_prob(x))
      if is_semi:
        y_true.append(np.argmax(y, -1))
        y_pred.append(np.argmax(get_ymean(P[1]), -1))
    # accuracy
    if is_semi:
      y_true = np.concatenate(y_true, axis=0)
      y_pred = np.concatenate(y_pred, axis=0)
      acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    else:
      acc = 0
    # log-likelihood
    llk = tf.reduce_mean(tf.concat(llk, 0))
    best_llk.append(llk)
    text = f'#{vae.step.numpy()} llk={llk:.2f} acc={acc:.2f}'
    if llk >= np.max(best_llk):
      vae.save_weights(f'{path}/model')
      vae.trainer.print(f'best llk {text}')
    else:
      vae.trainer.print(f'worse llk {text}')
    # tensorboard summary
    tf.summary.scalar('llk_valid', llk)
    tf.summary.scalar('acc_valid', acc)

  optim_info = get_optimizer_info(ds.name, batch_size=batch_size)
  if args.it > 0:
    optim_info['max_iter'] = int(args.it)
  vae.fit(
    train,
    skip_fitted=True,
    logdir=path,
    on_valid_end=callback,
    clipnorm=100,
    logging_interval=10,
    valid_interval=180,
    nan_gradients_policy='stop',
    **optim_info,
  )
  ## evaluating
  vae.load_weights(f'{path}/model', verbose=True)
  if args.eval:
    evaluate(vae, ds, path, f'{name}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('vae')
  parser.add_argument('ds')
  parser.add_argument('--override', action='store_true')
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('-alpha', type=float, default=10.)
  parser.add_argument('-coef', type=float, default=0.1)
  parser.add_argument('-ratio', type=float, default=0.0)
  parser.add_argument('-zdim', type=int, default=0)
  parser.add_argument('-it', type=int, default=0)
  ## select the right vae
  args = parser.parse_args()
  for vae in args.vae.split(','):
    for ds in args.ds.split(','):
      main(vae, ds, args, parser)
