# Helper for setting up and evaluate VAE experiments
import glob
import inspect
import os
import time
import timeit
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from functools import partial, wraps
from typing import Dict, Any, Tuple, Union, Callable, Optional, Sequence
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import Distribution

from odin.backend.keras_helpers import layer2text
from odin.bay import VariationalAutoencoder, VariationalModel, get_vae
from odin.fuel import ImageDataset, get_dataset
from odin.networks import get_optimizer_info, get_networks
from odin.utils import as_tuple
from odin import visual as vs
from matplotlib import pyplot as plt

import seaborn as sns

sns.set()

__all__ = [
  'set_cfg',
  'get_dir',
  'get_model',
  'get_model_path',
  'get_args',
  'train',
]

from odin.utils.decorators import schedule

_root_path: str = '/tmp/vae'
_logging_interval: float = 5.
_valid_interval: float = 120
_n_valid_batches: int = 200
_extra_kw = []

np.random.seed(1)
tf.random.set_seed(1)


# ===========================================================================
# Helpers for path
# ===========================================================================
def set_cfg(root_path: Optional[str] = None,
            logging_interval: Optional[float] = None,
            valid_interval: Optional[float] = None,
            n_valid_batches: Optional[int] = None):
  for k, v in locals().items():
    if v is None:
      continue
    if f'_{k}' in globals():
      globals()[f'_{k}'] = v
  print('Set configuration:')
  for k, v in sorted(globals().items()):
    if '_' == k[0] and '_' != k[1]:
      print(f'  {k[1:]:20s}', v)


def get_dir(args: Namespace) -> str:
  if not os.path.exists(_root_path):
    os.makedirs(_root_path)
  path = f'{_root_path}/{args.ds}/{args.vae}_z{args.zdim}_i{args.it}'
  if len(_extra_kw) > 0:
    for kw in _extra_kw:
      path += f'_{kw[0]}{getattr(args, kw)}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def get_model_path(args: Namespace) -> str:
  save_dir = get_dir(args)
  return f'{save_dir}/model'


def get_model(args: Namespace,
              return_dataset: bool = True,
              encoder: Any = None,
              decoder: Any = None,
              latents: Any = None,
              observation: Any = None,
              **kwargs) -> Union[VariationalModel,
                                 Tuple[VariationalModel, ImageDataset]]:
  ds = get_dataset(args.ds)
  vae_name = args.vae
  vae_cls = get_vae(vae_name)
  networks = get_networks(ds.name)
  for k, v in locals().items():
    if k in networks and v is not None:
      networks[k] = v
  vae = vae_cls(**networks, **kwargs)
  vae.build(ds.full_shape)
  if return_dataset:
    return vae, ds
  return vae


def get_args(extra: Optional[Dict[str, Tuple[type, Any]]] = None) -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('vae', type=str)
  parser.add_argument('ds', type=str)
  parser.add_argument('zdim', type=int)
  parser.add_argument('-it', type=int, default=80000)
  parser.add_argument('-bs', type=int, default=32)
  parser.add_argument('-clipnorm', type=float, default=100.)
  if extra is not None:
    for k, (t, d) in extra.items():
      _extra_kw.append(k)
      parser.add_argument(f'-{k}', type=t, default=d)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--override', action='store_true')
  return parser.parse_args()


# ===========================================================================
# Training and evaluate
# ===========================================================================
_best = defaultdict(lambda: -np.inf)
_attrs = defaultdict(lambda: defaultdict(lambda: None))


def _call(model: VariationalModel,
          x: tf.Tensor,
          y: tf.Tensor,
          decode: bool = True):
  model_id = id(model)
  if decode:
    def call_fn(inputs):
      return model(inputs, training=False)
  else:
    def call_fn(inputs):
      return model.encode(inputs, training=False)
  if _attrs[model_id]['labels_as_inputs']:
    rets = call_fn(y)
  else:
    try:
      rets = call_fn(x)
    except ValueError:
      _attrs[model_id]['labels_as_inputs'] = True
      rets = call_fn(y)
  return rets


class Callback:

  @staticmethod
  @schedule(5.)
  def latent_units(model: VariationalModel,
                   valid_ds: tf.data.Dataset):
    weights = model.weights
    Q = []
    for x, y in valid_ds.take(20):
      _call(model, x, y, decode=True)
      qz = model.get_latents()
      Q.append(as_tuple(qz))
    n_latents = len(Q[0])
    for i in range(n_latents):
      dists = [q[i] for q in Q]
      dists: Sequence[Distribution]
      # mean and stddev
      mean = tf.reduce_mean(tf.concat([d.mean() for d in dists], axis=0), 0)
      stddev = tf.reduce_mean(tf.concat([d.stddev() for d in dists], axis=0), 0)
      mean = tf.reshape(mean, -1).numpy()
      stddev = tf.reshape(stddev, -1).numpy()
      # the figure
      plt.figure(figsize=(8, 5), dpi=60)
      lines = []
      ids = np.argsort(stddev)
      styles = dict(marker='o', markersize=5, linewidth=1)
      lines += plt.plot(mean[ids], label='mean', color='r', **styles)
      lines += plt.plot(stddev[ids], label='stddev', color='b', **styles)
      plt.grid(False)
      # show weights if exists
      plt.twinx()
      zdim = mean.shape[0]
      for w in weights:
        name = w.name
        if w.shape.rank > 0 and w.shape[0] == zdim and '/kernel' in name:
          w = tf.linalg.norm(tf.reshape(w, (w.shape[0], -1)), axis=1).numpy()
          lines += plt.plot(w[ids], label=name.split(':')[0], linestyle='--',
                            alpha=0.6)
      plt.grid(False)
      plt.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
      # save summary
      tf.summary.image(f'z{i}', vs.plot_to_image(plt.gcf()), step=model.step)
      tf.summary.histogram(f'z{i}/mean', mean, step=model.step)
      tf.summary.histogram(f'z{i}/stddev', stddev, step=model.step)

  @staticmethod
  def save_best_llk(model: VariationalModel,
                    valid_ds: tf.data.Dataset):
    model_id = id(model)
    llk = []
    for x, y in valid_ds.take(_n_valid_batches):
      px, qz = _call(model, x, y)
      px: Distribution = as_tuple(px)[0]
      if px.event_shape == x.shape[1:]:  # VAE
        llk.append(px.log_prob(x))
      else:  # VIB
        llk.append(px.log_prob(y))
    llk = tf.reduce_mean(tf.concat(llk, 0)).numpy()
    if llk > _best[model_id]:
      _best[model_id] = llk
      model.save_weights(overwrite=True)
      model.trainer.print(f'best llk: {llk:.2f}')


def train(
    model: VariationalModel,
    ds: ImageDataset,
    args: Namespace,
    on_batch_end: Sequence[Callable[..., Any]] = (Callback.latent_units,),
    on_valid_end: Sequence[Callable[..., Any]] = (Callback.save_best_llk,),
    label_percent: float = 0,
    oversample_ratio: float = 0.5,
    debug: bool = False) -> VariationalModel:
  print(model)
  save_dir = get_dir(args)
  print('Save dir:', save_dir)
  model_path = get_model_path(args)
  model.save_path = model_path
  # === 0. check override
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
      model.load_weights(model_path)
      return model
  # === 1. create data
  train_ds = ds.create_dataset('train', batch_size=args.bs,
                               label_percent=label_percent,
                               oversample_ratio=oversample_ratio,
                               fixed_oversample=True)
  valid_ds = ds.create_dataset('valid', batch_size=64, label_percent=1.0)
  train_ds: tf.data.Dataset
  valid_ds: tf.data.Dataset

  # === 2. callback
  all_attrs = dict(locals())
  valid_callback = []
  for fn in as_tuple(on_valid_end):
    spec = inspect.getfullargspec(fn)
    fn = partial(fn, **{k: all_attrs[k] for k in spec.args + spec.kwonlyargs
                        if k in all_attrs})
    valid_callback.append(fn)
  batch_callback = []
  for fn in as_tuple(on_batch_end):
    spec = inspect.getfullargspec(fn)
    fn = partial(fn, **{k: all_attrs[k] for k in spec.args + spec.kwonlyargs
                        if k in all_attrs})
    batch_callback.append(fn)

  # === 3. training
  train_kw = get_optimizer_info(args.ds, batch_size=args.bs * 2)
  if args.it > 0:
    train_kw['max_iter'] = args.it
  model.fit(train_ds,
            on_batch_end=batch_callback,
            on_valid_end=valid_callback,
            logging_interval=_logging_interval,
            valid_interval=_valid_interval,
            global_clipnorm=args.clipnorm,
            logdir=save_dir,
            compile_graph=not debug,
            **train_kw)
  model.load_weights(model_path)
  return model

# ===========================================================================
# Helpers for plot
# ===========================================================================
