# Helper for setting up and evaluate VAE experiments
import glob
import inspect
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import product
from numbers import Number
from typing import Dict, Any, Tuple, Union, Callable, Optional, Sequence

import dataclasses
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from six import string_types
from tensorflow_probability.python.distributions import Distribution, Normal, \
  Blockwise, Categorical, Bernoulli
from typeguard import typechecked

from odin import visual as vs
from odin.bay import VariationalModel, get_vae, Vamprior
from odin.bay.distributions import Batchwise, QuantizedLogistic
from odin.fuel import ImageDataset, get_dataset
from odin.networks import get_optimizer_info, get_networks
from odin.utils import as_tuple
from odin.utils.decorators import schedule
from odin.utils.python_utils import defaultdictkey

sns.set()
np.random.seed(1)
tf.random.set_seed(1)

__all__ = [
  'Arguments',
  'prepare_images',
  'prepare_labels',
  'save_figs',
  'set_cfg',
  'get_output_dir',
  'get_model',
  'get_model_path',
  'get_results_path',
  'get_args',
  'run_multi',
  'train',
]

_root_path: str = '/tmp/model'
_logging_interval: float = 5.
_valid_interval: float = 80
_n_valid_batches: int = 200
_extra_kw = []
_DS: Dict[str, ImageDataset] = defaultdictkey(lambda name: get_dataset(name))


# ===========================================================================
# Helper for evaluation
# ===========================================================================
@dataclasses.dataclass
class Arguments:
  vae: str = ''
  ds: str = ''
  zdim: Union[str, int] = 32
  it: int = 80000
  bs: int = 32
  clipnorm: float = 100.
  dpi: int = 100
  points: int = 4000
  images: int = 36
  seed: int = 1
  eval: bool = False
  override: bool = False
  debug: bool = False
  # semi-supervised
  py: float = 0.0
  ratio: float = 0.1


def to_dist(p: Union[Distribution, Sequence[Distribution]]
            ) -> Union[Sequence[Distribution], Distribution]:
  """Convert DeferredTensor back to original Distribution"""
  if isinstance(p, (tuple, list)):
    return [to_dist(i) for i in p]
  p: Distribution
  return (p.parameters['distribution']
          if 'deferred_tensor' in str(type(p)) else p)


def get_ymean(py: Batchwise) -> np.ndarray:
  y_mean = []
  if isinstance(py, Batchwise):
    py = py.distributions
  else:
    py = [py]
  for p in py:
    p = to_dist(p)  # remove the DeferredTensor
    if isinstance(p, Blockwise):
      y = [i.mode() if isinstance(i, Categorical) else i.mean()
           for i in p.distributions.model]
      y = tf.stack(y, axis=-1)
    else:
      y = p.mode() if isinstance(p, Categorical) else p.mean()
    y_mean.append(y)
  return tf.concat(y_mean, 0).numpy()


def prepare_labels(y_true: np.ndarray,
                   y_pred: Optional[Distribution],
                   args: Arguments
                   ) -> Tuple[np.ndarray, np.ndarray,
                              Optional[np.ndarray], Optional[np.ndarray]]:
  """Prepare categorical labels

  Returns
  -------
  true_ids, true_labels, pred_ids, pred_labels
  """
  ds: ImageDataset = _DS[args.ds]
  label_type = ds.label_type
  pred = None
  labels_pred = None
  if label_type == 'categorical':
    labels_name = ds.labels
    true = np.argmax(y_true, axis=-1)
    labels_true = np.array([labels_name[i] for i in true])
    if y_pred is not None:
      pred = np.argmax(get_ymean(y_pred), axis=-1)
      labels_pred = np.array([labels_name[i] for i in pred])
  elif label_type == 'factor':  # dsprites, shapes3d
    labels_name = ['cube', 'cylinder', 'sphere', 'round'] \
      if 'shapes3d' in ds.name else ['square', 'ellipse', 'heart']
    true = y_true[:, 2].astype('int32')
    labels_true = np.array([labels_name[i] for i in true])
    if y_pred is not None:
      pred = get_ymean(y_pred)[:, 2].astype('int32')
      labels_pred = np.array([labels_name[i] for i in pred])
  else:  # CelebA
    raise NotImplementedError
  return true, labels_true, pred, labels_pred


def prepare_images(x, normalize=False):
  """if normalize=True, normalize the image to [0, 1], used for the
  reconstructed or generated image, not the original one.
  """
  x = np.asarray(x)
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


# ===========================================================================
# Helper for plotting
# ===========================================================================
def save_figs(args: Arguments,
              name: str,
              figs: Optional[Sequence[plt.Figure]] = None):
  path = get_results_path(args)
  multi_figs = True
  if figs is not None and len(as_tuple(figs)) == 1:
    multi_figs = False
    figs = as_tuple(figs)
  path = f'{path}/{name}.{"pdf" if multi_figs else "png"}'
  vs.plot_save(path, figs, dpi=args.dpi, verbose=True)


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


def get_output_dir(args: Arguments) -> str:
  if not os.path.exists(_root_path):
    os.makedirs(_root_path)
  path = f'{_root_path}/{args.ds}/{args.vae}_z{args.zdim}_i{args.it}'
  if len(_extra_kw) > 0:
    for kw in _extra_kw:
      path += f'_{kw[0]}{getattr(args, kw)}'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def get_model_path(args: Arguments) -> str:
  save_dir = get_output_dir(args)
  return f'{save_dir}/model'


def get_results_path(args: Arguments) -> str:
  save_dir = get_output_dir(args)
  path = f'{save_dir}/results'
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def get_model(args: Arguments,
              return_dataset: bool = True,
              encoder: Any = None,
              decoder: Any = None,
              latents: Any = None,
              observation: Any = None,
              **kwargs) -> Union[VariationalModel,
                                 Tuple[VariationalModel, ImageDataset]]:
  ds = _DS[args.ds]
  vae_name = args.vae
  vae_cls = get_vae(vae_name)
  networks = get_networks(ds.name, zdim=None if args.zdim < 1 else args.zdim)
  for k, v in locals().items():
    if k in networks and v is not None:
      networks[k] = v
  vae = vae_cls(**networks, **kwargs)
  vae.build(ds.full_shape)
  if return_dataset:
    return vae, ds
  return vae


def get_args(defaults: Optional[Dict[str, Any]] = None) -> Arguments:
  if defaults is None:
    defaults = {}
  final_args = Arguments()
  parser = ArgumentParser()
  parser.add_argument('vae', type=str)
  parser.add_argument('ds', type=str)
  parser.add_argument(
    'zdim',
    type=lambda t: int(t) if isinstance(t, Number) else str(t),
    default=32)
  for k, v in final_args.__annotations__.items():
    if k in ['vae', 'ds', 'zdim']:
      continue
    val = defaults.get(k, getattr(final_args, k))
    if v == bool:
      parser.add_argument(f'--{k}',
                          action=f'store_{"false" if val else "true"}')
    else:
      parser.add_argument(f'-{k}', type=v, default=val)
  args: Namespace = parser.parse_args()
  for k in dir(args):
    if '_' == k[0]:
      continue
    setattr(final_args, k, getattr(args, k))
  return final_args


@typechecked()
def run_multi(main_fn: Callable[[Arguments], Any],
              args: Arguments):
  vae = args.vae.split(',')
  ds = args.ds.split(',')
  zdim = (map(int, args.zdim.split(','))
          if isinstance(args.zdim, string_types) else
          [args.zdim])
  for vae, ds, zdim in product(vae, ds, zdim):
    args_copy = deepcopy(args)
    args_copy.vae = vae
    args_copy.ds = ds
    args_copy.zdim = zdim
    print('*** Running configuration:', args_copy)
    try:
      main_fn(args_copy)
    except Exception:
      from traceback import print_exc
      print_exc()


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
  @schedule(5)
  def llk_pixels(model: VariationalModel, valid_ds: tf.data.Dataset):
    llk = []
    for x, y in valid_ds.take(5):
      px, _ = _call(model, x, y, decode=True)
      px = as_tuple(px)[0]
      if hasattr(px, 'distribution'):
        px = px.distribution
      if isinstance(px, Bernoulli):
        px = Bernoulli(logits=px.logits)
      elif isinstance(px, Normal):
        px = Normal(loc=px.loc, scale=px.scale)
      elif isinstance(px, QuantizedLogistic):
        px = QuantizedLogistic(loc=px.loc, scale=px.scale,
                               low=px.low, high=px.high,
                               inputs_domain=px.inputs_domain,
                               reinterpreted_batch_ndims=None)
      else:
        return  # nothing to do
      llk.append(px.log_prob(x))
    # average over all channels
    llk_image = tf.reduce_mean(tf.reduce_mean(tf.concat(llk, 0), axis=0),
                               axis=-1)
    llk = tf.reshape(llk_image, -1)
    tf.summary.histogram('valid/llk_pixels', llk, step=model.step)
    # show the image heatmap of llk pixels
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    im = ax.pcolormesh(llk_image.numpy(),
                       cmap='Spectral',
                       vmin=np.min(llk),
                       vmax=np.max(llk))
    ax.axis('off')
    ax.margins(0.)
    # color bar
    ticks = np.linspace(np.min(llk), np.max(llk), 5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in ticks])
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    tf.summary.image('llk_heatmap', vs.plot_to_image(fig, dpi=100))

  @staticmethod
  @schedule(5)
  def latent_units(model: VariationalModel,
                   valid_ds: tf.data.Dataset):
    weights = model.weights
    Qz = []
    Pz = []
    for x, y in valid_ds.take(10):
      _call(model, x, y, decode=True)
      qz, pz = model.get_latents(return_prior=True)
      Qz.append(as_tuple(qz))
      Pz.append(as_tuple(pz))
    n_latents = len(Qz[0])
    for i in range(n_latents):
      qz: Sequence[Distribution] = [q[i] for q in Qz]
      pz: Sequence[Distribution] = [p[i] for p in Pz]
      # tracking kl
      kld = []
      for q, p in zip(qz, pz):
        q = Normal(loc=q.mean(), scale=q.stddev())
        z = q.sample()
        if isinstance(p, Vamprior):
          C = p.C
          p = p.distribution  # [n_components, zdim]
          p = Normal(loc=p.mean(), scale=p.stddev())
          kld.append(
            q.log_prob(z) -
            (tf.reduce_logsumexp(p.log_prob(tf.expand_dims(z, 1)), 1) - C))
        else:
          p = Normal(loc=p.mean(), scale=p.stddev())
          kld.append(q.log_prob(z) - p.log_prob(z))
      kld = tf.reshape(tf.reduce_mean(tf.concat(kld, 0), axis=0), -1).numpy()
      # mean and stddev
      mean = tf.reduce_mean(tf.concat([d.mean() for d in qz], axis=0), 0)
      stddev = tf.reduce_mean(tf.concat([d.stddev() for d in qz], axis=0), 0)
      mean = tf.reshape(mean, -1).numpy()
      stddev = tf.reshape(stddev, -1).numpy()
      zdim = mean.shape[0]
      # the figure
      plt.figure(figsize=(12, 5), dpi=50)
      lines = []
      ids = np.argsort(stddev)
      styles = dict(marker='o', markersize=2, linewidth=0, alpha=0.8)
      lines += plt.plot(mean[ids], label='mean', color='r', **styles)
      lines += plt.plot(stddev[ids], label='stddev', color='b', **styles)
      plt.grid(False)
      # show weights if exists
      plt.twinx()
      lines += plt.plot(kld[ids], label='KL(q|p)', linestyle='--',
                        linewidth=1.0, alpha=0.6)
      for w in weights:
        name = w.name
        if w.shape.rank > 0 and w.shape[0] == zdim and '/kernel' in name:
          w = tf.linalg.norm(tf.reshape(w, (w.shape[0], -1)), axis=1).numpy()
          lines += plt.plot(w[ids], label=name.split(':')[0], linestyle='--',
                            alpha=0.6)
      plt.grid(False)
      plt.legend(lines, [ln.get_label() for ln in lines], fontsize=6)
      # save summary
      tf.summary.image(f'z{i}', vs.plot_to_image(plt.gcf()), step=model.step)
      tf.summary.histogram(f'z{i}/mean', mean, step=model.step)
      tf.summary.histogram(f'z{i}/stddev', stddev, step=model.step)
      tf.summary.histogram(f'z{i}/kld', kld, step=model.step)

  @staticmethod
  def save_best_llk(model: VariationalModel,
                    valid_ds: tf.data.Dataset):
    model_id = id(model)
    llk = []
    for x, y in valid_ds.take(_n_valid_batches):
      px, qz = _call(model, x, y)
      px: Distribution = as_tuple(px)[0]
      if len(px.event_shape) == 0 and px.batch_shape[1:] == x.shape[1:]:
        llk.append(tf.reduce_sum(px.log_prob(x), tf.range(1, x.shape.rank)))
      elif px.event_shape == x.shape[1:]:  # VAE
        llk.append(px.log_prob(x))
      else:  # VIB
        llk.append(px.log_prob(y))
    llk = tf.reduce_mean(tf.concat(llk, 0)).numpy()
    tf.summary.scalar('valid/llk', llk, step=model.step)
    if llk > _best[model_id]:
      _best[model_id] = llk
      model.save_weights(overwrite=True)
      model.trainer.print(f'best llk: {llk:.2f}')
    else:
      model.trainer.print(
        f'worse llk: {llk:.2f} vs best: {_best[model_id]:.2f}')


def train(
    model: VariationalModel,
    ds: ImageDataset,
    args: Arguments,
    on_batch_end: Sequence[Callable[..., Any]] = (Callback.latent_units,
                                                  Callback.llk_pixels),
    on_valid_end: Sequence[Callable[..., Any]] = (Callback.save_best_llk,),
    label_percent: float = 0,
    oversample_ratio: float = 0.5,
    train_data_map: Optional[Callable] = None,
    valid_data_map: Optional[Callable] = None,
    verbose: bool = True) -> VariationalModel:
  if verbose:
    print(model)
  save_dir = get_output_dir(args)
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

  if train_data_map is not None:
    train_ds = train_ds.map(train_data_map)
  if valid_data_map is not None:
    valid_ds = valid_ds.map(valid_data_map)

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
            compile_graph=not args.debug,
            **train_kw)
  model.load_weights(model_path)
  return model
