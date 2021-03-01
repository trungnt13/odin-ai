import os
import shutil
from itertools import product
import argparse
from typing import Optional

import numpy as np
import tensorflow as tf

from odin.backend.interpolation import linear
from odin.bay.vi import (VAE, beta10VAE, conditionalM2VAE, factorVAE, miVAE,
                         multiheadVAE, semafoVAE, remafoVAE, skiptaskVAE)
from odin.utils import MPI, as_tuple

N_CPU = 2
OVERRIDE = False

all_vaes = [
    # semafoVAE,
    # remafoVAE,
    # miVAE,
    # factorVAE,
    # beta10VAE,
    VAE,
    # multiheadVAE,
    skiptaskVAE,
    # conditionalM2VAE,
]
all_py = [
    0.004,
    # 0.06,
    # 0.2,
    # 0.95,
]
# dsname -> (mi_coef, decay steps)
all_dsinfo = dict(
    mnist=linear(vmin=0.5, vmax=0.05, length=20000, cyclical=True),
    fashionmnist=linear(vmin=0.5, vmax=0.05, length=20000, cyclical=True),
    shapes3d=linear(vmin=0.5, vmax=0.05, length=20000, cyclical=True),
    dsprites=linear(vmin=0.2, vmax=0.02, length=20000, cyclical=True),
    celeba=linear(vmin=0.5, vmax=0.05, length=20000, cyclical=True),
)
# Higher MI_coef for shapes3d and celeba
# Set MI_Coef to zero when labels present
outdir = '/home/trung/exp/transitive'
if not os.path.exists(outdir):
  os.makedirs(outdir)


# ===========================================================================
# Training
# ===========================================================================
def train_task(args):
  from odin.fuel import get_dataset
  from odin.networks import get_networks, get_optimizer_info
  model, dsname, py, is_semi_supervised = args
  ## special case
  if (model is conditionalM2VAE and
      dsname not in ('mnist', 'fashionmnist', 'celeba')):
    return
  ######## prepare path
  logdir = f'{outdir}/{dsname}/{model.__name__.lower()}_{py}'
  if OVERRIDE and os.path.exists(logdir):
    shutil.rmtree(logdir)
    print(f'Override model at path: {logdir}')
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  modelpath = f'{logdir}/model'
  ######## dataset
  normalize = 'tanh' if dsname == 'celeba' else 'probs'
  ds = get_dataset(dsname)
  train = ds.create_dataset('train',
                            batch_size=32,
                            inc_labels=py,
                            normalize=normalize)
  valid = ds.create_dataset('valid',
                            batch_size=32,
                            inc_labels=True,
                            normalize=normalize\
    ).shuffle(1000, seed=1, reshuffle_each_iteration=True)
  ######## model
  networks = get_networks(dsname, is_semi_supervised=is_semi_supervised)
  kw = {}
  if is_semi_supervised:
    kw['alpha'] = 1 / py
    if model == semafoVAE:
      kw['mi_coef'] = all_dsinfo[dsname]
  vae = model(**networks, **kw)
  vae.build((None,) + ds.shape)

  ######## training
  best_llk = [-np.inf]

  def callback():
    llk = []
    for x, y in valid.take(300):
      P, Q = vae(x, training=False)
      px_z = as_tuple(P)[0].log_prob(x)
      llk.append(px_z)
    llk = tf.reduce_mean(tf.concat(llk, axis=0))
    if llk > best_llk[0]:
      vae.save_weights(modelpath)
      best_llk[0] = llk
      vae.trainer.print(
          f'{model.__name__} {dsname} {py} '
          f'best weights at iter#{vae.step.numpy()} llk={llk:.2f}')
    else:
      vae.trainer.print(
          f'{model.__name__} {dsname} {py} '
          f'worse weights at iter#{vae.step.numpy()} llk={llk:.2f}')

  opt_info = get_optimizer_info(dsname)
  # extra 20000 iterations for semafoVAE
  if isinstance(vae, semafoVAE):
    opt_info['max_iter'] += 20000
  vae.load_weights(modelpath)
  vae.fit(
      train,
      logdir=logdir,
      callback=callback,
      valid_interval=60,
      logging_interval=2,
      nan_gradients_policy='stop',
      compile_graph=True,
      skip_fitted=True,
      **opt_info,
  )
  print(f'Trained {model.__name__} {dsname} {py} {vae.step.numpy()}(steps)')


def main(ds: Optional[str] = ''):
  jobs = []
  ds = set([i for i in ds.split(',') if len(i) > 0])
  for model, dsname in product(all_vaes, all_dsinfo.keys()):
    # filter the dataset
    if len(ds) > 0 and dsname not in ds:
      continue
    # prepare model
    is_semi_supervised = model.is_semi_supervised()
    py_list = [0.0] if not is_semi_supervised else list(all_py)
    for py in py_list:
      jobs.append([model, dsname, py, is_semi_supervised])
  print(f'Start training {len(jobs)} jobs ...')
  for _ in MPI(jobs, train_task, ncpu=N_CPU, batch=1):
    pass


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--override', action='store_true')
  parser.add_argument('-ncpu', type=int, default=2)
  parser.add_argument('-ds', type=str, default='')
  args = parser.parse_args()
  OVERRIDE = args.override
  N_CPU = args.ncpu
  main(ds=args.ds)
