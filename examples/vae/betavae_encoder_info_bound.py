import os.path
import pickle
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Dense, Flatten, Reshape

from odin.fuel import MNIST
from odin.networks import get_networks, SequentialNetwork, CenterAt0
from odin.bay.vi import BetaGammaVAE
from argparse import ArgumentParser
from dataclasses import dataclass
from odin.utils import MPI
from utils import prepare_images
import seaborn as sns
from matplotlib import pyplot as plt
from odin import visual as vs

sns.set()
# ===========================================================================
# Constants
# ===========================================================================
PATH = os.path.expanduser('~/exp/beta_encoder')
BS = 32
MAX_ITER = 200000
BETA = [0.5, 1, 2, 5, 10, 20, 40]
ZDIM = 64


@dataclass
class Arguments:
  zdim: int = 32
  beta: float = 1
  gamma: float = 1
  finetune: bool = True
  overwrite: bool = False


# ===========================================================================
# Main
# ===========================================================================
def get_path(args: Arguments):
  path = os.path.join(PATH, f'z{args.zdim:g}_b{args.beta:g}_g{args.gamma:g}_' +
                      ('finetune' if args.finetune else 'none'))
  if os.path.exists(path) and args.overwrite:
    print('Overwrite:', path)
    shutil.rmtree(path)
  if not os.path.exists(path):
    os.makedirs(path)
  model_path = os.path.join(path, 'model')
  return path, model_path


def get_cache_path():
  return os.path.join(PATH, 'results')


def get_dense_networks(args: Arguments):
  networks = get_networks('mnist',
                          is_semi_supervised=False,
                          is_hierarchical=False,
                          zdim=args.zdim)
  networks['encoder'] = SequentialNetwork([
    InputLayer(input_shape=[28, 28, 1]),
    CenterAt0(),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
  ], name='Encoder')
  networks['decoder'] = SequentialNetwork([
    InputLayer(input_shape=[args.zdim]),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(28 * 28 * 1, activation='linear'),
    Reshape([28, 28, 1]),
  ], name='Decoder')
  return networks


def train(args: Arguments):
  np.random.seed(1)
  tf.random.set_seed(1)
  path, model_path = get_path(args)
  ds = MNIST()
  model = BetaGammaVAE(
    **get_dense_networks(args),
    gamma=float(args.gamma), beta=float(args.beta),
    name=f'Z{args.zdim}B{args.beta}G{args.gamma}'.replace('.', ''))
  model.build(ds.full_shape)
  print(model)
  optim1 = tf.optimizers.Adam(learning_rate=5e-4)
  optim2 = tf.optimizers.Adam(learning_rate=1e-4)
  # === 0. helper
  best_llk = [-np.inf, 0]
  valid = ds.create_dataset('valid')

  def callback():
    llk = tf.reduce_mean(
      tf.concat([model(x)[0].log_prob(x) for x in valid.take(100)], 0)
    ).numpy()
    if llk > best_llk[0]:
      best_llk[0] = llk
      best_llk[1] = model.step.numpy()
      model.trainer.print('*Save weights at:', model_path)
      model.save_weights(model_path, overwrite=True)
    model.trainer.print(
      f'Current:{llk:.2f} Best:{best_llk[0]:.2f} Step:{int(best_llk[1])}')
    for k, v in model.last_train_metrics.items():
      if '_' == k[0]:
        print(k, v.shape)

  # === 1. training
  train_kw = dict(on_valid_end=callback, valid_interval=30,
                  track_gradients=False)

  def train_ds():
    return ds.create_dataset('train', batch_size=BS)

  ## two-stage training
  if args.finetune:
    initial_weights = [model.decoder.get_weights(),
                       model.observation.get_weights()]
    model.fit(train_ds(), max_iter=MAX_ITER // 2, optimizer=optim1, **train_kw)
    model.decoder.set_weights(initial_weights[0])
    model.observation.set_weights(initial_weights[1])
    model.encoder.trainable = False
    model.latents.trainable = False
    print('Fine-tuning .....')
    model.fit(train_ds(), max_iter=MAX_ITER // 2 + MAX_ITER // 4,
              optimizer=optim2, **train_kw)
  ## full training
  else:
    model.fit(train_ds(), max_iter=MAX_ITER, optimizer=optim1, **train_kw)


def evaluate(args: Arguments):
  np.random.seed(1)
  tf.random.set_seed(1)
  path, model_path = get_path(args)
  if not os.path.exists(model_path + '.index'):
    return None
  ds = MNIST()
  model = BetaGammaVAE(
    **get_dense_networks(args),
    gamma=float(args.gamma), beta=float(args.beta),
    name=f'Z{args.zdim}B{args.beta}G{args.gamma}'.replace('.', ''))
  model.build(ds.full_shape)
  model.load_weights(model_path, raise_notfound=True, verbose=True)
  #
  test = ds.create_dataset('test', batch_size=32)
  for x in test.take(1):
    px, qz = model(x, training=False)
  x = prepare_images(px.mean().numpy(), True)[0]

  llk = tf.reduce_mean(
    tf.concat(
      [model(x, training=False)[0].log_prob(x) for x in test.take(200)], 0)
  ).numpy()
  return dict(beta=args.beta, gamma=args.gamma, zdim=args.zdim,
              finetune=args.finetune, step=model.step.numpy(),
              llk=llk, image=x)


if __name__ == '__main__':
  config = ArgumentParser()
  config.add_argument('mode', type=int)
  config.add_argument('--overwrite', action='store_true')
  config.add_argument('-ncpu', type=int, default=1)
  config = config.parse_args()
  jobs = [Arguments(beta=b, gamma=1, zdim=ZDIM, finetune=True,
                    overwrite=config.overwrite) for b in BETA] + \
         [Arguments(beta=b, gamma=1, zdim=ZDIM, finetune=False,
                    overwrite=config.overwrite) for b in BETA]
  mode = config.mode
  # === 1. train
  if mode == 0:
    for r in MPI(jobs=jobs, func=train, ncpu=config.ncpu):
      pass
  # === 2. eval
  elif mode == 1:
    cache_path = get_cache_path()
    if os.path.exists(cache_path) and config.overwrite:
      os.remove(cache_path)
    if not os.path.exists(cache_path):
      df = []
      for r in MPI(jobs=jobs, func=evaluate, ncpu=config.ncpu):
        if r is not None:
          df.append(r)
      df = sorted(df, key=lambda x: x['beta'])
      df = pd.DataFrame(df)
      with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    else:
      with open(cache_path, 'rb') as f:
        df = pickle.load(f)
    print(df)
    #
    plt.figure(figsize=(6, 5), dpi=150)
    sns.scatterplot(x='beta', y='llk', hue='finetune', data=df, alpha=0.5, s=80)
    plt.gca().set_xscale('log')
    plt.xticks(BETA, [f'{b:g}' for b in BETA])
    #
    n_images = len(df)
    n_col = 10
    n_row = int(np.ceil(n_images / 10))
    plt.figure(figsize=(1.5 * n_col, 1.5 * n_row), dpi=150)
    for i, (beta, gamma, zdim, finetune, step, llk, image) in enumerate(
        df.values):
      plt.subplot(n_row, n_col, i + 1)
      plt.imshow(image, cmap='Greys_r')
      plt.axis('off')
      plt.title(f'b={beta} g={gamma} z={zdim} t={"T" if finetune else "F"}',
                fontsize=8)
    plt.tight_layout()
    vs.plot_save(os.path.join(PATH, 'figures.pdf'), verbose=True)
  # === 3. no support
  else:
    raise NotImplementedError
