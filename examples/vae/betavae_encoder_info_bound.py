import os.path
import shutil

import numpy as np
import tensorflow as tf

from odin.fuel import MNIST
from odin.networks import get_networks, get_optimizer_info
from odin.bay.vi import BetaGammaVAE
from argparse import ArgumentParser
from dataclasses import dataclass

# ===========================================================================
# Constants
# ===========================================================================
PATH = os.path.expanduser('~/exp/beta_encoder')
BS = 64
MAX_ITER = 200000


@dataclass
class Arguments:
  zdim: int = 32
  beta: int = 1
  gamma: int = 1
  finetune: bool = False
  overwrite: bool = False
  eval: bool = False

  def parse(self):
    args = ArgumentParser()
    for k, v in self.__dict__.items():
      if isinstance(v, bool):
        args.add_argument(f'--{k}', action=f'store_{"false" if v else "true"}')
      else:
        args.add_argument(f'-{k}', type=type(v), default=v)
    args = args.parse_args()
    for k, v in args.__dict__.items():
      setattr(self, k, v)
    return self


# ===========================================================================
# Main
# ===========================================================================
def main(args: Arguments):
  ds = MNIST()
  model = BetaGammaVAE(**get_networks('mnist',
                                      is_semi_supervised=False,
                                      is_hierarchical=False,
                                      zdim=args.zdim),
                       gamma=float(args.gamma), beta=float(args.beta),
                       name=f'Z{args.zdim}B{args.beta}G{args.gamma}')
  model.build(ds.full_shape)
  print(model)
  lr1 = get_optimizer_info('mnist', batch_size=BS)['learning_rate']
  lr2 = get_optimizer_info('mnist', batch_size=BS)['learning_rate']
  # === 0. prepare path
  path = os.path.join(PATH, f'z{args.zdim}_b{args.beta}_g{args.gamma}_'
                            f'{"finetune" if args.finetune else "none"}')
  if os.path.exists(path) and args.overwrite:
    shutil.rmtree(path)
  if not os.path.exists(path):
    os.makedirs(path)
  model_path = os.path.join(path, 'model')

  # === 1.1 helper
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
  if not args.eval:
    train_kw = dict(on_valid_end=callback, valid_interval=30,
                    track_gradients=False)
    if args.finetune:
      initial_weights = [model.decoder.get_weights(),
                         model.observation.get_weights()]
      model.fit(ds.create_dataset('train', batch_size=BS),
                max_iter=MAX_ITER // 2,
                learning_rate=lr1,
                **train_kw)
      model.decoder.set_weights(initial_weights[0])
      model.observation.set_weights(initial_weights[1])
      model.encoder.trainable = False
      model.latents.trainable = False
      print('Fine-tuning .....')
      model.fit(ds.create_dataset('train', batch_size=BS),
                max_iter=MAX_ITER // 2 + MAX_ITER // 4,
                learning_rate=lr2,
                **train_kw)
    else:
      model.fit(ds.create_dataset('train', batch_size=BS),
                max_iter=MAX_ITER,
                learning_rate=lr1,
                **train_kw)
  # === 2. eval
  else:
    model.load_weights(model_path, raise_notfound=True, verbose=True)


if __name__ == '__main__':
  main(Arguments().parse())
