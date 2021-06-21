import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from odin.bay import DistributionDense, MVNDiagLatents, BetaGammaVAE, \
  DisentanglementGym
from odin.fuel import MNIST
from odin.utils import MPI
from multiprocessing import cpu_count
import seaborn as sns
from odin import  visual as vs

sns.set()

# ===========================================================================
# Const and helper
# ===========================================================================
root_path = '/home/trung/exp/rate_distortion_onehot'
if not os.path.exists(root_path):
  os.makedirs(root_path)


# ===========================================================================
# Create dataset and some test
# ===========================================================================
def test_vae_y(args):
  ds = MNIST()
  train_y = ds.create_dataset('train', label_percent=1.0).map(lambda x, y: y)
  valid_y = ds.create_dataset('valid', label_percent=1.0).map(
    lambda x, y: (y, y))

  gamma, beta = args
  basedir = os.path.join(root_path, 'vaey')
  save_path = os.path.join(basedir, f'{gamma}_{beta}')
  logdir = os.path.join(basedir, f'{gamma}_{beta}_log')
  vae_y = BetaGammaVAE(
    encoder=Sequential([Dense(256, 'relu'),
                        Dense(256, 'relu')],
                       name='Encoder'),
    decoder=Sequential([Dense(256, 'relu'),
                        Dense(256, 'relu')],
                       name='Decoder'),
    latents=MVNDiagLatents(10),
    observation=DistributionDense([10], posterior='onehot', projection=True,
                                  name='Digits'),
    gamma=gamma,
    beta=beta
  )
  vae_y.build((None, 10))
  vae_y.load_weights(save_path)
  vae_y.fit(train_y, max_iter=20000, logdir=logdir, skip_fitted=True)
  vae_y.save_weights(save_path)

  gym = DisentanglementGym(model=vae_y, valid=valid_y)
  with gym.run_model(partition='valid'):
    y_true = np.argmax(gym.y_true, -1)
    y_pred = np.argmax(gym.px_z[0].mode(), -1)
    acc = accuracy_score(y_true, y_pred)
    results = dict(acc=acc,
                   llk=gym.log_likelihood()[0],
                   kl=gym.kl_divergence()[0],
                   au=gym.active_units()[0],
                   gamma=gamma,
                   beta=beta)
    gym.plot_correlation()
    gym.plot_latents_stats()
    gym.plot_latents_tsne()
  gym.save_figures(save_path + '.pdf', verbose=True)
  return results


results_path = os.path.join(root_path, 'results')
jobs = list(itertools.product(np.linspace(0.1, 100, num=30),
                              np.linspace(0.1, 100, num=30)))
if not os.path.exists(results_path):
  data = []
  for results in MPI(jobs, func=test_vae_y, ncpu=cpu_count() - 1):
    data.append(results)
  df = pd.DataFrame(data)
  with open(results_path, 'wb') as f:
    pickle.dump(df, f)
else:
  with open(results_path, 'rb') as f:
    df = pickle.load(f)

df: pd.DataFrame
print(df)

for name in ['acc', 'llk', 'kl', 'au']:
  plt.figure(figsize=(9, 8), dpi=150)
  splot = sns.scatterplot(x='beta', y='gamma', hue=name, size=name,
                          data=df, sizes=(20, 200), alpha=0.95,
                          linewidth=0, palette='coolwarm')
  plt.title(name)
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)

vs.plot_save(verbose=True)