from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import search
from odin import visual as vs
from odin.bay import RVmeta
from odin.bay.vi import Criticizer
from odin.bay.vi.autoencoder import (Factor2VAE, FactorDiscriminator,
                                     SemiFactor2VAE, SemifactorVAE, factorVAE)
from odin.training import Experimenter, pretty_config
from odin.fuel import get_dataset
from odin.utils import md5_folder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python import keras
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(1)
np.random.seed(1)
sns.set()

# ===========================================================================
# Helpers
# ===========================================================================
CONFIG = \
r"""
ds: mnist
vae: factor
pretrain: 0
finetune: 10000
alpha: 10.
beta: 1.
gamma: 6.
lamda: 1.
semi: 0.1
maxtc: False
strategy: logsumexp
verbose: False
gpu: False
"""


def cal_mllk(vae, data, gpu=False):
  device = 'GPU' if gpu else 'CPU'
  with tf.device(f"/{device}:0"):
    prog = tqdm(data.repeat(1), desc="Calculating MarginalLLK")
    mllk = np.mean(
        np.concatenate(
            [vae.marginal_log_prob(x, sample_shape=50) for x in prog], axis=0))
    prog.clear()
    prog.close()
    return mllk


# ===========================================================================
# Experiments
# ===========================================================================
class Factor(Experimenter):

  def __init__(self):
    super().__init__(save_path='~/exp/factorexp',
                     config_path=CONFIG,
                     exclude_keys=['verbose', 'gpu'],
                     hash_length=5)

  def on_load_data(self, cfg):
    ds = get_dataset(cfg.ds)
    ds.sample_images(save_path=os.path.join(self.save_path, 'samples.png'))
    kw = dict(batch_size=128, drop_remainder=True)
    train = ds.create_dataset(partition='train',
                              inc_labels=float(cfg.semi),
                              **kw)
    train_u = ds.create_dataset(partition='train', inc_labels=False, **kw)
    valid = ds.create_dataset(partition='valid', inc_labels=1.0, **kw)
    valid_u = ds.create_dataset(partition='valid', inc_labels=False, **kw)
    # reduce batch_size here, otherwise, mllk take ~ 7GB VRAM
    kw['batch_size'] = 8
    test = ds.create_dataset(partition='test', inc_labels=1.0, **kw)
    test_u = ds.create_dataset(partition='test', inc_labels=False, **kw)
    self.ds = ds
    self.train, self.train_u = train, train_u
    self.valid, self.valid_u = valid, valid_u
    self.test, self.test_u = test, test_u
    if cfg.verbose:
      print("Dataset:", ds)
      print(" train:", train)
      print(" train_u:", train_u)

  def on_create_model(self, cfg, model_dir, md5):
    kw = dict(alpha=cfg.alpha,
              beta=cfg.beta,
              gamma=cfg.gamma,
              lamda=cfg.lamda,
              maximize_tc=bool(cfg.maxtc),
              path=os.path.join(model_dir, 'weight'))
    if cfg.vae == 'factor':
      del kw['alpha']
      model = factorVAE(
          encoder=cfg.ds,
          outputs=RVmeta(self.ds.shape, 'bern', name="Image"),
          latents=RVmeta(20, 'mvndiag', projection=True, name="Latents"),
          **kw,
      )
    elif cfg.vae == 'factor2':
      del kw['alpha']
      model = Factor2VAE(
          encoder=cfg.ds,
          outputs=RVmeta(self.ds.shape, 'bern', name="Image"),
          latents=RVmeta(10, 'mvndiag', projection=True, name='Latents'),
          factors=RVmeta(10, 'mvndiag', projection=True, name='Factors'),
          **kw,
      )
    elif cfg.vae == 'semi':
      model = SemifactorVAE(
          encoder=cfg.ds,
          outputs=RVmeta(self.ds.shape, 'bern', name="Image"),
          latents=RVmeta(20, 'mvndiag', projection=True, name="Latents"),
          n_labels=self.ds.n_labels,
          ss_strategy=cfg.strategy,
          **kw,
      )
    elif cfg.vae == 'semi2':
      model = SemiFactor2VAE(
          encoder=cfg.ds,
          outputs=RV(self.ds.shape, 'bern', name="Image"),
          latents=RV(10, 'mvndiag', projection=True, name='Latents'),
          factors=RV(10, 'mvndiag', projection=True, name='Factors'),
          n_labels=self.ds.n_labels,
          ss_strategy=cfg.strategy,
          **kw,
      )
    else:
      raise NotImplementedError("No support for model: %s" % cfg.vae)
    # store the model
    self.model = model
    if cfg.verbose:
      print(model)
      print(model_dir)
      if md5 is not None:
        print("MD5 saved: ", md5)
        print("MD5 loaded:", md5_folder(model_dir))

  def on_train(self, cfg, output_dir, model_dir):
    if cfg.verbose:
      print("Configurations:")
      for k, v in cfg.items():
        print('%-10s' % k, ':', v)
    # start training
    kw = {}
    if cfg.pretrain > 0:
      self.model.pretrain().fit(self.train_u,
                                valid=self.valid_u,
                                max_iter=cfg.pretrain)
      kw['optimizer'] = None
    if cfg.finetune > 0:
      train, valid = (self.train, self.valid) if self.model.is_semi_supervised \
        else (self.train_u, self.valid_u)
      self.model.finetune().fit(train, valid=valid, max_iter=cfg.finetune, **kw)
    self.model.plot_learning_curves(os.path.join(output_dir,
                                                 'learning_curves.png'),
                                    title=self.model.__class__.__name__)
    self.model.save_weights(os.path.join(model_dir, 'weight'))

  def on_eval(self, cfg, output_dir):
    # marginal log-likelihood
    mllk = cal_mllk(self.model, self.test_u, gpu=cfg.gpu)
    # Criticizer
    crt = Criticizer(vae=self.model)
    crt.sample_batch(inputs=self.test, n_samples=[10000, 5000], verbose=True)
    # clustering scores
    scores = crt.cal_clustering_scores()
    # downstream scores
    beta = np.mean(crt.cal_betavae_score(n_samples=10000, verbose=True))
    factor = np.mean(crt.cal_factorvae_score(n_samples=10000, verbose=True))
    tc = np.mean(crt.cal_total_correlation())
    d, c, i = crt.cal_dci_scores()
    sap = np.mean(crt.cal_separated_attr_predictability())
    mig = np.mean(crt.cal_mutual_info_gap())
    # save to database
    scores = dict(
        beta=beta,
        factor=factor,
        sap=sap,
        tc=tc,
        d=d,
        c=c,
        i=i,
        mig=mig,
        mllk=mllk,
        asw=scores['ASW'],
        ari=scores['ARI'],
        nmi=scores['NMI'],
        uca=scores['UCA'],
    )
    self.save_scores(table="score", override=True, **scores)

  def on_plot(self, cfg, output_dir):
    average = lambda train_test: (train_test[0] + train_test[1]) / 2.
    path = os.path.join(output_dir, 'matrix.png')
    decode = lambda mat: search.diagonal_beam_search(mat.T)
    stats = lambda mat: " mean:%.2f mean(max):%.2f" % (np.mean(
        mat), np.mean(np.max(mat, axis=0)))

    crt = Criticizer(vae=self.model)
    crt.sample_batch(inputs=self.test,
                     n_samples=[10000, 5000],
                     factor_names=self.ds.labels,
                     verbose=True)
    n_codes = crt.n_codes
    n_factors = crt.n_factors

    mi = average(crt.create_mutualinfo_matrix())
    spearman = average(crt.create_correlation_matrix(method='spearman'))
    pearson = average(crt.create_correlation_matrix(method='pearson'))

    height = 16
    fig = plt.figure(figsize=(height * n_factors / n_codes * 3 + 2, height + 2))
    kw = dict(cbar=True, annotation=True, fontsize=8)

    ids = decode(mi)
    vs.plot_heatmap(mi[ids],
                    xticklabels=crt.factor_names,
                    yticklabels=crt.code_names[ids],
                    cmap="Blues",
                    ax=(1, 3, 1),
                    title="[MutualInformation]" + stats(mi),
                    **kw)

    ids = decode(spearman)
    vs.plot_heatmap(spearman[ids],
                    xticklabels=crt.factor_names,
                    yticklabels=crt.code_names[ids],
                    cmap="bwr",
                    ax=(1, 3, 2),
                    title="[Spearman]" + stats(spearman),
                    **kw)

    ids = decode(pearson)
    vs.plot_heatmap(pearson[ids],
                    xticklabels=crt.factor_names,
                    yticklabels=crt.code_names[ids],
                    cmap="bwr",
                    ax=(1, 3, 3),
                    title="[Pearson]" + stats(pearson),
                    **kw)
    fig.tight_layout()
    fig.savefig(path, dpi=120)

  def on_compare(self, models, save_path):
    scores = [
        'mllk', 'mig', 'beta', 'factor', 'uca', 'nmi', 'sap', 'd', 'c', 'i'
    ]
    scores = {
        name: self.get_scores('score', [i.hash for i in models
                                       ], name) for name in scores
    }
    ncol = 5
    nrow = int(np.ceil(len(scores) / ncol))

    df = models.to_dataframe()
    for dsname, group in df.groupby('ds'):
      name = group['vae'] + '-' + group['strategy'] + '-' + group[
          'semi'].astype(str)
      colors = sns.color_palette(n_colors=group.shape[0])
      X = np.arange(group.shape[0])

      fig = plt.figure(figsize=(3 * ncol, 3 * nrow))
      for idx, (key, val) in enumerate(scores.items()):
        y = np.array([val[hash_code] for hash_code in group['hash']])
        vmin, vmax = np.min(y), np.max(y)
        y = y - np.min(y)
        ax = plt.subplot(nrow, ncol, idx + 1)
        points = [
            ax.scatter(x_, y_, s=32, color=c_, alpha=0.8)
            for x_, y_, c_ in zip(X, y, colors)
        ]
        ax.set_title(key)
        plt.yticks(np.linspace(0., np.max(y), 5),
                   ["%.2f" % i for i in np.linspace(vmin, vmax, 5)])
        ax.tick_params(bottom=False, labelbottom=False, labelsize=8)
        # show legend:
        if idx == 0:
          ax.legend(points, [i for i in name],
                    fontsize=6,
                    fancybox=False,
                    framealpha=0.)
      fig.suptitle(dsname)
      fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    vs.plot_save(os.path.join(save_path, 'compare.pdf'), dpi=100)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  exp = Factor()
  exp.train().run()
