from __future__ import absolute_import, division, print_function

import glob
import os
import pickle
import shutil
import urllib
from functools import partial
from typing import Dict, Tuple

import numpy as np
import scipy as sp
import tensorflow as tf
from odin.bay import concat_distributions
from odin.bay.layers import DenseDistribution
from odin.bay.vi import (AmortizedLDA, BetaVAE, FactorVAE,
                         LatentDirichletDecoder, MutualInfoVAE, NetworkConfig,
                         RandomVariable, TwoStageLDA, VariationalAutoencoder)
from odin.exp import Trainer, get_current_trainer
from odin.exp.experimenter import get_output_dir, run_hydra, save_to_yaml
from odin.fuel import (Cortex, LeukemiaATAC, Newsgroup5, Newsgroup20,
                       Newsgroup20_clean)
from odin.utils import ArgController, clean_folder
from sklearn.decomposition import LatentDirichletAllocation
from tensorflow.python import keras
from tensorflow.python.ops import summary_ops_v2
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import DistributionLambda
from tqdm import tqdm

tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.random.set_seed(1)
np.random.seed(1)

# ===========================================================================
# Configs
# ===========================================================================
CONFIG = \
r"""
ds: news5
n_topics: 20
n_iter: 180000
beta: 1
warmup: 120000
posterior: dirichlet
distribution: onehot
override: False
model: lda
"""
# support dataset: news5, news20, news20clean, cortex, leukemia
# support model: em, lda, lda2
learning_rate = 3e-4
batch_size = 128
valid_freq = 2500

DATA_DIR = "/tmp/lda_vae_data"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

OUTPUT_DIR = "/tmp/lda"
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)


# ===========================================================================
# Helpers
# ===========================================================================
def get_topics_text(topics_words, vocabulary, show_word_prob=False):
  topics = topics_words
  alpha = np.sum(topics, axis=1)
  alpha = alpha / np.sum(alpha)
  topics = topics / np.sum(topics, axis=1, keepdims=True)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  text = []
  for idx, topic_idx in enumerate(np.argsort(-alpha, kind="mergesort")[:20]):
    words = topics[topic_idx]
    desc = " ".join(f"{vocabulary[i]}_{words[i]:.2f}"
                    if show_word_prob else f"{vocabulary[i]}"
                    for i in np.argsort(-words)[:10])
    text.append(
        f"[#{idx}]index:{topic_idx:3d} alpha={alpha[topic_idx]:.2f} {desc}")
  return np.asarray(text)


# ===========================================================================
# Callback function
# ===========================================================================
def callback(
    vae: AmortizedLDA,
    test: tf.data.Dataset,
    vocabulary: Dict[int, str],
):
  print(f"[{type(vae).__name__}]{vae.lda.posterior}-{vae.lda.distribution}")
  # end of training
  if not get_current_trainer().is_training:
    vae.save_weights(overwrite=True)
  return dict(topics=vae.get_topics_string(vocabulary, n_topics=20),
              perplexity=vae.perplexity(test, verbose=False))


def callback1(
    vae: VariationalAutoencoder,
    test: tf.data.Dataset,
    vocabulary: Dict[int, str],
):
  px = []
  qz = []
  docs_words = []
  for inputs in test:
    docs_words.append(inputs)
    dists = vae(inputs, training=False)
    px.append(dists[0])
    qz.append(dists[1])
  px = concat_distributions(px, axis=0, name="Docs")
  qz = concat_distributions(qz, axis=0, name="Topics")
  stddev = tf.reduce_mean(qz.stddev(), axis=0)
  docs_words = tf.concat(docs_words, axis=0)
  if isinstance(qz, tfd.MultivariateNormalDiag):
    docs_topics = tf.nn.softmax(qz.mean(), axis=-1)
  elif isinstance(qz, tfd.Dirichlet):
    docs_topics = qz.mean()
  topics_words = tf.matmul(tf.transpose(docs_topics), docs_words)
  topics = get_topics_text(topics_words, vocabulary)
  return dict(stddev=stddev, topics=topics)


# ===========================================================================
# Create AmortizedLDA
# ===========================================================================
@run_hydra(output_dir=OUTPUT_DIR, exclude_keys=['override'])
def main(cfg):
  save_to_yaml(cfg)
  if cfg.ds == 'news5':
    ds = Newsgroup5()
  elif cfg.ds == 'news20':
    ds = Newsgroup20()
  elif cfg.ds == 'news20clean':
    ds = Newsgroup20_clean()
  elif cfg.ds == 'cortex':
    ds = Cortex()
  elif cfg.ds == 'lkm':
    ds = LeukemiaATAC()
  else:
    raise NotImplementedError(f"No support for dataset: {cfg.ds}")
  train = ds.create_dataset(batch_size=batch_size,
                            partition='train',
                            drop_remainder=True)
  valid = ds.create_dataset(batch_size=batch_size, partition='valid')
  test = ds.create_dataset(batch_size=batch_size, partition='test')
  n_words = ds.vocabulary_size
  vocabulary = ds.vocabulary
  ######## prepare the path
  output_dir = get_output_dir()
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_path = os.path.join(output_dir, 'model')
  if cfg.override:
    clean_folder(output_dir, verbose=True)
  ######### preparing all layers
  lda = LatentDirichletDecoder(
      posterior=cfg.posterior,
      distribution=cfg.distribution,
      n_words=n_words,
      n_topics=cfg.n_topics,
      warmup=cfg.warmup,
  )
  fit_kw = dict(train=train,
                valid=valid,
                max_iter=cfg.n_iter,
                optimizer='adam',
                learning_rate=learning_rate,
                batch_size=batch_size,
                valid_freq=valid_freq,
                compile_graph=True,
                logdir=output_dir,
                skip_fitted=True)
  output_dist = RandomVariable(
      n_words,
      cfg.distribution,
      projection=True,
      preactivation='softmax' if cfg.distribution == 'onehot' else 'linear',
      kwargs=dict(probs_input=True) if cfg.distribution == 'onehot' else {},
      name="Words")
  latent_dist = RandomVariable(cfg.n_topics,
                               'diag',
                               projection=True,
                               name="Latents")
  ######## AmortizedLDA
  if cfg.model == 'lda':
    vae = AmortizedLDA(lda=lda,
                       encoder=NetworkConfig([300, 300, 300], name='Encoder'),
                       decoder='identity',
                       latents='identity',
                       path=model_path)
    vae.fit(callback=partial(callback,
                             vae=vae,
                             test=test,
                             vocabulary=vocabulary),
            **fit_kw)
  ######## VDA - Variational Dirichlet Autoencoder
  elif cfg.model == 'vda':
    vae = BetaVAE(
        beta=cfg.beta,
        encoder=NetworkConfig([300, 150], name='Encoder'),
        decoder=NetworkConfig([150, 300], name='Decoder'),
        latents=RandomVariable(cfg.n_topics,
                               'dirichlet',
                               projection=True,
                               prior=None,
                               name="Topics"),
        outputs=output_dist,
        # important, MCMC KL for Dirichlet is very unstable
        analytic=True,
        path=model_path,
        name="VDA")
    vae.fit(callback=partial(callback1,
                             vae=vae,
                             test=test,
                             vocabulary=vocabulary),
            **dict(fit_kw,
                   valid_freq=1000,
                   optimizer=tf.optimizers.Adam(learning_rate=1e-4)))
  ######## VAE
  elif cfg.model == 'vae':
    vae = BetaVAE(beta=cfg.beta,
                  encoder=NetworkConfig([300, 300], name='Encoder'),
                  decoder=NetworkConfig([300], name='Decoder'),
                  latents=latent_dist,
                  outputs=output_dist,
                  path=model_path,
                  name="VAE")
    callback1(vae, test, vocabulary)
    vae.fit(callback=partial(callback1,
                             vae=vae,
                             test=test,
                             vocabulary=vocabulary),
            **dict(fit_kw,
                   valid_freq=1000,
                   optimizer=tf.optimizers.Adam(learning_rate=1e-4)))
  ######## FactorVAE
  elif cfg.model == 'fvae':
    vae = FactorVAE(gamma=6.0,
                    beta=cfg.beta,
                    encoder=NetworkConfig([300, 150], name='Encoder'),
                    decoder=NetworkConfig([150, 300], name='Decoder'),
                    latents=latent_dist,
                    outputs=output_dist,
                    path=model_path)
    vae.fit(callback=partial(callback1,
                             vae=vae,
                             test=test,
                             vocabulary=vocabulary),
            **dict(fit_kw,
                   valid_freq=1000,
                   optimizer=[
                       tf.optimizers.Adam(learning_rate=1e-4,
                                          beta_1=0.9,
                                          beta_2=0.999),
                       tf.optimizers.Adam(learning_rate=1e-4,
                                          beta_1=0.5,
                                          beta_2=0.9)
                   ]))
  ######## TwoStageLDA
  elif cfg.model == 'lda2':
    vae0_iter = 10000
    vae0 = BetaVAE(beta=1.0,
                   encoder=NetworkConfig(units=[300], name='Encoder'),
                   decoder=NetworkConfig(units=[300, 300], name='Decoder'),
                   outputs=DenseDistribution(
                       (n_words,),
                       posterior='onehot',
                       posterior_kwargs=dict(probs_input=True),
                       activation='softmax',
                       name="Words"),
                   latents=RandomVariable(cfg.n_topics,
                                          'diag',
                                          projection=True,
                                          name="Latents"),
                   input_shape=(n_words,),
                   path=model_path + '_vae0')
    vae0.fit(callback=lambda: None
             if get_current_trainer().is_training else vae0.save_weights(),
             **dict(fit_kw,
                    logdir=output_dir + "_vae0",
                    max_iter=vae0_iter,
                    learning_rate=learning_rate,
                    track_gradients=False))
    vae = TwoStageLDA(lda=lda,
                      encoder=vae0.encoder,
                      decoder=vae0.decoder,
                      latents=vae0.latent_layers,
                      warmup=cfg.warmup - vae0_iter,
                      path=model_path)
    vae.fit(callback=partial(callback,
                             vae=vae,
                             test=test,
                             vocabulary=vocabulary),
            **dict(fit_kw,
                   max_iter=cfg.n_iter - vae0_iter,
                   track_gradients=False))
  ######## EM-LDA
  elif cfg.model == 'em':
    if os.path.exists(model_path):
      with open(model_path, 'rb') as f:
        lda = pickle.load(f)
    else:
      writer = tf.summary.create_file_writer(output_dir)
      lda = LatentDirichletAllocation(n_components=cfg.n_topics,
                                      doc_topic_prior=0.7,
                                      learning_method='online',
                                      verbose=True,
                                      n_jobs=4,
                                      random_state=1)
      with writer.as_default():
        prog = tqdm(train.repeat(-1), desc="Fitting LDA")
        for n_iter, x in enumerate(prog):
          lda.partial_fit(x)
          if n_iter % 500 == 0:
            text = get_topics_text(lda.components_, vocabulary)
            perp = lda.perplexity(test)
            tf.summary.text("topics", text, n_iter)
            tf.summary.scalar("perplexity", perp, n_iter)
            prog.write(f"[#{n_iter}]Perplexity: {perp:.2f}")
            prog.write("\n".join(text))
          if n_iter >= 20000:
            break
      with open(model_path, 'wb') as f:
        pickle.dump(lda, f)
    # final evaluation
    text = get_topics_text(lda, vocabulary)
    final_score = lda.perplexity(data['test'])
    tf.summary.scalar("perplexity", final_score, step=n_iter + 1)
    print(f"Perplexity:", final_score)
    print("\n".join(text))


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
