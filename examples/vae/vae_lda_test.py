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
from sklearn.decomposition import LatentDirichletAllocation
from tensorflow.python import keras
from tensorflow.python.ops import summary_ops_v2
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import DistributionLambda
from tqdm import tqdm

from odin.bay.vi import AmortizedLDA, LatentDirichletDecoder, NetworkConfig
from odin.exp import Trainer
from odin.exp.experimenter import get_output_dir, run_hydra
from odin.utils import ArgController, clean_folder

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
n_topics: 50
n_iter: 180000
warmup: 120000
posterior: dirichlet
distribution: categorical
override: False
em: False
"""
learning_rate = 3e-4
batch_size = 128
valid_freq = 5000

DATA_DIR = "/tmp/lda_vae_data"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

OUTPUT_DIR = "/tmp/lda"
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)


# ===========================================================================
# Download data
# ===========================================================================
def download_newsgroup20(
    data_dir) -> Tuple[Dict[str, np.ndarray], Dict[int, str]]:
  root_path = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
  file_template = "{split}.txt.npy"
  filename = [
      ("vocab.pkl", os.path.join(root_path, "vocab.pkl")),
      ("train", os.path.join(root_path, file_template.format(split="train"))),
      ("test", os.path.join(root_path, file_template.format(split="test"))),
  ]
  data = {}
  for name, url in filename:
    filepath = os.path.join(data_dir, name)
    # download
    if not os.path.exists(filepath):
      print(f"Download file {filepath}")
      urllib.request.urlretrieve(url, filepath)
    # load
    if '.pkl' in name:
      with open(filepath, 'rb') as f:
        words_to_idx = pickle.load(f)
      n_words = len(words_to_idx)
      data[name.split(".")[0]] = words_to_idx
    else:
      x = np.load(filepath, allow_pickle=True, encoding="latin1")[:-1]
      n_documents = x.shape[0]
      indices = np.array([(row_idx, column_idx)
                          for row_idx, row in enumerate(x)
                          for column_idx in row])
      sparse_matrix = sp.sparse.coo_matrix(
          (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
          shape=(n_documents, n_words),
          dtype=np.float32)
      sparse_matrix = sparse_matrix.tocsr()
      data[name] = sparse_matrix
  vocabulary = {idx: word for word, idx in words_to_idx.items()}
  return data, vocabulary


def create_tfds(x):
  x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                      values=x.data,
                      dense_shape=x.shape)
  x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).map(
      lambda y: tf.cast(tf.sparse.to_dense(y), tf.float32))
  shape = tf.data.experimental.get_structure(x).shape[1:]
  return x, shape


data, vocabulary = download_newsgroup20(DATA_DIR)
n_words = len(vocabulary)
train_ds, input_shape = create_tfds(data['train'])
test_ds, _ = create_tfds(data['test'])


# ===========================================================================
# Helpers
# ===========================================================================
def get_topics_text(lda: LatentDirichletAllocation, show_word_prob=False):
  topics = lda.components_
  alpha = np.sum(topics, axis=1)
  alpha = alpha / np.sum(alpha)
  topics = topics / np.sum(topics, axis=1, keepdims=True)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  text = []
  for topic_idx in np.argsort(-alpha, kind="mergesort")[:20]:
    words = topics[topic_idx]
    desc = " ".join(f"{vocabulary[i]}_{words[i]:.2f}"
                    if show_word_prob else f"{vocabulary[i]}"
                    for i in np.argsort(-words)[:10])
    text.append(f"Topic#{topic_idx:3d} alpha={alpha[topic_idx]:.2f} {desc}")
  return text


def callback(vae: AmortizedLDA):
  print(f"*** {vae.lda.posterior}-{vae.lda.distribution} ***")
  text = vae.get_topics_string(vocabulary, n_topics=20)
  tf.summary.text("topics", text)
  perplexity = vae.perplexity(data['test'])
  tf.summary.scalar("perplexity", perplexity)
  print("\n".join(text))
  print(f"[#{int(vae.lda.step.numpy())}]Perplexity:", float(perplexity))


# ===========================================================================
# Create AmortizedLDA
# ===========================================================================
@run_hydra(output_dir=OUTPUT_DIR, exclude_keys=['override', 'em'])
def main(cfg):
  output_dir = get_output_dir()
  if cfg.override:
    clean_folder(output_dir, verbose=True)
  if not cfg.em:
    # VAE with Dirichlet latent posterior
    lda = LatentDirichletDecoder(
        posterior=cfg.posterior,
        distribution=cfg.distribution,
        n_words=n_words,
        n_topics=cfg.n_topics,
        warmup=cfg.warmup,
    )
    vae = AmortizedLDA(lda=lda,
                       encoder=NetworkConfig(units=[300, 300, 300],
                                             name='Encoder'),
                       decoder='identity',
                       latents='identity')
    vae.fit(train=train_ds,
            valid=test_ds,
            max_iter=cfg.n_iter,
            optimizer='adam',
            learning_rate=learning_rate,
            batch_size=batch_size,
            valid_freq=valid_freq,
            compile_graph=True,
            logdir=output_dir,
            checkpoint=output_dir,
            callback=partial(callback, vae=vae),
            skip_fitted=True)
    with vae.summary_writer.as_default():
      callback(vae)
  ######## EM-LDA
  else:
    if os.path.exists(LOGDIR + ".pkl"):
      with open(LOGDIR + ".pkl", 'rb') as f:
        lda = pickle.load(f)
    else:
      writer = tf.summary.create_file_writer(LOGDIR)
      lda = LatentDirichletAllocation(n_components=n_topics,
                                      doc_topic_prior=0.7,
                                      learning_method='online',
                                      verbose=True,
                                      n_jobs=4,
                                      random_state=1)
      with writer.as_default():
        prog = tqdm(train_ds.repeat(-1), desc="Fitting LDA")
        for n_iter, x in enumerate(prog):
          lda.partial_fit(x)
          if n_iter % 500 == 0:
            text = get_topics_text(lda)
            perp = lda.perplexity(data['test'])
            tf.summary.text("topics", text, n_iter)
            tf.summary.scalar("perplexity", perp, n_iter)
            prog.write(f"[#{n_iter}]Perplexity: {perp:.2f}")
            prog.write("\n".join(text))
          if n_iter >= 20000:
            break
      with open(LOGDIR + ".pkl", 'wb') as f:
        pickle.dump(lda, f)
    # final evaluation
    text = get_topics_text(lda)
    print(f"Perplexity:", lda.perplexity(data['test']))
    print("\n".join(text))


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
  main(CONFIG)
