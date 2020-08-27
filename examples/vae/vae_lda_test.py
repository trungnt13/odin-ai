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

from odin.bay.vi import AmortizedLDA, NetworkConfig
from odin.exp import Trainer
from odin.utils import ArgController

tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.random.set_seed(1)
np.random.seed(1)

args = ArgController(\
  ).add("-warmup", "number of iteration for warmup", 120000 \
  ).add("-niter", "maximum number of iteration", 180000 \
  ).add("-posterior", "latents distribution", 'dirichlet' \
  ).add("-distribution", "words distribution", 'onehot' \
  ).add("--override", "override exist model", False \
  ).add("--em", "Using sklearn and EM algorithm", False \
  ).parse()

# --em --override
# python vae_lda_test.py -posterior gaussian -distribution onehot --override
# python vae_lda_test.py -posterior gaussian -distribution poisson --override
# python vae_lda_test.py -posterior gaussian -distribution negativebinomial --override
# python vae_lda_test.py -posterior gaussian -distribution zinb --override
# python vae_lda_test.py -posterior dirichlet -distribution onehot --override
# python vae_lda_test.py -posterior dirichlet -distribution poisson --override
# python vae_lda_test.py -posterior dirichlet -distribution negativebinomial --override
# python vae_lda_test.py -posterior dirichlet -distribution zinb --override

# -posterior gaussian -distribution binomial --override
# -posterior dirichlet -distribution binomial --override
# ===========================================================================
# Configs
# ===========================================================================
n_topics = 50
warmup = int(args.warmup)
max_iter = int(args.niter)
assert max_iter > warmup
learning_rate = 3e-4
batch_size = 128
valid_freq = 5000

# ===========================================================================
# Path
# ===========================================================================
CACHE_DIR = "/tmp/lda_vae_data"
if not os.path.exists(CACHE_DIR):
  os.makedirs(CACHE_DIR)
# model path
if args.em:
  LOGDIR = f"/tmp/lda_vae_model/em_dirichlet_multinomial"
else:
  LOGDIR = f"/tmp/lda_vae_model/vae_{args.posterior}_{args.distribution}"
if os.path.exists(LOGDIR):
  if args.override:
    print("Override path:", LOGDIR)
    shutil.rmtree(LOGDIR)
  for f in glob.glob(f"{LOGDIR}*"):
    if os.path.isfile(f):
      print("Override file:", f)
      os.remove(f)
else:
  os.makedirs(LOGDIR)

print("Save path:", LOGDIR)


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


data, vocabulary = download_newsgroup20(CACHE_DIR)
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
  print(f"*** {vae.lda_posterior}-{vae.word_distribution} ***")
  text = vae.get_topics_string(vocabulary, n_topics=20)
  tf.summary.text("topics", text)
  perplexity = vae.perplexity(data['test'])
  tf.summary.scalar("perplexity", perplexity)
  print("\n".join(text))
  print(f"[#{int(vae.step.numpy())}]Perplexity:", float(perplexity))


# ===========================================================================
# Create AmortizedLDA
# ===========================================================================
if not args.em:
  # VAE with Dirichlet latent posterior
  vae = AmortizedLDA(lda_posterior=args.posterior,
                     word_distribution=args.distribution,
                     path=LOGDIR,
                     n_words=n_words,
                     n_topics=n_topics,
                     prior_init=0.7,
                     prior_warmup=warmup,
                     encoder=NetworkConfig(units=[300, 300, 300]))
  print(vae)
  vae.fit(train=train_ds,
          valid=test_ds,
          max_iter=max_iter,
          optimizer='adam',
          learning_rate=learning_rate,
          batch_size=batch_size,
          valid_freq=valid_freq,
          compile_graph=True,
          logdir=LOGDIR,
          checkpoint=LOGDIR,
          callback=partial(callback, vae=vae),
          skip_fitted=True)
  with vae.summary_writer.as_default():
    callback(vae)

# ===========================================================================
# LDA
# ===========================================================================
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
