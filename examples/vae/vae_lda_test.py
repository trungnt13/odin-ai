from __future__ import absolute_import, division, print_function

import os
import pickle
import shutil
import urllib
from functools import partial

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

from odin.bay.vi import LDAVAE, NetworkConfig
from odin.utils import ArgController

tf.debugging.set_log_device_placement(False)
tf.autograph.set_verbosity(0)
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True)
tf.random.set_seed(1)
np.random.seed(1)

args = ArgController().add("--override", "override exist model", False).parse()

# ===========================================================================
# Download dataset
# ===========================================================================
ROOT_PATH = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
FILE_TEMPLATE = "{split}.txt.npy"
CACHE_DIR = "/tmp/newsgroup20"
if not os.path.exists(CACHE_DIR):
  os.makedirs(CACHE_DIR)
# model path
MODEL_PATH = os.path.join(CACHE_DIR, "vae_lda")
if args.override and os.path.exists(MODEL_PATH):
  print("Override model at:", MODEL_PATH)
  shutil.rmtree(MODEL_PATH)
if not os.path.exists(MODEL_PATH):
  os.makedirs(MODEL_PATH)
GVAE_PATH = os.path.join(MODEL_PATH, "gvae")
DVAE_PATH = os.path.join(MODEL_PATH, "dvae")
# LDA path
LDA_PATH = os.path.join(CACHE_DIR, "lda.pkl")
if args.override and os.path.exists(LDA_PATH):
  print("Override model at:", LDA_PATH)
  os.remove(LDA_PATH)

## download data
filename = [
    ("vocab.pkl", os.path.join(ROOT_PATH, "vocab.pkl")),
    ("train", os.path.join(ROOT_PATH, FILE_TEMPLATE.format(split="train"))),
    ("test", os.path.join(ROOT_PATH, FILE_TEMPLATE.format(split="test"))),
]
data = {}
for name, url in filename:
  filepath = os.path.join(CACHE_DIR, name)
  # download
  if not os.path.exists(filepath):
    print(f"Download file {filepath}")
    urllib.request.urlretrieve(url, filepath)
  # load
  if '.pkl' in name:
    with open(filepath, 'rb') as f:
      words_to_idx = pickle.load(f)
    num_words = len(words_to_idx)
    data[name.split(".")[0]] = words_to_idx
  else:
    x = np.load(filepath, allow_pickle=True, encoding="latin1")[:-1]
    num_documents = x.shape[0]
    indices = np.array([(row_idx, column_idx)
                        for row_idx, row in enumerate(x)
                        for column_idx in row])
    sparse_matrix = sp.sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
        shape=(num_documents, num_words),
        dtype=np.float32)
    sparse_matrix = sparse_matrix.tocsr()
    data[name] = sparse_matrix

vocabulary = {idx: word for word, idx in words_to_idx.items()}


# ===========================================================================
# Create LDA VAE
# ===========================================================================
def callback(vae: LDAVAE, top_topics=10):
  print(f"*** {vae.lda_posterior} ***")
  vae.print_topics(vocabulary, top_topics=top_topics)
  print(f"[#{int(vae.step.numpy())}]Perplexity:", vae.perplexity(data['test']))


n_topics = 50
warmup = 120000
max_iter = 180000
learning_rate = 3e-4
batch_size = 32

train = tf.SparseTensor(indices=sorted(zip(*data['train'].nonzero())),
                        values=data['train'].data,
                        dense_shape=data['train'].shape)
train = tf.data.Dataset.from_tensor_slices(train).batch(batch_size).map(
    lambda y: tf.cast(tf.sparse.to_dense(y), tf.float32)).repeat(-1)

kwargs = dict(n_words=num_words,
              n_topics=n_topics,
              prior_init=0.7,
              prior_warmup=warmup,
              encoder=NetworkConfig(units=[300, 300, 300]))
train_kw = dict(train=train,
                max_iter=max_iter,
                optimizer='adam',
                learning_rate=learning_rate,
                batch_size=batch_size,
                valid_interval=30,
                compile_graph=True,
                skip_fitted=True)

# VAE with Logistic-Normal latent posterior
gvae = LDAVAE(lda_posterior="gaussian", path=GVAE_PATH, **kwargs)
gvae.fit(callback=partial(callback, vae=gvae), checkpoint=GVAE_PATH, **train_kw)
callback(gvae, top_topics=20)
gvae.plot_learning_curves(os.path.join(CACHE_DIR, "gvae.png"),
                          summary_steps=1000)

# VAE with Dirichlet latent posterior
dvae = LDAVAE(lda_posterior="dirichlet", path=DVAE_PATH, **kwargs)
dvae.fit(callback=partial(callback, vae=dvae), checkpoint=DVAE_PATH, **train_kw)
callback(dvae, top_topics=20)
dvae.plot_learning_curves(os.path.join(CACHE_DIR, "dvae.png"),
                          summary_steps=1000)


# ===========================================================================
# LDA
# ===========================================================================
def print_topics(lda: LatentDirichletAllocation,
                 top_topics=10,
                 top_words=10,
                 show_word_prob=False):
  topics = lda.components_
  alpha = np.sum(topics, axis=1)
  alpha = alpha / np.sum(alpha)
  topics = topics / np.sum(topics, axis=1, keepdims=True)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  for topic_idx in np.argsort(-alpha, kind="mergesort")[:int(top_topics)]:
    words = topics[topic_idx]
    desc = " ".join(f"{vocabulary[i]}_{words[i]:.2f}"
                    if show_word_prob else f"{vocabulary[i]}"
                    for i in np.argsort(-words)[:int(top_words)])
    print(f"Topic#{topic_idx:3d} alpha={alpha[topic_idx]:.2f} {desc}")


if os.path.exists(LDA_PATH):
  with open(LDA_PATH, 'rb') as f:
    lda = pickle.load(f)
else:
  lda = LatentDirichletAllocation(n_components=n_topics,
                                  doc_topic_prior=0.7,
                                  max_iter=100,
                                  batch_size=32,
                                  learning_method='online',
                                  verbose=True,
                                  n_jobs=4,
                                  random_state=1)
  for n_iter, x in enumerate(tqdm(train, desc="Fitting LDA")):
    lda.partial_fit(x)
    if n_iter > 20000:
      break
    if n_iter % 1000 == 0:
      print()
      print_topics(lda, top_topics=10)
      print(f"[#{n_iter}] Perplexity:", lda.perplexity(data['test']))
  with open(LDA_PATH, 'wb') as f:
    pickle.dump(lda, f)
# final evaluation
print_topics(lda, top_topics=20)
print(f"Perplexity:", lda.perplexity(data['test']))
