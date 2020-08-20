from __future__ import absolute_import, division, print_function

import glob
import os
import pickle
import shutil
import urllib
from functools import partial

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.decomposition import LatentDirichletAllocation
from tensorflow.python import keras
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.layers import DistributionLambda
from tqdm import tqdm

from odin.bay.vi import LDAVAE, NetworkConfig
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
  ).add("--override", "override exist model", False \
  ).parse()

# ===========================================================================
# Configs
# ===========================================================================
n_topics = 50
warmup = int(args.warmup)
max_iter = int(args.niter)
assert max_iter > warmup
learning_rate = 3e-4
batch_size = 128

# ===========================================================================
# Path
# ===========================================================================
ROOT_PATH = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
FILE_TEMPLATE = "{split}.txt.npy"
CACHE_DIR = "/tmp/lda_vae"
if not os.path.exists(CACHE_DIR):
  os.makedirs(CACHE_DIR)
# model path
GVAE_PATH = os.path.join(CACHE_DIR, "gvae")
DVAE_PATH = os.path.join(CACHE_DIR, "dvae")
# logdir
TFP_LOGDIR = os.path.join(CACHE_DIR, 'tfplog')
if args.override:
  print("Override model and remove:")
  for f in glob.glob(f"{GVAE_PATH}*") + glob.glob(f"{DVAE_PATH}*"):
    print("", f)
    if os.path.isfile(f):
      os.remove(f)
    else:
      shutil.rmtree(f)
  if os.path.exists(TFP_LOGDIR):
    shutil.rmtree(TFP_LOGDIR)
    print("", TFP_LOGDIR)

# LDA path
LDA_PATH = os.path.join(CACHE_DIR, "lda.pkl")
if args.override and os.path.exists(LDA_PATH):
  print("Override model at:", LDA_PATH)
  os.remove(LDA_PATH)

# ===========================================================================
# Download data
# ===========================================================================
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


def create_tfds(x):
  x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                      values=x.data,
                      dense_shape=x.shape)
  x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size).map(
      lambda y: tf.cast(tf.sparse.to_dense(y), tf.float32))
  shape = tf.data.experimental.get_structure(x).shape[1:]
  return x, shape


train_ds, input_shape = create_tfds(data['train'])
test_ds, _ = create_tfds(data['test'])
train_ds.repeat(-1)


# ===========================================================================
# Helpers
# ===========================================================================
def _clip_dirichlet_parameters(x):
  return tf.clip_by_value(x, 1e-3, 1e3)


def _softplus_inverse(x):
  return np.log(np.expm1(x))


def make_encoder(projection=False) -> tf.keras.Sequential:
  encoder_net = tf.keras.Sequential(name="Encoder")
  for num_hidden_units in [300, 300, 300]:
    encoder_net.add(
        tf.keras.layers.Dense(
            num_hidden_units,
            activation="relu",
            kernel_initializer=tf.initializers.glorot_normal()))
  if projection:
    encoder_net.add(
        tf.keras.layers.Dense(
            n_topics,
            activation=tf.nn.softplus,
            kernel_initializer=tf.initializers.glorot_normal()))
  return encoder_net


class LDADecoder(DistributionLambda):

  def __init__(self):
    super().__init__(make_distribution_fn=self.create_distribution,
                     name="Decoder")

  def build(self, input_shape):
    self.topics_words_logits = self.add_weight(
        name="topics_words_logits",
        shape=[n_topics, n_words],
        initializer=tf.initializers.glorot_normal(),
        trainable=True)

  def create_distribution(self, topics, *args, **kwargs):
    topics_words = tf.nn.softmax(self.topics_words_logits, axis=-1)
    word_probs = tf.matmul(topics, topics_words)
    return tfd.OneHotCategorical(probs=word_probs, name="bag_of_words")


def make_prior(logit_concentration):
  concentration = _clip_dirichlet_parameters(
      tf.nn.softplus(logit_concentration))
  return tfd.Dirichlet(concentration=concentration, name="topics_prior")


class LDAModel(keras.Model):

  def __init__(self):
    super().__init__()
    self.step = self.add_weight(name="step",
                                shape=(),
                                initializer=tf.initializers.constant(0),
                                trainable=False)
    self.encoder = make_encoder(projection=True)
    self.decoder = LDADecoder()
    self.latents = DistributionLambda(lambda x: tfd.Dirichlet(
        concentration=_clip_dirichlet_parameters(x), name="topics_posterior"))
    self.logit_concentration = self.add_weight(
        name="logit_concentration",
        initializer=tf.initializers.constant(_softplus_inverse(0.7)),
        shape=[1, n_topics],
        trainable=True)
    self(keras.Input(shape=(n_words,)))

  def call(self, features, *args, **kwargs):
    logit_concentration = tf.cond(
        self.step <= warmup,
        true_fn=lambda: tf.stop_gradient(self.logit_concentration),
        false_fn=lambda: self.logit_concentration)
    topics_prior = make_prior(logit_concentration)
    alpha = topics_prior.concentration

    e = self.encoder(features)
    topics_posterior = self.latents(e)
    topics = topics_posterior.sample()
    random_reconstruction = self.decoder(topics)
    reconstruction = random_reconstruction.log_prob(features)
    kl = tfd.kl_divergence(topics_posterior, topics_prior)

    tf.assert_greater(kl, -1e-3, message="kl")
    elbo = reconstruction - kl
    avg_elbo = tf.reduce_mean(input_tensor=elbo)
    loss = -avg_elbo

    tf.keras.callbacks.TensorBoard
    words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
    log_perplexity = -elbo / words_per_document
    log_perplexity_tensor = tf.reduce_mean(log_perplexity)
    perplexity_tensor = tf.exp(log_perplexity_tensor)

    return dict(loss=loss,
                alpha=alpha,
                llk=reconstruction,
                kl=kl,
                perplexity=perplexity_tensor)


def train_tfp_model(model, optimizer):
  writer = tf.summary.create_file_writer(TFP_LOGDIR)

  @tf.function
  def train_step(x):
    model.step.assign_add(1)
    with tf.GradientTape() as tape:
      outputs = model(x)
    grads = tape.gradient(outputs['loss'], model.trainable_variables)
    optimizer.apply_gradients([
        (g, v) for g, v in zip(grads, model.trainable_variables)
    ])
    return outputs

  for it, x in tqdm(enumerate(train_ds.repeat(-1))):
    outputs = train_step(x)
    if it % 5000 == 0:
      with writer.as_default():
        alpha = outputs.pop('alpha')
        alpha = np.squeeze(alpha, axis=0)
        highest_weight_topics = np.argsort(-alpha, kind="mergesort")
        top_words = np.argsort(
            -tf.nn.softmax(model.decoder.topics_words_logits, axis=-1), axis=1)
        print(f"#Iter: {it} {int(model.step)}")
        # metrics
        for k, v in sorted(outputs.items()):
          v = tf.reduce_mean(v)
          print(f" {k:10s}:{v:.4f}")
          tf.summary.scalar(k, v, step=it)
        # test data
        outputs = model(data['test'].toarray())
        outputs.pop('alpha')
        for k, v in sorted(outputs.items()):
          v = tf.reduce_mean(v)
          k = f"val_{k}"
          print(f" {k:10s}:{v:.4f}")
          tf.summary.scalar(k, v, step=it)
        # topics
        topic_text = []
        for idx, topic_idx in enumerate(highest_weight_topics[:20]):
          l = [
              "[#{}]index={} alpha={:.2f}".format(idx, topic_idx,
                                                  alpha[topic_idx])
          ]
          l += [vocabulary[word] for word in top_words[topic_idx, :10]]
          l = " ".join(l)
          print("", l)
          topic_text.append(l)
        tf.summary.text("topics", np.array(topic_text), step=it)
    if it >= max_iter:
      break


model = LDAModel()
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
# train_tfp_model(model, optimizer)


# ===========================================================================
# Create LDAVAE
# ===========================================================================
def callback(vae: LDAVAE, n_topics=10):
  print(f"*** {vae.lda_posterior} ***")
  text = vae.get_topics_string(vocabulary, n_topics=n_topics)
  print("\n".join(text))
  print(f"[#{int(vae.step.numpy())}]Perplexity:",
        float(vae.perplexity(data['test'])))
  vae.plot_learning_curves(os.path.join(CACHE_DIR, f"{vae.lda_posterior}.png"),
                           summary_steps=1000)


init_kw = dict(n_words=n_words,
               n_topics=n_topics,
               prior_init=0.7,
               prior_warmup=warmup,
               encoder=make_encoder)
train_kw = dict(train=train_ds,
                valid=test_ds,
                max_iter=max_iter,
                optimizer='adam',
                learning_rate=learning_rate,
                batch_size=batch_size,
                valid_interval=20,
                compile_graph=True,
                skip_fitted=True)

# VAE with Dirichlet latent posterior
dvae = LDAVAE(lda_posterior="dirichlet", path=DVAE_PATH, **init_kw)
print(dvae)
dvae.fit(callback=partial(callback, vae=dvae), checkpoint=DVAE_PATH, **train_kw)
callback(dvae, n_topics=20)
exit()
# VAE with Logistic-Normal latent posterior
gvae = LDAVAE(lda_posterior="gaussian", path=GVAE_PATH, **init_kw)
print(gvae)
gvae.fit(callback=partial(callback, vae=gvae), checkpoint=GVAE_PATH, **train_kw)
callback(gvae, n_topics=20)


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
