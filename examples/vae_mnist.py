import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from sklearn.decomposition import PCA

from odin import fuel as F, backend as K
from odin.utils import batching
from odin import visual as V
from odin.ml import fast_tsne
# ===========================================================================
# CONFIG
# ===========================================================================
CODE_SIZE = 2
NUM_SAMPLES = 16
NUM_EPOCH = 20
NUM_ITER = None
# Whether or not to use the analytic version of the KL. When set to
# False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO
# will be used. Otherwise the -KL(q(Z|X) || p(Z)) +
# E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True,
# then you must also specify `mixture_components=1`.
ANALYTIC_KL = False
BATCH_SIZE = 32
# ===========================================================================
# Load data
# Input shape: [num_batch, 28, 28]
# ===========================================================================
ds = F.MNIST.load()
X_train, y_train = ds['X_train'][:], ds['y_train'][:]
X_valid, y_valid = ds['X_valid'][:], ds['y_valid'][:]
X_test, y_test = ds['X_test'][:], ds['y_test'][:]
input_shape = X_train.shape
# ====== convert to binary ====== #
X_train = np.where(X_train >= 0.5, 1, 0).astype('int32')
X_valid = np.where(X_valid >= 0.5, 1, 0).astype('int32')
X_test = np.where(X_test >= 0.5, 1, 0).astype('int32')
X_sample, y_sample = X_train[:128], y_train[:128]
print('X:', X_train.shape, np.min(X_train), np.max(X_train))
print('y:', y_train.shape)
print("Sample binary image:", y_train[0])
print(X_train[0])
# ===========================================================================
# Network
# ===========================================================================
def softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.expm1(x))

def make_encoder(data, code_size):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  stats = tf.layers.dense(x, 2 * code_size,
                          activation=None)
  return tfd.MultivariateNormalDiag(
      loc=stats[:, :code_size],
      scale_diag=tf.nn.softplus(stats[:, code_size:] + softplus_inverse(1.0)),
      name="qZX")

def make_decoder(z):
  z_ndim = z.get_shape().ndims
  assert z_ndim == 3, "#Dims: %d" % z_ndim
  x = tf.layers.dense(z, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  logit = tf.layers.dense(x, np.prod(input_shape[1:]),
                          activation=None)
  # During training: [num_sample, num_batch, num_code]
  # Inference: [num_sample, num_code]
  original_shape = tf.shape(z)
  img = tf.reshape(logit,
                   shape=tf.concat([original_shape[:-1], input_shape[1:]],
                                   axis=0))
  # the Independent distribution composed of a collection of
  # Bernoulli distributions might define a distribution over
  # an image (where each Bernoulli is a distribution over each pixel).
  # batch: (?, 28, 28); event: ()
  tmp = tfd.Bernoulli(img)
  # batch: (?,); event: (28, 28)
  tmp1 = tfd.Independent(tmp, reinterpreted_batch_ndims=2, name="pXZ")
  return tmp1

def make_prior(code_size):
  prior = tfd.MultivariateNormalDiag(
      loc=tf.zeros(shape=[1, code_size]),
      scale_identity_multiplier=1.0,
      name="pZ")
  return prior
# ===========================================================================
# Helper
# ===========================================================================
def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')

def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')
# ===========================================================================
# Run
# ===========================================================================
X = tf.placeholder(tf.float32, [None, 28, 28])
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)
# ====== Define the model ====== #
approx_posterior = make_encoder(X, code_size=CODE_SIZE) # [num_batch, num_code]
approx_posterior_sample = approx_posterior.sample(NUM_SAMPLES) # [num_sample, num_batch, num_code]
# ====== Distortion ====== #
decoder_likelihood = make_decoder(approx_posterior_sample) # [num_sample, num_batch, 28, 28]
# `distortion` is just the negative log likelihood.
distortion = -decoder_likelihood.log_prob(X) # [num_sample, num_batch]
avg_distortion = tf.reduce_mean(distortion) # for monitoring
# ====== KL ====== #
latent_prior = make_prior(code_size=CODE_SIZE)
# divergence between posterior and the prior
if ANALYTIC_KL:
  rate = tfd.kl_divergence(approx_posterior, latent_prior) # [num_batch]
else:
  rate = (approx_posterior.log_prob(approx_posterior_sample) -
          latent_prior.log_prob(approx_posterior_sample)) # [num_sample, num_batch]
avg_rate = tf.reduce_mean(rate) # for monitoring
# ====== ELBO ====== #
elbo_local = -(rate + distortion)
elbo = tf.reduce_mean(elbo_local) # maximize evidence-lower-bound
loss = -elbo # minimize loss
# IWAE
importance_weighted_elbo = tf.reduce_mean(
    tf.reduce_logsumexp(elbo_local, axis=0) -
    tf.log(tf.to_float(NUM_SAMPLES)))
# sample images: Decode samples from the prior
# for visualization.
latent_prior_sample = latent_prior.sample(16) # [16, 1, num_code]
random_image = make_decoder(latent_prior_sample) # [16, 1, 28, 28]
random_image_sample = tf.squeeze(random_image.sample(), axis=1) # [16, 28, 28]
random_image_mean = tf.squeeze(random_image.mean(), axis=1) # [16, 28, 28]
# ===========================================================================
# Optimizing and training
# ===========================================================================
update_op = tf.train.AdamOptimizer(0.001).minimize(-elbo)
K.initialize_all_variables()

V.plot_figure(nrow=NUM_EPOCH + 2, ncol=4)
num_iter = 0
for epoch in range(NUM_EPOCH):
  # ====== evaluating ====== #
  scores = K.eval([elbo, avg_rate, avg_distortion,
                   random_image_sample, random_image_mean,
                   approx_posterior_sample],
                  feed_dict={X: X_test})
  print('#Epoch:', epoch, "#Iter:", num_iter,
        ' elbo:', scores[0], ' rate:', scores[1], ' distortion:', scores[2])
  img_sample = scores[-3]
  img_mean = scores[-2]
  code_sample = scores[-1].mean(axis=0)
  # ====== plotting ====== #
  num_row = epoch * 3
  ax = plt.subplot(NUM_EPOCH, 3, num_row + 1)
  ax.scatter(code_sample[:, 0], code_sample[:, 1], s=2, c=y_test, alpha=0.1)
  ax.axis('off')

  ax = plt.subplot(NUM_EPOCH, 3, num_row + 2)
  ax.imshow(V.tile_raster_images(img_sample), cmap=plt.cm.Greys_r)
  ax.axis('off')

  ax = plt.subplot(NUM_EPOCH, 3, num_row + 3)
  ax.imshow(V.tile_raster_images(img_mean), cmap=plt.cm.Greys_r)
  ax.axis('off')
  # ====== training ====== #
  for s, e in batching(batch_size=BATCH_SIZE, n=X_train.shape[0],
                       seed=5218 + epoch):
    feed = {X: X_train[s:e]}
    K.eval(update_op, feed_dict=feed)
    num_iter += 1
  # ====== upper bound for #iter ====== #
  if NUM_ITER is not None and num_iter >= NUM_ITER:
    break
V.plot_save('/tmp/tmp.pdf', tight_plot=False)
