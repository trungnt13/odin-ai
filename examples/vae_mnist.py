import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'gpu,float32,seed=5218'

from odin import fuel as F, backend as K
from odin.utils import batching
from odin import visual as V

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_probability import distributions as tfd
# ===========================================================================
# CONFIG
# ===========================================================================
CODE_SIZE = 12
NUM_SAMPLES = 16
NUM_EPOCH = 8
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
decoder_likelihood = make_decoder(approx_posterior_sample) # [num_sample, num_batch, 28, 28]
latent_prior = make_prior(code_size=CODE_SIZE)
# ====== Define the loss ====== #
# `distortion` is just the negative log likelihood.
distortion = -decoder_likelihood.log_prob(X) # [num_sample, num_batch]
avg_distortion = tf.reduce_mean(distortion) # for monitoring
# divergence between posterior and the prior
if ANALYTIC_KL:
  rate = tfd.kl_divergence(approx_posterior, latent_prior) # [num_batch]
else:
  rate = (approx_posterior.log_prob(approx_posterior_sample) -
          latent_prior.log_prob(approx_posterior_sample)) # [num_sample, num_batch]
avg_rate = tf.reduce_mean(rate) # for monitoring
# ELBO
elbo_local = -(rate + distortion)
elbo = tf.reduce_mean(elbo_local)
loss = -elbo
# IWAE
importance_weighted_elbo = tf.reduce_mean(
    tf.reduce_logsumexp(elbo_local, axis=0) -
    tf.log(tf.to_float(NUM_SAMPLES)))
# sample images: Decode samples from the prior
# for visualization.
latent_prior_sample = latent_prior.sample(16)
random_image = make_decoder(latent_prior_sample)
random_image_sample = tf.squeeze(random_image.sample(), axis=1)
random_image_mean = tf.squeeze(random_image.mean(), axis=1)
# ===========================================================================
# Optimizing and training
# ===========================================================================
global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.cosine_decay(0.001, global_step, 5001)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

K.initialize_all_variables()

for epoch in range(20):
  scores = K.eval([elbo, avg_rate, avg_distortion,
                   random_image_sample, random_image_mean],
                  feed_dict={X: X_test})
  print('Epoch:', epoch,
        ' elbo:', scores[0], ' rate:', scores[1], ' distortion:', scores[2])
  exit()
  # ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
  # plot_codes(ax[epoch, 0], test_codes, y_test)
  # plot_samples(ax[epoch, 1:], test_samples)
  for s, e in batching(batch_size=128, n=X_train.shape[0],
                       seed=5218 + epoch):
    feed = {X: X_train[s:e]}
    K.eval(train_op, feed_dict=feed)
# plt.savefig('/tmp/tmp.pdf', dpi=300, transparent=True, bbox_inches='tight')
