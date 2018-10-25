from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,seed=5218'
import timeit

import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd, bijectors as tfb

from odin import (nnet as N, backend as K, fuel as F,
                  visual as V, training as T, ml)
from odin.utils import args_parse, ctext, batching, Progbar

args = args_parse(descriptions=[
    ('-dim', 'latent dimension', None, 2),
    ('-hid', 'number of hidden units', None, 1024),
    ('-data', 'dataset mnist or fmnist', ('mnist', 'fmnist', 'cifar10'), 'mnist'),
    ('-loss', 'huber, mse, ce (cross-entropy), lglo (log loss)', ('huber', 'mse', 'ce', 'lglo'), 'ce'),
    ('-s', 'number of posterior samples', None, 25),
    ('-batch', 'batch size', None, 128),
    ('-epoch', 'batch size, if negative stop based on valid loss', None, -1),
    ('--analytic', 'using Analytic KL or not', None, False)
])
# ===========================================================================
# Load dataset
# ===========================================================================
if args.data == 'fmnist':
  ds = F.FMNIST_original.load()
  X_train, y_train = ds['X_train'][:], ds['y_train'][:]
  ids = np.random.permutation(len(X_train))
  X_train, y_train = X_train[ids], y_train[ids]

  X_valid, y_valid = X_train[50000:], y_train[50000:]
  X_train, y_train = X_train[:50000], y_train[:50000]
  X_test, y_test = ds['X_test'][:], ds['y_test'][:]
  # normalize value to [0, 1]
  X_train = X_train / 255.
  X_valid = X_valid / 255.
  X_test = X_test / 255.
elif args.data == 'mnist':
  ds = F.MNIST.load()
  X_train, y_train = ds['X_train'][:], ds['y_train'][:]
  X_valid, y_valid = ds['X_valid'][:], ds['y_valid'][:]
  X_test, y_test = ds['X_test'][:], ds['y_test'][:]
elif args.data == 'cifar10':
  ds = F.CIFAR10.load()
  X_train, y_train = ds['X_train'][:], ds['y_train'][:]
  X_test, y_test = ds['X_test'][:], ds['y_test'][:]

  ids = np.random.permutation(len(X_train))
  X_train, y_train = X_train[ids], y_train[ids]

  X_valid, y_valid = X_train[40000:], y_train[40000:]
  X_train, y_train = X_train[:40000], y_train[:40000]
  # normalize value to [0, 1]
  X_train = X_train / 255.
  X_valid = X_valid / 255.
  X_test = X_test / 255.
print(ds)
# ====== print data info ====== #
X_samples, y_samples = X_train[:25], y_train[:25]
input_shape = ds['X_train'].shape
print("Train shape:", ctext(X_train.shape, 'cyan'))
print("Valid shape:", ctext(X_valid.shape, 'cyan'))
print("Test  shape:", ctext(X_test.shape, 'cyan'))
# ====== create basic tensor ====== #
X = K.placeholder(shape=(None,) + input_shape[1:], name='X_input')
y = K.placeholder(shape=(None,), name='y_input')
# ===========================================================================
# Create the network
# ===========================================================================
num_units = int(args.hid)
with N.args_scope([N.Dense, dict(b_init=None, activation=K.linear)]):
  f_encoder = N.Sequence([
      N.Flatten(outdim=2),
      N.Dropout(level=0.3),
      N.Dense(num_units),
      N.BatchNorm(axes=0, activation=K.relu),
      N.Dense(num_units),
      N.BatchNorm(axes=0, activation=K.relu),
      N.Dense(num_units=args.dim * 2, activation=K.linear)
  ], debug=True, name='EncoderNetwork')

  f_decoder = N.Sequence([
      N.Reshape(shape=(-1, [-1])),
      N.Dense(num_units, activation=K.relu),
      N.BatchNorm(axes=0, activation=K.relu),
      N.Dense(num_units, activation=K.relu),
      N.BatchNorm(axes=0, activation=K.relu),
      N.Dense(num_units=np.prod(input_shape[1:]), activation=K.linear),
      N.Reshape(shape=([0],) + input_shape[1:])
  ], debug=True, name='DecoderNetwork')
# ===========================================================================
# Create statistical model
# ===========================================================================
# ====== posterior ====== #
loc_scale = f_encoder(X)
loc = loc_scale[:, :args.dim]
scale = loc_scale[:, args.dim:]
qZ_X = tfd.MultivariateNormalDiag(
    loc=loc, scale_diag=tf.nn.softplus(scale + K.softplus_inverse(1.0)),
    name="qZ_X")
qZ_X_samples = qZ_X.sample(args.s) # [num_samples, batch_size, dim]
# ====== prior ====== #
pZ = tfd.MultivariateNormalDiag(
    loc=tf.zeros(shape=(1, args.dim)), scale_identity_multiplier=1.0,
    name="pZ")
pZ_samples = pZ.sample(args.s)
# ===========================================================================
# Generator and Distortion
# The Independent distribution composed of a collection of
#   Bernoulli distributions might define a distribution over
#   an image (where each Bernoulli is a distribution over each pixel).
#   batch: (?, 28, 28); event: () -> batch: (?); event: (28, 28)
# Rule for broadcasting `log_prob`:
#  * If omitted batch_shape, add (1,) to the batch_shape places
#  * Broadcast the n rightmost dimensions of t' against the [batch_shape, event_shape] of the distribution you're computing a log_prob for. In more detail: for the dimensions where t' already matches the distribution, do nothing, and for the dimensions where t' has a singleton, replicate that singleton the appropriate number of times. Any other situation is an error. (For scalar distributions, we only broadcast against batch_shape, since event_shape = [].)
#  * Now we're finally able to compute the log_prob. The resulting tensor will have shape [sample_shape, batch_shape], where sample_shape is defined to be any dimensions of t or t' to the left of the n-rightmost dimensions: sample_shape = shape(t)[:-n].
# ===========================================================================
X_logits_qZ_X = f_decoder(qZ_X_samples) # [num_sample * num_batch, 28, 28]
X_probas_qZ_X = tf.nn.sigmoid(X_logits_qZ_X)
X_true = K.repeat(X, n=args.s, axes=0, name='X_true') # [num_batch * num_sample, 28, 28]
# ====== `distortion` is the negative log likelihood ====== #
if args.loss == 'ce':
  pX_Z = tfd.Independent(tfd.Bernoulli(logits=X_logits_qZ_X),
                         reinterpreted_batch_ndims=len(input_shape) - 1,
                         name="pX_Z")
  distortion = -pX_Z.log_prob(X_true) # [num_sample * num_batch]
else:
  if args.loss == 'mse':
    distortion = tf.losses.mean_squared_error(labels=X_true, predictions=X_probas_qZ_X,
                                              reduction=tf.losses.Reduction.NONE)
  elif args.loss == 'huber':
    distortion = tf.losses.huber_loss(labels=X_true, predictions=X_probas_qZ_X,
                                      reduction=tf.losses.Reduction.NONE)
  elif args.loss == 'lglo':
    distortion = tf.losses.log_loss(labels=X_true, predictions=X_probas_qZ_X,
                                    reduction=tf.losses.Reduction.NONE)
  distortion = tf.reduce_mean(distortion, axis=np.arange(1, len(input_shape)))
# reshape and avg
distortion = K.reshape(distortion, shape=(args.s, -1)) # [num_sample, num_batch]
avg_distortion = tf.reduce_mean(distortion) # for monitoring
# ====== Sampling ====== #
X_logits_pZ = f_decoder(pZ_samples) # [num_sample, 28, 28]
X_ = tfd.Independent(tfd.Bernoulli(logits=X_logits_pZ),
                     reinterpreted_batch_ndims=len(input_shape) - 1,
                     name="X_")
pX_Z_samples = X_.sample(args.s)
pX_Z_mean = X_.mean()
# ===========================================================================
# ELBO
# ===========================================================================
# ====== rate is KL objective ====== #
# Whether or not to use the analytic version of the KL. When set to
# False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO
# will be used. Otherwise the -KL(q(Z|X) || p(Z)) +
# E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True,
# then you must also specify `mixture_components=1`.
if args.analytic:
  rate = tfd.kl_divergence(qZ_X, pZ) # [num_batch]
else:
  rate = (qZ_X.log_prob(qZ_X_samples) - pZ.log_prob(qZ_X_samples)) # [num_sample, num_batch]
avg_rate = tf.reduce_mean(rate)
# ====== ELBO ====== #
elbo_local = -(rate + distortion)
elbo = tf.reduce_mean(elbo_local) # maximize evidence-lower-bound
loss = -elbo # minimize loss
# IWAE
importance_weighted_elbo = tf.reduce_mean(
    tf.reduce_logsumexp(elbo_local, axis=0) -
    tf.log(tf.to_float(args.s)))
# ===========================================================================
# Optimizing the network
# ===========================================================================
update_ops = K.optimizers.Adam(lr=0.001).minimize(-elbo)
K.initialize_all_variables()
# ====== helper ====== #
def calc_loss_and_code(dat):
  losses = []
  for start, end in batching(batch_size=2048, n=dat.shape[0]):
    losses.append(K.eval([distortion, rate, qZ_X_samples],
                         feed_dict={X: dat[start:end]}))
  d = np.concatenate([i[0] for i in losses], axis=1)
  r = np.concatenate([i[1] for i in losses], axis=0 if args.analytic else 1)
  code_samples = np.concatenate([i[-1] for i in losses], axis=1).mean(axis=0)
  return code_samples, np.mean(d), np.mean(r), np.mean(d + r)
# ====== intitalize ====== #
record_train_loss = []
record_valid_loss = []
patience = 3
epoch = 0
# We want the rate to go up but the distortion to go down
while True:
  # ====== training ====== #
  train_losses = []
  prog = Progbar(target=X_train.shape[0], name='Epoch%d' % epoch)
  start_time = timeit.default_timer()
  for start, end in batching(batch_size=args.batch, n=X_train.shape[0],
                             seed=K.get_rng().randint(10e8)):
    _ = K.eval([avg_distortion, avg_rate, loss],
               feed_dict={X: X_train[start:end]},
               update_after=update_ops)
    prog.add(end - start)
    train_losses.append(_)
  # ====== training log ====== #
  train_losses = np.mean(np.array(train_losses), axis=0).tolist()
  print(ctext("[Epoch %d]" % epoch, 'yellow'), '%.2f(s)' % (timeit.default_timer() - start_time))
  print("[Training set] Distortion: %.4f    Rate: %.4f    Loss: %.4f" % tuple(train_losses))
  # ====== validation set ====== #
  code_samples, di, ra, lo = calc_loss_and_code(dat=X_valid)
  print("[Valid set]    Distortion: %.4f    Rate: %.4f    Loss: %.4f" % (di, ra, lo))
  # ====== record the history ====== #
  record_train_loss.append(train_losses[-1])
  record_valid_loss.append(lo)
  # ====== plotting ====== #
  if args.dim > 2:
    code_samples = ml.fast_pca(code_samples, n_components=2,
                               random_state=K.get_rng().randint(10e8))
  samples = K.eval([pX_Z_samples, pX_Z_mean])
  img_samples = samples[0]
  img_mean = samples[1]

  V.plot_figure(nrow=3, ncol=12)

  ax = plt.subplot(1, 4, 1)
  ax.scatter(code_samples[:, 0], code_samples[:, 1], s=2, c=y_valid, alpha=0.3)
  ax.set_title('Epoch %d' % epoch)
  ax.set_aspect('equal', 'box')
  ax.axis('off')

  ax = plt.subplot(1, 4, 2)
  ax.imshow(V.tile_raster_images(img_samples.mean(axis=0)), cmap=plt.cm.Greys_r)
  ax.axis('off')

  ax = plt.subplot(1, 4, 3)
  ax.imshow(V.tile_raster_images(img_samples[np.random.randint(0, len(img_samples))]), cmap=plt.cm.Greys_r)
  ax.axis('off')

  ax = plt.subplot(1, 4, 4)
  ax.imshow(V.tile_raster_images(img_mean), cmap=plt.cm.Greys_r)
  ax.axis('off')
  # ====== check exit condition ====== #
  if args.epoch > 0:
    if epoch >= args.epoch:
      break
  elif len(record_valid_loss) >= 2 and record_valid_loss[-1] > record_valid_loss[-2]:
    print(ctext("Dropped generalization loss `%.4f` -> `%.4f`" %
                (record_valid_loss[-2], record_valid_loss[-1]), 'yellow'))
    patience -= 1
    if patience == 0:
      break
  epoch += 1
# ====== print summary training ====== #
text = V.merge_text_graph(V.print_bar(record_train_loss, title="Train Loss"),
                          V.print_bar(record_valid_loss, title="Valid Loss"))
print(text)
# ====== testing ====== #
code_samples, di, ra, lo = calc_loss_and_code(dat=X_test)
if args.dim > 2:
  code_samples = ml.fast_pca(code_samples, n_components=2,
                             random_state=K.get_rng().randint(10e8))
print("[Test set]     Distortion: %.4f    Rate: %.4f    Loss: %.4f" % (di, ra, lo))
# plot test code samples
V.plot_figure(nrow=6, ncol=6)
ax = plt.subplot(1, 1, 1)
ax.scatter(code_samples[:, 0], code_samples[:, 1], s=2, c=y_valid, alpha=0.5)
ax.set_title('Test set')
ax.set_aspect('equal', 'box')
ax.axis('off')

V.plot_save('/tmp/tmp_vae.pdf')
