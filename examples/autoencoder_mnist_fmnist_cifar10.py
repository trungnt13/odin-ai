from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
os.environ['ODIN'] = 'float32,gpu,seed=1234'
import timeit

import numpy as np

import tensorflow as tf

from odin import (nnet as N, backend as K, fuel as F,
                  visual as V, training as T, ml)
from odin.utils import args_parse, ctext, minibatch, Progbar

args = args_parse(descriptions=[
    ('-dim', 'latent dimension', None, 2),
    ('-data', 'dataset mnist or fmnist', ('mnist', 'fmnist', 'cifar10'), 'mnist'),
    ('-loss', 'huber, mse, ce (cross-entropy), lglo (log loss)', ('huber', 'mse', 'ce', 'lglo'), 'mse'),
    ('-bs', 'batch size', None, 128),
    ('-epoch', 'batch size, if negative stop based on valid loss', None, -1),
    ('--cnn', 'using convolutional network instead of dense network', None, False)
])
# ===========================================================================
# Load dataset
# ===========================================================================
is_cifar10 = False
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
  is_cifar10 = True
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
# ====== others ====== #
X_samples, y_samples = X_train[:25], y_train[:25]
input_shape = ds['X_train'].shape
input_ndim = len(input_shape)
print("Train shape:", ctext(X_train.shape, 'cyan'))
print("Valid shape:", ctext(X_valid.shape, 'cyan'))
print("Test  shape:", ctext(X_test.shape, 'cyan'))
# ====== create basic tensor ====== #
X = K.placeholder(shape=(None,) + input_shape[1:], name='X_input')
y = K.placeholder(shape=(None,), name='y_input')
# ===========================================================================
# Create the network
# ===========================================================================
LATENT_DROPOUT = 0.3
if args.cnn:
  with N.args_scope(([N.Conv, N.Dense], dict(b_init=None, activation=K.linear)),
                    (N.BatchNorm, dict(activation=tf.nn.elu)),
                    (N.Pool, dict(mode='max', pool_size=2))):
    f_encoder = N.Sequence([
        N.Dropout(level=0.5),
        N.Dimshuffle((0, 2, 3, 1)) if is_cifar10 else N.Dimshuffle((0, 1, 2, 'x')),

        N.Conv(num_filters=32, filter_size=3, pad='valid'),
        N.Pool(),
        N.BatchNorm(),

        N.Conv(num_filters=64, filter_size=3, pad='same'),
        N.BatchNorm(),

        N.Conv(num_filters=64, filter_size=3, pad='valid'),
        N.BatchNorm(activation=tf.nn.elu),
        N.Pool(),

        N.Flatten(outdim=2),
        N.Dense(num_units=args.dim)
    ], debug=True, name='EncoderNetwork')

    f_decoder = N.Sequence([
        N.Dropout(level=LATENT_DROPOUT, noise_type='uniform'),
        N.Noise(level=1.0, noise_type='gaussian'),
        N.Dimshuffle((0, 'x', 'x', 1)),

        N.TransposeConv(num_filters=64, filter_size=3, pad='valid'),
        N.Upsample(size=2, axes=(1, 2)),
        N.BatchNorm(),

        N.TransposeConv(num_filters=64, filter_size=3, pad='same'),
        N.BatchNorm(),

        N.TransposeConv(num_filters=32, filter_size=3, pad='valid'),
        N.Upsample(size=2, axes=(1, 2),
                   desire_shape=None if is_cifar10 else (None, 14, 14, None)),
        N.BatchNorm(),

        N.TransposeConv(num_filters=3 if is_cifar10 else 1,
                        filter_size=3, strides=2, pad='same'),
        N.Dimshuffle((0, 3, 1, 2)) if is_cifar10 else N.Squeeze(axis=-1)
    ], debug=True, name='DecoderNetwork')
else:
  with N.args_scope(N.Dense, b_init=None, activation=K.linear):
    f_encoder = N.Sequence([
        N.Dropout(level=0.5),
        N.Flatten(outdim=2),
        N.Dense(num_units=512),
        N.BatchNorm(axes=0, activation=K.relu),
        N.Dense(num_units=512),
        N.BatchNorm(axes=0, activation=K.relu),
        N.Dense(num_units=args.dim)
    ], debug=True, name='EncoderNetwork')

    f_decoder = N.Sequence([
        N.Dropout(level=LATENT_DROPOUT, noise_type='uniform'),
        N.Noise(level=1.0, noise_type='gaussian'),
        N.Dense(num_units=512, activation=K.relu),
        N.BatchNorm(axes=0, activation=K.relu),
        N.Dense(num_units=512, activation=K.relu),
        N.BatchNorm(axes=0, activation=K.relu),
        N.Dense(num_units=np.prod(input_shape[1:]), activation=K.linear),
        N.Reshape(shape=([0],) + input_shape[1:])
    ], debug=True, name='DecoderNetwork')
# ===========================================================================
# Create model and objectives
# ===========================================================================
Z = f_encoder(X)
X_logits = f_decoder(Z)

X_probas = tf.nn.sigmoid(X_logits)
f_X = K.function(inputs=X, outputs=X_probas,
                 training=True)

X_samples = f_decoder(tf.random_normal(shape=(25, args.dim),
                      dtype=X_probas.dtype))
f_samples = K.function(inputs=[], outputs=X_samples, training=False)
# ====== `distortion` is the negative log likelihood ====== #
if args.loss == 'ce':
  loss = tf.losses.softmax_cross_entropy(onehot_labels=X, logits=X_logits)
elif args.loss == 'mse':
  loss = tf.losses.mean_squared_error(labels=X, predictions=X_probas)
elif args.loss == 'huber':
  loss = tf.losses.huber_loss(labels=X, predictions=X_probas)
elif args.loss == 'lglo':
  loss = tf.losses.log_loss(labels=X, predictions=X_probas)
# ===========================================================================
# Optimizing the network
# ===========================================================================
update_ops = K.optimizers.Adam(lr=0.001).minimize(loss)
K.initialize_all_variables()
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
  for start, end in minibatch(batch_size=args.bs, n=X_train.shape[0],
                              seed=K.get_rng().randint(10e8)):
    _ = K.eval(loss, feed_dict={X: X_train[start:end]},
               update_after=update_ops)
    prog.add(end - start)
    train_losses.append(_)
  # ====== training log ====== #
  print(ctext("[Epoch %d]" % epoch, 'yellow'), '%.2f(s)' % (timeit.default_timer() - start_time))
  print("[Training set] Loss: %.4f" % np.mean(train_losses))
  # ====== validation set ====== #
  code_samples, lo = K.eval([Z, loss], feed_dict={X: X_valid})
  print("[Valid set]    Loss: %.4f" % lo)
  # ====== record the history ====== #
  record_train_loss.append(np.mean(train_losses))
  record_valid_loss.append(lo)
  # ====== plotting ====== #
  if args.dim > 2:
    code_samples = ml.fast_pca(code_samples, n_components=2,
                               random_state=K.get_rng().randint(10e8))
  img_samples = f_samples()
  img_mean = f_X(X_valid[:25])

  V.plot_figure(nrow=3, ncol=12)

  ax = plt.subplot(1, 3, 1)
  ax.scatter(code_samples[:, 0], code_samples[:, 1], s=2, c=y_valid, alpha=0.3)
  ax.set_title('Epoch %d' % epoch)
  ax.set_aspect('equal', 'box')
  ax.axis('off')

  ax = plt.subplot(1, 3, 2)
  ax.imshow(V.tile_raster_images(img_samples), cmap=plt.cm.Greys_r)
  ax.axis('off')

  ax = plt.subplot(1, 3, 3)
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
code_samples, lo = K.eval([Z, loss], feed_dict={X: X_valid})
if args.dim > 2:
  code_samples = ml.fast_pca(code_samples, n_components=2,
                             random_state=K.get_rng().randint(10e8))
print("[Test set]     Loss: %.4f" % lo)
# plot test code samples
V.plot_figure(nrow=8, ncol=8)
ax = plt.subplot(1, 1, 1)
ax.scatter(code_samples[:, 0], code_samples[:, 1], s=2, c=y_valid, alpha=0.5)
ax.set_title('Test set')
ax.set_aspect('equal', 'box')
ax.axis('off')

V.plot_save('/tmp/tmp_ae.pdf')
