from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("Agg")

import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np
import edward as ed
import tensorflow as tf

from odin import nnet as N, backend as K, fuel as F, training as T
from odin.utils import Progbar, get_modelpath, get_logpath

# ===========================================================================
# Const
# ===========================================================================
NUM_HIDDEN = 128
NUM_LATENT = 2
NDIM = 28 * 28
# ===========================================================================
# Load data
# ===========================================================================
ds = F.MNIST.load()
print(ds)
X_train, y_train = ds['X_train'][:], ds['y_train'][:]
X_test, y_test = ds['X_test'][:], ds['y_test'][:]
X_train = X_train.reshape(-1, NDIM)
X_test = X_test.reshape(-1, NDIM)
# ====== Create placeholder ====== #
X = K.placeholder(shape=(None, NDIM), dtype='float32', name='X')
BATCH_SIZE = tf.shape(X)[0]
# ===========================================================================
# Create the network
# ===========================================================================
# ====== INFERENCE ====== #
f_encoder = N.Sequence(ops=[
    N.Dense(num_units=NUM_HIDDEN, b_init=0, activation=K.relu),
    N.Dense(num_units=NUM_LATENT * 2, b_init=0, activation=K.linear)
], debug=True, name='Encoder')
z = f_encoder(X)
loc = z[:, :NUM_LATENT]
scale = tf.nn.softplus(z[:, NUM_LATENT:])
qz = ed.models.Normal(loc=scale, scale=scale)
# ====== MODEL ====== #
z = ed.models.Normal(loc=tf.zeros([BATCH_SIZE, NUM_LATENT]),
                     scale=tf.ones([BATCH_SIZE, NUM_LATENT]))
f_decoder = N.Sequence(ops=[
    N.Dense(num_units=NUM_HIDDEN, b_init=0, activation=K.relu),
    N.Dense(num_units=NDIM, b_init=0, activation=K.linear)
], debug=True, name='Decoder')
x = ed.models.Bernoulli(logits=f_decoder(z))
# Bind p(x,z) and q(z|x) to the same TensorFlow placeholder for x.
inference = ed.KLqp({z: qz}, data={x: tf.cast(X, 'int32')})
# ===========================================================================
# Training
# ===========================================================================
inference.initialize(optimizer=tf.train.RMSPropOptimizer(0.01, epsilon=1.0))
tf.global_variables_initializer().run()
# K.initialize_all_variables()

def generator(array, batch_size):
  """Generate batch with respect to array's first axis."""
  start = 0  # pointer to where we are in iteration
  while True:
    stop = start + batch_size
    diff = stop - array.shape[0]
    if diff <= 0:
      batch = array[start:stop]
      start += batch_size
    else:
      batch = np.concatenate((array[start:], array[:diff]))
      start = diff
    batch = np.random.binomial(1, batch)  # binarize images
    yield batch
x_train_generator = generator(X_train, batch_size=128)

n_iter_per_epoch = X_train.shape[0] // 128
for epoch in range(1, 100 + 1):
  print("Epoch: {0}".format(epoch))
  avg_loss = 0.0

  for t in range(1, n_iter_per_epoch + 1):
    x_batch = next(x_train_generator)
    info_dict = inference.update(feed_dict={X: x_batch})
    avg_loss += info_dict['loss']

  # Print a lower bound to the average marginal likelihood for an
  # image.
  avg_loss /= n_iter_per_epoch
  avg_loss /= 128
  print("-log p(x) <= {:0.3f}".format(avg_loss))

  # # Prior predictive check.
  # images = x.eval()
  # for m in range(FLAGS.M):
  #   imsave(os.path.join(FLAGS.out_dir, '%d.png') % m,
  #          images[m].reshape(28, 28))
