from __future__ import absolute_import, division, print_function

import os
import timeit

import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from odin import backend as K
from odin import bay as B
from odin import fuel as F
from odin import ml
from odin import nnet as N
from odin import training as T
from odin import visual as V
from odin.ml import fast_pca
from odin.stats import describe
from odin.utils import Progbar, args_parse, batching, ctext

matplotlib.use('Agg')

os.environ['ODIN'] = 'float32,gpu,seed=1234'




args = args_parse(descriptions=[
    ('-ds', 'dataset', None, 'mnist_original'),

    ('-zdim', 'latent dimension', None, 64),
    ('-hdim', 'number of hidden units', None, 256),

    ('-xdist', 'distribution of input X', None, 'poisson'),
    ('-zdist', 'posterior distribution of latent Z', None, 'normal'),
    ('-zprior', 'prior distribution of latent Z', None, 'normal01'),

    ('-xdrop', 'dropout on input X', None, 0),
    ('-edrop', 'dropout on the encoder E', None, 0),
    ('-zdrop', 'dropout on latent Z', None, 0),
    ('-ddrop', 'dropout on the decoder D', None, 0),

    ('-nsample-train', 'number of posterior samples', None, 1),
    ('-nsample-test', 'number of posterior samples', None, 1000),
    ('-batch', 'batch size', None, 64),
    ('-epoch', 'number of epoch', None, 200),

    ('--no-monitor', 'turn off epoch monitoring, significantly faster', None, False),
    ('--no-batchnorm', 'turn off batch normalization', None, False),
    ('--analytic', 'using analytic KL or sampling', None, False),
    ('--iw', 'enable important weights sampling', None, False),
])

FIGURE_PATH = '/tmp'
# ===========================================================================
# Load dataset
# ===========================================================================
ds = F.parse_dataset(args.ds)
print(ds)
# ====== print data info ====== #
if 'X' in ds and 'y' in ds:
  X, y = ds['X'], ds['y']
  rand = np.random.RandomState(seed=1234)
  n = X.shape[0]
  perm = rand.permutation(n)
  X, y = X[perm], y[perm]
  X_train, y_train = X[:int(0.8 * n)], y[:int(0.8 * n)]
  X_test, y_test = X[int(0.8 * n):], y[int(0.8 * n):]
elif 'X_train' in ds and 'X_test' in ds:
  X_train, y_train = ds['X_train'], ds['y_train']
  X_test, y_test = ds['X_test'], ds['y_test']
else:
  raise RuntimeError('No support for dataset: "%s"' % args.ds)
# ====== post processing ====== #
if y_train.ndim > 1:
  y_train = np.argmax(y_train, axis=-1)
if y_test.ndim > 1:
  y_test = np.argmax(y_test, axis=-1)
input_shape = (None,) + X_train.shape[1:]
n_classes = len(np.unique(y_train))
print("Train:", ctext(X_train.shape, 'cyan'), describe(X_train, shorten=True))
print("Test :", ctext(X_test.shape, 'cyan'), describe(X_test, shorten=True))
# ====== create basic tensor ====== #
X = K.placeholder(shape=(None,) + input_shape[1:], name='X')
W = K.placeholder(shape=(None,) + input_shape[1:], name='W')
y = K.placeholder(shape=(None,), name='y')
nsample = K.placeholder(shape=(), dtype='int32', name='nsample')
# ===========================================================================
# Create the network
# ===========================================================================
index = [0]
def dense_creator():
  net = N.Sequence([
      N.Dense(int(args.hdim),
              b_init=0 if args.no_batchnorm else None,
              activation=K.relu if args.no_batchnorm else K.linear),
      None if args.no_batchnorm else N.BatchNorm(activation=K.relu)
  ], debug=True, name="DenseBatchNorm%d" % index[0])
  index[0] += 1
  return net

f_encoder = N.Sequence([
    N.Flatten(outdim=2),
    N.Dropout(level=args.xdrop) if args.xdrop > 0 else None,
    dense_creator(),
    dense_creator(),
    N.Dropout(level=args.edrop) if args.edrop > 0 else None,
], debug=True, name='Encoder')

f_decoder = N.Sequence([
    N.Dropout(level=args.zdrop) if args.zdrop > 0 else None,
    dense_creator(),
    dense_creator(),
    N.Dropout(level=args.ddrop) if args.ddrop > 0 else None,
], debug=True, name='Decoder')
# ===========================================================================
# Create statistical model
# ===========================================================================
# ====== encoder ====== #
E = f_encoder(X)
# ====== latent ====== #
q_Z_given_X = B.parse_distribution(
    args.zdist, E, int(args.zdim),
    name='Z')
# [n_sample, n_batch, zdim]
q_Z_given_X_samples = q_Z_given_X.sample(nsample)

Z = [
    q_Z_given_X.mean(),
    tf.concat([q_Z_given_X.mean(), tf.sqrt(q_Z_given_X.variance())],
            axis=-1),
    K.flatten(tf.transpose(q_Z_given_X_samples, perm=(1, 0, 2)),
            outdim=2)
]
Z_names = [
    "posterior mean",
    "statistic pooling",
    "all samples flatten"
]
# ====== Z prior ====== #
p_Z = B.parse_distribution(dist_name=args.zprior)
# ====== decoder ====== #
D = f_decoder(q_Z_given_X_samples)
# ====== reconstruction ====== #
p_X_given_Z = B.parse_distribution(
    args.xdist, D, int(np.prod(input_shape[1:])),
    n_eventdim=1, name='W')
# [n_sample, n_batch, feat_dim]
p_X_given_Z_mean = p_X_given_Z.mean()
# [n_batch, feat_dim]
p_X_mean = tf.reduce_mean(p_X_given_Z_mean, axis=0)
# MCMC variance [n_batch, feat_dim] (/ tf.to_float(nsample) ?)
stdev_of_p_X_given_Z_mean = tf.sqrt(
    tf.reduce_mean(
        tf.square(p_X_given_Z_mean - tf.expand_dims(p_X_mean, axis=0)),
        axis=0)
)
# analytical variance
p_X_stdev = tf.sqrt(
    tf.reduce_mean(p_X_given_Z.variance(), axis=0))
# ===========================================================================
# Variational inference (ELBO)
# The Independent distribution composed of a collection of
#   Bernoulli distributions might define a distribution over
#   an image (where each Bernoulli is a distribution over each pixel).
#   batch: (?, 28, 28); event: () -> batch: (?); event: (28, 28)
# Rule for broadcasting `log_prob`:
#  * If omitted batch_shape, add (1,) to the batch_shape places
#  * Broadcast the n rightmost dimensions of t' against the [batch_shape, event_shape]
#    of the distribution you're computing a log_prob for. In more detail:
#    for the dimensions where t' already matches the distribution, do nothing,
#    and for the dimensions where t' has a singleton, replicate that singleton
#    the appropriate number of times. Any other situation is an error.
#    (For scalar distributions, we only broadcast against batch_shape,
#    since event_shape = [].)
#  * Now we're finally able to compute the log_prob. The resulting tensor will have shape [sample_shape, batch_shape], where sample_shape is defined to be any dimensions of t or t' to the left of the n-rightmost dimensions: sample_shape = shape(t)[:-n].
# ===========================================================================
print("=" * 48)
print("Creating ELBO")
print("=" * 48)
# ====== KL divergence ====== #
if args.analytic:
  # [n_batch, n_latent]
  KL = tfd.kl_divergence(q_Z_given_X, p_Z)
  KL = tf.expand_dims(KL, axis=0)
else:
  # [n_sample_train, n_batch, n_latent] - [n_sample_train, n_batch, n_latent]
  KL = (q_Z_given_X.log_prob(q_Z_given_X_samples) -
        p_Z.log_prob(q_Z_given_X_samples))
# latent variables are independent
KL = tf.reduce_sum(KL, axis=-1)
KL_mean = tf.reduce_mean(KL, name="KL_divergence")
print("KL  :", ctext(KL, 'cyan'))
# ====== negative log likelihood ====== #
W_2D = K.flatten(W, outdim=2)
NLLK = -p_X_given_Z.log_prob(tf.expand_dims(W_2D, axis=0))
NLLK_mean = tf.reduce_mean(NLLK, name="Negative_LLK")
print("NLLK:", ctext(NLLK, 'cyan'))
# ====== ELBO ====== #
# we want to maximize the evident lower bound
ELBO = tf.identity(-NLLK - KL, name="ELBO")
# but minimize the loss
loss = tf.identity(tf.reduce_mean(-ELBO), name="loss")
# important weights ELBO, logsumexp among sampling dimension
IW_ELBO = tf.identity(
    tf.reduce_logsumexp(ELBO, axis=0) - tf.log(tf.to_float(nsample)),
    name="ImportantWeight_ELBO")
iw_loss = tf.identity(tf.reduce_mean(-IW_ELBO), name="iw_loss")
print("ELBO:", ctext(ELBO, 'cyan'))
print("loss:", ctext(loss, 'cyan'))
print("IW-ELBO :", ctext(IW_ELBO, 'cyan'))
print("IW-loss:", ctext(iw_loss, 'cyan'))
# ===========================================================================
# Create the optimizer and function
# ===========================================================================
optz = K.optimizers.Adam(lr=0.001)
updates = optz.minimize(iw_loss if args.iw else loss, verbose=1)
global_norm = optz.norm
K.initialize_all_variables()
# ====== create functions ====== #
input_plh = [X, W]
f_train = K.function(inputs=input_plh,
                     outputs=[loss, iw_loss, KL_mean, NLLK_mean, global_norm],
                     updates=updates,
                     defaults={nsample: args.nsample_train},
                     training=True)
f_score = K.function(inputs=input_plh,
                     outputs=[loss, iw_loss, KL_mean, NLLK_mean],
                     defaults={nsample: args.nsample_train},
                     training=False)

f_z = K.function(inputs=X,
                 outputs=Z,
                 defaults={nsample: args.nsample_test},
                 batch_size=args.batch,
                 training=False)
f_w = K.function(inputs=X,
                 outputs=[p_X_mean, stdev_of_p_X_given_Z_mean, p_X_stdev],
                 defaults={nsample: args.nsample_test},
                 batch_size=args.batch,
                 training=False)
# ===========================================================================
# Training
# ===========================================================================
# ====== epoch visualization ====== #
def plot_epoch(task):
  if task is None:
    curr_epoch = 0
  else:
    curr_epoch = task.curr_epoch
    if not (curr_epoch < 5 or curr_epoch % 5 == 0):
      return
  rand = np.random.RandomState(seed=1234)

  X, y = X_test, y_test
  n_data = X.shape[0]
  Z = f_z(X)
  W, W_stdev_mcmc, W_stdev_analytic = f_w(X)

  X_pca, W_pca_1 = fast_pca(X, W, n_components=2,
                            random_state=rand.randint(10e8))
  W_pca_2 = fast_pca(W, n_components=2,
                     random_state=rand.randint(10e8))
  X_count_sum = np.sum(X, axis=tuple(range(1, X.ndim)))
  W_count_sum = np.sum(W, axis=-1)

  n_visual_samples = 8
  nrow = 13 + n_visual_samples * 3
  V.plot_figure(nrow=int(nrow * 1.8), ncol=18)
  with V.plot_gridSpec(nrow=nrow + 3, ncol=6, hspace=0.8) as grid:
    # plot the latent space
    for i, (z, name) in enumerate(zip(Z, Z_names)):
      if z.shape[1] > 2:
        z = fast_pca(z, n_components=2, random_state=rand.randint(10e8))
      ax = V.subplot(grid[:3, (i * 2):(i * 2 + 2)])
      V.plot_scatter(x=z[:, 0], y=z[:, 1],
                     color=y, marker=y, n_samples=4000,
                     ax=ax, legend_enable=False, legend_ncol=n_classes)
      ax.set_title(name, fontsize=12)
    # plot the reconstruction
    for i, (x, name) in enumerate(zip(
            [X_pca, W_pca_1, W_pca_2],
            ['Original data', 'Reconstruction', 'Reconstruction (separated PCA)'])):
      ax = V.subplot(grid[3:6, (i * 2):(i * 2 + 2)])
      V.plot_scatter(x=x[:, 0], y=x[:, 1],
                     color=y, marker=y, n_samples=4000,
                     ax=ax, legend_enable=i == 1, legend_ncol=n_classes,
                     title=name)
    # plot the reconstruction count sum
    for i, (x, count_sum, name) in enumerate(zip(
            [X_pca, W_pca_1],
            [X_count_sum, W_count_sum],
            ['Original data (Count-sum)', 'Reconstruction (Count-sum)'])):
      ax = V.subplot(grid[6:10, (i * 3):(i * 3 + 3)])
      V.plot_scatter_heatmap(x=x[:, 0], y=x[:, 1], val=count_sum,
                             n_samples=2000, marker=y, ax=ax, size=8,
                             legend_enable=i == 0, legend_ncol=n_classes,
                             title=name, colorbar=True, fontsize=10)
    # plot the count-sum series
    count_sum_observed = np.sum(X, axis=0).ravel()
    count_sum_expected = np.sum(W, axis=0)
    count_sum_stdev_explained = np.sum(W_stdev_mcmc, axis=0)
    count_sum_stdev_total = np.sum(W_stdev_analytic, axis=0)
    for i, kws in enumerate([dict(xscale='linear', yscale='linear', sort_by=None),
                             dict(xscale='linear', yscale='linear', sort_by='expected'),
                             dict(xscale='log', yscale='log', sort_by='expected')]):
      ax = V.subplot(grid[10:10 + 3, (i * 2):(i * 2 + 2)])
      V.plot_series_statistics(count_sum_observed, count_sum_expected,
                               explained_stdev=count_sum_stdev_explained,
                               total_stdev=count_sum_stdev_total,
                               fontsize=8, title="Count-sum" if i == 0 else None,
                               **kws)
    # plot the mean and variances
    curr_grid_index = 13
    ids = rand.permutation(n_data)
    ids = ids[:n_visual_samples]
    for i in ids:
      observed, expected, stdev_explained, stdev_total = \
          X[i], W[i], W_stdev_mcmc[i], W_stdev_analytic[i]
      observed = observed.ravel()
      for j, kws in enumerate([dict(xscale='linear', yscale='linear', sort_by=None),
                               dict(xscale='linear', yscale='linear', sort_by='expected'),
                               dict(xscale='log', yscale='log', sort_by='expected')]):
        ax = V.subplot(grid[curr_grid_index:curr_grid_index + 3,
                            (j * 2):(j * 2 + 2)])
        V.plot_series_statistics(observed, expected,
                explained_stdev=stdev_explained, total_stdev=stdev_total,
                fontsize=8, title="Test Sample #%d" % i if j == 0 else None,
                **kws)
      curr_grid_index += 3
  V.plot_save(os.path.join(FIGURE_PATH, 'latent_%d.png' % curr_epoch),
              dpi=200, log=True)
  exit()
# ====== training ====== #
runner = T.MainLoop(batch_size=args.batch,
                    seed=1234, shuffle_level=2,
                    allow_rollback=False, verbose=2)
runner.set_callbacks([
    T.NaNDetector(task_name=None, patience=-1, detect_inf=True),
    None if args.no_monitor else
    T.EpochSummary(task_name=('train', 'valid'),
                   output_name=(loss, iw_loss, KL_mean, NLLK_mean),
                   print_plot=False,
                   save_path=os.path.join(FIGURE_PATH, 'summary.png')),
    T.LambdaCallback(fn=plot_epoch, task_name='train')
])
runner.set_train_task(func=f_train, data=[X_train, X_train],
                      epoch=args.epoch,
                      name='train')
runner.set_valid_task(func=f_score, data=[X_test, X_test],
                      name='valid')
runner.run()
