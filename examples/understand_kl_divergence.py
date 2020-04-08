from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow import keras

from odin import visual as vs
from odin.bay import kl_divergence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)
sns.set()


# ===========================================================================
# Helper functions
# ===========================================================================
def minimize(loss_func,
             params,
             verbose=False,
             print_params=True,
             learning_rate=0.1,
             epochs=500):
  opt = tf.optimizers.Adam(learning_rate=learning_rate)
  benchmark = []
  history = []
  for i in range(epochs):
    start_time = time.time()
    with tf.GradientTape() as tape:
      tape.watch(params)
      loss = tf.reduce_mean(loss_func())
    grad = tape.gradient(loss, params)
    benchmark.append(time.time() - start_time)
    if verbose and (i + 1) % (epochs // 2) == 0:
      print("#%-4d Loss:%.4f (%.2f sec/100)" %
            (i + 1, loss, np.mean(benchmark) * 100))
      if print_params:
        for p in params:
          print(' * %s: %s' % (p.name, str(p.numpy())))
    history.append([loss.numpy()] + [p.numpy() for p in params])
    opt.apply_gradients(grads_and_vars=zip(grad, params))
  return history


create_posterior = lambda: tfp.distributions.Normal(
    loc=tf.Variable(0., dtype='float32', trainable=True, name='loc'),
    scale=tf.Variable(1., dtype='float32', trainable=True, name='scale'),
    name='Normal')

# NOTE: it important to get the loc spread wide enough to prevent mode collapse
# however, the scale must be small enough for not exploding the gradients
create_mixture_posterior = lambda n, loc_min=0, loc_max=100: \
  tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[1. / n] * n),
    components_distribution=tfp.distributions.Normal(
        loc=tf.Variable(
            np.linspace(loc_min, loc_max, n),
            dtype='float32', trainable=True, name='loc'),
        scale=tf.Variable(
            [1.] * n, dtype='float32', trainable=True, name='scale')))


def plot_posteriors(posterior, prior, n=1000):
  # this is very hard-coded function
  plt.figure(figsize=(12, 8))
  sns.kdeplot(prior.sample(int(n)).numpy(), label="Prior")
  for post, analytic, reverse, sample_shape in posterior:
    sns.kdeplot(post.sample(int(n)).numpy(),
                linestyle='-' if reverse else '--',
                label='%s-%s mcmc:%d' % ('KL(q||p)' if reverse else 'KL(p||q)',
                                         'A' if analytic else 'S', sample_shape))


def plot_histories(posterior, histories):
  plt.figure(figsize=(24, 5))
  for idx, (post, analytic, reverse, sample_shape) in enumerate(posterior):
    ax = plt.subplot(1, len(posterior), idx + 1)
    hist = histories[idx]
    name = '%s-%s mcmc:%d' % \
        ('KL(q||p)' if reverse else 'KL(p||q)', 'A' if analytic else 'S', sample_shape)
    loc = np.asarray([i[1] for i in hist])
    plt.plot(loc, label='loc', linestyle='-' if reverse else '--')
    scale = np.asarray([i[2] for i in hist])
    plt.plot(scale, label='scale', linestyle='-' if reverse else '--')
    plt.legend()
    ax = ax.twinx()
    plt.plot([i[0] for i in hist], label='loss', color='r')
    plt.title(name)
  plt.tight_layout()


# ===========================================================================
# Can deep network fix posterior mode collapse due to loc initialization
# * Appropriate learning rate is essential
# * High amount of components help, but not too high
# * Too deep network will make overfitting to the first components.
# * If input features are useless, deep network cannot help
# * maximum likelihood might end up with more modes
# ===========================================================================
prior = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[1.0 / 3] * 3),
    components_distribution=tfp.distributions.Normal(loc=[0, 25, 80],
                                                     scale=[1, 12, 4]))
n_components = 3
X = np.zeros(shape=(1, n_components)).astype('float32')
X = np.linspace(0, 80, num=n_components, dtype='float32')[None, :]
# X = np.random.rand(1, 3).astype('float32')
outputs = {}
for reverse in (True, False):
  loc = keras.Sequential([
      keras.layers.Dense(16, activation='relu', input_shape=(n_components,)),
      keras.layers.Dense(n_components,
                         activation='linear',
                         input_shape=(n_components,)),
  ])
  scale = tf.Variable([1.] * n_components,
                      dtype='float32',
                      trainable=True,
                      name='scale')
  history = minimize(lambda: kl_divergence(tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
          probs=[1. / n_components] * n_components),
      components_distribution=tfp.distributions.Normal(loc=loc(X), scale=scale
                                                      )),
                                           prior,
                                           reverse=reverse,
                                           q_sample=100),
                     params=loc.trainable_variables + [scale],
                     verbose=True,
                     print_params=False,
                     learning_rate=0.01,
                     epochs=1200)
  posterior = tfp.distributions.MixtureSameFamily(
      mixture_distribution=tfp.distributions.Categorical(
          probs=[1. / n_components] * n_components),
      components_distribution=tfp.distributions.Normal(loc=loc(X), scale=scale))
  outputs[reverse] = [posterior, history]

plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
sns.kdeplot(prior.sample(10000).numpy(), label='Prior')
sns.kdeplot(outputs[True][0].sample(10000).numpy().ravel(),
            label='Posterior-KL(q||p)')
sns.kdeplot(outputs[False][0].sample(10000).numpy().ravel(),
            label='Posterior-KL(p||q)',
            linestyle='--')
plt.legend()
ax = plt.subplot(1, 2, 2)
l1 = plt.plot([i[0] for i in outputs[True][1]], label='KL(q||p)')
ax.twinx()
l2 = plt.plot([i[0] for i in outputs[False][1]],
              label='KL(p||q)',
              linestyle='--')
plt.title("KL loss")
plt.legend(handles=[l1[0], l2[0]])

# ===========================================================================
# Mixture with Mixture Posterior
# ===========================================================================
prior = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[1.0 / 3] * 3),
    components_distribution=tfp.distributions.Normal(loc=[0, 32, 80],
                                                     scale=[1, 12, 4]))

for n in [2, 3, 5]:
  # analytic, reverse, nmcmc
  posterior = [
      (create_mixture_posterior(n=n), False, True, 1),
      (create_mixture_posterior(n=n), False, False, 1),
      (create_mixture_posterior(n=n), False, True, 100),
      (create_mixture_posterior(n=n), False, False, 100),
  ]
  histories = []
  for post, analytic, reverse, sample_shape in posterior:
    print("Training:", analytic, reverse, sample_shape)
    h = minimize(lambda: kl_divergence(
        q=post, p=prior, analytic=analytic, reverse=reverse, q_sample=sample_shape), [
            post.components_distribution.loc, post.components_distribution.scale
        ],
                 verbose=False)
    histories.append(h)
  # for more complicated distribution, need more samples
  plot_posteriors(posterior, prior, n=10000)
  plt.title("Prior:3-mixture Posterior:%d-mixture" % n)
  plot_histories(posterior, histories)
vs.plot_save()
exit()
# ===========================================================================
# Mixture with Normal Posterior
# ===========================================================================
prior = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[0.5, 0.5]),
    components_distribution=tfp.distributions.Normal(loc=[2, 20], scale=[1, 4]))

posterior = [
    (create_posterior(), False, True, 1),  # analytic, reverse, nmcmc
    (create_posterior(), False, False, 1),
    (create_posterior(), False, True, 100),
    (create_posterior(), False, False, 100),
]
histories = []
for post, analytic, reverse, sample_shape in posterior:
  print("Training:", analytic, reverse, sample_shape)
  h = minimize(lambda: kl_divergence(
      q=post, p=prior, analytic=analytic, reverse=reverse, q_sample=sample_shape),
               [post.loc, post.scale],
               verbose=False)
  histories.append(h)

plot_posteriors(posterior, prior)
plt.title("Prior:2-mixture Posterior:Normal")
plot_histories(posterior, histories)

# ===========================================================================
# Simple distribution
# ===========================================================================
prior = tfp.distributions.Normal(loc=8, scale=12)

posterior = [
    (create_posterior(), True, True, 1),  # analytic, reverse, nmcmc
    (create_posterior(), True, False, 1),
    (create_posterior(), False, True, 1),
    (create_posterior(), False, True, 100),
    (create_posterior(), False, False, 1),
    (create_posterior(), False, False, 100)
]
histories = []
for post, analytic, reverse, sample_shape in posterior:
  print("Training:", analytic, reverse, sample_shape)
  h = minimize(lambda: kl_divergence(
      q=post, p=prior, analytic=analytic, reverse=reverse, q_sample=sample_shape),
               [post.loc, post.scale],
               verbose=False)
  histories.append(h)

plot_posteriors(posterior, prior)
plt.title("Prior:Normal Posterior:Normal")
plot_histories(posterior, histories)
