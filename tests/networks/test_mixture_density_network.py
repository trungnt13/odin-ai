from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from odin import visual as vis
from odin.networks import MixtureDensityNetwork
from scipy import stats
from sklearn.mixture import GaussianMixture
from tensorflow.python.keras import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

n = 1200
n_components = 12
x = []
for i in range(n_components):
  x.append(
      stats.norm.rvs(size=(n, 1), loc=i * 12,
                     scale=np.random.randint(1, 6)).astype('float32'))
x = np.concatenate(x, axis=0)

# ====== gmm ====== #
gmm = GaussianMixture(n_components=n_components,
                      covariance_type='spherical',
                      random_state=8)
gmm.fit(x)
gmm_llk = gmm.score(x)
gmm_mean = gmm.means_.ravel().astype('float32')


# ====== mdn ====== #
def fn_loss(y_true, y_pred):
  # negative log-likelihood
  nllk = tf.reduce_mean(-y_pred.log_prob(y_true))
  return nllk


mdn = MixtureDensityNetwork(1,
                            n_components=n_components,
                            covariance_type='none')
model = Sequential([mdn])
model.compile(optimizer='adam', loss=fn_loss)
model.fit(x=x, y=x, epochs=48, batch_size=32, verbose=True)

y = model(x)
mdn_llk = tf.reduce_mean(y.log_prob(x)).numpy()
mdn_mean = tf.reduce_mean(y.components_distribution.mean(),
                          axis=(0, -1)).numpy()

# ====== visualizing ====== #
fig = plt.figure()
sns.distplot(x, bins=80)
plt.title('Data')

fig = plt.figure()
sns.distplot(gmm.sample(n * n_components)[0], bins=80)
plt.title('GMM - llk: %.2f' % gmm_llk)

fig = plt.figure()
sns.distplot(y.sample().numpy(), bins=80)
plt.title('MDN - llk: %.2f' % mdn_llk)

vis.plot_save()
