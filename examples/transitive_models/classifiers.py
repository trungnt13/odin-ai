# This code try to find the best classifier for the SemafoVAE latents
import os
import numpy as np
from typing import List
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import cloudpickle as pickle
from tensorflow_probability.python.distributions import Distribution
from collections import Counter

seed = 1
n_samples = 100
np.random.seed(seed)
tf.random.set_seed(seed)

######## load the arrays
model_name = 'mnist_variationalautoencoder'
model_name = 'mnist_semafovae_a10_r0_c0.1'
model_name = 'mnist_skiptaskvae_a10_r0'
model_name = 'mnist_skiptask2vae_a10_r0'
model_name = 'mnist_semafod_a10_r0_c0.1'

path = f'/home/trung/exp/fastexp/{model_name}/arrays'
if not os.path.exists(path):
  raise ValueError(f'path not found "{path}"')
with open(path, 'rb') as f:
  arrs = pickle.load(f)
### load arrays
z_train = arrs['z_train']
y_pred_train = arrs['y_pred_train']
y_true_train = arrs['y_true_train']

z_test = arrs['z_test']
y_pred_test = arrs['y_pred_test']
y_true_test = arrs['y_true_test']

z_train: List[Distribution]
y_pred_train: Distribution
z_test: List[Distribution]
y_pred_test: Distribution

labels = arrs['labels']
ds = arrs['ds']
label_type = arrs['label_type']

is_semi = isinstance(y_pred_train, Distribution)

n_train = y_true_train.shape[0]
n_test = y_true_test.shape[0]
zdim = [int(np.prod(z.event_shape)) for z in z_train]

######## load the arrays
print('z_train:', z_train)
print('z_test :', z_test)
print('y_train:')
print('  pred :', y_pred_train)
print('  true :', y_true_train.shape)
print('y_test:')
print('  pred :', y_pred_test)
print('  true :', y_true_test.shape)
print('Meta:')
print('  labels     :', labels)
print('  ds         :', ds)
print('  label_type :', label_type)

if is_semi:
  print('Train:', accuracy_score(np.argmax(y_pred_train.mean(), -1),
                                 np.argmax(y_true_train, -1)))
  print('Test :', accuracy_score(np.argmax(y_pred_test.mean(), -1),
                                 np.argmax(y_true_test, -1)))


# === Helper
def feat(zs: List[Distribution], py: Distribution) -> np.ndarray:
  x = np.concatenate([np.concatenate([z.mean(), z.stddev()], 1)
                      for z in zs], 1)
  if is_semi:
    y = np.concatenate([py.mean(), py.stddev()], 1)
    x = np.concatenate([x, y], 1)
  return x


######## Testing the semi-supervised
for m in (
    # LogisticRegression(max_iter=3000, random_state=seed),
    # LinearSVC(max_iter=3000, random_state=seed),
    SVC(C=1.5,
        kernel='rbf',
        max_iter=-1,
        probability=True,
        decision_function_shape='ovr',
        random_state=seed),
    # MLPClassifier(
    #     hidden_layer_sizes=[64, 64, 64],
    #     learning_rate='invscaling',
    #     early_stopping=True,  # helped when p > 0.2
    #     max_iter=3000,
    #     random_state=seed),
    # RandomForestClassifier(n_estimators=50, random_state=seed),
    # KNeighborsClassifier(n_neighbors=5),
):
  print(m.__class__.__name__)
  for p in [
    100,
    0.004,
    0.06,
    0.2,
    # 0.5,
    # 0.99,
  ]:
    print(' ****', p, '****')
    ids = np.arange(n_train)
    train_ids, valid_ids = train_test_split(
      ids,
      train_size=int(p * n_train) if p <= 1.0 else int(p),
      random_state=seed,
      shuffle=True,
      stratify=np.argmax(y_true_train, -1))
    x = feat(z_train, y_pred_train)
    x_train = x[train_ids]
    x_valid = x[valid_ids]
    y_train = np.argmax(y_true_train[train_ids], -1)
    y_valid = np.argmax(y_true_train[valid_ids], -1)
    x_test = feat(z_test, y_pred_test)
    y_test = np.argmax(y_true_test, -1)
    m.fit(x_train, y_train)
    print('  Train:', accuracy_score(y_true=y_train, y_pred=m.predict(x_train)))
    print('  Valid:', accuracy_score(y_true=y_valid, y_pred=m.predict(x_valid)))
    print('  Test :', accuracy_score(y_true=y_test, y_pred=m.predict(x_test)))
