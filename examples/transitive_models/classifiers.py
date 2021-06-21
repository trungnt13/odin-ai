# This code try to find the best classifier for the SemafoVAE latents
import os
import numpy as np
from typing import List, Optional
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
from odin import visual as vs
from matplotlib import pyplot as plt
import seaborn as sns

seed = 1
sns.set()
np.random.seed(seed)
tf.random.set_seed(seed)

# with open(f'{expdir}/results.txt', 'w') as f:
#   f.write(f'Steps: {vae.step.numpy()}\n')
#   f.write(f'[LLK] Train:\n')
#   f.write(f' X: {llkx_train:.4f}\n')
#   f.write(f' Y: {llky_train:.4f}\n')
#   f.write(f'[LLK] Test :\n')
#   f.write(f' X: {llkx_test:.4f}\n')
#   f.write(f' Y: {llky_test:.4f}\n')
#   for p in [0.004, 0.06]:
#     x_train, x_valid, y_train, y_valid = train_test_split(
#       z_mean_train,
#       np.argmax(y_true_train, axis=-1),
#       train_size=p,
#       random_state=1)
#     x_test = z_mean_test
#     y_test = np.argmax(y_true_test, axis=-1)
#     n_dims = min(x_train.shape[0], x_train.shape[1], 512)
#     m = Pipeline([
#       ('zscore', StandardScaler()),
#       ('pca', PCA(n_dims, random_state=seed)),
#       ('svc', SVC(kernel='rbf', max_iter=3000, random_state=seed))])
#     m.fit(x_train, y_train)
#     # write the report
#     f.write(f'{m.__class__.__name__} Number of labels: '
#             f'{p} {x_train.shape[0]}/{x_valid.shape[0]}')
#     f.write('\nTraining:\n')
#     f.write(classification_report(y_train, m.predict(x_train)))
#     f.write('\nValidation:\n')
#     f.write(classification_report(y_valid, m.predict(x_valid)))
#     f.write('\nTest:\n')
#     f.write(classification_report(y_test, m.predict(x_test)))
#     f.write('------------\n')
#   print(f'Exported results to "{expdir}/results.txt"')


# ===========================================================================
# Helper
# ===========================================================================
def flatten(x: np.ndarray) -> np.ndarray:
  if len(x.shape) > 2:
    x = np.reshape(x, (x.shape[0], -1))
  return x


def features(zs: List[Distribution],
             py: Optional[Distribution] = None) -> np.ndarray:
  x = np.concatenate(
    [np.concatenate([flatten(z.mean()), flatten(z.stddev())], 1)
     for z in zs], 1)
  if isinstance(py, Distribution):
    y = np.concatenate([flatten(py.mean()), flatten(py.stddev())], 1)
    x = np.concatenate([x, y], 1)
  return x


######## load the arrays
# model_name = 'mnist_skiptaskvae_a10_r0'
# model_name = 'mnist_skiptask2vae_a10_r0'
# model_name = 'mnist_semafod_a10_r0_c0.1'
# model_name = 'mnist_semafovae_a10_r0_c0.1'
# model_name = 'mnist_variationalautoencoder'


def run_classification(model_name):
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
      x = features(z_train, y_pred_train)
      x_train = x[train_ids]
      x_valid = x[valid_ids]
      y_train = np.argmax(y_true_train[train_ids], -1)
      y_valid = np.argmax(y_true_train[valid_ids], -1)
      x_test = features(z_test, y_pred_test)
      y_test = np.argmax(y_true_test, -1)
      m.fit(x_train, y_train)
      print('  Train:',
            accuracy_score(y_true=y_train, y_pred=m.predict(x_train)))
      print('  Valid:',
            accuracy_score(y_true=y_valid, y_pred=m.predict(x_valid)))
      print('  Test :', accuracy_score(y_true=y_test, y_pred=m.predict(x_test)))


# run_classification('mnist_variationalautoencoder')
# run_classification('mnist_skiptaskvae_a10_r0')
# run_classification('mnist_hierarchicalvae')
run_classification('cifar10_hierarchicalvae')
