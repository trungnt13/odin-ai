# This code try to find the best classifier for the SemafoVAE latents
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

seed = 1

######## load the arrays
basedir = '/home/trung/exp/hyperparams'
dsdir = 'mnist_0.004'
expdir = 'semafovae_linear_0.2_0.02_20000_0_0_lin'
path = f'{basedir}/{dsdir}/{expdir}/analysis/arrays.npz'
if not os.path.exists(path):
  raise ValueError(f'path not found "{path}"')

### load arrays
arrs = np.load(path)
x_train = arrs['x_train']
y_train = arrs['y_train']
x_test = arrs['x_test']
y_test = arrs['y_test']

zdim = arrs['zdim']
labels = arrs['labels']

z_train = x_train[:, :zdim]
y_pred_train = x_train[:, zdim:]
z_test = x_test[:, :zdim]
y_pred_test = x_test[:, zdim:]

######## load the arrays
print('X_train:', x_train.shape)
print('z_train:', z_train.shape)
print('y_train:', y_train.shape)
print('X_test :', x_test.shape)
print('z_test :', z_test.shape)
print('y_test :', y_test.shape)
print('zdim   :', zdim)
print('labels :', labels)

if (x_train.shape[-1] - zdim) == len(np.unique(y_test)):
  print('Train:', accuracy_score(np.argmax(x_train[:, zdim:], -1), y_train))
  print('Test :', accuracy_score(np.argmax(x_test[:, zdim:], -1), y_test))


######## Testing the semi-supervised
def fusion(ypred1, ypred2):
  ypred = (ypred1 + ypred2) / 2.0
  return np.argmax(ypred, axis=-1)


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
      0.004,
      0.06,
      0.2,
      # 0.5,
      # 0.99,
  ]:
    print(' ****', p, '****')
    x1, x2, y1, y2 = train_test_split(x_train,
                                      y_train,
                                      train_size=int(p * x_train.shape[0]),
                                      random_state=seed,
                                      shuffle=True)
    # y_pred1 = x1[:, zdim:]
    # y_pred2 = x2[:, zdim:]
    # x1 = x1[:, :zdim]
    # x2 = x2[:, :zdim]
    m.fit(x1, y1)
    print('  Valid       :', accuracy_score(y_true=y2, y_pred=m.predict(x2)))
    # print('  Valid Pred  :',
    #       accuracy_score(y_true=y2, y_pred=np.argmax(y_pred2, axis=1)))
    # print(
    #     '  Valid Fusion:',
    #     accuracy_score(y_true=y2, y_pred=fusion(y_pred2, m.predict_proba(x2))))
    print('  Test        :',
          accuracy_score(y_true=y_test, y_pred=m.predict(x_test)))
    # print(
    #     '  Test  Fusion:',
    #     accuracy_score(y_true=y_test,
    #                    y_pred=fusion(y_pred_test, m.predict_proba(z_test))))
