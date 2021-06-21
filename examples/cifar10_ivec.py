from __future__ import print_function, division, absolute_import
import os
os.environ['ODIN'] = 'gpu,float32'
import shutil

import numpy as np
import tensorflow as tf

from odin import backend as K, nnet as N, visual as V, fuel as F
from odin.utils import minibatch, Progbar, get_exppath, crypto
from odin import ml

from sklearn.svm import SVC
from sklearn.metrics import classification_report

EXP_PATH = get_exppath('cifar10_ivec')
# ===========================================================================
# Load the dataset
# ===========================================================================
ds = F.CIFAR10.load()
print(ds)
X_train, y_train = ds['X_train'][:].reshape(-1, 3 * 32 * 32), ds['y_train'][:]
X_test, y_test = ds['X_test'][:].reshape(-1, 3 * 32 * 32), ds['y_test'][:]
# ====== normalize the data ====== #
X_train = X_train / 255.
X_test = X_test / 255.
print("Input:", X_train.shape, X_test.shape)
# ===========================================================================
# Training the GMM
# ===========================================================================
ivec = ml.Ivector(path=EXP_PATH, nmix=32, tv_dim=16,
                  niter_gmm=8, niter_tmat=8)
ivec.fit(X_train)
I_train = ivec.transform(X_train, save_ivecs=True, name='train')[:]
I_test = ivec.transform(X_test, save_ivecs=True, name='test')[:]
print(ivec)
# ===========================================================================
# Classifier
# ===========================================================================
svm = SVC()
svm.fit(I_train, y_train)
print(classification_report(y_true=y_test,
                            y_pred=svm.predict(I_test)))
