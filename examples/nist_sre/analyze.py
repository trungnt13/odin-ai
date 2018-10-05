from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
import shutil
from collections import defaultdict

import numpy as np
import tensorflow as tf

from odin import fuel as F
from odin import nnet as N, backend as K
from odin.utils import ctext, mpi, Progbar, catch_warnings_ignore

from sklearn.metrics import accuracy_score, log_loss, f1_score

from helpers import (FEATURE_RECIPE, FEATURE_NAME, PATH_ACOUSTIC_FEATURES,
                     MINIMUM_UTT_DURATION,
                     filter_utterances, prepare_dnn_data)

BASE_DIR = '/home/trung/exp/sre/xvec_mfccmusanrirs_mfcc_4_fisher'
assert FEATURE_RECIPE.replace('_', '') in os.path.basename(BASE_DIR)
assert FEATURE_NAME in os.path.basename(BASE_DIR)
assert str(MINIMUM_UTT_DURATION) in os.path.basename(BASE_DIR)
MODEL = 'model.ai.0'
# ===========================================================================
# Load the data
# ===========================================================================
train, valid, all_speakers, ds = prepare_dnn_data(save_dir=BASE_DIR,
                                                  return_dataset=True)
print(ds)

label2spk = {i: spk for i, spk in enumerate(all_speakers)}
labels = np.arange(len(all_speakers))

X = ds[FEATURE_NAME]
indices = ds['indices_%s' % FEATURE_NAME]
spkid = ds['spkid']
# ===========================================================================
# Load the model
# ===========================================================================
# ====== load the network ====== #
x_vec = N.deserialize(path=os.path.join(BASE_DIR, MODEL),
                      force_restore_vars=True)
# ====== get output tensors ====== #
y_logit = x_vec()
y_proba = tf.nn.softmax(y_logit)
X = K.ComputationGraph(y_proba).placeholders[0]
z = K.ComputationGraph(y_proba).get(roles=N.Dense, scope='LatentOutput',
                                    beginning_scope=False)[0]
f_prob = K.function(inputs=X, outputs=y_proba, training=False)
f_z = K.function(inputs=X, outputs=z, training=False)
print('Inputs:', ctext(X, 'cyan'))
print('Predic:', ctext(y_proba, 'cyan'))
print('Latent:', ctext(z, 'cyan'))
# ===========================================================================
# Helper
# ===========================================================================
def evaluate_prediction(name_list, y_pred, y_true, title):
  def _report(y_, y, pad=''):
    with catch_warnings_ignore(Warning):
      print(pad, "#Samples:", ctext(len(y), 'cyan'))
      print(pad, "Log loss:", log_loss(y_true=y, y_pred=y_, labels=labels))
      print(pad, "Accuracy:", accuracy_score(y_true=y, y_pred=np.argmax(y_, axis=-1)))
      print(pad, "F1 score:", f1_score(y_true=y, y_pred=np.argmax(y_, axis=-1),
                                  labels=labels, average='macro'))

  datasets_2_samples = defaultdict(list)
  for name, y_, y in zip(name_list, y_pred, y_true):
    dsname = ds['dsname'][name]
    datasets_2_samples[dsname].append((name, y_, y))

  y_pred = np.concatenate(y_pred, axis=0)
  y_true = np.array(y_true)
  print('=' * 12, ctext(title, 'lightyellow'), '=' * 12)
  _report(y_=y_pred, y=y_true)

  for dsname, data in sorted(datasets_2_samples.items(),
                             key=lambda x: x[0]):
    print(ctext(dsname, 'lightcyan'), ':')
    name_list = np.array([i[0] for i in data])
    y_pred = np.concatenate([i[1] for i in data], axis=0)
    y_true = np.array([i[2] for i in data])
    _report(y_=y_pred, y=y_true, pad='  ')

# ===========================================================================
# make prediction
# ===========================================================================
def make_prediction(feeder, title):
  prog = Progbar(target=len(feeder), name=title)
  name_list = []
  y_pred = []
  y_true = []
  for name, idx, X, y in feeder.set_batch(batch_size=100000,
                                         batch_mode='file',
                                         seed=None, shuffle_level=0):
    name_list.append(name)

    y = np.argmax(y, axis=-1)
    assert len(np.unique(y)) == 1, name
    spk = label2spk[y[0]]
    assert spkid[name] == spk, name
    y_true.append(y[0])

    y_ = f_prob(X)
    if y_.shape[0] > 1:
      y_ = np.mean(y_, axis=0, keepdims=True)
    y_pred.append(y_)

    prog.add(X.shape[0])
  evaluate_prediction(name_list, y_pred, y_true, title=title)
# ====== do it ====== #
make_prediction(train, title="Train Data")
make_prediction(valid, title="Valid Data")
