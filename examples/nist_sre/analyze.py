from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'
import shutil
from collections import defaultdict

import numpy as np
import tensorflow as tf

from odin import fuel as F
from odin import nnet as N, backend as K
from odin.utils import (ctext, mpi, Progbar, catch_warnings_ignore, stdio,
                        get_logpath)

from sklearn.metrics import accuracy_score, log_loss, f1_score

from helpers import (FEATURE_RECIPE, FEATURE_NAME, PATH_ACOUSTIC_FEATURES,
                     MINIMUM_UTT_DURATION, ANALYSIS_DIR, EXP_DIR,
                     filter_utterances, prepare_dnn_data)

MODEL_ID = 'xvec_mfccmusanrirs.mfcc.5_pad_5_8.fisher_swb_voxceleb1_voxceleb2'
MODEL_ID = 'xvec_mfccmusanrirs.mfcc.5_pad_5_8.fisher_voxceleb1_voxceleb2'
MODEL_ID = 'xvec_mfccmusanrirs.mfcc.5_pad_5_8.fisher_sre10_swb_voxceleb1_voxceleb2'

info = MODEL_ID.split('.')
feat_name = info[1]
utt_length, seq_mode, min_dur, min_utt = info[2].split('_')
exclude_datasets = info[-1].split('_')
# ====== base dir ====== #
BASE_DIR = os.path.join(EXP_DIR, MODEL_ID)
assert FEATURE_RECIPE.replace('_', '') in os.path.basename(BASE_DIR)
assert FEATURE_NAME in os.path.basename(BASE_DIR)
# ====== get the last model ====== #
all_model = sorted([name
                    for name in os.listdir(BASE_DIR)
                    if 'model.ai.' in name],
                   key=lambda x: int(x.split('.')[-1]))
assert len(all_model) > 0, "Cannot find any model.ai. at path: %s" % BASE_DIR
MODEL = os.path.join(BASE_DIR, all_model[-1])
# ====== prepare log ====== #
stdio(get_logpath(name="analyze.log", increasing=True,
                  odin_base=False, root=ANALYSIS_DIR))
print(ctext(BASE_DIR, 'lightyellow'))
print(ctext(MODEL, 'lightyellow'))
print("Feature name:", ctext(feat_name, 'lightyellow'))
print("Utt length  :", ctext(utt_length, 'lightyellow'))
print("Seq mode    :", ctext(seq_mode, 'lightyellow'))
print("Min Duration:", ctext(min_dur, 'lightyellow'))
print("Min #Utt    :", ctext(min_utt, 'lightyellow'))
print("Excluded    :", ctext(exclude_datasets, 'lightyellow'))
# ===========================================================================
# Load the data
# ===========================================================================
train, valid, all_speakers, ds = prepare_dnn_data(save_dir=BASE_DIR,
    feat_name=feat_name, utt_length=int(utt_length), seq_mode=str(seq_mode),
    min_dur=int(min_dur), min_utt=int(min_utt),
    exclude=exclude_datasets, train_proportion=0.5,
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
x_vec = N.deserialize(path=MODEL, force_restore_vars=True)
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
  def _report(y_p, y_t, pad=''):
    with catch_warnings_ignore(Warning):
      z_ = np.concatenate(y_p, axis=0)
      z = np.concatenate(y_t, axis=0)
      print(pad, '*** %s ***' % ctext('Frame-level', 'lightcyan'))
      print(pad, "#Samples:", ctext(len(z), 'cyan'))
      print(pad, "Log loss:", log_loss(y_true=z, y_pred=z_, labels=labels))
      print(pad, "Accuracy:", accuracy_score(y_true=z, y_pred=np.argmax(z_, axis=-1)))

      z_ = np.concatenate([np.mean(i, axis=0, keepdims=True) for i in y_p],
                          axis=0)
      z = np.array([i[0] for i in y_t])
      print(pad, '*** %s ***' % ctext('Utterance-level', 'lightcyan'))
      print(pad, "#Samples:", ctext(len(z), 'cyan'))
      print(pad, "Log loss:", log_loss(y_true=z, y_pred=z_, labels=labels))
      print(pad, "Accuracy:", accuracy_score(y_true=z, y_pred=np.argmax(z_, axis=-1)))

  datasets_2_samples = defaultdict(list)
  for name, y_p, y_t in zip(name_list, y_pred, y_true):
    dsname = ds['dsname'][name]
    datasets_2_samples[dsname].append((name, y_p, y_t))

  print('=' * 12, ctext(title, 'lightyellow'), '=' * 12)
  _report(y_p=y_pred, y_t=y_true)

  for dsname, data in sorted(datasets_2_samples.items(),
                             key=lambda x: x[0]):
    print(ctext(dsname, 'yellow'), ':')
    y_pred = [i[1] for i in data]
    y_true = [i[2] for i in data]
    _report(y_p=y_pred, y_t=y_true, pad='  ')
# ===========================================================================
# make prediction
# ===========================================================================
def make_prediction(feeder, title):
  prog = Progbar(target=len(feeder), print_summary=True, name=title)
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
    y_true.append(y)

    y_ = f_prob(X)
    y_pred.append(y_)

    assert len(y) == len(y_)
    prog.add(X.shape[0])
  evaluate_prediction(name_list, y_pred, y_true, title=title)
# ====== do it ====== #
make_prediction(train, title="Train Data")
make_prediction(valid, title="Valid Data")
