# ===========================================================================
# Using TIDIGITS dataset to predict gender (Boy, Girl, Woman, Man)
# ===========================================================================
# Saved WAV file format:
#     0) [train|test]
#     1) [m|w|b|g] (alias for man, women, boy, girl)
#     2) [age]
#     3) [dialectID]
#     4) [speakerID]
#     5) [production]
#     6) [digit_sequence]
#     => "train_g_08_17_as_a_4291815"
# ===========================================================================
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'gpu,float32,seed=12082518'
import shutil
import pickle

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score

from odin import backend as K, nnet as N, fuel as F, visual as V
from odin.stats import train_valid_test_split, freqcount
from odin import ml
from odin import training
from odin import preprocessing as pp
from odin.visual import print_dist, print_confusion, print_hist
from odin.utils import (get_logpath, get_modelpath, get_datasetpath, get_exppath,
                        Progbar, unique_labels, chain, get_formatted_datetime,
                        as_tuple_of_shape, stdio, ctext, ArgController)

# ===========================================================================
# Input arguments
# ===========================================================================
args = ArgController(
).add('-nmix', "Number of GMM mixture", 256
).add('-tdim', "Dimension of t-matrix", 128
).add('-feat', "Acoustic feature: spec, mspec, mfcc, bnf, sdc", 'mspec'
).add('--gmm', "Force re-run training GMM", False
).add('--stat', "Force re-extraction of centered statistics", False
).add('--tmat', "Force re-run training Tmatrix", False
).add('--ivec', "Force re-run extraction of i-vector", False
).add('--all', "Run all the system again, just a shortcut", False
).parse()
args.gmm |= args.all
args.stat |= args.all | args.gmm
args.tmat |= args.all | args.stat
args.ivec |= args.all | args.tmat
FEAT = args.feat
# ===========================================================================
# Const
# ===========================================================================
DTYPE = 'float64'
# ====== GMM trainign ====== #
NMIX = args.nmix
GMM_NITER = 10
GMM_DOWNSAMPLE = 4
GMM_STOCHASTIC = True
# ====== IVEC training ====== #
TV_DIM = args.tdim
TV_NITER = 10
# ===========================================================================
# path and dataset
# ===========================================================================
# path to preprocessed dataset
ds = F.TIDIGITS_feat.load()
assert FEAT in ds, "Cannot find feature with name: %s" % FEAT
indices = list(ds['indices'].items())
# ====== general path ====== #
EXP_DIR = get_exppath(tag='TIDIGITS_ivec',
                      name='%s_%d_%d' % (FEAT, args.nmix, args.tdim))
if not os.path.exists(EXP_DIR):
  os.mkdir(EXP_DIR)
# ====== start logging ====== #
LOG_PATH = os.path.join(EXP_DIR,
                        'log_%s.txt' % get_formatted_datetime(only_number=True))
stdio(LOG_PATH)
print("Exp-dir:", ctext(EXP_DIR, 'cyan'))
print("Log path:", ctext(LOG_PATH, 'cyan'))
# ===========================================================================
# Helper
# ===========================================================================
def is_train(x):
  return x.split('_')[0] == 'train'

def extract_gender(x):
  return x.split('_')[1]

print("Task:", ctext("gender", 'cyan'))
fn_extract = extract_gender
fn_label, labels = unique_labels([i[0] for i in indices],
                                 key_func=fn_extract,
                                 return_labels=True)
print("Labels:", ctext(labels, 'cyan'))
# ===========================================================================
# Preparing data
# ===========================================================================
train_files = [] # (name, (start, end)) ...
test_files = []
for name, (start, end) in indices:
  if is_train(name):
    train_files.append((name, (start, end)))
  else:
    test_files.append((name, (start, end)))
# name for each dataset, useful for later
data_name = ['train', 'test']
print("#Train:", len(train_files))
print("#Test:", len(test_files))
# ===========================================================================
# Training Ivector
# ===========================================================================
ivec = ml.Ivector(path=EXP_DIR, nmix=NMIX, tv_dim=TV_DIM,
                  nmix_start=1, niter_gmm=GMM_NITER, niter_tmat=TV_NITER,
                  allow_rollback=True, exit_on_error=False,
                  downsample=GMM_DOWNSAMPLE, stochastic_downsample=GMM_STOCHASTIC,
                  device='gpu', ncpu=1, gpu_factor_gmm=80, gpu_factor_tmat=3,
                  dtype=DTYPE, seed=5218, name=None)
ivec.fit(X=ds[FEAT], indices=train_files,
         extract_ivecs=True, keep_stats=True)
ivec.transform(X=ds[FEAT], indices=test_files,
               save_ivecs=True, keep_stats=True, name='test')
# ====== i-vector ====== #
I_train = F.MmapData(path=ivec.get_i_path(None), read_only=True)
F_train = F.MmapData(path=ivec.get_f_path(None), read_only=True)
name_train = np.genfromtxt(ivec.get_name_path(None), dtype=str)
y_train = [fn_label(i) for i in name_train]

I_test = F.MmapData(path=ivec.get_i_path(name='test'), read_only=True)
F_test = F.MmapData(path=ivec.get_f_path(name='test'), read_only=True)
name_test = np.genfromtxt(ivec.get_name_path(name='test'), dtype=str)
y_test = [fn_label(i) for i in name_test]
# ===========================================================================
# I-vector
# ===========================================================================
X_train = I_train
X_test = I_test
# ====== cosine scoring ====== #
print(ctext("==== '%s'" % "Ivec cosine-scoring", 'cyan'))
scorer = ml.Scorer(centering=True, wccn=True, lda=True, method='cosine')
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== GMM scoring ====== #
print(ctext("==== '%s'" % "Ivec GMM-scoring-ova", 'cyan'))
scorer = ml.GMMclassifier(strategy="ova",
                          n_components=3, covariance_type='full',
                          centering=True, wccn=True, unit_length=True,
                          lda=False, concat=False)
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== GMM scoring ====== #
print(ctext("==== '%s'" % "Ivec GMM-scoring-all", 'cyan'))
scorer = ml.GMMclassifier(strategy="all", covariance_type='full',
                          centering=True, wccn=True, unit_length=True,
                          lda=False, concat=False)
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== plda scoring ====== #
print(ctext("==== '%s'" % "Ivec PLDA-scoring", 'cyan'))
scorer = ml.PLDA(n_phi=100, n_iter=12,
                 centering=True, wccn=True, unit_length=True,
                 random_state=5218)
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== svm scoring ====== #
print(ctext("==== '%s'" % "Ivec SVM-scoring", 'cyan'))
scorer = ml.Scorer(wccn=True, lda=True, method='svm')
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ===========================================================================
# Super-vector
# ===========================================================================
X_train = F_train
X_test = F_test
X_train, X_test = ml.fast_pca(X_train, X_test, n_components=args.tdim,
                              algo='ppca', random_state=5218)
# ====== GMM scoring ====== #
print(ctext("==== '%s'" % "Super-Vector GMM-scoring-ova", 'cyan'))
scorer = ml.GMMclassifier(strategy="ova",
                          n_components=3, covariance_type='full',
                          centering=True, wccn=True, unit_length=True,
                          lda=False, concat=False)
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== plda scoring ====== #
print(ctext("==== '%s'" % "Super-Vector PLDA-scoring", 'cyan'))
scorer = ml.PLDA(n_phi=100, n_iter=12,
                 centering=True, wccn=True, unit_length=True,
                 random_state=5218)
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
# ====== svm scoring ====== #
print(ctext("==== '%s'" % "Super-Vector SVM-scoring", 'cyan'))
scorer = ml.Scorer(wccn=True, lda=True, method='svm')
scorer.fit(X=X_train, y=y_train)
scorer.evaluate(X_test, y_test, labels=labels)
