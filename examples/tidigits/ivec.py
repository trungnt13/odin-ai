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
from odin.utils import (Progbar, unique_labels, chain, get_formatted_datetime,
                        as_tuple_of_shape, stdio, ctext, ArgController)

from utils import prepare_data, get_exp_path
# ===========================================================================
# Input arguments
# ===========================================================================
args = ArgController(
).add('-nmix', "Number of GMM mixture", 256
).add('-tdim', "Dimension of t-matrix", 128
).add('-feat', "Acoustic feature: spec, mspec, mfcc, bnf, sdc", 'bnf'
).add('-task', 'gender, age, dialect, speaker, digit', 'gender'
).add('--retrain', "deleted trained model, and re-train everything", False
).parse()
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
EXP_DIR, MODEL_PATH, LOG_PATH = get_exp_path('ivec', args, override=args.retrain)
stdio(LOG_PATH)
X, train, y_train, test, y_test, labels = prepare_data(
    feat=args.feat, label=args.task, for_ivec=True)
# ===========================================================================
# Training Ivector
# ===========================================================================
ivec = ml.Ivector(path=EXP_DIR, nmix=NMIX, tv_dim=TV_DIM,
                  nmix_start=1, niter_gmm=GMM_NITER, niter_tmat=TV_NITER,
                  allow_rollback=True, exit_on_error=False,
                  downsample=GMM_DOWNSAMPLE, stochastic_downsample=GMM_STOCHASTIC,
                  device='gpu', ncpu=1, gpu_factor_gmm=80, gpu_factor_tmat=3,
                  dtype=DTYPE, seed=5218, name=None)
ivec.fit(X=X, indices=train,
         extract_ivecs=True, keep_stats=True)
ivec.transform(X=X, indices=test,
               save_ivecs=True, keep_stats=True, name='test')
# ====== i-vector ====== #
I_train = F.MmapData(path=ivec.get_i_path(None), read_only=True)
F_train = F.MmapData(path=ivec.get_f_path(None), read_only=True)
name_train = np.genfromtxt(ivec.get_name_path(None), dtype=str)

I_test = F.MmapData(path=ivec.get_i_path(name='test'), read_only=True)
F_test = F.MmapData(path=ivec.get_f_path(name='test'), read_only=True)
name_test = np.genfromtxt(ivec.get_name_path(name='test'), dtype=str)
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
