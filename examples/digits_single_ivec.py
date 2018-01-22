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
from odin.utils import (get_logpath, get_modelpath, get_datasetpath,
                        Progbar, unique_labels, chain,
                        as_tuple_of_shape, stdio, ctext, ArgController,
                        unique_labels)

# ===========================================================================
# Input arguments
# ===========================================================================
args = ArgController(
).add('-task', '0-gender,1-dialect,2-digit', 0
).add('-nmix', "Number of GMM mixture", 512
).add('-tdim', "Dimension of t-matrix", 20
).add('--gmm', "Force re-run training GMM", False
).add('--stat', "Force re-extraction of centered statistics", False
).add('--tmat', "Force re-run training Tmatrix", False
).add('--ivec', "Force re-run extraction of i-vector", False
).add('--all', "Run all the system again, just a shortcut", False
).parse()
if args.all:
  args.gmm = True
  args.stat = True
  args.tmat = True
  args.ivec = True
# ===========================================================================
# path
# ===========================================================================
# path to preprocessed dataset
PATH = get_datasetpath('digits', override=False)

SAVE_PATH = os.path.join(
    os.path.expanduser('~'),
    'digit_ivec%d' % args.task
)
print("Save path:", ctext(SAVE_PATH, 'cyan'))
if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

GMM_PATH = os.path.join(SAVE_PATH, 'gmm')
TMAT_PATH = os.path.join(SAVE_PATH, 'tmat')

Z_PATH = (
    os.path.join(SAVE_PATH, 'Z_train'),
    os.path.join(SAVE_PATH, 'Z_valid'),
    os.path.join(SAVE_PATH, 'Z_test')
)
F_PATH = (
    os.path.join(SAVE_PATH, 'F_train'),
    os.path.join(SAVE_PATH, 'F_valid'),
    os.path.join(SAVE_PATH, 'F_test')
)
I_PATH = (
    os.path.join(SAVE_PATH, 'I_train'),
    os.path.join(SAVE_PATH, 'I_valid'),
    os.path.join(SAVE_PATH, 'I_test')
)
LABELS_PATH = (
    os.path.join(SAVE_PATH, 'L_train'),
    os.path.join(SAVE_PATH, 'L_valid'),
    os.path.join(SAVE_PATH, 'L_test')
)

LOG_PATH = get_logpath('digit_ivec.log', override=True)
# ===========================================================================
# Const
# ===========================================================================
FEAT = 'mspec'
ds = F.Dataset(PATH, read_only=True)
stdio(LOG_PATH)
# ====== GMM trainign ====== #
NMIX = 32
GMM_NITER = 8
GMM_DOWNSAMPLE = 4
GMM_DTYPE = 'float64'
# ====== IVEC training ====== #
TV_DIM = 20
TV_NITER = 16
TV_DTYPE = 'float64'

# ===========================================================================
# Helper
# ===========================================================================
def is_train(x):
  return x.split('_')[0] == 'train'

def extract_gender(x):
  return x.split('_')[1]

def extract_dialect(x):
  return x.split('_')[3]

def extract_spk(x):
  return x.split('_')[4]

def extract_digit(x):
  return x.split('_')[6]

if args.task == 0:
  extract_fn = extract_gender
elif args.task == 1:
  extract_fn = extract_dialect
elif args.task == 2:
  extract_fn = extract_digit

fn_label, labels = unique_labels(list(ds['indices'].keys()),
                                 key_func=extract_fn,
                                 return_labels=True)
print("Labels:", ctext(labels, 'cyan'))
# ===========================================================================
# Preparing data
# ===========================================================================
train_files = []
test_files = []
for name, (start, end) in ds['indices']:
  if is_train(name):
    train_files.append((name, (start, end)))
  else:
    test_files.append((name, (start, end)))
train_files, valid_files = train_valid_test_split(
    train_files, train=0.8,
    cluster_func=lambda x: extract_fn(x[0]),
    idfunc=lambda x: extract_spk(x[0]),
    inc_test=False,
    seed=5218)
# name for each dataset, useful for later
data_name = ['train', 'valid', 'test']
print("#Train:", len(train_files))
print("#Valid:", len(valid_files))
print("#Test:", len(test_files))
# ===========================================================================
# GMM
# ===========================================================================
if not os.path.exists(GMM_PATH) or args.gmm:
  gmm = ml.GMM(nmix=NMIX, nmix_start=1,
               niter=GMM_NITER,
               dtype=GMM_DTYPE,
               allow_rollback=False,
               exit_on_error=False,
               batch_size_cpu='auto', batch_size_gpu='auto',
               downsample=GMM_DOWNSAMPLE,
               stochastic_downsample=True,
               device='gpu',
               seed=5218, path=GMM_PATH)
  gmm.fit((ds[FEAT], train_files))
else:
  with open(GMM_PATH, 'rb') as f:
    gmm = pickle.load(f)
print(gmm)
# ===========================================================================
# Extract Zero and first order statistics
# ===========================================================================
stats = {}
y_true = {}
for files, z_path, f_path, l_path, name in zip(
        (train_files, valid_files, test_files),
        Z_PATH,
        F_PATH,
        LABELS_PATH,
        data_name):
  # extracting zeroth and first order statistics
  if not all(os.path.exists(i) for i in (z_path, f_path, l_path)) or\
      args.stat:
    print('========= Extracting statistics for: "%s" =========' % name)
    name_list = gmm.transform_to_disk(X=ds[FEAT], indices=files,
                                      pathZ=z_path, pathF=f_path,
                                      name_path=None,
                                      dtype=GMM_DTYPE,
                                      device='cpu',
                                      override=True)
    # save the labels
    labels = np.array([fn_label(i) for i in name_list], dtype='int32')
    np.savetxt(fname=l_path, X=labels, fmt='%d')
  # load the statistics in MmapData
  y_true[name] = np.genfromtxt(fname=l_path, dtype='int32')
  stats[name] = (F.MmapData(path=z_path, read_only=True),
                 F.MmapData(path=f_path, read_only=True))
# ====== print the stats ====== #
for name in data_name:
  print(stats[name])
# ===========================================================================
# Training T-matrix
# ===========================================================================
if not os.path.exists(TMAT_PATH) or args.tmat:
  tmat = ml.Ivector(tv_dim=TV_DIM, gmm=gmm,
                    niter=TV_NITER,
                    dtype=TV_DTYPE,
                    batch_size_cpu='auto', batch_size_gpu='auto',
                    device='mix', ncpu=1, gpu_factor=3,
                    path=TMAT_PATH)
  tmat.fit((stats['train'][0], # Z_train
            stats['train'][1])) # F_train
else:
  with open(TMAT_PATH, 'rb') as f:
    tmat = pickle.load(f)
print(tmat)
# ===========================================================================
# Extracting I-vectors
# ===========================================================================
ivecs = {}
for i_path, name in zip(I_PATH, data_name):
  if not os.path.exists(i_path) or args.ivec:
    print('========= Extracting ivecs for: "%s" =========' % name)
    tmat.transform_to_disk(path=i_path,
                           Z=stats[name][0],
                           F=stats[name][1],
                           name_path=None,
                           dtype=TV_DTYPE,
                           override=True)
  # load extracted ivec
  ivecs[name] = F.MmapData(i_path, read_only=True)
# ====== print the i-vectors ====== #
for name in data_name:
  print(ivecs[name])
# ===========================================================================
# Backend
# ===========================================================================
from sklearn.linear_model import LogisticRegression

if True:
  model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                             fit_intercept=True, intercept_scaling=1,
                             class_weight=None, random_state=None,
                             solver='liblinear', max_iter=100,
                             multi_class='ovr', verbose=0, warm_start=False,
                             n_jobs=1)
  X = ivecs['train'][:]
  model.fit(X, y_true['train'])
  y_valid = model.predict(ivecs['valid'][:])
  cm_valid = confusion_matrix(y_true['valid'], y_valid)
  print(V.print_confusion(cm_valid, labels=labels))

  y_test = model.predict(ivecs['test'][:])
  cm_test = confusion_matrix(y_true['test'], y_test)
  print(V.print_confusion(cm_test, labels=labels))
