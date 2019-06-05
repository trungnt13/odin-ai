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
os.environ['ODIN'] = 'gpu,float32,seed=1234'
import shutil
import pickle

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score

from odin import backend as K, nnet as N, fuel as F, visual as V
from odin.stats import train_valid_test_split, freqcount, describe
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
).add('-nmix', "Number of GMM mixture", 128
).add('-tdim', "Dimension of t-matrix", 64
).add('-feat', "Acoustic feature: spec, mspec, mfcc", 'mfcc'
).add('--gmm', "Force re-run training GMM", False
).add('--stat', "Force re-extraction of centered statistics", False
).add('--tmat', "Force re-run training Tmatrix", False
).add('--ivec', "Force re-run extraction of i-vector", False
).add('--all', "Run all the system again, just a shortcut", False
).add('--acous', "Force re-run acoustic feature extraction", False
).parse()
args.gmm |= args.all
args.stat |= args.all | args.gmm
args.tmat |= args.all | args.stat
args.ivec |= args.all | args.tmat
FEAT = args.feat
# ===========================================================================
# Const
# ===========================================================================
EXP_DIR = get_exppath('FSDD')
PATH_ACOUSTIC_FEATURES = os.path.join(EXP_DIR, 'features')
# ====== GMM trainign ====== #
NMIX = args.nmix
GMM_NITER = 12
GMM_DOWNSAMPLE = 1
GMM_STOCHASTIC = True
GMM_DTYPE = 'float64'
# ====== IVEC training ====== #
TV_DIM = args.tdim
TV_NITER = 16
TV_DTYPE = 'float64'
# ===========================================================================
# Extract acoustic features
# ===========================================================================
# path to preprocessed dataset
all_files, meta = F.FSDD.load()
if not os.path.exists(PATH_ACOUSTIC_FEATURES) or \
len(os.listdir(PATH_ACOUSTIC_FEATURES)) != 14 or \
bool(args.acous):
  extractors = pp.make_pipeline(steps=[
      pp.speech.AudioReader(sr_new=8000, best_resample=True, remove_dc=True),
      pp.speech.PreEmphasis(coeff=0.97),
      pp.base.Converter(converter=lambda x: os.path.basename(x).split('.')[0],
                        input_name='path', output_name='name'),
      # ====== STFT ====== #
      pp.speech.STFTExtractor(frame_length=0.025, step_length=0.005,
                              n_fft=512, window='hamm', energy=False),
      # ====== spectrogram ====== #
      pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
      pp.speech.MelsSpecExtractor(n_mels=24, fmin=64, fmax=4000,
                                  input_name=('spec', 'sr'), output_name='mspec'),
      pp.speech.MFCCsExtractor(n_ceps=20,
                               remove_first_coef=True, first_coef_energy=True,
                               input_name='mspec', output_name='mfcc'),
      pp.base.DeltaExtractor(input_name='mfcc', order=(0, 1, 2)),
      # ====== SAD ====== #
      pp.base.RenameFeatures(input_name='mfcc_energy', output_name='energy'),
      pp.speech.SADthreshold(energy_threshold=0.55, smooth_window=5,
                             input_name='energy', output_name='sad'),
      # ====== normalization ====== #
      pp.base.DeleteFeatures(input_name=('stft', 'spec', 'sad_threshold')),
      pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                             input_name=('mspec', 'mfcc')),
      # ====== post processing ====== #
      pp.base.AsType(dtype='float16'),
  ], debug=False)
  with np.warnings.catch_warnings():
    np.warnings.filterwarnings('ignore')
    processor = pp.FeatureProcessor(
        jobs=all_files,
        path=PATH_ACOUSTIC_FEATURES,
        extractor=extractors,
        n_cache=120,
        ncpu=None,
        override=True,
        identifier='name',
        log_path=os.path.join(EXP_DIR, 'processor.log'),
        stop_on_failure=True)
    processor.run()
    # pp.validate_features(processor,
    #                      nb_samples=12,
    #                      path=os.path.join(EXP_DIR, 'feature_validation'),
    #                      override=True)
ds = F.Dataset(PATH_ACOUSTIC_FEATURES, read_only=True)
print(ds)
indices = list(ds['indices_%s' % args.feat].items())
print("Utterances length:")
print("   ", describe([end - start for name, (start, end) in indices], shorten=True))
# ===========================================================================
# Basic path for GMM, T-matrix and I-vector
# ===========================================================================
EXP_DIR = os.path.join(EXP_DIR, '%s_%d_%d' % (FEAT, NMIX, TV_DIM))
LOG_PATH = get_logpath(name='log.txt', override=False, root=EXP_DIR, odin_base=False)
stdio(LOG_PATH)
print("Exp-dir:", ctext(EXP_DIR, 'cyan'))
print("Log path:", ctext(LOG_PATH, 'cyan'))
# ====== ivec path ====== #
GMM_PATH = os.path.join(EXP_DIR, 'gmm')
TMAT_PATH = os.path.join(EXP_DIR, 'tmat')
# zero order statistics
Z_PATH = (
    os.path.join(EXP_DIR, 'Z_train'),
    os.path.join(EXP_DIR, 'Z_test'))
# first order statistics
F_PATH = (
    os.path.join(EXP_DIR, 'F_train'),
    os.path.join(EXP_DIR, 'F_test'))
# i-vector path
I_PATH = (
    os.path.join(EXP_DIR, 'I_train'),
    os.path.join(EXP_DIR, 'I_test'))
# labels
L_PATH = ( # labels
    os.path.join(EXP_DIR, 'L_train'),
    os.path.join(EXP_DIR, 'L_test'))
# ===========================================================================
# Helper
# ===========================================================================
# jackson speaker for testing, all other speaker for training
def is_train(x):
  return x.split('_')[1] != 'jackson'

def extract_digit(x):
  return x.split('_')[0]

fn_extract = extract_digit
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
# GMM
# ===========================================================================
if not os.path.exists(GMM_PATH) or args.gmm:
  gmm = ml.GMM(nmix=NMIX, nmix_start=1,
               niter=GMM_NITER, dtype=GMM_DTYPE,
               allow_rollback=True, exit_on_error=True,
               batch_size_cpu=2048, batch_size_gpu=2048,
               downsample=GMM_DOWNSAMPLE,
               stochastic_downsample=GMM_STOCHASTIC,
               device='gpu',
               seed=1234, path=GMM_PATH)
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
for name, files, z_path, f_path, l_path in zip(
        data_name,
        (train_files, test_files),
        Z_PATH, F_PATH, L_PATH):
  # extracting zeroth and first order statistics
  if not all(os.path.exists(i) for i in (z_path, f_path, l_path)) or\
      args.stat:
    print('========= Extracting statistics for: "%s" =========' % name)
    gmm.transform_to_disk(X=ds[FEAT], indices=files,
                          pathZ=z_path, pathF=f_path, name_path=l_path,
                          dtype='float32', device='cpu', ncpu=None,
                          override=True)
  # load the statistics in MmapData
  y_true[name] = [fn_label(i) for i in np.genfromtxt(fname=l_path, dtype=str)]
  stats[name] = (F.MmapData(path=z_path, read_only=True),
                 F.MmapData(path=f_path, read_only=True))
for name, x in stats.items():
  print(ctext(name + ':', 'cyan'), x)
# ===========================================================================
# Training T-matrix
# ===========================================================================
if not os.path.exists(TMAT_PATH) or args.tmat:
  tmat = ml.Tmatrix(tv_dim=TV_DIM, gmm=gmm,
                    niter=TV_NITER, dtype=TV_DTYPE,
                    batch_size_cpu='auto', batch_size_gpu='auto',
                    device='gpu', ncpu=1, gpu_factor=3,
                    path=TMAT_PATH)
  tmat.fit(X=(stats['train'][0], # Z_train
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
    z, f = stats[name]
    tmat.transform_to_disk(path=i_path, Z=z, F=f,
                           dtype='float32', device='gpu', ncpu=1,
                           override=True)
  # load extracted ivec
  ivecs[name] = F.MmapData(i_path, read_only=True)
# ====== print the i-vectors ====== #
for name in data_name:
  print('========= %s =========' % name)
  print(ctext('i-vectors:', 'cyan'))
  print(ctext(' *', 'yellow'), ivecs[name])
  print(ctext('z-stats:', 'cyan'))
  print(ctext(' *', 'yellow'), stats[name][0])
  print(ctext('f-stats:', 'cyan'))
  print(ctext(' *', 'yellow'), stats[name][1])
  print(ctext('labels:', 'cyan'))
  print(ctext(' *', 'yellow'), len(y_true[name]))
# ==================== turn off all annoying warning ==================== #
with np.warnings.catch_warnings():
  np.warnings.filterwarnings('ignore')
  # ===========================================================================
  # I-vector
  # ===========================================================================
  X_train = ivecs['train']
  X_test = ivecs['test']
  # ====== cosine scoring ====== #
  print(ctext("==== '%s'" % "Ivec cosine-scoring", 'cyan'))
  scorer = ml.Scorer(centering=True, wccn=True, lda=True, method='cosine')
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== GMM scoring ====== #
  print(ctext("==== '%s'" % "Ivec GMM-scoring-ova", 'cyan'))
  scorer = ml.GMMclassifier(strategy="ova",
                            n_components=3, covariance_type='full',
                            centering=True, wccn=True, unit_length=True,
                            lda=False, concat=False)
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== GMM scoring ====== #
  print(ctext("==== '%s'" % "Ivec GMM-scoring-all", 'cyan'))
  scorer = ml.GMMclassifier(strategy="all", covariance_type='full',
                            centering=True, wccn=True, unit_length=True,
                            lda=False, concat=False)
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== plda scoring ====== #
  print(ctext("==== '%s'" % "Ivec PLDA-scoring", 'cyan'))
  scorer = ml.PLDA(n_phi=TV_DIM // 2, n_iter=12,
                   centering=True, wccn=True, unit_length=True,
                   random_state=1234)
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== svm scoring ====== #
  print(ctext("==== '%s'" % "Ivec SVM-scoring", 'cyan'))
  scorer = ml.Scorer(wccn=True, lda=True, method='svm')
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ===========================================================================
  # Super-vector
  # ===========================================================================
  X_train = stats['train'][1]
  X_test = stats['test'][1]
  X_train, X_test = ml.fast_pca(X_train, X_test, n_components=args.tdim,
                                algo='ppca', random_state=1234)
  # ====== GMM scoring ====== #
  print(ctext("==== '%s'" % "Super-Vector GMM-scoring-ova", 'cyan'))
  scorer = ml.GMMclassifier(strategy="ova",
                            n_components=3, covariance_type='full',
                            centering=True, wccn=True, unit_length=True,
                            lda=False, concat=False)
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== plda scoring ====== #
  print(ctext("==== '%s'" % "Super-Vector PLDA-scoring", 'cyan'))
  scorer = ml.PLDA(n_phi=TV_DIM // 2, n_iter=12,
                   centering=True, wccn=True, unit_length=True,
                   random_state=1234)
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
  # ====== svm scoring ====== #
  print(ctext("==== '%s'" % "Super-Vector SVM-scoring", 'cyan'))
  scorer = ml.Scorer(wccn=True, lda=True, method='svm')
  scorer.fit(X=X_train, y=y_true['train'])
  scorer.evaluate(X_test, y_true['test'], labels=labels)
