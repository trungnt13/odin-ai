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

import os
import shutil
from collections import defaultdict

import numpy as np

from odin import fuel as F, backend as K, visual as V
from odin.utils import (cache_disk, ctext, unique_labels,
                        Progbar, batching, get_exppath,
                        get_formatted_datetime)
from odin.stats import train_valid_test_split, sampling_iter

_support_label = {
    'other': 0,
    'gender': 1,
    'age': 2,
    'dialect': 3,
    'speaker': 4,
    'production': 5,
    'digit': 6,
}

# ===========================================================================
# Const for path and features configuration
# ===========================================================================
PATH_EXP = get_exppath(tag='TIDIGITS', override=False)
# ====== acoustic feature extraction ====== #
PATH_ACOUSTIC = os.path.join(PATH_EXP, 'TIDIGITS_feat')

class FeatureConfigs(object):
  padding = False
  sr = 8000
  window = 'hamm'
  n_fft = 512
  n_mels = 40
  n_ceps = 40
  fmin = 100
  fmax = 4000
  frame_length = 0.025
  step_length = 0.010
  dtype = 'float16'

def get_exp_path(system_name, args, override=False):
  """ Return: exp_dir, model_path, log_path """
  exp_dir = get_exppath(tag='TIDIGITS_%s_%s_%s' %
    (system_name, args.task, args.feat))
  if 'nmix' in args:
    exp_dir += '_%d' % args.nmix
  if 'tdim' in args:
    exp_dir += '_%d' % args.tdim
  # ====== check override ====== #
  if bool(override) and os.path.exists(exp_dir):
    shutil.rmtree(exp_dir)
  if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
  # ====== basic paths ====== #
  model_path = os.path.join(exp_dir, 'model.ai')
  log_path = os.path.join(exp_dir,
                         'log_%s.txt' % get_formatted_datetime(only_number=True))
  print("Exp dir:", ctext(exp_dir, 'cyan'))
  print("Model path:", ctext(model_path, 'cyan'))
  print("Log path:", ctext(log_path, 'cyan'))
  return exp_dir, model_path, log_path

# ===========================================================================
# For DNN
# ===========================================================================
def prepare_data(feat, label, utt_length=0.4, for_ivec=False):
  """

  Returns (i-vector)
  ------------------
  ds[feat]
  train_files
  y_train
  test_files
  y_test
  labels

  Returns (x-vector)
  ------------------
  train : Feeder
    feeder for training data for iterating over pair of (X, y)
  valid : Feeder
    feeder for validating data for iterating over pair of (X, y)
  X_test_name : list of file names
    file names are append with '.%d' for cut segment ID
  X_test_true : list of integer
    label of each sample
  X_test_data : array
    list of test data same length as X_test_name
  labels : list of string
    list of labels for classification task

  Example
  -------
  (train, valid,
   X_test_name, X_test_true, X_test_data,
   labels) = prepare_data_dnn(feat=FEAT, label='gender')

  """
  label = str(label).lower()
  assert label in _support_label, "No support for label: %s" % label
  assert 0 < utt_length <= 1.
  # ====== load dataset ====== #
  if not os.path.exists(PATH_ACOUSTIC):
    raise RuntimeError("Cannot find extracted acoustic features at path: '%s',"
                       "run the code speech_features_extraction.py!" % PATH_ACOUSTIC)
  ds = F.Dataset(PATH_ACOUSTIC, read_only=True)
  assert feat in ds, "Cannot find feature with name: %s" % feat
  indices = list(ds['indices'].items())
  K.get_rng().shuffle(indices)

  # ====== helper ====== #
  def is_train(x):
    return x.split('_')[0] == 'train'

  def extract_label(x):
    return x.split('_')[_support_label[label]]

  print("Task:", ctext(label, 'cyan'))
  fn_label, labels = unique_labels([i[0] for i in indices],
                                   key_func=extract_label,
                                   return_labels=True)
  print("Labels:", ctext(labels, 'cyan'))
  # ====== training and test data ====== #
  train_files = [] # (name, (start, end)) ...
  test_files = []
  for name, (start, end) in indices:
    if is_train(name):
      train_files.append((name, (start, end)))
    else:
      test_files.append((name, (start, end)))
  # name for each dataset, useful for later
  print("#Train:", ctext(len(train_files), 'cyan'))
  print("#Test:", ctext(len(test_files), 'cyan'))
  # ====== for i-vectors ====== #
  y_train = np.array([fn_label(i[0]) for i in train_files])
  y_test = np.array([fn_label(i[0]) for i in test_files])
  if bool(for_ivec):
    return ds[feat], train_files, y_train, test_files, y_test, labels
  # ====== length ====== #
  length = [(end - start) for _, (start, end) in indices]
  max_length = max(length)
  frame_length = int(max_length * utt_length)
  step_length = frame_length
  print("Max length  :", ctext(max_length, 'yellow'))
  print("Frame length:", ctext(frame_length, 'yellow'))
  print("Step length :", ctext(step_length, 'yellow'))
  # ====== split dataset ====== #
  # split by speaker ID
  train_files, valid_files = train_valid_test_split(
      x=train_files, train=0.8,
      cluster_func=None,
      idfunc=lambda x: x[0].split('_')[4], # splited by speaker
      inc_test=False)
  print("#File train:", ctext(len(train_files), 'cyan'))
  print("#File valid:", ctext(len(valid_files), 'cyan'))
  print("#File test :", ctext(len(test_files), 'cyan'))

  recipes = [
      F.recipes.Sequencing(frame_length=frame_length, step_length=step_length,
                           end='pad', pad_mode='post', pad_value=0),
      F.recipes.Name2Label(converter_func=fn_label),
      F.recipes.LabelOneHot(nb_classes=len(labels), data_idx=-1)
  ]
  feeder_train = F.Feeder(F.IndexedData(ds[feat], indices=train_files),
                          ncpu=6, batch_mode='batch')
  feeder_valid = F.Feeder(F.IndexedData(ds[feat], indices=valid_files),
                          ncpu=4, batch_mode='batch')
  feeder_test = F.Feeder(F.IndexedData(ds[feat], indices=test_files),
                         ncpu=4, batch_mode='file')
  feeder_train.set_recipes(recipes)
  feeder_valid.set_recipes(recipes)
  feeder_test.set_recipes(recipes)
  print(feeder_train)

  # ====== process X_test, y_test in advance for faster evaluation ====== #
  @cache_disk
  def _extract_test_data(feat, label, utt_length):
    prog = Progbar(target=len(feeder_test),
                   print_summary=True, name="Preprocessing test set")
    X_test = defaultdict(list)
    for name, idx, X, y in feeder_test:
      # validate everything as expected
      assert fn_label(name) == np.argmax(y), name # label is right
      # save to list
      X_test[name].append((idx, X))
      prog.add(X.shape[0])
    # ====== create 1 array for data and dictionary for indices ====== #
    X_test_name = []
    X_test_data = []
    for name, X in X_test.items():
      X = np.concatenate([x[1] for x in sorted(X, key=lambda i: i[0])],
                         axis=0).astype('float16')
      X_test_name += [name + '.%d' % i for i in range(len(X))]
      X_test_data.append(X)
    X_test_name = np.array(X_test_name)
    X_test_data = np.concatenate(X_test_data, axis=0)
    return X_test_name, X_test_data
  # convert everything back to float32
  X_test_name, X_test_data = _extract_test_data(feat, label, utt_length)
  X_test_true = np.array([fn_label(i.split('.')[0])
                          for i in X_test_name])
  return feeder_train, feeder_valid, \
  X_test_name, X_test_true, X_test_data, labels

def make_dnn_prediction(functions, X, batch_size=256, title=''):
  return_list = True
  if not isinstance(functions, (tuple, list)):
    functions = [functions]
    return_list = False
  n_functions = len(functions)
  results = [[] for i in range(n_functions)]
  # ====== prepare progress bar ====== #
  n_samples = len(X)
  prog = Progbar(target=n_samples, print_summary=True,
                 name="Making prediction: %s" % str(title))
  # ====== for feeder ====== #
  if isinstance(X, F.Feeder):
    y_true = []
    for x, y in X.set_batch(batch_size=batch_size):
      for res, fn in zip(results, functions):
        res.append(fn(x))
      prog.add(x.shape[0])
      y_true.append(np.argmax(y, axis=-1) if y.ndim == 2 else y)
    results = [np.concatenate(res, axis=0)
               for res in results]
    y_true = np.concatenate(y_true, axis=0)
    if return_list:
      return results, y_true
    return results[0], y_true
  # ====== for numpy array ====== #
  else:
    for start, end in batching(batch_size=batch_size, n=n_samples):
      y = X[start:end]
      for res, fn in zip(results, functions):
        res.append(fn(y))
      prog.add(end - start)
    results = [np.concatenate(res, axis=0)
               for res in results]
    if return_list:
      return results
    return results[0]

# ===========================================================================
# visualization
# ===========================================================================
def visualize_latent_space(X_org, X_latent, name, labels, title):
  """
  X_org : [n_samples, n_timesteps, n_features]
  X_latent : [n_samples, n_timesteps, n_latents]
  """
  assert X_org.shape[0] == X_latent.shape[0] == len(name) == len(labels)
  assert not np.any(np.isnan(X_org))
  assert not np.any(np.isnan(X_latent))
  X_org = X_org.astype('float32')
  X_latent = X_latent.astype('float32')
  # ====== evaluation of the latent space ====== #
  n_channels = 1 if X_latent.ndim == 3 else int(np.prod(X_latent.shape[3:]))
  n_samples = X_org.shape[0]
  # 1 for original, 1 for mean channel, then the rest
  n_row = 1 + 1 + n_channels
  n_col = 3
  V.plot_figure(nrow=n_row + 1, ncol=16)
  # only select 3 random sample
  for i, idx in enumerate(
      sampling_iter(it=range(n_samples), k= n_col, seed=5218)):
    x = X_org[idx]
    # latent tensor can be 3D or 4D
    z = X_latent[idx]
    if z.ndim > 3:
      z = np.reshape(z, newshape=(z.shape[0], z.shape[1], -1))
    elif z.ndim == 2:
      z = np.reshape(z, newshape=(z.shape[0], z.shape[1], 1))
    elif z.ndim == 3:
      pass
    else:
      raise ValueError("No support for z value: %s" % str(z.shape))
    # plot original acoustic
    ax = V.plot_spectrogram(x.T, ax=(n_row, n_col, i + 1), title='Org')
    if i == 0:
      ax.set_title("[%s]'%s-%s'" % (str(title), str(name[idx]), str(labels[idx])),
                   fontsize=8)
    else:
      ax.set_title("'%s-%s'" % (str(name[idx]), str(labels[idx])),
                   fontsize=8)
    # plot the mean
    V.plot_spectrogram(np.mean(z, axis=-1).T,
                       ax=(n_row, n_col, i + 4), title='Zmean')
    # plot first 25 channels
    if n_channels > 1:
      for j in range(min(8, n_channels)):
        V.plot_spectrogram(z[:, :, j].T,
                           ax=(n_row, n_col, j * 3 + 7 + i),
                           title='Z%d' % j)
