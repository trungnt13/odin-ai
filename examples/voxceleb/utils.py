import os
import pickle

import numpy as np
from scipy.io import savemat

from odin import fuel as F, visual as V
from odin.utils import ctext, Progbar, get_exppath, select_path
from odin.stats import train_valid_test_split, sampling_iter
from odin.preprocessing.signal import segment_axis

HOME_PATH = os.path.expanduser('~')
# fixed path to 'voxceleb1_wav' folder
PATH_TO_WAV = select_path(
    '/media/data2/SRE_DATA/voxceleb',
    '/mnt/sdb1/SRE_DATA/voxceleb',
    os.path.join(HOME_PATH, 'data', 'voxceleb'),
    os.path.join(HOME_PATH, 'voxceleb'),
    create_new=False
)
# path to folder contains experiment results
PATH_EXP = get_exppath('voxceleb')
# output path for acoustic features directory
PATH_ACOUSTIC_FEAT = os.path.join(PATH_EXP, 'voxceleb_feat')
if not os.path.exists(PATH_ACOUSTIC_FEAT):
  os.mkdir(PATH_ACOUSTIC_FEAT)
# ====== remove '_quarter' if you want full training data ====== #
FILE_LIST = "voxceleb_files_quarter"
TRAIN_LIST = "voxceleb_sys_train_with_labels_quarter"
TRIAL_LIST = "voxceleb_trials"
# ====== Load the file list ====== #
ds = F.load_voxceleb_list()
WAV_FILES = {} # dictionary mapping 'file_path' -> 'file_name'
for path, channel, name in ds[FILE_LIST]:
  path = os.path.join(PATH_TO_WAV, path)
  # validate all files are exist
  assert os.path.exists(path), path
  WAV_FILES[path] = name
# some sampled files for testing
SAMPLED_WAV_FILE = sampling_iter(it=sorted(WAV_FILES.items(),
                                           key=lambda x: x[0]),
                                 k=8, seed=52181208)
# ====== extract the list of all train files ====== #
# mapping from name of training file to speaker label
TRAIN_DATA = {}
for x, y in ds[TRAIN_LIST]:
  TRAIN_DATA[x] = int(y)

# ===========================================================================
# Path helpers
# ===========================================================================
def get_model_path(system_name, args):
  """Return: exp_dir, model_path, log_path, train_path, test_path"""
  name = '_'.join([str(system_name).lower(), args.recipe, args.feat])
  if 'l' in args:
    name += '_' + str(int(args.l))
  if 'nmix' in args:
    name += '_' + str(int(args.nmix))
  if 'tdim' in args:
    name += '_' + str(int(args.tdim))
  save_path = os.path.join(PATH_EXP, name)
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  # ====== return path ====== #
  log_path = os.path.join(save_path, 'log.txt')
  model_path = os.path.join(save_path, 'model.ai')
  train_path = os.path.join(save_path, 'train.dat')
  test_path = os.path.join(save_path, 'test.dat')
  print("Model path:", ctext(model_path, 'cyan'))
  print("Log path:", ctext(log_path, 'cyan'))
  return save_path, model_path, log_path, train_path, test_path

# ===========================================================================
# Data helpers
# ===========================================================================
FRAME_SHIFT = 0.005

def prepare_dnn_data(recipe, feat, utt_length, seed=52181208):
  """
  Return
  ------
  train_feeder : Feeder for training
  valid_feeder : Feeder for validating
  test_ids : Test indices
  test_dat : Data array
  all_speakers : list of all speaker in training set
  """
  # Load dataset
  frame_length = int(utt_length / FRAME_SHIFT)
  ds = F.Dataset(os.path.join(PATH_ACOUSTIC_FEAT, recipe),
                 read_only=True)
  X = ds[feat]
  train_indices = {name: ds['indices'][name]
                   for name in TRAIN_DATA.keys()}
  test_indices = {name: start_end
                  for name, start_end in ds['indices'].items()
                  if name not in TRAIN_DATA}
  train_indices, valid_indices = train_valid_test_split(
      x=list(train_indices.items()), train=0.9, inc_test=False, seed=seed)
  all_speakers = sorted(set(TRAIN_DATA.values()))
  n_speakers = max(all_speakers) + 1
  print("#Train files:", ctext(len(train_indices), 'cyan'))
  print("#Valid files:", ctext(len(valid_indices), 'cyan'))
  print("#Test files:", ctext(len(test_indices), 'cyan'))
  print("#Speakers:", ctext(n_speakers, 'cyan'))
  recipes = [
      F.recipes.Sequencing(frame_length=frame_length, step_length=frame_length,
                           end='pad', pad_value=0, pad_mode='post',
                           data_idx=0),
      F.recipes.Name2Label(lambda name:TRAIN_DATA[name], ref_idx=0),
      F.recipes.LabelOneHot(nb_classes=n_speakers, data_idx=1)
  ]
  train_feeder = F.Feeder(
      data_desc=F.IndexedData(data=X, indices=train_indices),
      batch_mode='batch', ncpu=7, buffer_size=12)
  valid_feeder = F.Feeder(
      data_desc=F.IndexedData(data=X, indices=valid_indices),
      batch_mode='batch', ncpu=2, buffer_size=4)
  train_feeder.set_recipes(recipes)
  valid_feeder.set_recipes(recipes)
  print(train_feeder)
  # ====== cache the test data ====== #
  cache_dat = os.path.join(PATH_EXP, 'test_%s_%d.dat' % (feat, int(utt_length)))
  cache_ids = os.path.join(PATH_EXP, 'test_%s_%d.ids' % (feat, int(utt_length)))
  # validate cache files
  if os.path.exists(cache_ids):
    with open(cache_ids, 'rb') as f:
      ids = pickle.load(f)
    if len(ids) != len(test_indices):
      os.remove(cache_ids)
      if os.path.exists(cache_dat):
        os.remove(cache_dat)
  elif os.path.exists(cache_dat):
    os.remove(cache_dat)
  # caching
  if not os.path.exists(cache_dat):
    dat = F.MmapData(cache_dat, dtype='float16',
                     shape=(0, frame_length, X.shape[1]))
    ids = {}
    prog = Progbar(target=len(test_indices))
    s = 0
    for name, (start, end) in test_indices.items():
      y = X[start:end]
      y = segment_axis(y, axis=0,
                       frame_length=frame_length, step_length=frame_length,
                       end='pad', pad_value=0, pad_mode='post')
      dat.append(y)
      # update indices
      ids[name] = (s, s + len(y))
      s += len(y)
      # update progress
      prog.add(1)
    dat.flush()
    dat.close()
    with open(cache_ids, 'wb') as f:
      pickle.dump(ids, f)
  # ====== re-load ====== #
  dat = F.MmapData(cache_dat, read_only=True)
  with open(cache_ids, 'rb') as f:
    ids = pickle.load(f)
  # ====== save some sample ====== #
  sample_path = os.path.join(PATH_EXP,
                             'test_%s_%d.pdf' % (feat, int(utt_length)))
  V.plot_figure(nrow=9, ncol=6)
  for i, (name, (start, end)) in enumerate(
      sampling_iter(it=sorted(ids.items(), key=lambda x: x[0]), k=12, seed=52181208)):
    x = dat[start:end][:].astype('float32')
    ax = V.plot_spectrogram(x[np.random.randint(0, len(x))].T,
                            ax=(12, 1, i + 1), title='')
    ax.set_title(name)
  V.plot_save(sample_path)
  return (train_feeder, valid_feeder,
          ids, dat, all_speakers)

def prepare_ivec_data(recipe, feat):
  ds = F.Dataset(os.path.join(PATH_ACOUSTIC_FEAT, recipe),
                 read_only=True)
  X = ds[feat]
  train_indices = {name: ds['indices'][name]
                   for name in TRAIN_DATA.keys()}
  test_indices = {name: start_end
                  for name, start_end in ds['indices'].items()
                  if name not in TRAIN_DATA}
  print("#Train files:", ctext(len(train_indices), 'cyan'))
  print("#Test files:", ctext(len(test_indices), 'cyan'))
  return X, train_indices, test_indices

# ===========================================================================
# Data saving helpers
# ===========================================================================
def csv2mat(exp_dir):
  in_train = os.path.join(exp_dir, 'train.dat')
  out_train = os.path.join(exp_dir, 'train.mat')
  in_test = os.path.join(exp_dir, 'test.dat')
  out_test = os.path.join(exp_dir, 'test.mat')
  assert os.path.exists(in_train) and os.path.exists(in_test)

  train_dat = np.genfromtxt(in_train, dtype=str, delimiter='\t')
  X_train = train_dat[:, 1:].astype('float32').T
  y_train = train_dat[:, 0].ravel()
  print('Train:', X_train.shape, X_train.dtype, y_train.shape, y_train.dtype)
  savemat(file_name=out_train, mdict={'X': X_train.T, 'y': y_train})

  test_dat = np.genfromtxt(in_test, dtype=str, delimiter='\t')
  X_test = test_dat[:, 1:].astype('float32').T
  y_test = test_dat[:, 0].ravel()
  print('Test:', X_test.shape, X_test.dtype, y_test.shape, y_test.dtype)
  savemat(file_name=out_test, mdict={'X': X_test.T, 'y': y_test})
