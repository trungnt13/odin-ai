import os
import pickle

import numpy as np

from odin import fuel as F, visual as V
from odin.utils import ctext, Progbar
from odin.stats import train_valid_test_split, sampling_iter
from odin.preprocessing.signal import segment_axis

from const import PATH_ACOUSTIC_FEAT, TRAIN_DATA, PATH_EXP

# ===========================================================================
# Path helpers
# ===========================================================================
def get_model_path(system_name, args):
  """Return: model_path, log_path, train_path, test_path"""
  name = '_'.join([str(system_name).lower(), args.feat])
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
  return model_path, log_path, train_path, test_path

# ===========================================================================
# Data helpers
# ===========================================================================
FRAME_SHIFT = 0.005

def prepare_dnn_data(feat, utt_length, seed=52181208):
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
  ds = F.Dataset(PATH_ACOUSTIC_FEAT, read_only=True)
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
      F.recipes.Name2Label(lambda name:TRAIN_DATA[name], ref_idx=0),
      F.recipes.Sequencing(frame_length=frame_length, step_length=frame_length,
                           end='pad', pad_value=0, pad_mode='post',
                           data_idx=0, label_idx=1),
      F.recipes.LabelOneHot(nb_classes=n_speakers, data_idx=1)
  ]
  train_feeder = F.Feeder(
      data_desc=F.DataDescriptor(data=X, indices=train_indices),
      batch_mode='batch', ncpu=7, buffer_size=12)
  valid_feeder = F.Feeder(
      data_desc=F.DataDescriptor(data=X, indices=valid_indices),
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

def prepare_ivec_data(feat):
  ds = F.Dataset(PATH_ACOUSTIC_FEAT, read_only=True)
  X = ds[feat]
  train_indices = {name: ds['indices'][name]
                   for name in TRAIN_DATA.keys()}
  test_indices = {name: start_end
                  for name, start_end in ds['indices'].items()
                  if name not in TRAIN_DATA}
  print("#Train files:", ctext(len(train_indices), 'cyan'))
  print("#Test files:", ctext(len(test_indices), 'cyan'))
  return X, ds['sad'], train_indices, test_indices
