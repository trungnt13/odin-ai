from __future__ import print_function, division, absolute_import

import os
import pickle
from enum import Enum
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.io import wavfile

from odin import visual as V
from odin.preprocessing.signal import anything2wav
from odin.utils import (Progbar, get_exppath, cache_disk, ctext,
                        mpi, args_parse, select_path, get_logpath,
                        get_script_name, get_script_path, get_module_from_path)
from odin.stats import freqcount, sampling_iter
from odin import fuel as F

# ===========================================================================
# Configuration
# ===========================================================================
class Config(object):
  # ====== Acoustic features ====== #
  FRAME_LENGTH = 0.025
  STEP_LENGTH = 0.01
  SAMPLE_RATE = 8000
  WINDOW = 'hamm'
  NFFT = 512
  NMELS = 24
  NCEPS = 40
  FMIN = 100
  FMAX = 4000
  dtype = 'float16'
  # Random seed for reproducibility
  SUPER_SEED = 52181208

class SystemStates(Enum):
  """ SystemStates """
  UNKNOWN = 0
  EXTRACT_FEATURES = 1
  TRAINING = 2
  SCORING = 3

# ===========================================================================
# General arguments for all experiments
# ===========================================================================
_args = args_parse(descriptions=[
    ('recipe', 'recipe is the name of acoustic Dataset defined in feature_recipes.py', None),
    ('-aug', 'augmentation dataset: musan, rirs; could be multiple dataset for training: "musan,rirs"', None, 'None'),
    ('-downsample', 'downsampling all the dataset for testing code', None, 0),
    ('-ncpu', 'number of CPU to be used, if <= 0, auto-select', None, 0),
    # for scoring
    ('-sys', 'name of the system for scoring: xvec, ivec, e2e ...', None, 'xvec'),
    ('-sysid', 'when a system is saved multiple checkpoint (e.g. sys.0.ai)', None, '-1'),
    ('-score', 'name of dataset for scoring, multiple dataset splited by ","', None, 'sre18'),
    # for ivector
    ('-nmix', 'for i-vector training, number of Gaussian components', None, 2048),
    ('-tdim', 'for i-vector training, number of latent dimension for i-vector', None, 600),
    # for DNN
    ('-utt', 'for x-vector training, maximum utterance length', None, 4),
    ('-batch', 'batch size, for training DNN', None, 256),
    ('-epoch', 'number of epoch, for training DNN', None, 25),
    ('--train', 'if the model is already trained, forced running the fine-tuning', None, False),
    ('--debug', 'enable debugging', None, False),
])
IS_DEBUGGING = _args.debug
IS_TRAINING = _args.train
# this variable determine which state is running
CURRENT_STATE = SystemStates.UNKNOWN
# ====== Features ====== #
FEATURE_RECIPE = str(_args.recipe)
AUGMENTATION_NAME = _args.aug
# ====== DNN ====== #
BATCH_SIZE = int(_args.batch)
EPOCH = int(_args.epoch)
# ====== system ====== #
NCPU = min(18, mpi.cpu_count() - 2) if _args.ncpu <= 0 else int(_args.ncpu)
# ====== helper for checking the requirement ====== #
def _check_feature_extraction_requirement():
  # check requirement for feature extraction
  from shutil import which
  if which('sox') is None:
    raise RuntimeError("`sox` was not installed")
  if which('sph2pipe') is None:
    raise RuntimeError("`sph2pipe` was not installed")
  if which('ffmpeg') is None:
    raise RuntimeError("`ffmpeg` was not installed")

def _check_recipe_name_for_extraction():
  # check the requirement of recipe name for feature extraction
  if '_' in FEATURE_RECIPE:
    raise ValueError("'_' can appear in recipe name which is: '%s'" % FEATURE_RECIPE)
# ====== check the running script to determine the current running states ====== #
_script_name = get_script_name()
if _script_name in ('speech_augmentation', 'speech_features_extraction'):
  CURRENT_STATE = SystemStates.EXTRACT_FEATURES
  _check_feature_extraction_requirement()
  _check_recipe_name_for_extraction()
elif _script_name in ('train_xvec', 'train_ivec', 'train_tvec'):
  CURRENT_STATE = SystemStates.TRAINING
elif _script_name in ('make_score'):
  CURRENT_STATE = SystemStates.SCORING
  _check_feature_extraction_requirement()
else:
  raise RuntimeError("Unknown states for current running script: %s/%s" %
    (get_script_path(), get_script_name()))
# some fancy log of current state
print(ctext('====================================', 'red'))
print(ctext("System state:", 'cyan'), ctext(CURRENT_STATE, 'yellow'))
print(ctext('====================================', 'red'))
# ===========================================================================
# FILE LIST PATH
# ===========================================================================
EXP_DIR = get_exppath('sre', override=False)
BASE_DIR = select_path(
    '/media/data2/SRE_DATA',
    '/mnt/sdb1/SRE_DATA',
default='')
# path to directory contain following folders:
#  * mx6_speech
#  * voxceleb
#  * voxceleb2
#  * SRE04
#  * SRE05
#  * SRE06
#  * SRE08
#  * SRE10
#  * SRE18
#  * Switchboard
#  * fisher
#  * musan
#  * RIRS_NOISES
PATH_RAW_DATA = {
    'mx6': BASE_DIR,
    'voxceleb1': BASE_DIR,
    'voxceleb2': BASE_DIR,
    'swb': BASE_DIR,
    'fisher': BASE_DIR,
    'sre04': os.path.join(BASE_DIR, 'NIST1996_2008/SRE02_SRE06'),
    'sre05': os.path.join(BASE_DIR, 'NIST1996_2008/SRE96_SRE05'),
    'sre06': os.path.join(BASE_DIR, 'NIST1996_2008/SRE02_SRE06'),
    'sre08': BASE_DIR,
    'sre10': BASE_DIR,
    'sre18': os.path.join(BASE_DIR, 'SRE18'),
    # noise datasets
    'musan': BASE_DIR,
    'rirs': BASE_DIR,
}
# all features will be stored here
OUTPUT_DIR = select_path(
    '/home/trung/data',
    '/media/data1',
    '/mnt/sda1'
)
PATH_ACOUSTIC_FEATURES = os.path.join(OUTPUT_DIR, "SRE_FEAT")
if not os.path.exists(PATH_ACOUSTIC_FEATURES):
  os.mkdir(PATH_ACOUSTIC_FEATURES)
# ===========================================================================
# Load the file list
# ===========================================================================
sre_file_list = F.load_sre_list()
print('README at:', ctext(sre_file_list['README.txt'], 'cyan'))
sre_file_list = {k: v
                 for k, v in sre_file_list.items()
                 if isinstance(v, np.ndarray)}
print("Original dataset:")
for k, v in sorted(sre_file_list.items(), key=lambda x: x[0]):
  print(' ', ctext('%-12s' % k, 'yellow'), ':',
    ctext(v.shape, 'cyan'))
# ====== check dataset for scoring ====== #
if CURRENT_STATE == SystemStates.SCORING:
  assert len(_args.score) > 0, \
  "No dataset are provided for scoring, specify '-score' option"

  def validate_scoring_dataset(in_path_raw, score_dataset):
    all_files = {}
    for dsname in score_dataset:
      if dsname not in sre_file_list:
        raise ValueError("Cannot find dataset with name: '%s' in the file list" % dsname)
      if dsname not in in_path_raw:
        raise ValueError("Cannot find dataset with name: '%s' in provided path" % dsname)

      base_path = in_path_raw[dsname]
      ds = []
      for row in sre_file_list[dsname]:
        path = os.path.join(base_path, row[0])
        # every file must exist
        if not os.path.exists(path):
          raise RuntimeError("File not exist at path: %s" % path)
        ds.append([path] + row[1:].tolist() + [dsname])
      all_files[dsname] = np.array(ds)
    # Header:
    #  0       1      2        3           4
    # path, channel, name, something, dataset_name
    return all_files

  SCORING_DATASETS = validate_scoring_dataset(
      in_path_raw=PATH_RAW_DATA,
      score_dataset=_args.score.split(','))
  print("Processed scoring data:")
  for dsname, dsarray in SCORING_DATASETS.items():
    print('  ', ctext('%-10s' % dsname, 'yellow'), ':',
          '%s' % ctext(dsarray.shape, 'cyan'))

  # searching for the appropriate system
  SCORE_SYSTEM_NAME = _args.sys
  SCORE_SYSTEM_ID = int(_args.sysid)
# ===========================================================================
# Validating the Noise dataset for augmentation
# ===========================================================================
@cache_disk
def validating_noise_data(in_path_raw):
  # preparing
  noise_dataset = ['musan', 'rirs']
  all_files = defaultdict(list)
  n_files = sum(len(sre_file_list[i])
                for i in noise_dataset
                if i in sre_file_list)
  n_non_exist = 0
  n_exist = 0
  prog = Progbar(target=n_files, print_summary=True,
                 name="Validating noise dataset")
  prog.set_summarizer(key='#Non-exist', fn=lambda x: x[-1])
  prog.set_summarizer(key='#Exist', fn=lambda x: x[-1])
  # check all dataset
  for ds_name in noise_dataset:
    if ds_name not in sre_file_list:
      continue
    if ds_name not in in_path_raw:
      continue
    base_path = in_path_raw[ds_name]
    base_ds = all_files[ds_name]
    # start validating
    for row in sre_file_list[ds_name]:
      # check file
      path, channel, name, noise_type, duration = row[:5]
      path = os.path.join(base_path, path)
      if os.path.exists(path):
        base_ds.append([path, channel, name, noise_type, duration])
        n_exist += 1
      else:
        n_non_exist += 1
      # update progress
      prog['ds'] = ds_name
      prog['#Exist'] = n_exist
      prog['#Non-exist'] = n_non_exist
      prog.add(1)
  # ====== return ====== #
  # Header:
  #  0       1      2         3         4
  # path, channel, name, noise_type, duration
  return {key: np.array(sorted(val, key=lambda x: x[0]))
          for key, val in all_files.items()}
# ==================== run the validation ==================== #
if CURRENT_STATE == SystemStates.EXTRACT_FEATURES:
  ALL_NOISE = validating_noise_data(
      in_path_raw=PATH_RAW_DATA)
  print("Processed noise data:")
  for ds_name, noise_list in ALL_NOISE.items():
    print(" ", ctext(ds_name, 'yellow'), ':', noise_list.shape)
    if len(noise_list) == 0:
      continue
    for name, count in sorted(freqcount(noise_list[:, 3]).items(),
                              key=lambda x: x[0]):
      print('  ', ctext('%-10s' % name, 'yellow'), ':',
            '%s(files)' % ctext('%-6d' % count, 'cyan'))
# ===========================================================================
# Validating the file list of training data
# ===========================================================================
@cache_disk
def validating_training_data(in_path_raw, training_dataset):
  file_list = {ds: sre_file_list[ds]
               for ds in training_dataset
               if ds in sre_file_list}
  # ====== meta info ====== #
  all_files = []
  non_exist_files = []
  extension_count = defaultdict(int)
  total_data = sum(v.shape[0]
                   for k, v in file_list.items()
                   if k not in('musan', 'rirs'))
  # ====== progress ====== #
  prog = Progbar(target=total_data,
                 print_summary=True, print_report=True,
                 name="Preprocessing File List")
  prog.set_summarizer('#Files', fn=lambda x: x[-1])
  prog.set_summarizer('#Non-exist', fn=lambda x: x[-1])
  # ====== iterating ====== #
  for ds_name, data in sorted(file_list.items(),
                              key=lambda x: x[0]):
    if ds_name in ('musan', 'rirs'):
      continue
    for row in data:
      path, channel, name, spkid = row[:4]
      assert channel in ('0', '1')
      # check path provided
      if ds_name in in_path_raw:
        path = os.path.join(in_path_raw[ds_name], path)
      # create new row
      start_time = '-'
      end_time = '-'
      if ds_name == 'mx6':
        start_time, end_time = row[-2:]
      new_row = [path, channel, name,
                 ds_name + '_' + spkid, ds_name,
                 start_time, end_time]
      # check file exist
      if os.path.exists(path):
        all_files.append(new_row)
      else:
        non_exist_files.append(new_row)
      # extension
      ext = os.path.splitext(path)[-1]
      extension_count[ext + '-' + ds_name] += 1
      # update progress
      prog['Dataset'] = ds_name
      prog['#Files'] = len(all_files)
      prog['#Non-exist'] = len(non_exist_files)
      prog.add(1)
  # final results
  all_files = np.array(all_files)
  if len(all_files) == 0:
    return all_files, np.array(non_exist_files), extension_count
  # ====== check no duplicated name ====== #
  n_files = len(all_files)
  n_unique_files = len(np.unique(all_files[:, 2]))
  assert n_files == n_unique_files, \
  'Found duplicated name: %d != %d' % (n_files, n_unique_files)
  # ====== check no duplicated speaker ====== #
  n_spk = sum(len(np.unique(dat[:, 3]))
              for name, dat in file_list.items()
              if name not in ('musan', 'rirs'))
  n_unique_spk = len(np.unique(all_files[:, 3]))
  assert n_spk == n_unique_spk, \
  'Found duplicated speakers: %d != %d' % (n_spk, n_unique_spk)
  # ====== return ====== #
  # Header:
  #  0       1      2      3       4          5         6
  # path, channel, name, spkid, dataset, start_time, end_time
  return all_files, np.array(non_exist_files), extension_count
# ==================== run the validation process ==================== #
if CURRENT_STATE == SystemStates.EXTRACT_FEATURES:
  (ALL_FILES, NON_EXIST_FILES, ext_count) = validating_training_data(
      in_path_raw=PATH_RAW_DATA,
      training_dataset=['mx6', 'voxceleb1', 'voxceleb2', 'swb', 'fisher',
                        'sre04', 'sre05', 'sre06', 'sre08', 'sre10']
  )
  if len(ALL_FILES) == 0:
    raise RuntimeError("No files found for feature extraction")

  # list of all dataset
  ALL_DATASET = sorted(np.unique(ALL_FILES[:, 4]))
  print("All extensions:")
  for name, val in sorted(ext_count.items(), key=lambda x: x[0]):
    print('  ', '%-16s' % name, ':', ctext('%-6d' % val, 'cyan'), '(files)')
  print("#Speakers:", ctext(len(np.unique(ALL_FILES[:, 3])), 'cyan'))

  # map Dataset_name -> speaker_ID
  DS_SPK = defaultdict(list)
  for row in ALL_FILES:
    DS_SPK[row[4]].append(row[3])
  DS_SPK = {k: sorted(set(v))
            for k, v in DS_SPK.items()}

  print("Processed datasets:")
  for name, count in sorted(freqcount(ALL_FILES[:, 4]).items(),
                            key=lambda x: x[0]):
    print('  ', ctext('%-10s' % name, 'yellow'), ':',
          '%s(files)' % ctext('%-6d' % count, 'cyan'),
          '%s(spk)' % ctext('%-4d' % len(DS_SPK[name]), 'cyan'))
# ===========================================================================
# PATH HELPER
# ===========================================================================
def get_model_path(system_name):
  """
  Parameters
  ----------
  args_name : list of string
    list of name for parsed argument, taken into account for creating
    model name

  Return
  ------
  exp_dir, model_path, log_path
  """
  if system_name == 'xvec':
    args_name = ['utt']
  elif system_name == 'ivec':
    args_name = ['nmix', 'tdim']
  else:
    raise ValueError("No support for system with name: %s" % system_name)
  # ====== prefix ====== #
  name = '_'.join([str(system_name).lower(), FEATURE_RECIPE])
  # ====== concat the attributes ====== #
  for i in sorted(str(i) for i in args_name):
    name += '_' + str(int(getattr(_args, i)))
  # ====== check save_path ====== #
  save_path = os.path.join(EXP_DIR, name)
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  # ====== return path ====== #
  log_path = get_logpath(name='log.txt', increasing=True,
                         odin_base=False, root=save_path)
  model_path = os.path.join(save_path, 'model.ai')
  print("Model path:", ctext(model_path, 'cyan'))
  print("Log path:", ctext(log_path, 'cyan'))
  return save_path, model_path, log_path
# ===========================================================================
# Data helper
# ===========================================================================
def prepare_dnn_feeder_recipe(name2label=None, n_speakers=None):
  frame_length = float(_args.utt) / Config.STEP_LENGTH
  recipes = [
      F.recipes.Sequencing(frame_length=frame_length,
                           step_length=frame_length,
                           end='cut', data_idx=0),
  ]
  if name2label is not None and \
  n_speakers is not None:
    recipes += [
        F.recipes.Name2Label(lambda name:name2label[name.split('/')[0]],
                             ref_idx=0),
        F.recipes.LabelOneHot(nb_classes=n_speakers, data_idx=1)
    ]
  elif (name2label is not None and n_speakers is None) or\
  (name2label is None and n_speakers is not None):
    raise RuntimeError("name2label and n_speakers must both be None, or not-None")
  return recipes

def prepare_dnn_data():
  path = os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE)
  assert os.path.exists(path), "Cannot find acoustic dataset at path: %s" % path
  ds = F.Dataset(path=path, read_only=True)
  # ====== find the right feature ====== #
  feature_name = FEATURE_RECIPE.split('_')[0]
  ids_name = 'indices_%s' % feature_name
  assert feature_name in ds, "Cannot find feature with name: %s" % feature_name
  assert ids_name in ds, "Cannot find indices with name: %s" % ids_name
  X = ds[feature_name]
  indices = ds[ids_name]
  # ====== all training file name ====== #
  rand = np.random.RandomState(seed=Config.SUPER_SEED)
  # modify here to train full dataset
  all_name = sorted(indices.keys())
  rand.shuffle(all_name)
  all_name = all_name
  n_files = len(all_name)
  print("#Files:", ctext(n_files, 'cyan'))
  # ====== speaker mapping ====== #
  name2spk = {name: ds['spkid'][name]
              for name in all_name}
  all_speakers = sorted(set(name2spk.values()))
  spk2label = {spk: i
               for i, spk in enumerate(all_speakers)}
  name2label = {name: spk2label[spk]
                for name, spk in name2spk.items()}
  print("#Speakers:", ctext(len(all_speakers), 'cyan'))
  # ====== stratify sampling based on speaker ====== #
  valid_name = []
  # create speakers' cluster
  label2name = defaultdict(list)
  for name, label in name2label.items():
    label2name[label].append(name)
  # for each speaker with >= 2 utterance, pick 1 utterance
  for label, name_list in label2name.items():
    if len(name_list) < 2:
      continue
    n = max(1, int(0.1 * len(name_list))) # 10% for validation
    valid_name += rand.choice(a=name_list, size=n).tolist()
  # train list is the rest
  _ = {name: 1 for name in valid_name}
  train_name = [i for i in all_name
                if i not in _]
  # ====== split training and validation ====== #
  train_indices = {name: indices[name]
                   for name in train_name}
  valid_indices = {name: indices[name]
                   for name in valid_name}

  print("#Train files:", ctext('%-8d' % len(train_indices), 'cyan'),
        "#spk:", ctext(len(set(name2label[name]
                               for name in train_name)), 'cyan'))

  print("#Valid files:", ctext('%-8d' % len(valid_indices), 'cyan'),
        "#spk:", ctext(len(set(name2label[name]
                               for name in valid_name)), 'cyan'))
  # ====== create the recipe ====== #
  recipes = prepare_dnn_feeder_recipe(name2label=name2label,
                                      n_speakers=len(all_speakers))
  train_feeder = F.Feeder(
      data_desc=F.IndexedData(data=X, indices=train_indices),
      batch_mode='batch', ncpu=NCPU, buffer_size=80)
  valid_feeder = F.Feeder(
      data_desc=F.IndexedData(data=X, indices=valid_indices),
      batch_mode='batch', ncpu=max(2, NCPU // 4), buffer_size=4)
  train_feeder.set_recipes(recipes)
  valid_feeder.set_recipes(recipes)
  print(train_feeder)
  print(valid_feeder)
  # ====== debugging ====== #
  if IS_DEBUGGING:
    prog = Progbar(target=len(valid_feeder), print_summary=True,
                   name="Iterating validation set")
    for X, y in valid_feeder.set_batch(BATCH_SIZE):
      prog['X'] = X.shape
      prog['y'] = y.shape
      prog.add(X.shape[0])
  # ====== return ====== #
  return train_feeder, valid_feeder, all_speakers
# ===========================================================================
# Evaluation HELPER
# ===========================================================================
def validate_feature_dataset(path, outpath):
  if os.path.exists(path):
    ds = F.Dataset(path, read_only=True)
    print(ds)

    for name, (start, end) in ds['indices'].items():
      if end - start == 0:
        print('Zero-length:', ctext(name, 'yellow'))

      if 'mspec' in ds:
        feat_name = 'mspec'
      elif 'spec' in ds:
        feat_name = 'spec'
      elif 'mfcc' in ds:
        feat_name = 'mfcc'
      elif 'bnf' in ds:
        feat_name = 'bnf'
      elif 'sad' in ds:
        raise NotImplementedError("No support for visualize SAD")
      else:
        raise RuntimeError()

    for name, (start, end) in sampling_iter(it=ds['indices'].items(),
                                            k=80, seed=Config.SUPER_SEED):
      dsname = ds['dsname'][name]
      if 'voxceleb2' == dsname and np.random.rand() < 0.95:
        continue
      if np.random.rand() < 0.8:
        continue
      spkid = ds['spkid'][name]
      dur = ds['duration'][name]
      # ====== search for the right features ====== #
      X = ds[feat_name][start:end][:1200].astype('float32')
      V.plot_figure(nrow=4, ncol=12)
      V.plot_spectrogram(X.T)
      V.plot_title(title='%s  %s  %s  %f' % (name.split('/')[0], spkid, dsname, dur))
    V.plot_save(outpath, tight_plot=True)
    ds.close()
