from __future__ import print_function, division, absolute_import

import os
import pickle
from shutil import which
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.io import wavfile

from odin.preprocessing.signal import anything2wav
from odin.utils import (Progbar, get_exppath, cache_disk, ctext,
                        mpi, args_parse, select_path)
from odin.stats import freqcount
from odin import fuel as F

# ===========================================================================
# Checking prerequisites
# ===========================================================================
if which('sox') is None:
  raise RuntimeError("`sox` was not installed")
if which('sph2pipe') is None:
  raise RuntimeError("`sph2pipe` was not installed")
if which('ffmpeg') is None:
  raise RuntimeError("`ffmpeg` was not installed")
# ===========================================================================
# General arguments for all experiments
# ===========================================================================
_args = args_parse(descriptions=[
    ('recipe', 'recipe is the name of acoustic Dataset defined in feature_recipes.py', None),
    ('-aug', 'Name of the augmentation dataset: musan, rirs', None, 'None'),
    ('-downsample', 'downsampling all the dataset for testing', None, 0),
    ('--debug', 'enable debugging', None, False),
])
FEATURE_RECIPE = str(_args.recipe)
IS_DEBUGGING = bool(_args.debug)
AUGMENTATION_NAME = _args.aug
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
  # ====== File list ====== #
  SUPER_SEED = 52181208
  # <= 0 mean no downsample, > 0 mean number of sample
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
  print(' ', ctext(k, 'yellow'), ':', ctext(v.shape, 'cyan'))
# ===========================================================================
# FILE LIST PATH
# ===========================================================================
EXP_DIR = get_exppath('sre', override=False)
BASE_DIR = select_path(
    '/media/data2/SRE_DATA',
    '/mnt/sdb1/SRE_DATA',
)
# path to directory contain following folders:
#  * mx6_speech
#  * voxceleb
#  * voxceleb2
#  * SRE04
#  * SRE05
#  * SRE06
#  * SRE08
#  * SRE10
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
    # noise datasets
    'musan': BASE_DIR,
    'rirs': BASE_DIR,
}
# all data will be down sampled to following
PATH_ACOUSTIC_FEATURES = '/media/data1/SRE_FEAT'
if not os.path.exists(PATH_ACOUSTIC_FEATURES):
  os.mkdir(PATH_ACOUSTIC_FEATURES)
# For augmentation data
PATH_AUGMENTATION = '/media/data1/SRE_AUG'
if not os.path.exists(PATH_AUGMENTATION):
  os.mkdir(PATH_AUGMENTATION)
# ===========================================================================
# Validating the datasets
# ===========================================================================
# ====== validating noise data for augmentation ====== #
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

ALL_NOISE = validating_noise_data(
    in_path_raw=PATH_RAW_DATA)
print("Processed noise data:")
for ds_name, noise_list in ALL_NOISE.items():
  print(" ", ctext(ds_name, 'yellow'), ':', noise_list.shape)
  for name, count in sorted(freqcount(noise_list[:, 3]).items(),
                            key=lambda x: x[0]):
    print('  ', ctext('%-10s' % name, 'yellow'), ':',
          '%s(files)' % ctext('%-6d' % count, 'cyan'))
# ====== validating the file list of training data ====== #
@cache_disk
def validating_all_data(in_path_raw, downsample):
  file_list = dict(sre_file_list)
  # ====== downsample for debugging ====== #
  if downsample > 0:
    np.random.seed(Config.SUPER_SEED)
    for k, v in list(file_list.items()):
      if not isinstance(v, np.ndarray):
        continue
      np.random.shuffle(v)
      file_list[k] = v[:int(downsample)]
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
(ALL_FILES, NON_EXIST_FILES, ext_count) = validating_all_data(
    in_path_raw=PATH_RAW_DATA,
    downsample=_args.downsample)
# list of all dataset
ALL_DATASET = sorted(np.unique(ALL_FILES[:, 4]))
print("All extensions:")
for name, val in sorted(ext_count.items(), key=lambda x: x[0]):
  print('  ', '%-16s' % name, ':', ctext('%-6d' % val, 'cyan'), '(files)')
print("#Speakers:", ctext(len(np.unique(ALL_FILES[:, 3])), 'cyan'))
DS_SPK = defaultdict(list)
for row in ALL_FILES:
  DS_SPK[row[4]].append(row[3])
DS_SPK = {k: sorted(set(v)) for k, v in DS_SPK.items()}
print("Processed datasets:")
for name, count in sorted(freqcount(ALL_FILES[:, 4]).items(),
                          key=lambda x: x[0]):
  print('  ', ctext('%-10s' % name, 'yellow'), ':',
        '%s(files)' % ctext('%-6d' % count, 'cyan'),
        '%s(spk)' % ctext('%-4d' % len(DS_SPK[name]), 'cyan'))
# ===========================================================================
# PATH HELPER
# ===========================================================================

# ===========================================================================
# DATA HELPER
# ===========================================================================
