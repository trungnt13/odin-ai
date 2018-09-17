from __future__ import print_function, division, absolute_import

import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.io import wavfile

from odin.preprocessing.signal import anything2wav
from odin.utils import (Progbar, get_exppath, cache_disk, ctext,
                        mpi)
from odin.stats import freqcount
from odin import fuel as F

sre_file_list = F.load_sre_list()
print(sre_file_list)
# ===========================================================================
# Exp path
# ===========================================================================
EXP_DIR = get_exppath('sre', override=False)
# ===========================================================================
# FILE LIST PATH
# ===========================================================================
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
PATH_RAW_DATA = {
    'mx6': '/media/data2/SRE_DATA',
    'voxceleb1': '/media/data2/SRE_DATA',
    'voxceleb2': '/media/data2/SRE_DATA',
    'swb': '/media/data2/SRE_DATA',
    'fisher': '/media/data2/SRE_DATA',
    'sre04': '/media/data2/SRE_DATA/NIST1996_2008/SRE02_SRE06',
    'sre05': '/media/data2/SRE_DATA/NIST1996_2008/SRE96_SRE05',
    'sre06': '/media/data2/SRE_DATA/NIST1996_2008/SRE02_SRE06',
    'sre08': '/media/data2/SRE_DATA',
    'sre10': '/media/data2/SRE_DATA',
    # noise datasets
    'musan': '/media/data2/SRE_DATA',
    'rirs': '/media/data2/SRE_DATA',
}

# all data will be down sampled to following
SAMPLE_RATE = 8000
PATH_ACOUSTIC_FEATURES = '/media/data1/SRE_FEAT'
if not os.path.exists(PATH_ACOUSTIC_FEATURES):
  os.mkdir(PATH_ACOUSTIC_FEATURES)
# ===========================================================================
# Validating the datasets
# ===========================================================================
@cache_disk
def validating_all_data(in_path_raw):
  # ====== meta info ====== #
  all_files = []
  non_exist_files = []
  extension_count = defaultdict(int)
  total_data = sum(x.shape[0]
                   for x in sre_file_list.values()
                   if isinstance(x, np.ndarray))
  # ====== progress ====== #
  prog = Progbar(target=total_data,
                 print_summary=True, print_report=True,
                 name="Preprocessing File List")
  prog.set_summarizer('#Files', fn=lambda x: x[-1])
  prog.set_summarizer('#Non-exist', fn=lambda x: x[-1])
  # ====== iterating ====== #
  for ds_name, data in sorted(sre_file_list.items(),
                              key=lambda x: x[0]):
    if 'README' in ds_name:
      continue
    for row in data:
      path, channel, name, spkid = row[:4]
      # check path provided
      if ds_name in in_path_raw:
        path = os.path.join(in_path_raw[ds_name], path)
      # check file exist
      if os.path.exists(path):
        all_files.append([path, channel, name, spkid, ds_name])
      else:
        non_exist_files.append([path, channel, name, spkid, ds_name])
      # extension
      ext = os.path.splitext(path)[-1]
      extension_count[ext + '-' + ds_name] += 1
      # update progress
      prog['Dataset'] = ds_name
      prog['#Files'] = len(all_files)
      prog['#Non-exist'] = len(non_exist_files)
      prog.add(1)
  # ====== final validation ====== #
  all_files = np.array(all_files)
  n_files = len(all_files)
  n_unique_files = len(np.unique(all_files[:, 2]))
  # check no duplicated name
  assert n_files == n_unique_files, \
  'Found duplicated name: %d != %d' % (n_files, n_unique_files)
  # ====== convert everything to .wav ====== #
  exit()
  jobs = defaultdict(list)
  for row in all_files:
    jobs[row[0]].append(row[1:])

  # data will be saved at: `SRE_CLEAN/name + '.wav'`
  def to_new_path(name):
    return os.path.join(out_path_clean, name + '.wav')

  def convert_to_wav(rows):
    new_rows = []
    for path, info in rows:
      # # already converted skip
      # if all(os.path.exists(to_new_path(i[1]))
      #        for i in info):
      #   for i in info:
      #     new_rows.append((to_new_path(i[1]), i[2], duration, ds_name))
      #   continue
      # processing
      ds_name = info[0][-1]
      y, sr = anything2wav(inpath=path, outpath=None,
                           dataset=ds_name,
                           sample_rate=out_sample_rate,
                           return_data=True)
      duration = max(y.shape) / sr
      for channel, name, spkid, ds_name in info:
        channel = int(channel)
        x = y[:, channel] if y.ndim == 2 else y
        outpath = to_new_path(name)
        with open(outpath, 'wb') as f:
          wavfile.write(f, sr, x)
        new_rows.append((outpath, spkid, duration, ds_name))
    yield new_rows
  # running the conversion
  new_files = []
  prog = Progbar(target=len(jobs), print_summary=True,
                 name="Convert to wav: %s" % out_path_clean)
  for rows in mpi.MPI(jobs=list(jobs.items()), func=convert_to_wav,
                      batch=12, ncpu=mpi.cpu_count() - 2):
    new_files += rows
    prog.add(len(rows))
  # outpath, spkid, duration, ds_name
  new_files = np.array(new_files)
  return new_files, np.array(non_exist_files), extension_count

# ==================== run the validation process ==================== #
(all_files, non_exist_files, ext_count) = validating_all_data(
    in_path_raw=PATH_RAW_DATA)
print("#Non-exist-files:", ctext(len(non_exist_files), 'cyan'))
print("Extensions:")
for name, val in sorted(ext_count.items(), key=lambda x: x[0]):
  print('  ', name, ':', ctext(val, 'cyan'))
print("All Datasets:")
for name, count in freqcount(all_files[:, -1], sort=True).items():
  print('  ', ctext(name, 'yellow'), ':', ctext(count, 'cyan'))
# ===========================================================================
# DATA PATH
# ===========================================================================

# ===========================================================================
# PATH HELPER
# ===========================================================================

# ===========================================================================
# DATA HELPER
# ===========================================================================
