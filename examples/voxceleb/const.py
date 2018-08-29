import os
from odin.utils import get_exppath
from odin.stats import sampling_iter

# ====== fixed path to 'voxceleb1_wav' folder ====== #
PATH_TO_WAV = '/mnt/sdb1/'
PATH_ACOUSTIC_FEAT = '/mnt/sdb1/voxceleb_feat'
# path to folder contains experiment results
PATH_EXP = get_exppath('voxceleb')
# ====== remove '_quarter' if you want full training data ====== #
FILE_LIST = "voxceleb_files_quarter"
TRAIN_LIST = "voxceleb_sys_train_with_labels_quarter"
TRIAL_LIST = "voxceleb_trials"

# ====== Load the file list ====== #
from odin import fuel as F
ds = F.load_voxceleb_list()
WAV_FILES = {} # dictionary mapping 'file_path' -> 'file_name'
for path, channel, name in ds[FILE_LIST]:
  path = os.path.join(PATH_TO_WAV, path)
  assert os.path.exists(path), path
  WAV_FILES[path] = name
# some sampled files for testing
SAMPLED_WAV_FILE = sampling_iter(it=WAV_FILES.items(), k=8, seed=5218)
# ====== extract the list of all train files ====== #
TRAIN_DATA = {} # mapping from name of training file to speaker label
for x, y in ds[TRAIN_LIST]:
  TRAIN_DATA[x] = int(y)
