from __future__ import print_function, division, absolute_import

import os
import numpy as np

from odin import fuel as F
from odin.utils import (ctext, get_logpath, get_module_from_path,
                        get_script_path)

from helpers import (ALL_FILES, ALL_NOISE, ALL_DATASET, IS_DEBUGGING,
                     PATH_AUGMENTATION, PATH_ACOUSTIC_FEATURES,
                     FEATURE_RECIPE, AUGMENTATION_NAME, Config)

if AUGMENTATION_NAME == 'None':
  raise ValueError("`-aug` option was not provided, choose: 'rirs' or 'musan'")
np.random.seed(Config.SUPER_SEED)
# ===========================================================================
# Constant
# ===========================================================================
AUGMENTATION_DATASET = [
    'swb', 'sre04', 'sre05', 'sre06', 'sre08', 'sre10']
AUGMENTATION_DATASET = [i for i in AUGMENTATION_DATASET
                        if i in ALL_DATASET]
print("Augmenting following dataset: %s" % ', '.join(AUGMENTATION_DATASET))
# cmd = input("(y) to continue ...")
# if cmd.lower() != 'y':
#   exit()
# ====== get the duration ====== #
path = os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE)
assert os.path.exists(path), \
"Acoustic feature must be extracted first at path: %s" % path
ds = F.Dataset(path, read_only=True)
all_duration = dict(ds['duration'].items())
ds.close()
# ====== select a new file list ====== #
AUG_FILES = []
for row in ALL_FILES:
  if row[4] not in AUGMENTATION_DATASET:
    continue
  if row[2] not in all_duration:
    print("Missing duration:", ctext('; '.join(row), 'yellow'))
    continue
  dur = all_duration[row[2]]
  AUG_FILES.append([i for i in row] + [dur])
assert len(AUG_FILES), "Cannot find any files for augmentation"
AUG_FILES = np.array(AUG_FILES)
org_shape = AUG_FILES.shape
# select half of the files
np.random.shuffle(AUG_FILES)
np.random.shuffle(AUG_FILES)
AUG_FILES = AUG_FILES[:(len(AUG_FILES) // 2)]
sel_shape = AUG_FILES.shape
# print the log
print("#Augmentation Files:")
print("  Found    :", ctext(org_shape, 'cyan'))
print("  Selected :", ctext(sel_shape, 'cyan'))
# ===========================================================================
# Get the recipe
# ===========================================================================
recipe = get_module_from_path(identifier=FEATURE_RECIPE,
                              path=get_script_path(),
                              prefix='feature_recipes')
assert len(recipe) > 0, "Cannot find recipe with name: %s" % recipe
recipe = recipe[0](augmentation=AUGMENTATION_NAME)
# ===========================================================================
# Debugging
# ===========================================================================
recipe.transform(AUG_FILES[0])
exit()
