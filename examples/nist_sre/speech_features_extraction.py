from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'
import sys
import time
from collections import defaultdict

import numpy as np

from odin import visual as V, nnet as N
from odin.utils import (ctext, unique_labels, UnitTimer,
                        Progbar, get_logpath, get_module_from_path,
                        get_script_path, mpi, catch_warnings_ignore)
from odin import fuel as F, preprocessing as pp
from odin.stats import sampling_iter

from helpers import (PATH_ACOUSTIC_FEATURES, EXP_DIR,
                     ALL_FILES, IS_DEBUGGING, FEATURE_RECIPE, FEATURE_NAME,
                     ALL_DATASET, Config, NCPU,
                     validate_features_dataset)
# ALL_FILES
# Header:
#  0       1      2      3       4          5         6
# path, channel, name, spkid, dataset, start_time, end_time
# ===========================================================================
# Extractor
# ===========================================================================
recipe = get_module_from_path(identifier=FEATURE_RECIPE,
                              path=get_script_path(),
                              prefix='feature_recipes')
if len(recipe) == 0:
  raise ValueError("Cannot find feature recipe with name: '%s'" % FEATURE_RECIPE)
recipe = recipe[0]()
# ===========================================================================
# Debug mode
# ===========================================================================
if IS_DEBUGGING:
  with np.warnings.catch_warnings():
    rand = np.random.RandomState(seed=Config.SUPER_SEED)
    np.warnings.filterwarnings('ignore')
    # ====== stratify sampling from each dataset ====== #
    clusters = defaultdict(list)
    clusters_count = defaultdict(int)
    samples = []
    for row in sorted(ALL_FILES,
                      key=lambda x: x[0]):
      clusters[row[4]].append(row)
      clusters_count[row[4]] += 1
    for k, v in clusters.items():
      rand.shuffle(v)
      samples += v[:18] # 18 files from each dataset
    # ====== run the MPI for feature extraction ====== #
    prog = Progbar(target=len(samples),
                   print_report=True, print_summary=False,
                   name=FEATURE_RECIPE)
    error_signal = []
    for feat in mpi.MPI(jobs=samples,
                        func=recipe.transform,
                        ncpu=NCPU, batch=1):
      assert FEATURE_NAME in feat
      # update progress
      if isinstance(feat, pp.base.ExtractorSignal):
        error_signal.append(feat)
        prog.add(1)
        continue
      prog['spkid'] = feat['spkid']
      prog['name'] = feat['name']
      prog['dsname'] = feat['dsname']
      prog['duration'] = feat['duration']
      prog.add(1)
      # 30% chance plotting
      if rand.rand() < 0.5:
        V.plot_multiple_features(feat, fig_width=20,
                                 title='[%s]%s' % (feat['dsname'], feat['name']))
    V.plot_save(os.path.join(EXP_DIR, 'debug_%s.pdf' % FEATURE_RECIPE),
                dpi=30)
    # ====== save the extractor debugging log ====== #
    pp.set_extractor_debug(recipe, debug=True)
    recipe.transform(samples[0])
    with open(os.path.join(EXP_DIR, 'debug_%s.log' % FEATURE_RECIPE), 'w') as f:
      for name, step in recipe.steps:
        f.write(step.last_debugging_text)
      # ====== print error signal ====== #
      for e in error_signal:
        f.write(str(e) + '\n')
        print(e)
  exit()
# ===========================================================================
# Running the extractor
# ===========================================================================
# ====== basic path ====== #
output_dataset_path = os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE)

processor_log_path = os.path.join(EXP_DIR, 'processor_%s.log' % FEATURE_RECIPE)
if os.path.exists(processor_log_path):
  os.remove(processor_log_path)
print("Log path:", ctext(processor_log_path, 'cyan'))

ds_validation_path = os.path.join(EXP_DIR, 'validate_%s.pdf' % FEATURE_RECIPE)
if os.path.exists(ds_validation_path):
  os.remove(ds_validation_path)
print("Validation path:", ctext(ds_validation_path, 'cyan'))

# ====== running the processing ====== #
with catch_warnings_ignore(Warning):
  processor = pp.FeatureProcessor(
      jobs=ALL_FILES,
      path=output_dataset_path,
      extractor=recipe,
      n_cache=320,
      ncpu=NCPU,
      override=True,
      identifier='name',
      log_path=processor_log_path,
      stop_on_failure=False)
  processor.run()
# ===========================================================================
# Make some visualization
# ===========================================================================
validate_features_dataset(output_dataset_path, ds_validation_path)
