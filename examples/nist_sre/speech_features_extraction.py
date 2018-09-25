from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,gpu'
import sys
import time
from collections import defaultdict

import numpy as np
import soundfile as sf

from odin import visual as V, nnet as N
from odin.utils import (ctext, unique_labels, UnitTimer,
                        Progbar, get_logpath,
                        get_module_from_path,
                        get_script_path, mpi)
from odin import fuel as F, preprocessing as pp
from odin.stats import sampling_iter

from helpers import (PATH_ACOUSTIC_FEATURES, EXP_DIR, BASE_DIR,
                     ALL_FILES, IS_DEBUGGING, FEATURE_RECIPE,
                     ALL_DATASET, Config, NCPU, validate_feature_dataset,
                     check_requirement_feature_processing)
# ALL_FILES
# Header:
#  0       1      2      3       4          5         6
# path, channel, name, spkid, dataset, start_time, end_time
np.random.seed(Config.SUPER_SEED)
check_requirement_feature_processing()
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
    np.warnings.filterwarnings('ignore')
    # ====== stratify sampling from each dataset ====== #
    clusters = defaultdict(list)
    clusters_count = defaultdict(int)
    samples = []
    for row in ALL_FILES:
      clusters[row[4]].append(row)
      clusters_count[row[4]] += 1
    for k, v in clusters.items():
      np.random.shuffle(v)
      samples += v[:12]

    # ====== run the MPI for feature extraction ====== #
    prog = Progbar(target=len(samples),
                   print_report=True, print_summary=False,
                   name=FEATURE_RECIPE)
    for feat in mpi.MPI(jobs=samples,
                        func=recipe.transform,
                        ncpu=NCPU, batch=1):
      # update progress
      prog['path'] = feat['path'].replace(BASE_DIR, '')
      prog['spkid'] = feat['spkid']
      prog['name'] = feat['name']
      prog['dsname'] = feat['dsname']
      prog['duration'] = feat['duration']
      prog.add(1)
      # 30% chance plotting
      if np.random.rand() < 0.3:
        feat[FEATURE_RECIPE] = feat[FEATURE_RECIPE][:1200]
        V.plot_multiple_features(feat,
                                 title=feat['path'])
    V.plot_save(os.path.join(EXP_DIR, 'debug_%s.pdf' % FEATURE_RECIPE))
    # ====== save the extractor debugging log ====== #
    pp.set_extractor_debug(recipe, debug=True)
    recipe.transform(samples[0])
    with open(os.path.join(EXP_DIR, 'debug_%s.log' % FEATURE_RECIPE), 'w') as f:
      for name, step in recipe.steps:
        f.write(step.last_debugging_text)
  exit()
# ===========================================================================
# Running the extractor
# ===========================================================================
# ====== basic path ====== #
output_dataset_path = os.path.join(PATH_ACOUSTIC_FEATURES, FEATURE_RECIPE)
processor_log_path = get_logpath(name='processor_%s.log' % FEATURE_RECIPE,
                                 increasing=True,
                                 odin_base=False,
                                 root=EXP_DIR)
ds_validation_path = os.path.join(EXP_DIR, 'validate_%s.pdf' % FEATURE_RECIPE)
# ====== running the processing ====== #
with np.warnings.catch_warnings():
  np.warnings.filterwarnings('ignore')
  # ====== start processing ====== #
  processor = pp.FeatureProcessor(
      jobs=ALL_FILES,
      path=output_dataset_path,
      extractor=recipe,
      n_cache=250,
      ncpu=NCPU,
      override=True,
      identifier='name',
      log_path=processor_log_path,
      stop_on_failure=False)
  processor.run()
# ===========================================================================
# Make some visualization
# ===========================================================================
validate_feature_dataset(path=output_dataset_path,
                         outpath=ds_validation_path)
