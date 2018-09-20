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
                     ALL_DATASET)
# ALL_FILES
# Header:
#  0       1      2      3       4          5         6
# path, channel, name, spkid, dataset, start_time, end_time
np.random.seed(52181208)
NCPU = min(18, mpi.cpu_count() - 2)
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
    def _benchmark_func(job):
      s = time.time()
      res = recipe.transform(job)
      return res, time.time() - s
    prog = Progbar(target=len(samples),
                   print_report=True, print_summary=False,
                   name=FEATURE_RECIPE)
    start_time = time.time()
    all_duration = []
    all_benchmark = defaultdict(list)
    for feat, benchmark in mpi.MPI(jobs=samples,
                        func=_benchmark_func,
                        ncpu=NCPU, batch=1):
      # update progress
      prog['path'] = feat['path'].replace(BASE_DIR, '')
      prog['spkid'] = feat['spkid']
      prog['name'] = feat['name']
      prog['dsname'] = feat['dsname']
      prog['duration'] = feat['duration']
      prog.add(1)
      # update benchmark
      all_benchmark[feat['dsname']].append(benchmark)
      all_duration.append(feat['duration'])
      # 30% chance plotting
      if np.random.rand() < 0.3:
        V.plot_multiple_features(feat,
                                 title=feat['path'])
    V.plot_save(os.path.join(EXP_DIR, 'debug_%s.pdf' % FEATURE_RECIPE))
    # ====== save the extractor debugging log ====== #
    pp.set_extractor_debug(recipe, debug=True)
    recipe.transform(samples[0])
    with open(os.path.join(EXP_DIR, 'debug_%s.log' % FEATURE_RECIPE), 'w') as f:
      for name, step in recipe.steps:
        f.write(step.last_debugging_text)
    # ====== summary ====== #
    print("Avg.Duration:", ctext(np.mean(all_duration), 'cyan'))
    end_time = time.time()
    print("Elapse:", ctext(end_time - start_time, 'cyan'))
    print("Avg.Speed:", ctext(len(samples) / (end_time - start_time), 'cyan'))
    # ====== estimate processing time ====== #
    est_time = 0
    for name in sorted(ALL_DATASET):
      t = all_benchmark[name]
      c = clusters_count[name]
      e = c * np.mean(t)
      print('%-12s' % name,
            ctext('Avg.Time: %.2f(s)' % np.mean(t), 'yellow'),
            ctext('Est.Time: %.2f(hour)' % (e / 3600), 'yellow'))
      est_time += e
    # this time is not precise, just for fun
    print("Total time:",
          'Est.: %s(hour)' % ctext('%.2f' % ((end_time - start_time) / len(samples) * len(ALL_FILES) / 3600), 'cyan'),
          '%d-cores: %s(hour)' % (NCPU, ctext('%.2f' % (est_time / 3600 / NCPU), 'cyan')),
    )
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
  # ====== shuffling all files so the jobs is more evenly distributed ====== #
  perm = np.random.permutation(len(ALL_FILES))
  jobs = ALL_FILES[perm]
  # ====== start processing ====== #
  processor = pp.FeatureProcessor(jobs=jobs,
      path=output_dataset_path,
      extractor=recipe,
      n_cache=250,
      ncpu=NCPU,
      override=True,
      identifier='name',
      log_path=processor_log_path,
      stop_on_failure=False)
  with UnitTimer():
    processor.run()
# ===========================================================================
# Make some visualization
# ===========================================================================
if os.path.exists(output_dataset_path):
  ds = F.Dataset(output_dataset_path, read_only=True)
  print(ds)
  for name, (start, end) in sampling_iter(it=ds['indices'].items(),
                                          k=30, seed=52181208):
    dsname = ds['ds'][name]
    if 'voxceleb2' == dsname and np.random.rand() < 0.95:
      continue
    spkid = ds['spkid'][name]
    dur = ds['duration'][name]
    X = ds['mspec'][start:end][:1200].astype('float32')
    V.plot_figure(nrow=4, ncol=12)
    V.plot_spectrogram(X.T, title='%s  %s  %s  %f' % (name, spkid, dsname, dur))
  V.plot_save(ds_validation_path)
  ds.close()
