# -*- coding: utf-8 -*-
# ===========================================================================
# Parallel features processing using multi-core CPU and multiprocessing
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import re
import os
import sys
import wave
import time
import random
import shutil
import warnings
from numbers import Number
from multiprocessing import Pool, cpu_count, Process, Queue
from six import add_metaclass, string_types
from six.moves import zip, zip_longest, range, cPickle
from abc import ABCMeta, abstractmethod, abstractproperty

from collections import defaultdict, Mapping

import numpy as np

from sklearn.pipeline import Pipeline

from odin.utils.mpi import MPI
from odin.utils import (Progbar, as_tuple, get_all_files, ctext,
                        get_tempdir, is_string, batching,
                        add_notification, defaultdictkey,
                        stdio, get_stdio_path, wprint,
                        add_notification, flatten_list)
from odin.fuel import Dataset, MmapDict, MmapData

from .base import Extractor
_default_module = re.compile('__.*__')


# ===========================================================================
# PCA calculation
# ===========================================================================
def calculate_pca(dataset, feat_name='auto', batch_size=5218, override=False):
  """ Using parallel MiniBatchPCA to do PCA for multiple features
  at once.

  """
  # TODO: add different pca prefix (e.g. pca_full_mspec, pca_sami_mspec)
  # add reading data from indices also
  # ====== check input dataset ====== #
  own_dataset = True
  if is_string(dataset) and os.path.isdir(dataset):
    dataset = Dataset(dataset, read_only=True)
  elif isinstance(dataset, Dataset):
    own_dataset = False
  elif isinstance(dataset, FeatureProcessor):
    dataset = Dataset(dataset.path, read_only=True)
  else:
    raise ValueError("Cannot acquire Dataset from input: %s" %
                     str(dataset))
  # ====== extract all feat_name ====== #
  if is_string(feat_name) and feat_name == 'auto':
    feat_name = []
    for k in dataset.keys():
      X = dataset[k]
      if hasattr(X, 'ndim') and X.ndim == 2 and X.shape[-1] > 1:
        feat_name.append(k)
  else:
    feat_name = [name
                 for name in as_tuple(feat_name, t=str)
                 if name in dataset]
  # ====== load PCA ====== #
  from odin.ml import MiniBatchPCA
  # init PCA
  nb_samples = 0
  for feat in feat_name:
    nb_samples += dataset[feat].shape[0]
  # ====== prepare MPI PCA ====== #
  add_notification("Selected features for PCA: " +
      ctext(', '.join(feat_name), 'yellow'))

  def map_pca(name):
    X = dataset[name]
    # found exist pca model
    if 'pca_' + feat in dataset and not override:
      pca = dataset['pca_' + feat]
    # create new PCA
    else:
      pca = MiniBatchPCA(n_components=None, whiten=False,
                         copy=True, batch_size=None)
    # No shuffling make iter much faster
    for x in X.set_batch(batch_size=batch_size, seed=None, shuffle_level=0):
      pca.partial_fit(x)
      yield x.shape[0]
    # save PCA model
    with open(os.path.join(dataset.path, 'pca_' + name), 'wb') as f:
      cPickle.dump(pca, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # finish return feature name
    yield name
  mpi = MPI(jobs=feat_name, func=map_pca,
            ncpu=None, batch=1, hwm=12082518,
            backend='python')
  # ====== running the MPI ====== #
  remain_features = list(feat_name)
  finished_features = []
  prog = Progbar(target=nb_samples, print_summary=True, print_report=True,
                 name='PCA')
  for n in mpi:
    if is_string(n):
      remain_features.remove(n)
      finished_features.append(n)
    else:
      prog['Remain'] = ', '.join(remain_features)
      prog['Finished'] = ', '.join(finished_features)
      prog.add(n)
  # ====== return ====== #
  if own_dataset:
    dataset.close()


# ===========================================================================
# Helper
# ===========================================================================
def _escape_file_name(file_name):
  file_name = file_name.replace('/', '_')
  file_name = file_name.replace(':', '_')
  return file_name


def _special_cases(X, feat_name, file_name, ds, path):
  """ Same special for checking the integrity of the features """
  if feat_name == 'raw':
    from .speech import save
    sr = ds['sr'][file_name]
    if '.wav' not in file_name:
      file_name += '.wav'
    save(os.path.join(path, _escape_file_name(file_name)),
         X.astype('float32'), sr=sr)
  elif feat_name == 'spec':
    from .speech import SpectraExtractor, _extract_frame_step_length, save
    from .signal import ispec
    sr = ds['sr'][file_name]
    extractor = [i for _, i in ds['pipeline'].steps
                 if isinstance(i, SpectraExtractor)][0]
    frame_length, step_length = _extract_frame_step_length(sr,
        extractor.frame_length, extractor.step_length)
    raw = ispec(X,
                frame_length=frame_length, step_length=step_length,
                window=extractor.window, padding=extractor.padding,
                db=extractor.log)
    file_name += '-ispec.wav'
    save(os.path.join(path, _escape_file_name(file_name)),
         raw.astype('float32'), sr=sr)


def validate_features(ds_or_processor, path, nb_samples=25,
                      override=False, seed=12082518, fig_width=4):
  # TODO: add PCA visualization
  def logger(title, tag, check):
    check = bool(check)
    text_color = 'yellow' if check else 'red'
    print(ctext('   *', 'cyan'),
          ctext(str(title), text_color),
          ctext(str(tag), 'magenta'),
          ctext("✓", text_color) if check else ctext("✗", text_color))
  import matplotlib
  matplotlib.use('Agg')
  from odin.visual import plot_save, plot_multiple_features
  # ====== check path to dataset ====== #
  should_close_ds = True
  if isinstance(ds_or_processor, FeatureProcessor):
    ds = Dataset(ds_or_processor.path, read_only=True)
  elif is_string(ds_or_processor):
    ds = Dataset(ds_or_processor, read_only=True)
  elif isinstance(ds_or_processor, Dataset):
    ds = ds_or_processor
    should_close_ds = False
  else:
    raise ValueError("`ds` can be None, string, or Dataset. No "
                     "support for given input type: %s" % str(type(ds)))
  print(ctext('Validating dataset:', 'yellow'), '"%s"' % ds.path)
  # ====== extract the config of the dataset ====== #
  if 'config' not in ds or 'pipeline' not in ds:
    raise RuntimeError("The `Dataset` must be generated by `FeatureProcessor` "
                       "which must contain `config` MmapDict of extracted "
                       "features configuration.")
  config = ds['config']
  pipeline = ds['pipeline']
  # ====== output path ====== #
  path = str(path)
  if not os.path.exists(path):
    os.mkdir(path)
  elif override:
    if os.path.isfile(path):
      os.remove(path)
    else:
      shutil.rmtree(path)
    os.mkdir(path)
  else:
    raise ValueError("`path`=%s exists, cannot override." % path)
  prev_stdio = get_stdio_path()
  stdio(path=os.path.join(path, 'log.txt'))
  nb_samples = int(nb_samples)
  # ====== get all features ====== #
  # [(name, dtype, statistic-able), ...]
  all_keys = [k for k in ds.keys() if k not in ('config', 'pipeline')]
  # store all features (included the features in external_indices
  all_features = []
  # the external indices can be: indices_mfcc_bnf
  external_indices = flatten_list([k.split('_')[1:] for k in all_keys
                                   if 'indices' in k and k != 'indices'])
  # ====== checking indices ====== #
  main_indices = {name: (start, end)
                  for name, (start, end) in ds['indices'].items()}
  for ids_name in (k for k in all_keys if 'indices' in k):
    ids = sorted([(name, start, end)
                  for name, (start, end) in ds[ids_name].items()],
                 key=lambda x: x[1])
    for prev, now in zip(ids, ids[1:]):
      assert prev[2] == now[1], "Zero length in indices"
      assert prev[2] - prev[1] > 0, "Zero length in indices"
      assert now[2] - now[1] > 0, "Zero length in indices"
    # final length match length of Data
    if ids_name != 'indices':
      for feat_name in ids_name.split('_')[1:]:
        assert now[-1] == len(ds[feat_name]), \
            "Indices and data length mismatch, indices:'%s' feat:'%s'" % \
            (ids_name, feat_name)
        all_features.append(feat_name)
    else:
      for feat_name in all_keys:
        if feat_name not in external_indices and \
        'sum1' != feat_name[-4:] and 'sum2' != feat_name[-4:] and \
        'mean' != feat_name[-4:] and 'std' != feat_name[-3:] and \
        isinstance(ds[feat_name], MmapData):
          assert now[-1] == len(ds[feat_name]), \
          "Length of indices and actual data mismatch, " + ids_name + ':' + feat_name
          all_features.append(feat_name)
    # logging
    logger("Checked all:", ids_name, True)
  # ====== check all dictionary types ====== #
  for name in all_keys:
    if isinstance(ds[name], MmapDict) and 'indices' not in name:
      data = ds[name]
      # special cases
      if name == 'sr':
        checking_func = lambda x: x > 0 # for sr
      else:
        checking_func = lambda x: True
      # check
      for key, val in data.items():
        assert key in main_indices, \
        "Dictionary with name:'%s' has key not found in indices." % name
        assert checking_func(val)
      logger("Checked dictionary: ", name, True)
  # ====== checking each type of data ====== #
  # get all stats name
  all_stats = defaultdict(list)
  for k in all_keys:
    if 'sum1' == k[-4:] or 'sum2' == k[-4:] or \
    'mean' == k[-4:] or 'std' == k[-3:]:
      all_stats[k[:-4].split('_')[0]].append(k)
  # get all pca name
  all_pca = {i: i + '_pca' for i in all_features
             if i + '_pca' in ds}
  # checking one-by-one numpy.ndarray features array
  for feat_name in all_features:
    dtype = str(ds[feat_name].dtype)
    # checking all data
    indices = ds.find_prefix(feat_name, 'indices')
    prog = Progbar(target=len(indices), interval=0.1,
                   print_report=True,
                   name='Checking: %s(%s)' % (feat_name, dtype))
    # start iterating over all data file
    fail_test = False
    for file_name, (start, end) in indices:
      dat = ds[feat_name][start:end]
      # No NaN value
      if np.any(np.isnan(dat)):
        logger("NaN values", file_name + ':' + feat_name, False)
        fail_test = True
      # not all value closed to zeros
      if np.all(np.isclose(dat, 0.)):
        logger("All-closed-zeros values", file_name + ':' + feat_name,
               False)
        fail_test = True
      prog['Name'] = file_name
      prog.add(1)
    if not fail_test:
      logger("Check data incredibility for: ", feat_name, True)
    # checking statistics
    if feat_name in all_stats:
      fail_test = False
      for stat_name in all_stats[feat_name]:
        X = ds[stat_name]
        if X.ndim >= 1:
          X = X[:]
        if np.any(np.isnan(X)):
          logger("NaN values", feat_name + ':' + stat_name, False)
          fail_test = True
        if np.all(np.isclose(X, 0.)):
          logger("All-closed-zeros values", feat_name + ':' + stat_name,
                 False)
          fail_test = True
      if not fail_test:
        logger("Check statistics for: ", feat_name, True)
    # check PCA
    if feat_name in all_pca:
      pca = ds[all_pca[feat_name]]
      n = ds[feat_name].shape[0]
      nb_feats = ds[feat_name].shape[-1]
      fail_test = False
      # performing PCA on random samples
      for i in range(nb_samples):
        start = np.random.randint(0, n - nb_samples - 1)
        X = pca.transform(
            ds[feat_name][start:(start + nb_samples)],
            n_components=max(nb_feats // 2, 1))
        if np.any(np.isnan(X)):
          logger("NaN values in PCA", feat_name, False)
          fail_test = True
          break
        if np.all(np.isclose(X, 0.)):
          logger("All-closed-zeros values in PCA", feat_name, False)
          fail_test = True
          break
      if not fail_test:
        logger("Check PCA for: ", feat_name, True)
  # ====== Do sampling ====== #
  np.random.seed(seed) # seed for reproceducible
  all_samples = np.random.choice(list(ds['indices'].keys()),
                                 size=nb_samples,
                                 replace=False)
  # plotting all samples
  for sample_id, file_name in enumerate(all_samples):
    X = {}
    for feat_name in all_features:
      start, end = ds.find_prefix(feat_name, 'indices')[file_name]
      feat = ds[feat_name][start:end]
      X[feat_name] = feat
      # some special handling
      try:
        _special_cases(X=feat, feat_name=feat_name, file_name=file_name,
                       ds=ds, path=path)
      except Exception as e:
        logger("Special case error: %s" % str(e),
               file_name + ':' + feat_name, False)
    plot_multiple_features(X, title=file_name, fig_width=fig_width)
    figure_path = os.path.join(path, '%s.pdf' % _escape_file_name(file_name))
    plot_save(figure_path, log=False, clear_all=True)
    logger("Sample figure saved at: ", figure_path, True)
  # plotting the statistic
  figure_path = os.path.join(path, 'stats.pdf')
  for feat_name, stat_name in all_stats.items():
    X = {name: ds[name][:]
         for name in stat_name
         if ds[name].ndim >= 1}
    if len(X) > 0:
      plot_multiple_features(X, title=feat_name, fig_width=fig_width)
  plot_save(figure_path, log=False, clear_all=True)
  logger("Stats figure save at: ", figure_path, True)
  logger("All reports at folder: ", os.path.abspath(path), True)
  # ====== cleaning ====== #
  stdio(path=prev_stdio)
  if should_close_ds:
    ds.close()


# ===========================================================================
# Features Processor
# ===========================================================================
class FeatureProcessor(object):

  """ FeatureProcessor

  Parameters
  ----------
  jobs: list, or tuple
      list of jobs for processing, keep the jobs specification simple
      additional information can be introduced via custom Extractor.
  extractor: sklearn.Pipeline, odin.preprocessing.Extractor
      list of extactor for creating a Pipeline
  path: str
      path to a folder for saving output Dataset
  ncache: float or int (> 0)
      pass
  ncpu: int (>0)
      number of Processes will be used to parallel the processor
  name: None or str
      identity for the Processor
  """

  def __init__(self, jobs, path, extractor,
               ncache=0.12, ncpu=1, name=None, override=False):
    super(FeatureProcessor, self).__init__()
    # ====== check outpath ====== #
    path = os.path.abspath(str(path))
    if os.path.isfile(path):
      raise ValueError("`path` must be path to a directory, but found a "
                       "path to file.")
    # check override
    if os.path.exists(path) and override:
      wprint("Remove existed Dataset at path: %s" % path)
      shutil.rmtree(path)
    # set path and name
    self.path = path
    if name is None:
      name = self.path
    self.name = name
    # ====== check jobs ====== #
    if not isinstance(jobs, (tuple, list, np.ndarray)):
      raise ValueError("Provided `jobs` must be instance of tuple, list or ndarray.")
    if isinstance(jobs, np.ndarray):
      jobs = jobs.tolist()
    self.jobs = tuple(jobs)
    # ====== check multiprocessing ====== #
    if ncpu is None: # auto select number of CPU
      ncpu = min(len(jobs), cpu_count() - 1)
    ncpu = int(ncpu)
    if ncpu <= 0 or ncache <= 0:
      raise ValueError('`ncpu` and `ncache` must be greater than 0, but '
                       'given values ncpu=%d ncache=%f' % (ncpu, ncache))
    self.ncpu = ncpu
    self.ncache = ncache
    # ====== internal control for feature processor ====== #
    if isinstance(extractor, Pipeline):
      pass
    elif isinstance(extractor, (tuple, list)):
      steps = [('%s_%d' % (e.__class__.__name__, i), e)
               for i, e in enumerate(extractor)]
      extractor = Pipeline(steps=steps)
    elif isinstance(extractor, Mapping):
      steps = [(str(n), e) for n, e in extractor.items()]
      extractor = Pipeline(steps=steps)
    elif isinstance(extractor, Extractor):
      extractor = Pipeline(
          steps=[(extractor.__class__.__name__, extractor)])
    self.extractor = extractor

  # ==================== Abstract properties ==================== #
  def run(self):
    njobs = len(self.jobs)
    dataset = Dataset(self.path)
    if self.ncache <= 1:
      cache_limit = max(2, int(0.12 * njobs))
    else:
      cache_limit = int(self.ncache)
    # ====== indices ====== #
    databases = defaultdictkey(lambda key:
        MmapDict(path=os.path.join(dataset.path, key), cache_size=10000,
                 read_only=False))
    # ====== statistic ====== #
    # load old statistics
    stats = defaultdict(lambda: [0, 0]) # name -> (sum1, sum2)
    for key in dataset.keys():
      if 'sum1' == key[-4]:
        stats[key[:-4]][0] = dataset[key][:]
      elif 'sum2' == key[-4:]:
        stats[key[:-4]][1] = dataset[key][:]
    # all data are cached for periodically flushed
    cache = defaultdict(list)
    nb_processed = [0] # store the value as reference
    last_start = defaultdict(int)

    # ====== helper ====== #
    def flush_feature(feat_name, X_cached):
      if len(X_cached) > 0:
        X_cached = np.concatenate(X_cached, 0)
        # flush data
        if feat_name in dataset:
          dataset[feat_name].append(X_cached)
        else:
          dataset[(feat_name, 'memmap')] = X_cached

    # ====== repeated for each result returned ====== #
    def post_processing(result):
      # returned result is always dictionary or None
      # if dictionary, it is mapping: name -> feature_matrix
      if result is None:
        return
      # search for file name
      for key in ('name', 'path'):
        file_name = result.get(key, None)
        if file_name is not None:
          break
      # invalid file_name
      if file_name is None or not is_string(file_name):
        raise RuntimeError("Cannot find file name in returned features "
            "list, the file name can be specified in key: 'name', 'path' "
            "and the type of the value must be string. All available "
            "keys are: %s" % str(result.keys()))
      # store all new indices
      all_indices = defaultdict(list)
      # processing
      for feat_name, X in result.items():
        # some invalid feat_name
        if feat_name in ('config', 'pipeline'):
          raise RuntimeError("Returned features' name cannot be one "
                             "of the following: 'config', 'pipeline'.")
        # ignore some feat_name
        if feat_name in ('name'):
          continue
        # if numpy ndarray, save to MmapData
        if isinstance(X, np.ndarray) or \
        'sum1' == feat_name[-4:] or 'sum2' == feat_name[-4:]:
          # save statistics instead
          if 'sum1' == feat_name[-4:]:
            stats[feat_name[:-4]][0] += X
          elif 'sum2' == feat_name[-4:]:
            stats[feat_name[:-4]][1] += X
          # save features array
          else:
            all_indices[X.shape[0]].append(feat_name)
            # cache data, only if we have more than 0 sample
            if X.shape[0] > 0:
              cache[feat_name].append(X)
        # else all other kind of data save to MmapDict
        else:
          databases[feat_name][file_name] = X
        # remove data
        del X
      # ====== update indices ====== #
      all_indices = sorted(all_indices.items(),
                           key=lambda x: len(x[1]), reverse=True)
      if len(all_indices) > 0:
        # the first dominant indices
        n, _ = all_indices[0]
        databases['indices'][file_name] = (last_start['indices'],
                                           last_start['indices'] + n)
        last_start['indices'] += n
        # the rest append feature name
        for n, feats in all_indices[1:]:
          ids_name = 'indices_%s' % '_'.join(feats)
          databases[ids_name][file_name] = (last_start[ids_name],
                                            last_start[ids_name] + n)
          last_start[ids_name] += n
      # ====== flush cache ====== #
      nb_processed[0] += 1
      if nb_processed[0] % cache_limit == 0: # 12 + 8
        for feat_name, X_cached in cache.items():
          flush_feature(feat_name, X_cached)
        cache.clear()
      # ====== update progress ====== #
      return file_name
    # ====== processing ====== #
    mpi = MPI(jobs=self.jobs,
              func=self.extractor.transform,
              ncpu=self.ncpu, batch=1,
              hwm=self.ncpu * 3,
              backend='python')
    prog = Progbar(target=njobs, name=self.name,
                   interval=0.12, print_report=True, print_summary=True)
    for result in mpi:
      name = post_processing(result)
      prog['File'] = '%-20s' % str(name)
      prog.add(1)
    # ====== end, flush the last time ====== #
    for feat_name, X_cached in cache.items():
      flush_feature(feat_name, X_cached)
    cache.clear(); cache = None
    dataset.flush()
    prog.add_notification("Flushed all data to disk")
    # ====== saving indices ====== #
    for name, db in databases.items():
      db.flush(save_all=True)
      db_size = len(db)
      db.close()
      prog.add_notification('Flush MmapDict "%s" to disk, size: %s' %
                            (ctext(name, 'yellow'),
                             ctext(str(db_size), 'yellow')))

    # ====== save mean and std ====== #
    def save_mean_std(sum1, sum2, name):
      N = dataset[name.split('_')[0]].shape[0]
      mean = sum1 / N
      std = np.sqrt(sum2 / N - np.power(mean, 2))
      assert not np.any(np.isnan(mean)), 'Mean contains NaN, name: %s' % name
      assert not np.any(np.isnan(std)), 'Std contains NaN, name: %s' % name
      dataset[name + 'sum1'] = sum1
      dataset[name + 'sum2'] = sum2
      dataset[name + 'mean'] = mean
      dataset[name + 'std'] = std
    # save all stats
    if len(stats) > 0:
      for feat_name, (sum1, sum2) in stats.items():
        save_mean_std(sum1, sum2, feat_name)
        prog.add_notification('Saved statistics of: %s, shape: %s' %
                              (ctext(feat_name.split('_')[0], 'yellow'),
                               ctext(str(sum1.shape), 'yellow')))
    # ====== dataset flush() ====== #
    dataset.flush()
    dataset.close()
    # ====== saving the extractor ====== #
    pipeline_path = os.path.join(dataset.path, 'pipeline')
    with open(pipeline_path, 'wb') as f:
      cPickle.dump(self.extractor, f, protocol=2)
    prog.add_notification("Saved Extractor pipeline at: %s" %
                          ctext(pipeline_path, 'yellow'))
    # ====== saving the configuration ====== #
    config_path = os.path.join(dataset.path, 'config')
    config = MmapDict(config_path)
    config['__configuration_time__'] = time.time()
    config['__processor__'] = self.name
    for i in dir(self):
      if _default_module.match(i) is not None:
        continue
      j = getattr(self, i)
      if isinstance(j, (Number, string_types, bool)):
        config[i] = j
    config.flush(save_all=True)
    config.close()
    prog.add_notification("Saved configuration at: %s" %
                          ctext(config_path, 'yellow'))
    # ====== final notification ====== #
    prog.add_notification("Closed all dataset.")
    prog.add_notification("Dataset at path: %s" % ctext(dataset.path, 'yellow'))
