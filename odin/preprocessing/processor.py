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
                        add_notification, flatten_list,
                        get_formatted_datetime)
from odin.fuel import Dataset, MmapDict, MmapData
from odin.preprocessing.base import Extractor, ExtractorSignal

_default_module = re.compile(r"__.*__")

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
    from odin.preprocessing.speech import save
    sr = ds['sr'][file_name]
    if '.wav' not in file_name:
      file_name += '.wav'
    save(os.path.join(path, _escape_file_name(file_name)),
         X.astype('float32'), sr=sr)
  elif feat_name == 'spec':
    from odin.preprocessing.speech import (SpectraExtractor, STFTExtractor, Power2Db,
                                          _extract_frame_step_length, save)
    from odin.preprocessing.signal import ispec
    sr = ds['sr'][file_name]
    extractor = [i for _, i in ds['pipeline'].steps
                 if isinstance(i, SpectraExtractor) or isinstance(i, STFTExtractor)][0]
    frame_length, step_length = _extract_frame_step_length(sr,
        extractor.frame_length, extractor.step_length)
    raw = ispec(X,
                frame_length=frame_length, step_length=step_length,
                window=extractor.window, padding=extractor.padding,
                db=extractor.log if hasattr(extractor, 'log') else
                any(isinstance(i, Power2Db) for i in ds['pipeline'].steps))
    file_name += '-ispec.wav'
    save(os.path.join(path, _escape_file_name(file_name)),
         raw.astype('float32'), sr=sr)


def validate_features(ds_or_processor, path, nb_samples=25,
                      override=False, seed=12082518, fig_width=4):
  # TODO: add PCA visualization
  # TODO: update to match new indices style
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
  if 'config' not in ds:
    raise RuntimeError("The `Dataset` must be generated by `FeatureProcessor` "
                       "which must contain `config` MmapDict of extracted "
                       "features configuration.")
  # config = ds['config']
  # pipeline = ds['pipeline']
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
def _check_logpath(log_path):
  main_path, ext = os.path.splitext(log_path)
  main_path = main_path.split('.')
  try:
    current_log_index = int(main_path[-1])
    main_path = main_path[:-1]
  except ValueError:
    current_log_index = 0
  main_path = '.'.join(main_path)
  # ====== increase log index until found a new file ====== #
  while True:
    path = main_path + '.' + str(current_log_index) + ext
    if not os.path.exists(path):
      break
    current_log_index += 1
  return main_path + '.' + str(current_log_index) + ext

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

  n_cache: float or int (> 0)
      pass

  ncpu: int (>0)
      number of Processes will be used to parallel the processor

  override : bool (default: False)
      if output folder already exist, deleted old features if
      `override=True`

  identifier : string (default: 'name')
      extractor will return a dictionary, `identifier` is the key
      of the entry that will be used to identify (or distinguish)
      between files (or images, or samples).
      Note this key must exist in the returned dictionary,
      otherwise, RuntimeError will be raised

  log_path : string
      path to log file

  stop_on_failure : bool (default: False)
      There are two types or error during running a FeatureProcessor.

      The first type is handled by `odin.preprocessing.base.Extractor`
      and return `odin.preprocessing.base.ExtractorSignal` which
      contained instruction for handling the Exception.

      The second type is non-handled by the `Extractor`, the
      `FeatureProcessor` will automatically handle this Exception
      getting the traceback and write everything to the log file
      before decided to continue or terminating the process.

      if True, terminate the processor if non-handled Exception
      appeared.
  """

  def __init__(self, jobs, path, extractor,
               n_cache=0.12, ncpu=1, override=False,
               identifier='name',
               log_path=None,
               stop_on_failure=False):
    super(FeatureProcessor, self).__init__()
    # ====== check outpath ====== #
    path = os.path.abspath(str(path))
    if os.path.isfile(path):
      raise ValueError("`path` must be path to a directory, but found a "
                       "path to file.")
    # check override
    if os.path.exists(path) and override:
      wprint("Remove existed Dataset at path: %s" % path)
      for i in os.listdir(path):
        i = os.path.join(path, i)
        if os.path.isdir(i): # remove folder
          shutil.rmtree(i)
        else: # remove file
          os.remove(i)
    # set path and name
    self.path = path
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
    if ncpu <= 0 or n_cache <= 0:
      raise ValueError('`ncpu` and `n_cache` must be greater than 0, but '
                       'given values ncpu=%d n_cache=%f' % (ncpu, n_cache))
    self.n_cpu = ncpu
    self.n_cache = n_cache
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
    # ====== check identifier and log path ====== #
    self._identifier = str(identifier)
    if log_path is None:
      log_path = os.path.join(self.path, 'log.txt')
    else:
      log_path = str(log_path)
    self._log_path = _check_logpath(log_path)
    # ====== others ====== #
    self.config = {}
    self._error_log = []
    self.stop_on_failure = bool(stop_on_failure)

  @property
  def identifier(self):
    return self._identifier

  @property
  def error_log(self):
    return list(self._error_log)

  # ==================== debugging ==================== #
  def __str__(self):
    s = ctext('============= FeatureProcessor: %s =============' % self.path, 'yellow') + '\n'
    padding = '  '
    # ====== basic info ====== #
    s += '- Jobs: ' + ctext(len(self.jobs), 'cyan') + '\n'
    s += '- #CPU: ' + ctext(self.n_cpu, 'cyan') + '\n'
    s += '- #Cache: ' + ctext(self.n_cache, 'cyan') + '\n'
    # ====== print pipeline ====== #
    s += ctext("* Pipeline:", 'yellow') + '\n'
    for _, extractor in self.extractor.steps:
      for line in str(extractor).split('\n'):
        s += padding + ' ' + line + '\n'
    # ====== print config ====== #
    s += ctext("* Configurations:", 'yellow') + '\n'
    for i, j in self.config.items():
      s += padding + str(i) + ' : ' + str(j) + '\n'
    return s

  def __repr__(self):
    return self.__str__()

  # ==================== Abstract properties ==================== #
  def run(self):
    njobs = len(self.jobs)
    dataset = Dataset(self.path)
    if self.n_cache <= 1:
      cache_limit = max(2, int(0.12 * njobs))
    else:
      cache_limit = int(self.n_cache)
    # ====== indices ====== #
    databases = defaultdictkey(lambda key:
        MmapDict(path=os.path.join(dataset.path, key), cache_size=10000,
                 read_only=False))
    last_start = defaultdict(int)
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
    n_processed = [0] # store the value as reference

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
      # search for file name
      if self.identifier not in result:
        raise RuntimeError(
            "Cannot find identifier '%s' in returned dictionary" % self.identifier)
      file_name = result[self.identifier]
      # invalid file_name
      if not is_string(file_name):
        raise RuntimeError("Cannot find file name in returned features "
            "list, the file name can be specified in key: 'name', 'path' "
            "and the type of the value must be string. All available "
            "keys are: %s" % str(result.keys()))
      # store all new indices
      # mapping [X.shape[0]] -> [feat_name, feat_name, ...]
      all_indices = {}
      # processing
      for feat_name, X in result.items():
        # some invalid feat_name
        if feat_name in ('config', 'pipeline', 'sum1', 'sum2'):
          raise RuntimeError("Returned features' name cannot be one "
                             "of the following: 'config', 'pipeline', 'sum1', 'sum2'.")
        # ignore some feat_name
        if feat_name in ('name'):
          continue
        # if numpy ndarray, save to MmapData
        if isinstance(X, np.ndarray) or \
        'sum1' == feat_name[-4:] or \
        'sum2' == feat_name[-4:]:
          # save statistics instead
          if 'sum1' == feat_name[-4:]:
            stats[feat_name[:-4]][0] += X
          elif 'sum2' == feat_name[-4:]:
            stats[feat_name[:-4]][1] += X
          # save features array
          else:
            all_indices[feat_name] = X.shape[0]
            # cache data, only if we have more than 0 sample
            if X.shape[0] > 0:
              cache[feat_name].append(X)
        # else all other kind of data save to MmapDict
        else:
          databases[feat_name][file_name] = X
        # remove data
        del X
      # ====== update indices ====== #
      if len(all_indices) > 0:
        for feat_name, n in all_indices.items():
          ids_name = 'indices_%s' % feat_name
          databases[ids_name][file_name] = (last_start[ids_name],
                                            last_start[ids_name] + n)
          last_start[ids_name] += n
      # ====== flush cache ====== #
      n_processed[0] += 1
      if n_processed[0] % cache_limit == 0: # 12 + 8
        for feat_name, X_cached in cache.items():
          flush_feature(feat_name, X_cached)
        cache.clear()
      # ====== update progress ====== #
      return file_name

    # ====== mapping function ====== #
    def _map_func(dat):
      try:
        ret = self.extractor.transform(dat)
      except Exception as e: # Non-handled exception
        ret = '\n========\n'
        ret += 'Time  : `%s`\n' % str(get_formatted_datetime(only_number=False))
        ret += 'Error : `%s`\n' % str(e)
        ret += 'Input : `%s`\n' % str(dat)
        import traceback
        etype, value, tb = sys.exc_info()
        for line in traceback.TracebackException(
                type(value), value, tb, limit=None).format(chain=True):
          ret += line
      return ret
    # ====== processing ====== #
    mpi = MPI(jobs=self.jobs,
              func=_map_func,
              ncpu=self.n_cpu,
              batch=1,
              hwm=self.n_cpu * 3,
              backend='python')
    # initialize
    prog = Progbar(target=njobs, name=self.path,
                   interval=0.12, print_report=True, print_summary=True)
    start_time = time.time()
    last_time = time.time()
    last_count = 0
    with open(self._log_path, 'w') as flog:
      # writing the log head
      flog.write('============================\n')
      flog.write('Start Time : %s\n' % get_formatted_datetime(only_number=False))
      flog.write('Outpath    : %s\n' % self.path)
      flog.write('Extractor  : %s\n' % '->'.join([s[-1].__class__.__name__
                                                  for s in self.extractor.steps]))
      flog.write('#Jobs      : %d\n' % njobs)
      flog.write('#CPU       : %d\n' % self.n_cpu)
      flog.write('#Cache     : %d\n' % cache_limit)
      flog.write('============================\n')
      flog.flush()
      # start processing the file list
      for count, result in enumerate(mpi):
        # Non-handled exception
        if isinstance(result, string_types):
          flog.write(result)
          flog.flush()
          self._error_log.append(result)
          if self.stop_on_failure:
            raise RuntimeError(result)
        # some error might happened
        elif isinstance(result, ExtractorSignal):
          flog.write(str(result)); flog.flush()
          if result.action == 'error':
            prog.add_notification(str(result))
            raise RuntimeError("ExtractorSignal requests terminating processor!")
          elif result.action == 'warn':
            prog.add_notification(str(result))
          elif result.action == 'ignore':
            self._error_log.append(result)
          else:
            raise RuntimeError("Unknown action from ExtractorSignal: %s" % result.action)
          prog['File'] = '%-48s' % result.message[:48]
        # otherwise, no error happened, do post-processing
        else:
          name = post_processing(result)
          prog['File'] = '%-48s' % str(name)[:48]
        # update progress
        prog.add(1)
        # manually write to external log file
        if (count + 1) % max(1, int(0.01 * njobs)) == 0:
          curr_time = time.time()
          elap = curr_time - start_time
          avg_speed = (count + 1) / elap
          cur_speed = (count + 1 - last_count) / (curr_time - last_time)
          avg_est = (njobs - count - 1) / avg_speed
          cur_est = (njobs - count - 1) / cur_speed
          flog.write('[%s] Processed: %d(files)   Remain: %d(files)   Elap.: %.2f(secs)\n'
                     '   Avg.Spd: %.2f(obj/sec)  Avg.Est.: %.2f(secs)\n'
                     '   Cur.Spd: %.2f(obj/sec)  Cur.Est.: %.2f(secs)\n' %
                     (get_formatted_datetime(only_number=False),
                      count + 1, njobs - count - 1, elap,
                      avg_speed, avg_est,
                      cur_speed, cur_est))
          flog.flush()
          last_time = curr_time
          last_count = count + 1
    # ====== end, flush the last time ====== #
    for feat_name, X_cached in cache.items():
      flush_feature(feat_name, X_cached)
    cache.clear()
    cache = None
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
      if np.any(np.isnan(mean)):
        wprint('Mean contains NaN, name: %s' % name)
      if np.any(np.isnan(std)):
        wprint('Std contains NaN, name: %s' % name)
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
    # not good idea to save the extractor all the time
    # pipeline_path = os.path.join(dataset.path, 'pipeline')
    # with open(pipeline_path, 'wb') as f:
    #   cPickle.dump(self.extractor, f, protocol=2)
    # prog.add_notification("Saved Extractor pipeline at: %s" %
    #                       ctext(pipeline_path, 'yellow'))
    # ====== saving the configuration ====== #
    config_path = os.path.join(dataset.path, 'config')
    config = MmapDict(config_path)
    config['__configuration_time__'] = time.time()
    config['__processor__'] = self.path
    for i in dir(self):
      if _default_module.match(i) is not None:
        continue
      j = getattr(self, i)
      if isinstance(j, (Number, string_types, bool)):
        config[i] = j
    config.flush(save_all=True)
    self.config = {i: j
                   for i, j in config}
    config.close()
    prog.add_notification("Saved configuration at: %s" %
                          ctext(config_path, 'yellow'))
    # ====== final notification ====== #
    prog.add_notification("Closed all dataset.")
    prog.add_notification("Dataset at path: %s" % ctext(dataset.path, 'yellow'))
