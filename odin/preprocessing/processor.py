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

from odin.ml import MiniBatchPCA
from odin.utils.mpi import MPI
from odin.utils import (Progbar, as_tuple, get_all_files, ctext,
                        get_tempdir, is_string, batching,
                        add_notification, keydefaultdict,
                        stdio, get_stdio_path, wprint)
from odin.fuel import Dataset, MmapDict

from .base import Extractor
_default_module = re.compile('__.*__')


# ===========================================================================
# Helper
# ===========================================================================
# ==================== For speech ==================== #
def _escape_filename(file_name):
    return file_name.replace('/', '-')


def _plot_data(data, feat_name, file_name,
               dataset, processor, outpath):
    from matplotlib import pyplot as plt
    from odin.visual import plot_save, plot_spectrogram
    from scipy.io.wavfile import write
    # ====== Audio signals ====== #
    if processor in (SpeechProcessor, WaveProcessor):
        # ====== raw  ====== #
        if feat_name == 'raw':
            path = os.path.join(outpath,
                    _escape_filename(file_name.split('.')[0] + '.wav'))
            data = data[:].astype('float32')
            data = (data - data.mean()) / data.std()
            write(path, rate=dataset['sr'][file_name], data=data)
            # saving figure
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                plt.figure(figsize=(10, 4))
                data = speech.resample(data, dataset['sr'][file_name], 8000,
                                       best_algorithm=True)
                if 'vad' in dataset:
                    plt.subplot(2, 1, 1)
                    plt.plot(data)
                    start, end = dataset['indices'][file_name]
                    plt.subplot(2, 1, 2)
                    plt.plot(dataset['vad'][start:end][:])
                else:
                    plt.plot(data)
                plt.suptitle(file_name)
        # ====== speech features ====== #
        elif feat_name in ('spec', 'mspec', 'mfcc', 'qspec', 'qmspec', 'qmfcc'):
            data = data[:].astype('float32')
            # saving figure
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                plt.figure(figsize=(10, 4))
                if 'vad' in dataset:
                    start, end = dataset['indices'][file_name]
                    plot_spectrogram(data.T, vad=dataset['vad'][start:end][:])
                else:
                    plot_spectrogram(data.T)
                plt.suptitle(file_name)
            # invert spectrogram
            if feat_name == 'spec':
                sr = dataset['sr'][file_name]
                hop = dataset['config']['hop']
                raw = signal.ispec(data, hop_length=hop * sr, nb_iter=48,
                                   db=dataset['config']['log'],
                                   normalize=True,
                                   center=dataset['config']['center'])
                path = os.path.join(outpath,
                        _escape_filename(file_name.split('.')[0] + '-ispec.wav'))
                write(path, rate=sr, data=raw)
        # ====== energy, f0, pitch ====== #
        elif feat_name in ('energy', 'f0', 'pitch'):
            data = data[:].astype('float32')
            plt.figure(figsize=(10, 4))
            plt.plot(data)
            plt.suptitle(file_name)


def validate_features(ds_or_processor, path, nb_samples=25,
                      override=False):
    def logger(title, tag, check):
        print(ctext('   *', 'cyan'),
              ctext(str(title), 'yellow'),
              ctext(str(tag), 'magenta'),
              "✓" if bool(check) else "✗")
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6
    from matplotlib import pyplot as plt
    from odin.visual import plot_save
    # ====== check path to dataset ====== #
    if isinstance(ds_or_processor, FeatureProcessor):
        ds = Dataset(ds_or_processor.output_path, read_only=True)
    elif is_string(ds_or_processor):
        ds = Dataset(ds_or_processor, read_only=True)
    elif isinstance(ds_or_processor, Dataset):
        ds = ds_or_processor
    else:
        raise ValueError("`ds` can be None, string, or Dataset. No "
                         "support for given input type: %s" % str(type(ds)))
    print(ctext('Validating dataset:', 'yellow'),
          '"%s"' % ds.path)
    # ====== extract the config of the dataset ====== #
    if 'config' not in ds:
        raise RuntimeError("The `Dataset` must be generated by `FeatureProcessor` "
                           "which must contain `config` MmapDict of extracted "
                           "features configuration.")
    config = ds['config']
    processor = eval(config['__processor__'])
    if not issubclass(processor, FeatureProcessor):
        raise RuntimeError("This Dataset was created by %s which is NOT "
                           "subclass of `FeatureProcessor`." % str(processor))
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
    features_properties = config['features_properties']
    external_indices = config['external_indices']
    excluded_pca = config['excluded_pca']
    all_keys = ds.keys()
    # ====== checking indices ====== #
    main_indices = {name: (start, end)
                    for name, (start, end) in ds['indices'].iteritems()}
    for ids_name in (k for k in all_keys if 'indices' in k):
        ids = sorted([(name, start, end)
                      for name, (start, end) in ds[ids_name].iteritems()],
                     key=lambda x: x[1])
        for prev, now in zip(ids, ids[1:]):
            assert prev[2] == now[1], "Zero length in indices"
            assert prev[2] - prev[1] > 0, "Zero length in indices"
            assert now[2] - now[1] > 0, "Zero length in indices"
        # final length match length of Dat
        if ids_name != 'indices':
            assert now[-1] == len(ds[ids_name.split('_')[-1]]), ids_name
        else:
            for name, dtype, _ in features_properties:
                if name not in external_indices and dtype != 'dict':
                    assert now[-1] == len(ds[name]), \
                    "Length of indices and actual data mismatch, " + ids_name + ':' + name
        # logging
        logger("Checked all:", ids_name, True)
    # ====== check all dictionary types ====== #
    for name, dtype, stats_able in features_properties:
        if dtype == 'dict':
            data = ds[name]
            # special cases
            if name == 'sr':
                checking_func = lambda x: x > 0 # for sr
            else:
                checking_func = lambda x: True
            # check
            for key, val in data.iteritems():
                assert key in main_indices, \
                "Dictionary key not found in indices (dict:%s dtype:%s)" % (name, str(dtype))
                assert checking_func(val)
            logger("Checked dictionary type: ", name, True)
    # ====== checking each type of data ====== #
    sampled_name = np.random.choice(ds['indices'].keys(),
                                    size=nb_samples,
                                    replace=False)
    for feat_name, dtype, stats_able in features_properties:
        if feat_name == 'vad' or dtype == 'dict':
            continue
        figure_path = os.path.join(path, '%s.pdf' % feat_name)
        # checking all data
        indices = ds['indices'] if feat_name not in external_indices else \
            ds['indices_%s' % feat_name]
        prog = Progbar(target=len(indices), interval=0.1,
                       print_report=True,
                       name='Checking: %s(%s)' % (feat_name, dtype))
        # start iterating over all data file
        for file_name, (start, end) in indices:
            dat = ds[feat_name][start:end]
            # picking sample for visualization
            if file_name in sampled_name:
                _plot_data(data=dat, feat_name=feat_name, file_name=file_name,
                           dataset=ds, processor=processor, outpath=path)
            # No NaN value
            assert not np.any(np.isnan(dat)), \
                "NaN values in file: %s, feat: %s" % (file_name, feat_name)
            # not all value closed to zeros
            assert not np.all(np.isclose(dat, 0.)),\
                "All-zeros values in file: %s, feat: %s" % (file_name, feat_name)
            prog['Name'] = file_name
            prog.add(1)
        logger("Check data incredibility for: ", feat_name, True)
        # checking statistics
        if stats_able:
            plt.figure()
            for i, stats in enumerate(['_mean', '_std', '_sum1', '_sum2']):
                stats_name = stats[1:]
                stats = ds[feat_name + stats][:]
                assert not np.any(np.isnan(stats)),\
                    "NaN values in stat: %s, feat: %s" % (stats_name, feat_name)
                assert not np.all(np.isclose(stats, 0.)),\
                    "All-zeros values in stats: %s, feat: %s" % (stats_name, feat_name)
                plt.subplot(4, 2, i * 2 + 1)
                plt.plot(stats)
                plt.ylabel(stats_name)
                plt.subplot(4, 2, i * 2 + 2)
                plt.scatter(np.arange(len(stats)), stats, s=1.)
            plt.suptitle('"%s" Statistics' % feat_name)
            logger("Check statistics for: ", feat_name, True)
            # check PCA
            if feat_name not in excluded_pca and \
            feat_name + '_pca' in ds:
                pca = ds[feat_name + '_pca']
                n = len(ds[feat_name])
                nb_feats = ds[feat_name].shape[-1]
                # performing PCA on random samples
                for i in range(nb_samples):
                    start = np.random.randint(0, n - nb_samples - 1)
                    X = pca.transform(
                        ds[feat_name][start:(start + nb_samples)],
                        n_components=max(nb_feats // 2, 1))
                    assert not np.any(np.isnan(X)),\
                        "NaN values in PCA of feat: %s" % feat_name
                    assert not np.all(np.isclose(X, 0.)),\
                        "All-zeros values in PCA of feat: %s" % feat_name
                logger("Check PCA for: ", feat_name, True)
        # saving all the figures
        plot_save(figure_path, dpi=80, log=False, clear_all=True)
        logger("Figure save at: ", figure_path, True)
    logger("All reports at folder: ", os.path.abspath(path), True)
    # ====== cleaning ====== #
    stdio(path=prev_stdio)
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
    pca: bool
        if True, calculte PCA for returned features as numpy.ndarray
        with `ndim > 1` and `dtype is float`.
    excluded_pca: tuple of str
        all features with given name in this list is ignored during
        computation of pca
    ncache: float or int (> 0)
        pass
    ncpu: int (>0)
        number of Processes will be used to parallel the processor
    name: None or str
        identity for the Processor
    """

    def __init__(self, jobs, extractor, path,
                 pca=True, excluded_pca=(),
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
        if not isinstance(jobs, (tuple, list)):
            raise ValueError("Provided `jobs` must be instance of tuple or list.")
        self.jobs = tuple(jobs)
        # ====== check PCA ====== #
        self.pca = bool(pca)
        self.excluded_pca = as_tuple(excluded_pca, t=str)
        # ====== check multiprocessing ====== #
        if ncpu is None: # auto select number of CPU
            ncpu = min(len(jobs), int(1.2 * cpu_count()))
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
            steps = [(str(n), e) for n, e in extractor.iteritems()]
            extractor = Pipeline(steps=steps)
        elif isinstance(extractor, Extractor):
            extractor = Pipeline(
                steps=[(extractor.__class__.__name__, extractor)])
        self.extractor = extractor

    # ==================== Abstract properties ==================== #
    def _map_multiple_works(self, jobs):
        for j in jobs:
            yield self.extractor.transform(j)

    def run(self):
        njobs = len(self.jobs)
        dataset = Dataset(self.path)
        if self.ncache <= 1:
            cache_limit = max(2, int(0.12 * njobs))
        else:
            cache_limit = int(self.ncache)
        # ====== indices ====== #
        databases = keydefaultdict(lambda key:
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
        # init PCA
        pca = defaultdict(lambda: MiniBatchPCA(n_components=None, whiten=False,
                                               copy=True, batch_size=None))
        # all data are cached for periodically flushed
        cache = defaultdict(list)
        nb_processed = [0] # store the value as reference
        last_start = defaultdict(int)

        # ====== helper ====== #
        def flush_feature(feat_name, X_cached):
            if len(X_cached) > 0:
                X_cached = np.concatenate(X_cached, 0)
                # NOTE: if nb_samples < nb_features,
                # fitting PCA will course error
                if self.pca and feat_name not in self.excluded_pca and \
                (X_cached.ndim >= 2 and all(s > 1 for s in X_cached.shape) and
                 'float' in str(X_cached.dtype).lower()):
                    pca[feat_name].partial_fit(X_cached)
                # flush data
                if feat_name in dataset:
                    dataset[feat_name].append(X_cached)
                else:
                    dataset[(feat_name, 'memmap')] = X_cached

        # ====== repeated for each result returned ====== #
        def wrapped_reduce(result):
            # returned result is always dictionary or None
            # if dictionary, it is mapping: name -> feature_matrix
            if result is None:
                return
            # search for file name
            file_name = result.get('name', None)
            if file_name is None:
                file_name = result.get('path', None)
            if file_name is None or not is_string(file_name):
                raise RuntimeError("Cannot find file name in returned features "
                    "list, the file name can be specified in key: 'name', 'path' "
                    "and the type of the value must be string. All available "
                    "keys are: %s" % str(result.keys()))
            # store all new indices
            all_indices = defaultdict(list)
            # processing
            for feat_name, X in result.iteritems():
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
            all_indices = sorted(all_indices.iteritems(),
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
                for feat_name, X_cached in cache.iteritems():
                    flush_feature(feat_name, X_cached)
                cache.clear()
            # ====== update progress ====== #
            return file_name
        # ====== processing ====== #
        mpi = MPI(jobs=self.jobs,
                  map_func=self._map_multiple_works,
                  reduce_func=wrapped_reduce,
                  ncpu=self.ncpu,
                  buffer_size=min(8, max(len(self.jobs) // self.ncpu, 1)),
                  maximum_queue_size=self.ncpu * 3,
                  chunk_scheduler=True)
        prog = Progbar(target=njobs, name=self.name,
                       interval=0.12, print_report=True, print_summary=True)
        for name in mpi:
            prog['File'] = '%-20s' % str(name)
            prog.add(1)
        # ====== end, flush the last time ====== #
        for feat_name, X_cached in cache.iteritems():
            flush_feature(feat_name, X_cached)
        cache.clear(); cache = None
        dataset.flush()
        prog.add_notification("Flushed all data to disk")
        # ====== saving indices ====== #
        for name, db in databases.iteritems():
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
            for feat_name, (sum1, sum2) in stats.iteritems():
                save_mean_std(sum1, sum2, feat_name)
                prog.add_notification('Saved statistics of: %s, shape: %s' %
                                      (ctext(feat_name.split('_')[0], 'yellow'),
                                       ctext(str(sum1.shape), 'yellow')))
        # ====== save all PCA ====== #
        for name, pca_model in pca.iteritems():
            if pca_model is not None and pca_model.is_fitted:
                dataset[name + '_pca'] = pca_model
                prog.add_notification('Stored PCA model of: %s' %
                                      ctext(name, 'yellow'))
        # ====== dataset flush() ====== #
        dataset.flush()
        dataset.close()
        # ====== saving the extractor ====== #
        pipeline_path = os.path.join(dataset.path, 'pipeline')
        with open(pipeline_path, 'w') as f:
            cPickle.dump(self.extractor, f, protocol=2)
        prog.add_notification("Saved Extractor pipeline at: %s" %
                              ctext(pipeline_path, 'yellow'))
        # ====== saving the configuration ====== #
        config_path = os.path.join(dataset.path, 'config')
        config = MmapDict(config_path)
        config['__configuration_time__'] = time.time()
        config['__processor__'] = self.name
        config['excluded_pca'] = self.excluded_pca
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
