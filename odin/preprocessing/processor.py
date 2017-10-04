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


# ===========================================================================
# Speech features
# ===========================================================================
def _valid_segment_name(segments):
    for _, i in segments:
        if '.' in i[0] or ':' in i[0]:
            raise ValueError("Segment name cannot contain: '.' or ':', the given"
                " name is: %s" % i[0])


def _segments_preprocessing(segments, audio_ext):
    """ Filter segments into map of jobs
    Return
    ------
    jobs: dict
        file_name -> [segments, ...]
    nb_jobs: int
        total number of segment found
    """

    audio_ext = as_tuple('' if audio_ext is None else audio_ext,
                         t=string_types)
    # ====== load jobs ====== #
    if isinstance(segments, Dataset):
        # WAVE dataset
        if ('indices' in segments and 'raw' in segments and 'sr' in segments):
            file_list = [(name, segments, 0., -1., 0)
                         for name, (start, end) in segments['indices'].iteritems()]
        else: # assume that each key in dataset is a files
            file_list = [(os.path.basename(segments[k]), segments[k], 0.0, -1.0, 0)
                         for k in segments.keys()] # segment, path, start, end
    # NOT loaded segments
    elif isinstance(segments, str):
        if not os.path.exists(segments):
            raise ValueError('Path to segments must exists, however, '
                             'exist(segments)={}'.format(os.path.exists(segments)))
        # given a directory
        if os.path.isdir(segments):
            file_list = get_all_files(segments)
            file_list = [(os.path.basename(i), i, 0.0, -1.0)
                         for i in file_list] # segment, path, start, end
        # given csv file
        else:
            file_list = np.genfromtxt(segments, dtype=str, delimiter=' ')
    # LOADED segments
    elif isinstance(segments, (tuple, list, np.ndarray)):
        # just a list of path to file
        if isinstance(segments[0], str):
            file_list = [(os.path.basename(i), os.path.abspath(i), 0.0, -1.0)
                         for i in segments]
        # list of all information
        elif isinstance(segments[0], (tuple, list)):
            if len(segments[0]) == 1: # only path is given
                segments = [(path, path, 0., -1., 0) for path in segments]
            elif len(segments[0]) == 2: # name and path are given
                segments = [(name, path, 0., -1., 0) for name, path in segments]
            elif len(segments[0]) != 4 and len(segments[0]) != 5:
                raise Exception('segments must contain information in following order:'
                                '[name] [path] [start] [end] [channel]')
            file_list = segments
    # filter using support audio extension
    file_list = [f for f in file_list
                 if ((isinstance(f[1], str) and
                    any(ext in f[1][-len(ext):] for ext in audio_ext)) or
                 isinstance(f[1], Dataset))]
    # if no channel is provided, append the channel
    file_list = [list(f) + [0] if len(f) == 4 else f for f in file_list]
    nb_jobs = len(file_list)
    # convert into: audio_path -> list_of_segments[(name, start, end, channel), ...]
    jobs = []
    file_jobs = defaultdict(list)
    for segment, path_or_ds, start, end, channel in file_list:
        # Dataset related jobs
        if isinstance(path_or_ds, Dataset):
            jobs.append((path_or_ds, [(segment, start, end, channel)]))
        # audio files jobs
        else:
            file_jobs[path_or_ds].append(
                (segment, float(start), float(end), int(channel)))
    file_jobs = sorted(file_jobs.items(), key=lambda x: x[0])
    jobs += file_jobs
    _valid_segment_name(jobs)
    # check empty jobs
    if len(jobs) == 0:
        raise Exception('NO jobs found for processing.')
    return jobs, nb_jobs


def _load_audio(path_or_ds, segments,
                sr, sr_info={}, sr_new=None, best_resample=True,
                maxlen=None, vad_split=False, vad_split_args={},
                remove_dc_offset=True, remove_zeros=True):
    """ Return iterator of (name, data, sr) """
    # directory path for Dataset
    if is_string(path_or_ds) and os.path.isdir(path_or_ds):
        path_or_ds = Dataset(path_or_ds)
    # iterate over file path
    if is_string(path_or_ds) or isinstance(path_or_ds, file):
        s, sr_orig = speech.read(path_or_ds,
                                 remove_dc_offset=remove_dc_offset)
        # check original sample rate
        if sr_orig is not None and sr is not None and sr_orig != sr:
            raise RuntimeError('Given sample rate (%d Hz) is different from '
                               'audio file sample rate (%d Hz).' %
                               (sr, sr_orig))
        # get given sr
        if sr_orig is None:
            sr_orig = sr
        # get from sr_info
        if sr_orig is None and is_string(path_or_ds):
            sr_orig = sr_info.get(path_or_ds, None)
        # still None, then exception
        if sr_orig is None:
            raise RuntimeError("Cannot acquire original sample rate from "
                               "loaded utterance, or from given arguments "
                               "of this Processor (file: '%s')." % str(path_or_ds))
        # check if audio file is not long enough, ignore it
        if len(s) < 25:
            raise RuntimeError("Audio at path: '%s' is too short, length: %f(s)"
                               % (str(path_or_ds), len(s) / sr_orig))
        # downsampling
        if sr_new is not None:
            s = speech.resample(s, sr_orig, sr_new, best_algorithm=best_resample)
            sr_orig = sr_new
        N = len(s)
    # vad_split_audio kwargs
    minimum_duration = vad_split_args.get('minimum_duration', None)
    frame_length = vad_split_args.get('frame_length', 128)
    nb_mixtures = vad_split_args.get('nb_mixtures', 3)
    threshold = vad_split_args.get('threshold', 0.6)
    # ====== cut into segments ====== #
    for name, start, end, channel in segments:
        # iterate over dataset
        if isinstance(path_or_ds, Dataset):
            st, en = path_or_ds['indices'][name]
            s = path_or_ds['raw'][st:en]
            N = len(s)
            sr_orig = path_or_ds['sr'][name]
        # start processing
        if 0. <= start < 1. and 0. < end <= 1.: # percentage
            start = int(start * N)
            end = int(np.ceil(end * N))
        else: # given the duration in second
            start = int(float(start) * sr_orig)
            end = int(N if end <= 0 else float(end) * sr_orig)
        # check maxlen (maxlen is in duration second now)
        if maxlen is not None and (end - start > maxlen * sr_orig):
            # using VAD information to split the audio
            if vad_split:
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                data = signal.vad_split_audio(data, sr=sr_orig,
                    maximum_duration=maxlen, minimum_duration=minimum_duration,
                    frame_length=frame_length, nb_mixtures=nb_mixtures,
                    threshold=threshold)
                accum_length = np.cumsum([0] + [len(i) for i in data[:-1]])
                for st, d in zip(accum_length, data):
                    st_ = ('%f' % (st / sr_orig)).rstrip('0').rstrip('.')
                    en_ = ('%f' % ((st + len(d)) / sr_orig)).rstrip('0').rstrip('.')
                    yield (name + ":%s:%s" % (st_, en_),
                           d,
                           sr_orig)
            # greedy cut into small segments
            else:
                max_frames = int(maxlen * sr_orig)
                sectors = list(range(start, end, max_frames)) + [end]
                # list with start and end
                sectors = [(i, j) for i, j in zip(sectors, sectors[1:])]
                # merge the very short last sector into previous one.
                if sectors[-1][1] - sectors[-1][0] < max_frames // 2:
                    sectors = sectors[:-2] + [(sectors[-2][0], sectors[-1][1])]
                # return sector-by-sector
                for st, en in sectors:
                    st_ = ('%f' % (st / sr_orig)).rstrip('0').rstrip('.')
                    en_ = ('%f' % (en / sr_orig)).rstrip('0').rstrip('.')
                    yield (name + ":%s:%s" % (st_, en_),
                           s[st:en, channel] if s.ndim > 1 else s[st:en],
                           sr_orig)
        # return normally
        else:
            yield (name,
                   s[start:end, channel] if s.ndim > 1 else s[start:end],
                   sr_orig)


class WaveProcessor(FeatureProcessor):
    """ Concatenate all Waveform data into single memmap (or HDF5) file
    with its meta-data information included in the indices

    The saved Dataset contains 3 Data:
     * "indices": MmapDict contain the mapping from file name to (start, end).
     * "raw": the big memmap contains all concatenated raw waveform.
     * "sr": MmapDict contains the mapping from file name to its sample rate.

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted), `start` and `end`
        is in second (if `start`, or `end` is smaller than 1.0, then they
        are understand as percentage)
            name                |     path             |start|end |channel
        ------------------------|----------------------|-----|----|---
        sw02001-A_000098-001156 | /path/to/sw02001.sph | 0.0 | -1 | 0
        sw02001-B_001980-002131 | /path/to/sw02001.sph | 0.0 | -1 | 1
    output_path: str
        path to output folder
    sr: int
        sample rate
    sr_info: dict
        mapping audio_file_path -> sampling_rate for each segment
        if provided.
    sr_new: int or None
        new sample rate (if you want to down or up sampling)
    best_resample: bool
        if True, use the best but slow algorithm for resampling
    remove_dc_offset: bool
        if True, substract the mean of the audio array
    remove_zeros: bool
        if True, remove all zeros values in the audio array
    maxlen: int
        maximum length of an utterances in second, if any file is longer than
        given length, it is divided into small segments and the start time and
        end time are concatenated to the name (e.g. file:0:30)
    vad_split: boolean (default: False)
        using VAD information to split the audio in most likely silence part.
    vad_split_args: dict
        kwargs for `odin.preprocessing.signal.vad_split_audio`, includes:
        (minimum_duration, frame_length, nb_mixtures, threshold)
    dtype: numpy.dtype, None, 'auto'
        if None or 'auto', keep the original dtype of audio
    ignore_error: boolean (default: False)
        if True, ignore error files during processing
    """

    def __init__(self, segments, output_path,
                sr=None, sr_info={}, sr_new=None, best_resample=True,
                audio_ext=None, pcm=False,
                remove_dc_offset=True, remove_zeros=True,
                maxlen=None, vad_split=False, vad_split_args={},
                dtype='float16', datatype='memmap',
                ignore_error=False, ncache=0.12, ncpu=1):
        super(WaveProcessor, self).__init__(output_path=output_path,
            datatype=datatype, pca=False, pca_whiten=False,
            save_stats=False, substitute_nan=False,
            ncache=ncache, ncpu=ncpu)
        if isinstance(segments, Dataset):
            raise ValueError("WaveProcessor does not support segments as a Dataset.")
        self.maxlen = None if maxlen is None else int(maxlen)
        self.vad_split = bool(vad_split)
        self.vad_split_args = vad_split_args
        self.jobs, self.njobs = _segments_preprocessing(segments, audio_ext)
        if dtype is None or (is_string(dtype) and dtype == 'auto'):
            s, _ = speech.read(self.jobs[0][0], pcm=pcm, dtype=None)
            dtype = s.dtype
            del s
        self.sr = sr
        self.sr_info = sr_info
        self.sr_new = sr_new
        self.best_resample = bool(best_resample)
        self.dtype = dtype
        self.pcm = pcm
        self.remove_dc_offset = remove_dc_offset
        self.remove_zeros = remove_zeros
        self._features_properties = [('raw', self.dtype, False),
                                     ('sr', 'dict', False),
                                     ('dtype', 'dict', False)]
        self.ignore_error = bool(ignore_error)

    def map(self, job):
        audio_path, segments = job[0] if len(job) == 1 else job
        nb_jobs = len(segments)
        try:
            # processing all segments
            ret = []
            for name, data, sr in _load_audio(audio_path, segments,
                        self.sr, self.sr_info, self.sr_new, self.best_resample,
                        self.maxlen, self.vad_split, self.vad_split_args,
                        remove_dc_offset=self.remove_dc_offset,
                        remove_zeros=self.remove_zeros):
                ret.append([name, 0, [data, int(sr), data.dtype.str]])
            # a hack to return proper amount of processed jobs
            ret[-1][1] = nb_jobs
            # return result
            return (i for i in ret)
        except Exception as e:
            import traceback; traceback.print_exc()
            msg = '\n[Error file]: %s, [Exception]: %s\n' % (audio_path, str(e))
            if self.ignore_error:
                add_notification(msg)
            else:
                raise RuntimeError(msg)


class SpeechProcessor(FeatureProcessor):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted), `start` and `end` is in second
        (if `start`, or `end` is smaller than 1. then they are understand as
        percentage)
            name                |     path             |start|end |channel
        ------------------------|----------------------|-----|----|---
        sw02001-A_000098-001156 | /path/to/sw02001.sph | 0.0 | -1 | 0
        sw02001-B_001980-002131 | /path/to/sw02001.sph | 0.0 | -1 | 1
    output_path: str
        path to output folder
    sr: int
        sample rate
    sr_info: dict
        mapping audio_file_path -> sampling_rate for each segment
        if provided.
    sr_new: int or None
        new sample rate (if you want to down or up sampling)
    best_resample: bool
        if True, use the best but slow algorithm for resampling
    win: float
        window length in millisecond
    hop: float
        hop length between windows, in millisecond
    nb_melfilters: int, or None
        number of Mel bands to generate, if None, mel-filter banks features
        won't be returned
    nb_ceps: int, or None
        number of MFCCs to return, if None, mfcc coefficients won't be
        returned
    get_spec: bool
        if True, include the log-power spectrogram
    get_qspec: bool
        if True, return Q-transform coefficients
    get_phase: bool
        if True, return phase components of STFT
    get_pitch:
        if True, include the Pitch frequency (F0)
    get_vad: int, bool
        if True, include the indicators of voice activities detection.
        if int, `get_vad` is the number of Gaussian mixture components for VAD.
        by default, use 2 distribution.
    get_energy: bool
        if True, include the log energy of each frames
    get_delta: bool or int
        if True and > 0, for each features append the delta with given order-th
        (e.g. delta=2 will include: delta1 and delta2)
    fmin : float > 0 [scalar]
        lower frequency cutoff.
    fmax : float > 0 [scalar]
        upper frequency cutoff.
    preemphasis: float `(0, 1)`
        pre-emphasis coefficience
    pitch_threshold: float in `(0, 1)`
        Voice/unvoiced threshold. Default is 0.3.
    pitch_fmax: float
        maximum frequency of pitch
    pitch_algo: 'swipe', 'rapt', 'avg'
        SWIPE - A Saw-tooth Waveform Inspired Pitch Estimation.
        RAPT - a robust algorithm for pitch tracking.
        avg - apply swipe and rapt at the same time, then take average.
        Default is 'SWIPE'
    vad_smooth: int, bool
        window length to smooth the vad indices.
        If True default window length is 3.
    vad_minlen: float (in second)
        the minimum length of audio segments that can be considered
        speech.
    cqt_bins : int > 0
        Number of frequency bins for constant Q-transform, starting at `fmin`
    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
    log: bool
        if True, convert all power spectrogram to DB
    backend: 'odin', 'sptk'
        support backend for calculating the spectra
    pca: bool
        save trained PCA for each features
    pca_whiten : bool
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
    maxlen: int
        maximum length of an utterances in second, if any file is longer than
        given length, it is divided into small segments and the start time and
        end time are concatenated to the name (e.g. file:0:30)
    vad_split: boolean (default: False)
        using VAD information to split the audio in most likely silence part.
    vad_split_args: dict
        kwargs for `odin.preprocessing.signal.vad_split_audio`, includes:
        (minimum_duration, frame_length, nb_mixtures, threshold)
    save_raw: bool
        if True, saving the raw signal together with all the acoustic features
    save_stats: bool
        same the first order and second order statistics, standard deviation
        of all features
    substitute_nan: bool
        if the statistics contain NaN, replace them with zero of given
        value
    dtype: 'float16', 'float32', 'float64'
        the dtype of saved features
    datatype: 'memmap', 'hdf5'
        store processed features in memmap or hdf5
    ignore_error: boolean (default: False)
        if True, ignore error files during processing
    ncache: float or int
        number of samples are kept until flush to the disk.
    ncpu: int
        number of CPU used for this task.

    Return
    ------
    spec, mspec, mfcc, pitch, vad_idx

    Note
    ----
    For using invert-spectrogram, smaller hop (e.g. 0.005) creates much much
    better voices.

    Example
    -------
    >>> feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', fs=8000,
    >>>                          win=0.025, hop=0.01, n_filters=40, n_ceps=13,
    >>>                          delta_order=2, energy=True, pitch_threshold=0.5,
    >>>                          get_spec=True, get_mspec=True, get_mfcc=True,
    >>>                          get_pitch=False, get_vad=True,
    >>>                          save_stats=True, substitute_nan=None,
    >>>                          dtype='float32', datatype='memmap', ncpu=4)
    >>> feat.run()
    '''

    def __init__(self, segments, output_path,
                sr=None, sr_info={}, sr_new=None, best_resample=True,
                win=0.02, hop=0.01, window='hann',
                nb_melfilters=None, nb_ceps=None,
                get_spec=True, get_qspec=False, get_phase=False,
                get_pitch=False, get_f0=False,
                get_vad=True, get_energy=False, get_delta=False,
                fmin=64, fmax=None,
                pitch_threshold=0.3, pitch_fmax=260, pitch_algo='swipe',
                vad_smooth=3, vad_minlen=0.1,
                cqt_bins=96, preemphasis=None,
                center=True, power=2, log=True, backend='odin',
                pca=True, pca_whiten=False,
                audio_ext=None,
                maxlen=None, vad_split=False, vad_split_args={},
                save_raw=False, save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap',
                ignore_error=False, ncache=0.12, ncpu=1):
        super(SpeechProcessor, self).__init__(output_path=output_path,
            datatype=datatype, pca=pca, pca_whiten=pca_whiten,
            save_stats=save_stats, substitute_nan=substitute_nan,
            ncache=ncache, ncpu=ncpu)
        self.maxlen = None if maxlen is None else int(maxlen)
        self.vad_split = bool(vad_split)
        self.vad_split_args = vad_split_args
        self.jobs, self.njobs = _segments_preprocessing(segments, audio_ext)
        # ====== which features to get ====== #
        features_properties = []
        if save_raw:
            features_properties.append(('raw', dtype, False))
        if get_spec: features_properties.append(('spec', dtype, True))
        if get_energy: features_properties.append(('energy', dtype, True))
        if nb_melfilters is not None:
            features_properties.append(('mspec', dtype, True))
        if nb_ceps is not None:
            features_properties.append(('mfcc', dtype, True))
        if get_qspec:
            features_properties.append(('qspec', dtype, True))
            if nb_melfilters is not None:
                features_properties.append(('qmspec', dtype, True))
            if nb_ceps is not None:
                features_properties.append(('qmfcc', dtype, True))
            if get_phase: features_properties.append(('qphase', dtype, True))
        if get_phase: features_properties.append(('phase', dtype, True))
        if get_pitch: features_properties.append(('pitch', dtype, True))
        if get_f0: features_properties.append(('f0', dtype, True))
        if get_vad:
            features_properties.append(('vad', 'uint8', False))
        # store the sample rate of each file also
        features_properties.append(('sr', 'dict', False))
        self._features_properties = features_properties
        # control FeatureProcessor behaviour
        self._external_indices = ['raw']
        self._excluded_pca = ['energy', 'vad']
        # ====== local variable ====== #
        self.get_spec = get_spec
        self.get_pitch = get_pitch
        self.get_f0 = get_f0
        self.get_qspec = get_qspec
        self.get_phase = get_phase
        self.get_vad = get_vad
        self.get_energy = get_energy
        self.get_delta = 0 if get_delta is None else int(get_delta)
        self.save_raw = save_raw
        # ====== feature information ====== #
        self.sr = sr
        self.sr_new = sr_new
        self.sr_info = sr_info
        self.best_resample = bool(best_resample)
        self.win = win
        self.hop = hop
        self.window = window
        self.nb_melfilters = nb_melfilters
        self.nb_ceps = nb_ceps
        # constraint pitch threshold in 0-1
        self.pitch_threshold = min(max(pitch_threshold, 0.), 1.)
        self.pitch_fmax = pitch_fmax
        self.pitch_algo = pitch_algo
        self.vad_smooth = vad_smooth
        self.vad_minlen = vad_minlen
        self.cqt_bins = cqt_bins
        self.fmin = fmin
        self.fmax = fmax
        self.preemphasis = preemphasis
        self.center = center
        self.power = power
        self.log = log
        self.backend = backend
        self.ignore_error = bool(ignore_error)

    # ==================== Abstract properties ==================== #
    def map(self, job):
        '''
        Return
        ------
        [(name, spec, mspec, mfcc, pitch, vad), ...]
        '''
        audio_path, segments = job[0] if len(job) == 1 else job
        nb_jobs = len(segments)
        try:
            ret = []
            for name, data, sr_orig in _load_audio(audio_path, segments,
                                self.sr, self.sr_info, self.sr_new, self.best_resample,
                                self.maxlen, self.vad_split, self.vad_split_args):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    features = speech.speech_features(data.ravel(), sr=sr_orig,
                        win=self.win, hop=self.hop, window=self.window,
                        nb_melfilters=self.nb_melfilters, nb_ceps=self.nb_ceps,
                        get_spec=self.get_spec, get_qspec=self.get_qspec,
                        get_phase=self.get_phase,
                        get_pitch=self.get_pitch, get_f0=self.get_f0,
                        get_vad=self.get_vad, get_energy=self.get_energy,
                        get_delta=self.get_delta,
                        pitch_threshold=self.pitch_threshold,
                        pitch_fmax=self.pitch_fmax,
                        pitch_algo=self.pitch_algo,
                        vad_smooth=self.vad_smooth, vad_minlen=self.vad_minlen,
                        cqt_bins=self.cqt_bins, fmin=self.fmin, fmax=self.fmax,
                        sr_new=None, preemphasis=self.preemphasis,
                        center=self.center, power=self.power, log=self.log,
                        return_raw=self.save_raw, backend=self.backend)
                if features is not None:
                    saved_features = []
                    found_NaN = False
                    for i in self.features_properties[:-1]:
                        feat = features[i[0]]
                        if isinstance(feat, np.ndarray) and \
                        sum(feat.shape) > 0 and np.isnan(np.min(feat)):
                            found_NaN = True
                        else:
                            saved_features.append(feat)
                    # append the sample rate
                    if found_NaN:
                        warnings.warn('Ignore segments: %s, error: NaN values' % name)
                    else:
                        saved_features.append(sr_orig if self.sr_new is None
                                              else self.sr_new)
                        ret.append([name, 0, saved_features])
                else:
                    warnings.warn('Ignore segments: %s, no features found' % name)
            # a hack to return proper amount of processed jobs
            ret[-1][1] = nb_jobs
            # return the results as a generator
            return (i for i in ret)
        except Exception as e:
            import traceback; traceback.print_exc()
            msg = '\n[Error file]: %s, [Exception]: %s\n' % (audio_path, str(e))
            if self.ignore_error:
                add_notification(msg)
            else:
                raise RuntimeError(msg)
