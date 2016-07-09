# ===========================================================================
# Parallel features processing using multi-core CPU and multiprocessing
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import sys
import os
import warnings
from multiprocessing import Pool, cpu_count
from six import add_metaclass
from six.moves import zip, zip_longest, range
from abc import ABCMeta

from collections import defaultdict
import numpy as np
try: # this library may not available
    from scikits.samplerate import resample
except:
    pass

from odin.preprocessing import speech
from odin.utils import queue, Progbar, segment_list, as_tuple, get_all_files
from odin.utils.decorators import functionable, abstractstatic, autoinit
from .dataset import Dataset


__all__ = [
    'MapReduce',
    'FeatureRecipe',
    'SpeechFeature'
]


# ===========================================================================
# MPI MapReduce
# ===========================================================================
class MapReduce(object):

    """ This class manage all MapReduce task by callback function:

    map_function : argmuents(static_data, job)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        job: is a single job that automatically scheduled to each MPI process

    reduce_function : arguments(static_data, results, finnished)
        static_data: dictionary, which initialized right after you set the
        init_funciotn
        results: list, of returned result from each map_function (None returned
        will be ignored)
        finnished: bool, whether this function is called at the end of MapReduce
        task

    Example
    -------
    >>> def function(a):
    ...     x, y = a
    ...     return x + y

    >>> def function1(x):
    ...     return x - 1

    >>> mr = MapReduce(2, 1)
    >>> mr.cache = 12
    >>> mr.push(zip(range(26), reversed(range(26))),
    ...         function, lambda x: x, name='hello')
    >>> mr.push('hello', function1, lambda x: x, name='hi')
    >>> mr()
    >>> print(mr['hello']) # [25, ...]
    >>> print(mr['hi']) # [24, ...]

    """

    def __init__(self, processes=8, cache=5, verbose=1):
        super(MapReduce, self).__init__()
        # variables
        self._cache = int(min(cache, 1))
        self._tasks = queue()
        self._processes = int(min(processes, cpu_count() - 1))
        self._pool = Pool(processes)
        self._results = defaultdict(list)

    # ==================== Get & set ==================== #
    @property
    def cache(self):
        return self._cache

    @property
    def processes(self):
        return self._processes

    # ==================== Task manager ==================== #
    def add_recipe(self, recipe):
        if isinstance(recipe, str): # path to pickled file or pickled string
            import cPickle
            if os.path.exists(recipe):
                recipe = cPickle.load(open(recipe, 'r'))
            else:
                recipe = cPickle.loads(recipe)

        if not isinstance(recipe, (tuple, list)):
            recipe = (recipe,)
        if not all(isinstance(i, FeatureRecipe) for i in recipe):
            raise ValueError('Given recipe is not instance of FeatureRecipe, '
                             'but has type={}'.format(map(type, recipe)))
        for i in recipe:
            self._tasks.append(i) # in this case, tasks contain recipe

    def add(self, jobs, map_func, reduce_func=None, finalize_func=None,
            init_func=None, name=None):
        ''' Wrapped preprocessing procedure in multiprocessing.
                ....root
                / / / | \ \ \ ,
                .mapping_func
                \ \ \ | / / /
                .reduce_func
                ......|
                .finalize_func

        Parameters
        ----------
        jobs : list
            [data_concern_job_1, job_2, ....]

        map_func : function(dict, job_i)
            function object to extract feature from each job, the dictionary
            will contain all static data initilized from set_init function

        reduce_func : function(dict, [job_i,...], finnished)
            transfer all data to process 0 as a list for saving to disk, the
            dictionary will contain all static data initilized from set_init
            function

        Notes
        -----
        Any None return by features_func will be ignored

        '''
        if not hasattr(map_func, '__call__') or \
            (reduce_func is not None and not hasattr(reduce_func, '__call__')) or \
            (finalize_func is not None and not hasattr(finalize_func, '__call__')) or \
                (init_func is not None and not hasattr(init_func, '__call__')):
            raise ValueError('map, reduce, finalize and init function must be callable'
                             ' object, but map_func={}, reduce_func={}, '
                             'finalize_func={} and init_func={}'
                             ''.format(type(map_func), type(reduce_func),
                                type(finalize_func), type(init_func)))
        self._tasks.append([jobs, map_func, reduce_func, finalize_func, init_func, name])
        return self

    # ==================== internal helper methods ==================== #
    def _flexible_init(self, init_func):
        import inspect
        # flexible init_func, accept 1 arg or None
        if inspect.ismethod(init_func) or \
            len(inspect.getargspec(init_func).args) == 1:
            init_func(self)
        else:
            init_func()

    def _run_mpi(self, task):
        #####################################
        # 0. parse task information.
        if isinstance(task, (tuple, list)):
            jobs_list, map_func, reduce_func, finalize_func, init_func, name = task
            if init_func is not None:
                self._flexible_init(init_func)
            seq_jobs = []
        elif isinstance(task, FeatureRecipe):
            self._flexible_init(task.initialize) # init first
            jobs_list, map_func, reduce_func, finalize_func, name = \
            task.jobs, task.map_func, task.reduce_func, task.finalize_func, task.name
            seq_jobs = task.seq_jobs
        else:
            raise ValueError('No support for type(task)={}.'.format(type(task)))

        #####################################
        # 1. Scatter jobs for all process.
        try:
            # str => the name of previous jobs
            if isinstance(jobs_list, str):
                jobs_list = len(self._results[jobs_list])
                if len(jobs_list) == 1: # only 1 result in result list
                    jobs_list = jobs_list[0]
            # if still no jobs
            if not isinstance(jobs_list, (tuple, list)) or \
            len(jobs_list) + len(seq_jobs) == 0:
                raise ValueError('no job for running task!')
            # create progbar
            progbar = Progbar(target=len(jobs_list) + len(seq_jobs),
                              title='Task:' + str(name))
            progbar.add(0) # update progress-bar
            # ====== start segment and process jobs ====== #
            jobs = segment_list(jobs_list, size=self._cache * self.processes)
            jobs.append(seq_jobs) # append seq jobs
            final_results = []
            for count, j in enumerate(jobs):
                if len(j) == 0: continue
                elif len(j) > self.processes and count < len(jobs) - 1:
                    results = self._pool.map(map_func, j, chunksize=self._cache)
                else: # execute sequently
                    results = [map_func(i) for i in j]
                # reduce all the results
                results = [i for i in results if i is not None]
                results = (reduce_func(results)
                           if reduce_func is not None else None)
                progbar.add(len(j)) # update progress-bar
                if results is not None:
                    final_results.append(results)
            # finalize all reduced results
            if finalize_func is not None:
                final_results = finalize_func(final_results)
            # store results
            if isinstance(final_results, dict):
                self._results.update(final_results)
            else:
                self._results[name].append(final_results)
        except Exception, e:
            sys.stderr.write("\nError! Ignored given task: name={}, error='{}'\n"
                             ''.format(name, e))
            import traceback; traceback.print_exc()

    def __getitem__(self, key):
        x = self._results.__getitem__(key)
        if isinstance(x, (tuple, list)) and len(x) == 1:
            return x[0]
        return x

    def get(self, key):
        return self.__getitem__(key)

    def run(self):
        while not self._tasks.empty():
            self._run_mpi(self._tasks.get())

    def __del__(self):
        try:
            self._pool.close()
            self._pool.join()
            del self._pool
        except:
            pass # already closed


# ===========================================================================
# Predefined tasks
# ===========================================================================
@add_metaclass(ABCMeta)
class FeatureRecipe(object):

    ''' Pickle-able recipe for extracting object, that can be used with
    MapReduce

    '''

    def __init__(self, name=None):
        self.name = name
        self._map_func = None
        self._reduce_func = None
        self._finalize_func = None
        self.jobs = []
        self.seq_jobs = []

    # ==================== helper function ==================== #
    def update(self, key, value):
        '''Update all argument with given name to given value'''
        for i in [self._map_func, self._reduce_func, self._finalize_func]:
            if isinstance(i, functionable):
                i[key] = value

    def wrap_map(self, *args, **kwargs):
        self._map_func = functionable(self._map, *args, **kwargs)
        return self

    def wrap_reduce(self, *args, **kwargs):
        self._reduce_func = functionable(self._reduce, *args, **kwargs)
        return self

    def wrap_finalize(self, *args, **kwargs):
        self._finalize_func = functionable(self._finalize, *args, **kwargs)
        return self

    def initialize(self, mr):
        ''' This function will be called before the recipe is executed '''
        pass

    # ==================== non-touchable properties ==================== #
    @property
    def map_func(self):
        if not isinstance(self._map_func, functionable):
            raise ValueError('map_func must be instance of functionable')
        return self._map_func

    @property
    def reduce_func(self):
        if not isinstance(self._reduce_func, functionable):
            raise ValueError('reduce_func must be instance of functionable')
        return self._reduce_func

    @property
    def finalize_func(self):
        if not isinstance(self._finalize_func, functionable) and \
           self._finalize_func is not None:
            raise ValueError('finalize_func only can be None or functionable')
        return self._finalize_func

    # ==================== main function ==================== #
    @abstractstatic
    def _map(*args, **kwargs):
        raise NotImplementedError

    @abstractstatic
    def _reduce(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _finalize(*args, **kwargs):
        raise NotImplementedError

    # ==================== load from yaml ==================== #
    @classmethod
    def load(cls, path):
        if isinstance(path, str):
            if os.path.isfile(path):
                data = open(path, 'r').read()
            else:
                data = path
            import yaml
            from StringIO import StringIO
            data = yaml.load(StringIO(data))
            if isinstance(data, dict):
                if cls.__name__ in data:
                    data = data[cls.__name__]
                return cls(**data)
        raise Exception('Cannot load yaml recipe from path:%s' % path)

    def dump(self, path=None):
        """ Return yaml string represent this class """
        if not hasattr(self, '_arguments'):
            raise Exception('This method only support @autoinit class, which '
                            'store all its parameters in _arguments.')
        import yaml
        data = {self.__class__.__name__: self._arguments}
        styles = {'default_flow_style': False, 'encoding': 'utf-8'}
        if path is not None:
            yaml.dump(data, open(path, 'w'), **styles)
            return path
        return yaml.dump(data, **styles)


# ===========================================================================
# Speech features
# ===========================================================================
def _append_energy_and_deltas(s, energy, delta_order):
    # s.shape = [Time, Dimension]
    if s is None:
        return None
    if energy is not None:
        s = np.hstack((s, energy[:, None]))
    # compute delta
    if delta_order > 0:
        deltas = speech.compute_delta(s.T, order=delta_order)
        # tranpose back to [Time, Dim]
        s = np.hstack([s] + [i.T for i in deltas])
    return s


def speech_features_extraction(s, fs, n_filters, n_ceps, win, shift,
                               delta_order, energy, vad, dtype,
                               get_spec, get_mspec, get_mfcc):
    import sidekit
    """ return: spec, mspec, and mfcc """
    if s.ndim >= 2:
        raise Exception('Speech Feature Extraction only accept 1-D signal')
    # speech features, shape: [Time, Dimension]
    mfcc, logEnergy, spec, mspec = sidekit.frontend.mfcc(
        s, fs=fs, lowfreq=64, maxfreq=fs // 2, nlogfilt=n_filters,
        nwin=win, shift=shift, nceps=n_ceps,
        get_spec=get_spec, get_mspec=get_mspec)
    # any nan value in MFCC ignore the whole file
    if np.any(np.isnan(mfcc)):
        return None
    mfcc = mfcc if get_mfcc else None
    # VAD
    vad_idx = None
    if vad:
        distribNb, nbTrainIt = 8, 12
        if isinstance(vad, (tuple, list)):
            distribNb, nbTrainIt = vad
        # vad_idx = sidekit.frontend.vad.vad_snr(s, threshold,
        # fs=fs, shift=shift, nwin=int(fs * win)).astype('int8')
        vad_idx = sidekit.frontend.vad.vad_energy(logEnergy,
            distribNb=distribNb, nbTrainIt=nbTrainIt).astype('int8')
    # Energy
    logEnergy = logEnergy if energy else None

    # everything is (T, D)
    mfcc = (_append_energy_and_deltas(mfcc, logEnergy, delta_order)
            if mfcc is not None else None)
    # we don't calculate deltas for spectrogram features
    spec = (_append_energy_and_deltas(spec, logEnergy, 0)
            if spec is not None else None)
    mspec = (_append_energy_and_deltas(mspec, logEnergy, delta_order)
            if mspec is not None else None)
    # normalization
    mfcc = (mfcc.astype(dtype),
            np.sum(mfcc, axis=0, dtype='float64'),
            np.sum(mfcc**2, axis=0, dtype='float64')) if mfcc is not None else None
    spec = (spec.astype(dtype),
            np.sum(spec, axis=0, dtype='float64'),
            np.sum(spec**2, axis=0, dtype='float64')) if spec is not None else None
    mspec = (mspec.astype(dtype),
             np.sum(mspec, axis=0, dtype='float64'),
             np.sum(mspec**2, axis=0, dtype='float64')) if mspec is not None else None
    return spec, mspec, mfcc, vad_idx


class SpeechFeature(FeatureRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted)
            name                |     path             |start|end |channel
        ------------------------|----------------------|-----|----|---
        sw02001-A_000098-001156 | /path/to/sw02001.sph | 0.0 | -1 | 0
        sw02001-B_001980-002131 | /path/to/sw02001.sph | 0.0 | -1 | 1
    win : float
        frame or window length in second
    shift : float
        frame or window, or hop length in second
    n_filters : int
        number of log-mel filter banks
    n_ceps : int
        number of cepstrum for MFCC
    delta_order : int
        compute deltas featues (e.g 2 means delta1 and delta2)
    energy : bool
        if True, append log energy to features
    vad : bool, tuple or list
        save Voice Activities Detection mask
        if tuple or list provodied, it must represents (distribNb, nbTrainIt)
        where distribNb is number of distribution, nbTrainIt is number of iteration
        (default: distribNb=8, nbTrainIt=12)
    downsample : str
        One of the following algorithms:
        sinc_medium : Band limited sinc interpolation, medium quality, 121dB SNR, 90% BW.
        linear : Linear interpolator, very fast, poor quality.
        sinc_fastest : Band limited sinc interpolation, fastest, 97dB SNR, 80% BW.
        zero_order_hold : Zero order hold interpolator, very fast, poor quality.
        sinc_best : Band limited sinc interpolation, best quality, 145dB SNR, 96% BW.
        (default: best quality algorithm is used)
    get_spec : bool
        return spectrogram
    get_mspec : bool
        return log-mel filterbank
    get_mfcc : bool
        return mfcc features
    robust : bool
        run in robust mode, auto ignore error files
    datatype : memmap, hdf5

    Example
    -------
    '''

    @autoinit
    def __init__(self, segments, output, audio_ext=None, fs=8000,
                 win=0.025, shift=0.01, n_filters=40, n_ceps=13,
                 downsample='sinc_best', delta_order=2, energy=True, vad=True,
                 datatype='memmap', dtype='float32',
                 get_spec=False, get_mspec=True, get_mfcc=False,
                 robust=True):
        super(SpeechFeature, self).__init__('SpeechFeatures')

    def initialize(self, mr):
        if not self.get_spec and not self.get_mspec and not self.get_mfcc:
            raise Exception('You must specify which features you want: spectrogram'
                            'filter-banks, or MFCC.')
        # ====== super function should be called at the beginning ====== #
        segments = self.segments
        output = self.output
        audio_ext = as_tuple('' if self.audio_ext is None else self.audio_ext, 1, str)
        datatype = self.datatype

        # ====== load jobs ====== #
        if isinstance(segments, str):
            if not os.path.exists(segments):
                raise ValueError('Path to segments must exists, however, '
                                 'exist(segments)={}'.format(os.path.exists(segments)))
            if os.path.isdir(segments):
                file_list = get_all_files(segments)
                file_list = [(os.path.basename(i), i, 0.0, -1.0)
                             for i in file_list] # segment, path, start, end
            else: # csv file
                file_list = np.genfromtxt(segments, dtype=str, delimiter=' ')
        elif isinstance(segments, (tuple, list)):
            if isinstance(segments[0], str): # just a list of path to file
                file_list = [(os.path.basename(i), os.path.abspath(i), 0.0, -1.0)
                             for i in segments]
            elif isinstance(segments[0], (tuple, list)):
                if len(segments[0]) != 4 and len(segments[0]) != 5:
                    raise Exception('segments must contain information in following for:'
                                    '[name] [path] [start] [end]')
                file_list = segments
        # filter using support audio extension
        file_list = [f for f in file_list if any(ext in f[1] for ext in audio_ext)]
        # if no channel is provided, append the channel
        file_list = [list(f) + [0] if len(f) == 4 else f for f in file_list]
        # convert into audio_path -> segment
        self.jobs = defaultdict(list)
        for segment, file, start, end, channel in file_list:
            self.jobs[file].append((segment, float(start), float(end), int(channel)))
        self.jobs = sorted(self.jobs.items(), key=lambda x: x[0])
        # ====== check output ====== #
        dataset = Dataset(output)
        # create map_func
        self.wrap_map(n_filters=self.n_filters, n_ceps=self.n_ceps,
                      fs=self.fs, downsample=self.downsample,
                      win=self.win, shift=self.shift,
                      delta_order=self.delta_order, energy=self.energy,
                      vad=self.vad, dtype=self.dtype,
                      get_spec=self.get_spec, get_mspec=self.get_mspec,
                      get_mfcc=self.get_mfcc, robust=self.robust)
        # create reduce
        self.wrap_reduce(dataset=dataset, datatype=datatype)
        # create finalize
        self.wrap_finalize(dataset=dataset, get_spec=self.get_spec,
                           get_mspec=self.get_mspec, get_mfcc=self.get_mfcc)

    @staticmethod
    def _map(f, n_filters=40, n_ceps=13, fs=8000, downsample='sinc_best',
             win=0.025, shift=0.01, delta_order=2, energy=True, vad=True,
             dtype='float32', get_spec=False, get_mspec=True, get_mfcc=False,
             robust=True):
        '''
        Return
        ------
        (name, features, vad, sum1, sum2)

        '''
        try:
            audio_path, segments = f
            # load audio data
            s, _ = speech.read(audio_path)
            # check frequency for downsampling (if necessary)
            if _ is not None:
                if fs is not None and fs != _:
                    if fs < _: # downsample
                        s = resample(s, fs / _, 'sinc_best')
                    else:
                        raise ValueError('Cannot perform upsample from frequency: '
                                         '{}Hz to {}Hz'.format(_, fs))
                else:
                    fs = _
            N = len(s)
            features = []
            for name, start, end, channel in segments:
                start = int(float(start) * fs)
                end = int(N if end < 0 else end * fs)
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                tmp = speech_features_extraction(data.ravel(), fs=fs,
                    n_filters=n_filters, n_ceps=n_ceps,
                    win=win, shift=shift, delta_order=delta_order,
                    energy=energy, vad=vad, dtype=dtype,
                    get_spec=get_spec, get_mspec=get_mspec, get_mfcc=get_mfcc)
                if tmp is not None:
                    features.append((name,) + tmp)
                else:
                    msg = 'Ignore segments: %s, error: NaN values' % name
                    warnings.warn(msg)
            return features
        except Exception, e:
            msg = 'Ignore file: %s, error: %s' % (f[0], str(e))
            warnings.warn(msg)
            if robust:
                return None
            else:
                import traceback; traceback.print_exc()
                raise e

    @staticmethod
    def _reduce(results, dataset, datatype):
        # contains (name, spec, mspec, mfcc, vad)
        index = []
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        n = 0
        for r in results:
            for name, spec, mspec, mfcc, vad in r:
                if spec is not None:
                    X, sum1, sum2 = spec
                    _ = dataset.get_data('spec', dtype=X.dtype,
                                         shape=(0,) + X.shape[1:],
                                         datatype=datatype)
                    _.append(X)
                    spec_sum1 += sum1; spec_sum2 += sum2
                    n = X.shape[0]; del X
                if mspec is not None:
                    X, sum1, sum2 = mspec
                    _ = dataset.get_data('mspec', dtype=X.dtype,
                                         shape=(0,) + X.shape[1:],
                                         datatype=datatype)
                    _.append(X)
                    mspec_sum1 += sum1; mspec_sum2 += sum2
                    n = X.shape[0]; del X
                if mfcc is not None:
                    X, sum1, sum2 = mfcc
                    _ = dataset.get_data('mfcc', dtype=X.dtype,
                                         shape=(0,) + X.shape[1:],
                                         datatype=datatype)
                    _.append(X)
                    mfcc_sum1 += sum1; mfcc_sum2 += sum2
                    n = X.shape[0]; del X
                # index
                index.append([name, n])
                # VAD
                if vad is not None:
                    assert vad.shape[0] == n,\
                        'VAD mismatch features shape: %d != %d' % (vad.shape[0], n)
                    _ = dataset.get_data('vad', dtype=vad.dtype,
                                         shape=(0,) + vad.shape[1:],
                                         datatype=datatype)
                    _.append(vad)
                    del vad
        dataset.flush()
        return ((spec_sum1, spec_sum2),
                (mspec_sum1, mspec_sum2),
                (mfcc_sum1, mfcc_sum2), index)

    @staticmethod
    def _finalize(results, dataset, get_spec, get_mspec, get_mfcc):
        # contains (sum1, sum2, n)
        path = dataset.path
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        n = 0
        indices = []
        for spec, mspec, mfcc, index in results:
            # spec
            spec_sum1 += spec[0]
            spec_sum2 += spec[1]
            # mspec
            mspec_sum1 += mspec[0]
            mspec_sum2 += mspec[1]
            # mfcc
            mfcc_sum1 += mfcc[0]
            mfcc_sum2 += mfcc[1]
            for name, size in index:
                # name, start, end
                indices.append([name, int(n), int(n + size)])
                n += size
        # ====== saving indices ====== #
        with open(os.path.join(path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d\n' % (name, start, end))

        # ====== helper ====== #
        def save_mean_std(sum1, sum2, n, name, dataset):
            mean = sum1 / n
            std = np.sqrt(sum2 / n - mean**2)
            assert not np.any(np.isnan(mean)), 'Mean contains NaN'
            assert not np.any(np.isnan(std)), 'Std contains NaN'
            _ = dataset.get_data(name + '_mean', dtype=mean.dtype, shape=mean.shape)
            _[:] = mean

            _ = dataset.get_data(name + '_std', dtype=std.dtype, shape=std.shape)
            _[:] = std
        # ====== save mean and std ====== #
        if get_spec:
            save_mean_std(spec_sum1, spec_sum2, n, 'spec', dataset)
        if get_mspec:
            save_mean_std(mspec_sum1, mspec_sum2, n, 'mspec', dataset)
        if get_mfcc:
            save_mean_std(mfcc_sum1, mfcc_sum2, n, 'mfcc', dataset)
        dataset.flush()
        dataset.close()
        return {'dataset': path}
