# ===========================================================================
# Parallel features processing using multi-core CPU and multiprocessing
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import sys
import os
import types
import warnings
import cPickle
from numbers import Number
from multiprocessing import Pool, cpu_count, Process, Queue
from six import add_metaclass
from six.moves import zip, zip_longest, range
from abc import ABCMeta, abstractmethod

from collections import defaultdict
import numpy as np
try: # this library may not available
    from scikits.samplerate import resample
except:
    pass

from odin.preprocessing import speech, video
from odin.utils import (queue, Progbar, segment_list, as_tuple,
                        get_all_files, get_tempdir)
from odin.utils.decorators import autoinit
from .dataset import Dataset


__all__ = [
    'MapReduce',
    'FeatureRecipe',
    'SpeechFeature',
    'speech_features_extraction',
    'VideoFeature'
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

    def __init__(self, ncpu=8, buffer_size=5):
        super(MapReduce, self).__init__()
        # variables
        self._tasks = queue()
        self._stop_now = False
        self._task_results = {}

        self.buffer_size = int(max(buffer_size, 1))
        self.ncpu = max(min(ncpu, cpu_count() - 1), 1)

    def stop(self):
        self._stop_now = True
        return self

    # ==================== Task manager ==================== #
    def add_recipe(self, recipe):
        # path to pickled file or pickled string
        if isinstance(recipe, str):
            if os.path.exists(recipe):
                recipe = cPickle.load(open(recipe, 'r'))
            else:
                recipe = cPickle.loads(recipe)
        # recipe must be list
        if not isinstance(recipe, (tuple, list)):
            recipe = (recipe,)
        if not all(isinstance(i, FeatureRecipe) for i in recipe):
            raise ValueError('Given recipe is not instance of FeatureRecipe, '
                             'but has type={}'.format(map(type, recipe)))
        for i in recipe:
            self._tasks.append(i) # in this case, tasks contain recipe

    def add_task(self, jobs, map_func, reduce_func=None, finalize_func=None,
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
    def _run_mpi(self, task):
        #####################################
        # 0. parse task information.
        if isinstance(task, (tuple, list)):
            (jobs_list, map_func, reduce_func,
             finalize_func, init_func, name) = task
            if init_func is not None:
                init_func()
        elif isinstance(task, FeatureRecipe):
            task.initialize() # init first
            (jobs_list, map_func, reduce_func, finalize_func, name) = \
            task.jobs, task.map_func, task.reduce_func, task.finalize_func, task.name
        else:
            raise ValueError('No support for type(task)={}.'.format(type(task)))

        #####################################
        # 1. Scatter jobs for all process.
        try:
            # str => the name of previous jobs
            if len(jobs_list) == 0:
                raise Exception('Found no jobs to execute.')
            n_jobs = len(jobs_list)
            jobs_list = segment_list(jobs_list, n_seg=self.ncpu)

            # ====== work ====== #
            def work_multi(jobs, res):
                for j in jobs:
                    j = map_func(j)
                    # Generator: return a list of results
                    if isinstance(j, types.GeneratorType):
                        [res.put(_) for _ in j if _ is not None]
                    elif j is not None:
                        res.put(j)
            # ====== create processes ====== #
            results = Queue()
            processes = [Process(target=work_multi, args=(j, results))
                         for j in jobs_list]
            [p.start() for p in processes]
            # ====== start segment and process jobs ====== #
            final_results = []
            batch = []
            exit_on_stop = False
            # create progbar
            progbar = Progbar(target=n_jobs, title='Task:' + str(name))
            progbar.add(0) # update progress-bar
            # running the jobs
            for _ in range(n_jobs):
                # check if immediately stop
                if self._stop_now:
                    exit_on_stop = True
                    break
                # processing
                progbar.add(1)
                batch.append(results.get())
                # reduce the results
                if len(batch) == self.buffer_size or _ == (n_jobs - 1):
                    final_results.append(reduce_func(batch))
                    batch = []
            # ====== end all processes ====== #
            # end the worker
            if not exit_on_stop:
                [p.join() for p in processes]
            else:
                [p.terminate() for p in processes if p.is_alive()]
            results.close()
            # ====== finalize the results ====== #
            if finalize_func is not None:
                print('Finalizing task: %s ...' % name)
                final_results = finalize_func(final_results)
                self._task_results[name] = final_results
        except Exception, e:
            sys.stderr.write("\nError! Ignored given task: name={}, error='{}'\n"
                             ''.format(name, e))
            import traceback; traceback.print_exc()

    def __getitem__(self, key):
        return self._task_results[key]

    def __len__(self):
        return len(self._tasks)

    def run(self):
        while not self._tasks.empty():
            self._run_mpi(self._tasks.get())
            # check if stop now
            if self._stop_now:
                self._stop_now = False
                break
        return self


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
        self.jobs = []

    @abstractmethod
    def initialize(self):
        """ This method is important, it enable a sequential
        execution of many recipes
        """
        pass

    # ==================== non-touchable properties ==================== #
    @abstractmethod
    def map_func(self, job):
        pass

    @abstractmethod
    def reduce_func(self, list_of_job):
        pass

    @abstractmethod
    def finalize_func(self, results):
        pass


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
                               delta_order, energy, vad, dtype, pitch_threshold,
                               get_spec, get_mspec, get_mfcc, get_pitch):
    """ return spec(X, sum, sum2),
               mspec(X, sum, sum2),
               mfcc(X, sum, sum2),
               vad_idx """
    import sidekit
    if s.ndim >= 2:
        raise Exception('Speech Feature Extraction only accept 1-D signal')
    # speech features, shape: [Time, Dimension]
    mfcc, logEnergy, spec, mspec = sidekit.frontend.mfcc(
        s, fs=fs, lowfreq=64, maxfreq=fs // 2, nlogfilt=n_filters,
        nwin=win, shift=shift, nceps=n_ceps,
        get_spec=True, get_mspec=get_mspec, prefac=0.97)
    # geting pitch if required (using librosa)
    pitch = None
    if get_pitch:
        import librosa
        pitch_freq, pitch_mag = librosa.piptrack(S=spec.T, sr=fs,
            n_fft=2 ** int(np.ceil(np.log2(int(round(win * fs))))),
            hop_length=shift * fs,
            fmin=150, fmax=fs / 2,
            threshold=pitch_threshold)
        # no append log energy or delta for pitch features
        pitch = np.hstack([pitch_freq.T, pitch_mag.T])
    # reset spec to true value
    spec = None if not get_spec else spec
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
            distrib_nb=distribNb, nb_train_it=nbTrainIt)[0].astype('int8')
    # Energy
    logEnergy = logEnergy if energy else None
    # everything is (T, D)
    mfcc = (_append_energy_and_deltas(mfcc, logEnergy, delta_order)
            if mfcc is not None else None)
    mspec = (_append_energy_and_deltas(mspec, logEnergy, delta_order)
            if mspec is not None else None)
    # we don't calculate deltas for spectrogram features
    spec = (_append_energy_and_deltas(spec, logEnergy, 0)
            if spec is not None else None)

    # for future normalization
    mfcc = (mfcc.astype(dtype),
            np.sum(mfcc, axis=0, dtype='float64'),
            np.sum(mfcc**2, axis=0, dtype='float64')) if mfcc is not None else None
    pitch = (pitch.astype(dtype),
             np.sum(pitch, axis=0, dtype='float64'),
             np.sum(pitch**2, axis=0, dtype='float64')) if pitch is not None else None
    spec = (spec.astype(dtype),
            np.sum(spec, axis=0, dtype='float64'),
            np.sum(spec**2, axis=0, dtype='float64')) if spec is not None else None
    mspec = (mspec.astype(dtype),
             np.sum(mspec, axis=0, dtype='float64'),
             np.sum(mspec**2, axis=0, dtype='float64')) if mspec is not None else None
    return spec, mspec, mfcc, pitch, vad_idx


class SpeechFeature(FeatureRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted), start and end is in second
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
    get_pitch : bool
        return pitch frequencies, and pitch energy (which is
        horizontal-stacked) into big features matrix (i.e. expected
        double number of features of spectrogram).
    pitch_threshold : float (0.0,1.0)
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`. If delta is added, the order will be
        [pitch_freq + pitch_freq_delta + pitch_mag + pitch_mag_delta]
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
                 get_pitch=False, pitch_threshold=0.5, robust=True):
        super(SpeechFeature, self).__init__('SpeechFeatures')

    def initialize(self):
        if not self.get_spec and not self.get_mspec and not self.get_mfcc:
            raise Exception('You must specify which features you want: '
                            'spectrogram, filter-banks, or MFCC.')
        # ====== super function should be called at the beginning ====== #
        segments = self.segments
        audio_ext = as_tuple('' if self.audio_ext is None else self.audio_ext, 1, str)

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
        # convert into: audio_path -> segment(name, start, end, channel)
        self.jobs = defaultdict(list)
        for segment, file, start, end, channel in file_list:
            self.jobs[file].append((segment, float(start), float(end), int(channel)))
        self.jobs = sorted(self.jobs.items(), key=lambda x: x[0])
        # ====== check output ====== #
        self.dataset = Dataset(self.output)
        # constraint pitch threshold in 0-1
        self.pitch_threshold = min(max(self.pitch_threshold, 0.), 1.)

    def map_func(self, f):
        '''
        Return
        ------
        [(name, spec(x, sum1, sum2), # if available, otherwise None
                mspec(x, sum1, sum2), # if available, otherwise None
                mfcc(x, sum1, sum2), # if available, otherwise None
                pitch(x, sum1, sum2), # if available, otherwise None
                vad), ...]
        '''
        fs = self.fs
        try:
            audio_path, segments = f
            # load audio data
            s, orig_fs = speech.read(audio_path)
            orig_fs = fs if orig_fs is None else orig_fs
            # check frequency for downsampling (if necessary)
            if fs is None:
                fs = orig_fs
            elif fs < orig_fs: # downsample
                s = resample(s, fs / orig_fs, 'sinc_best')
            elif fs > orig_fs:
                raise ValueError('Cannot perform upsample from frequency: '
                                 '{}Hz to {}Hz'.format(orig_fs, fs))
            N = len(s)
            features = []
            for name, start, end, channel in segments:
                start = int(float(start) * fs)
                end = int(N if end < 0 else end * fs)
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                tmp = speech_features_extraction(data.ravel(), fs=fs,
                    n_filters=self.n_filters, n_ceps=self.n_ceps,
                    win=self.win, shift=self.shift, delta_order=self.delta_order,
                    energy=self.energy, vad=self.vad, dtype=self.dtype,
                    pitch_threshold=self.pitch_threshold,
                    get_spec=self.get_spec, get_mspec=self.get_mspec,
                    get_mfcc=self.get_mfcc, get_pitch=self.get_pitch)
                if tmp is not None:
                    features.append((name,) + tmp)
                else:
                    msg = 'Ignore segments: %s, error: NaN values' % name
                    warnings.warn(msg)
            # return an iterator of features
            for f in features:
                yield f
        except Exception, e:
            msg = 'Ignore file: %s, error: %s' % (f[0], str(e))
            warnings.warn(msg)
            if self.robust:
                yield None
            else:
                import traceback; traceback.print_exc()
                raise e

    def reduce_func(self, results):
        # contains (name, spec, mspec, mfcc, vad)
        dataset = self.dataset
        datatype = self.datatype

        index = []
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        pitch_sum1, pitch_sum2 = 0., 0.

        n = 0
        for name, spec, mspec, mfcc, pitch, vad in results:
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
            if pitch is not None:
                X, sum1, sum2 = pitch
                _ = dataset.get_data('pitch', dtype=X.dtype,
                                     shape=(0,) + X.shape[1:],
                                     datatype=datatype)
                _.append(X)
                pitch_sum1 += sum1; pitch_sum2 += sum2
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
                (mfcc_sum1, mfcc_sum2),
                (pitch_sum1, pitch_sum2), index)

    def finalize_func(self, results):
        # contains (sum1, sum2, n)
        dataset = self.dataset
        path = dataset.path
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        pitch_sum1, pitch_sum2 = 0., 0.
        n = 0
        indices = []
        for spec, mspec, mfcc, pitch, index in results:
            # spec
            spec_sum1 += spec[0]
            spec_sum2 += spec[1]
            # mspec
            mspec_sum1 += mspec[0]
            mspec_sum2 += mspec[1]
            # mfcc
            mfcc_sum1 += mfcc[0]
            mfcc_sum2 += mfcc[1]
            # pitch
            pitch_sum1 += pitch[0]
            pitch_sum2 += pitch[1]
            # indices
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
        if self.get_spec:
            save_mean_std(spec_sum1, spec_sum2, n, 'spec', dataset)
        if self.get_mspec:
            save_mean_std(mspec_sum1, mspec_sum2, n, 'mspec', dataset)
        if self.get_mfcc:
            save_mean_std(mfcc_sum1, mfcc_sum2, n, 'mfcc', dataset)
        if self.get_pitch:
            save_mean_std(pitch_sum1, pitch_sum2, n, 'pitch', dataset)
        dataset.flush()
        dataset.close()
        return path

    def __setstate__(self, states):
        self.name = states[0]
        for name, value in states[1].iteritems():
            setattr(self, name, value)

    def __getstate__(self):
        return self.name, self._arguments


# ===========================================================================
# Video features
# ===========================================================================
def video_features_extraction(X, boundingbox, desire_size):
    finalX = [X]
    dtype = X.dtype
    if boundingbox is not None:
        finalX = [list() for i in range(len(boundingbox[0]) // 4)]
        # travel through each frames
        for x, bound in zip(X, boundingbox):
            # get each bounding box
            for i, box in enumerate(np.reshape(bound, (-1, 4))):
                x_, y_, w_, h_ = box
                # zero area, ignore it
                if w_ == 0 or h_ == 0:
                    if desire_size is None: continue
                    tmp = np.zeros(desire_size, dtype=dtype)
                # ====== get the bounding ====== #
                else:
                    if desire_size is not None:
                        # crop in the center
                        x_ = x_ + w_ // 2 - desire_size[-2] // 2
                        w_ = desire_size[-2] # width
                        y_ = y_ + h_ // 2 - desire_size[-1] // 2
                        h_ = desire_size[-1] # height
                    tmp = x[:, x_:x_ + w_, y_:y_ + h_]
                    # if actual size smaller than desire_size
                    # perform padding with 0.
                    if tmp.shape[-2] != w_ or tmp.shape[-1] != h_:
                        _ = np.zeros(desire_size, dtype=dtype)
                        startX = int(w_ // 2 - tmp.shape[-2] / 2)
                        startY = int(h_ // 2 - tmp.shape[-1] / 2)
                        _[:, startX: startX + tmp.shape[-2],
                          startY: startY + tmp.shape[-1]] = tmp
                        tmp = _
                # add to final results
                finalX[i].append(tmp)
        # create 1 big array hold all images
        finalX = [np.asarray(x) for x in finalX]
        finalX = (finalX[0] if len(finalX) == 1
                  else np.concatenate(finalX, axis=1))
    return (finalX,
            np.sum(finalX, axis=0, dtype='float64'),
            np.sum(finalX**2, axis=0, dtype='float64'))


class VideoFeature(FeatureRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Parameters
    ----------
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted), start and end is in second
            name                |     path             |start|end |
        ------------------------|----------------------|-----|----|
        sw02001-A_000098-001156 | /path/to/sw02001.mp4 | 0.0 | -1 |
        sw02001-B_001980-002131 | /path/to/sw02001.mp4 | 0.0 | -1 |
    size : tuple(width, height)
        desire size of the return features images
    boundingbox : None, dict
        mapping from filename to sequence of bounding box
        (region of interest), name -> [x(from left),y(from top),width,height]
        For example: if is multiple of 4, then extract multiple regions
        sw02001-A_000098-001156 ->  [[30, 40, 15, 20, .... ], ...]
        sw02001-B_001980-002131 ->  [[30, 40, 15, 20, .... ], ...]
    robust : bool
        run in robust mode, auto ignore error files

    datatype : memmap, hdf5

    Example
    -------
    '''

    @autoinit
    def __init__(self, segments, output, size=None,
                 boundingbox=None, video_ext=None,
                 datatype='memmap', robust=True):
        super(VideoFeature, self).__init__('VideoFeature')

    def initialize(self):
        # reversed to height width for easy processing
        if self.size is not None:
            self.size = as_tuple(self.size, N=2, t=int)
        segments = self.segments
        video_ext = as_tuple('' if self.video_ext is None
                             else self.video_ext, 1, str)
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
                if len(segments[0]) != 4:
                    raise Exception('segments must contain information in following for:'
                                    '[name] [path] [start] [end]')
                file_list = segments
        # filter using support audio extension
        file_list = [f for f in file_list if any(ext in f[1] for ext in video_ext)]
        # convert into: audio_path -> segment(name, start, end, channel)
        self.jobs = defaultdict(list)
        names = []
        for segment, file, start, end in file_list:
            self.jobs[file].append((segment, float(start), float(end)))
            names.append(segment)
        self.jobs = sorted(self.jobs.items(), key=lambda x: x[0])
        # ====== load bounding box ====== #
        if self.boundingbox is not None:
            if not isinstance(self.boundingbox, dict):
                raise ValueError('Bounding box must be a dictionary')
            if set(names) != set(self.boundingbox.keys()):
                raise Exception('Segments names and boundingbox keys mismatch.')
        # ====== check output ====== #
        self.dataset = Dataset(self.output)
        self._temp_path = get_tempdir()
        print('Temporary dir created at:', self._temp_path)
        # remove old cache files
        for p in os.listdir(self._temp_path):
            os.remove(os.path.join(self._temp_path, p))

    def map_func(self, f):
        '''
        Return
        ------
        [(name, spec(x, sum1, sum2), # if available, otherwise None
                mspec(x, sum1, sum2), # if available, otherwise None
                mfcc(x, sum1, sum2), # if available, otherwise None
                pitch(x, sum1, sum2), # if available, otherwise None
                vad), ...]
        '''
        video_path, segments = f
        # read the whole video
        frames, fps = video.read(video_path)
        size = self.size
        if size is not None:
            size = (frames.shape[1],) + size
        # generating features
        features = []
        for name, start, end in segments:
            start = int(float(start) * fps)
            end = int(frames.shape[0] if end < 0 else end * fps)
            data = frames[start:end]
            # ====== check bounding box ====== #
            box = (None if self.boundingbox is None
                   else self.boundingbox[name])
            tmp = video_features_extraction(data, box, size)
            if tmp is not None:
                features.append((name,) + tmp)
            else:
                msg = 'Ignore segments: %s, error: NaN values' % name
                warnings.warn(msg)
        # return an iterator of features
        del frames
        for name, x, sum1, sum2 in features:
            path = os.path.join(self._temp_path, name)
            # save big array, because the video can be very
            # big so we don't transfer it to Queue
            f = open(path, 'w'); np.save(f, x); f.close()
            yield name, path, sum1, sum2

    def reduce_func(self, results):
        # contains (name, spec, mspec, mfcc, vad)
        dataset = self.dataset
        datatype = self.datatype

        index = []
        sum1, sum2 = 0., 0.

        n = 0
        for name, path, s1, s2 in results:
            # load big array
            f = open(path, 'r'); X = np.load(f); f.close()
            _ = dataset.get_data('frames', dtype=X.dtype,
                                 shape=(0,) + X.shape[1:],
                                 datatype=datatype)
            _.append(X)
            sum1 += s1
            sum2 += s2
            n = X.shape[0]
            # index
            index.append([name, n])
            os.remove(path)
        dataset.flush()
        return (sum1, sum2, index)

    def finalize_func(self, results):
        # contains (sum1, sum2, n)
        dataset = self.dataset
        path = dataset.path
        sum1, sum2 = 0., 0.
        n = 0
        indices = []
        for s1, s2, index in results:
            # spec
            sum1 += s1
            sum2 += s2
            # indices
            for name, size in index:
                # name, start, end
                indices.append([name, int(n), int(n + size)])
                n += size
        # ====== saving indices ====== #
        with open(os.path.join(path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d\n' % (name, start, end))

        # ====== helper ====== #
        mean = sum1 / n
        std = np.sqrt(sum2 / n - mean**2)
        assert not np.any(np.isnan(mean)), 'Mean contains NaN'
        assert not np.any(np.isnan(std)), 'Std contains NaN'

        _ = dataset.get_data(name + '_mean', dtype=mean.dtype, shape=mean.shape)
        _[:] = mean

        _ = dataset.get_data(name + '_std', dtype=std.dtype, shape=std.shape)
        _[:] = std

        # ====== clean up and release cv2 ====== #
        dataset.flush()
        dataset.close()
        # remove all temp file
        if os.path.exists(self._temp_path):
            os.remove(self._temp_path)
            self._temp_path = get_tempdir()
        return path

    def __setstate__(self, states):
        self.name = states[0]
        for name, value in states[1].iteritems():
            setattr(self, name, value)

    def __getstate__(self):
        return self.name, self._arguments
