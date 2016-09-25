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

from odin.preprocessing import speech, video
from odin.utils import (queue, Progbar, segment_list, as_tuple,
                        get_all_files, get_tempdir)
from odin.utils.decorators import autoinit
from .dataset import Dataset
from .recipes import FeederRecipe

try:
    import sidekit
except:
    warnings.warn('The speech processing framework "sidekit" is '
                  'NOT available, hence, you cannot use SpeechFeatures.')
try: # this library may not available
    from scikits.samplerate import resample
except:
    warnings.warn('The sampling framework "scikits.samplerate" is '
                  'NOT available, hence, downsampling features will be ignored.')

__all__ = [
    'SpeechFeature',
    'SpeechFeaturesSaver',
    'speech_features_extraction',
    'VideoFeature'
]


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


class SpeechFeature(FeederRecipe):

    ''' Extract speech features from all audio files in given directory or
    file list, then saves them to a `keras.ext.dataset.Dataset`

    Require `indices` format for this task:
    segments : path, list
        if path, directory of all audio file, or segment csv file in
        following format (channel can be omitted), start and end is in second
            name                |     path             |start|end |channel
        ------------------------|----------------------|-----|----|---
        sw02001-A_000098-001156 | /path/to/sw02001.sph | 0.0 | -1 | 0
        sw02001-B_001980-002131 | /path/to/sw02001.sph | 0.0 | -1 | 1

    Parameters
    ----------
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

    Example
    -------
    >>> recipe = F.SpeechFeature(segments, OUTPUT_PATH, audio_ext='.sph',
    >>>                          fs=8000, win=0.025, shift=0.01,
    >>>                          n_filters=40, n_ceps=13, delta_order=2, energy=True,
    >>>                          vad=True, get_spec=False, get_mspec=True, get_mfcc=True,
    >>>                          datatype='memmap', dtype='float16')
    >>> mr = F.MapReduce(12)
    >>> mr.set_cache(3)
    >>> mr.add_recipe(recipe)
    >>> mr.run()

    '''

    def __init__(self, audio_ext=None, fs=8000,
                 win=0.025, shift=0.01, n_filters=40, n_ceps=13,
                 downsample='sinc_best', delta_order=2,
                 energy=True, vad=True, dtype='float32',
                 get_spec=False, get_mspec=True, get_mfcc=False,
                 get_pitch=False, pitch_threshold=0.5, robust=True):
        super(SpeechFeature, self).__init__()
        # ====== which features to get ====== #
        if not get_spec and not get_mspec \
            and not get_mfcc and not get_pitch:
            raise Exception('You must specify which features you want: '
                            'spectrogram, filter-banks, MFCC, or pitch.')
        self.get_spec = get_spec
        self.get_mspec = get_mspec
        self.get_mfcc = get_mfcc
        self.get_pitch = get_pitch
        # ====== other ====== #
        self.audio_ext = as_tuple('' if audio_ext is None else audio_ext, 1, str)
        # ====== feature infor ====== #
        self.fs = fs
        self.win = win
        self.shift = shift
        self.n_filters = n_filters
        self.n_ceps = n_ceps
        self.downsample = downsample
        self.delta_order = int(delta_order)
        self.energy = energy
        self.vad = vad
        # constraint pitch threshold in 0-1
        self.pitch_threshold = min(max(pitch_threshold, 0.), 1.)
        # ====== check output ====== #
        self.dtype = dtype
        self.robust = robust
        # ====== intermediate variable ====== #
        self.index = [] # contain name, start, end
        self.index_end = 0
        self.spec_sum1, self.spec_sum2 = 0., 0.
        self.mspec_sum1, self.mspec_sum2 = 0., 0.
        self.mfcc_sum1, self.mfcc_sum2 = 0., 0.
        self.pitch_sum1, self.pitch_sum2 = 0., 0.

    def shape_transform(self, shape):
        """ Return the new shape that transformed by this Recipe """
        return self.shape

    def preprocess_indices(self, indices):
        # store original indices shape because nwe indices
        # only contain the number of files not all the segments
        self.shape = (len(indices),)
        # ====== load jobs ====== #
        if isinstance(indices, np.ndarray):
            if indices.ndim == 1: # just a list of path to file
                file_list = [(os.path.basename(i), os.path.abspath(i), 0.0, -1.0)
                             for i in indices]
            else:
                if indices.shape[1] != 4 and indices.shape[1] != 5:
                    raise Exception('indices must contain information in following for:'
                                    '[name] [path] [start] [end] [channel(optional)]')
                file_list = indices
        # filter using support audio extension
        file_list = np.array([f for f in file_list
                              if any(ext in f[1] for ext in self.audio_ext)])
        # if no channel is provided, append the channel
        if file_list.shape[1] == 4:
            file_list = np.hstack([file_list,
                                   np.zeros(shape=(file_list.shape[0], 1), dtype='int32')])
        # convert into: audio_path -> segment(name, start, end, channel)
        jobs = defaultdict(list)
        for segment, file, start, end, channel in file_list:
            jobs[file].append((segment, float(start), float(end), int(channel)))
        return np.array(jobs.items(), dtype=object)

    def map(self, audio_path, segments):
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
            # processing all segments
            for name, start, end, channel in segments:
                start = int(float(start) * fs)
                end = int(N if end < 0 else end * fs)
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                features = speech_features_extraction(data.ravel(), fs=fs,
                    n_filters=self.n_filters, n_ceps=self.n_ceps,
                    win=self.win, shift=self.shift, delta_order=self.delta_order,
                    energy=self.energy, vad=self.vad, dtype=self.dtype,
                    pitch_threshold=self.pitch_threshold,
                    get_spec=self.get_spec, get_mspec=self.get_mspec,
                    get_mfcc=self.get_mfcc, get_pitch=self.get_pitch)
                if features is not None:
                    yield (name, features)
                else:
                    msg = 'Ignore segments: %s, error: NaN values' % name
                    warnings.warn(msg)
        except Exception, e:
            msg = 'Ignore file: %s, error: %s' % (audio_path, str(e))
            warnings.warn(msg)
            if self.robust:
                yield None
            else:
                import traceback; traceback.print_exc()
                raise e

    def reduce(self, results):
        # contains (name, (spec, mspec, mfcc, vad))
        for r in results:
            yield r


class SpeechFeaturesSaver(object):
    """ SpeechFeatureSaver
    This function take output from Feeder with SpeechFeatures recipe
    and update output dataset

    Parameters
    ----------
    outpath: str
        path to output dataset
    datatype : memmap, hdf5
        datatype to save given task
    """

    def __init__(self, outpath, datatype):
        super(SpeechFeaturesSaver, self).__init__()
        self.dataset = Dataset(outpath)
        if datatype not in ('memmap', 'hdf5'):
            raise ValueError('datatype must be "memmap", or "hdf5"')
        self.datatype = datatype

    def run(self, feeder):
        dataset = self.dataset
        datatype = self.datatype
        # ====== indices ====== #
        indices = []
        start = 0
        # ====== statistic ====== #
        spec_sum1, spec_sum2 = 0., 0.
        mspec_sum1, mspec_sum2 = 0., 0.
        mfcc_sum1, mfcc_sum2 = 0., 0.
        pitch_sum1, pitch_sum2 = 0., 0.
        # ====== cache ====== #
        # all data are cached and periodically flushed
        spec_cache = []
        mspec_cache = []
        mfcc_cache = []
        pitch_cache = []
        vad_cache = []

        # ====== helper ====== #
        def flush_feature(name, cache):
            if len(cache) > 0:
                cache = np.concatenate(cache, 0)
                if name in dataset:
                    dataset[name].append(cache)
                else:
                    dataset[(name, datatype)] = cache
            return []
        # ====== processing ====== #
        prog = Progbar(target=feeder.shape[0])
        for count, (name, (spec, mspec, mfcc, pitch, vad)) in enumerate(feeder):
            # ====== spec ====== #
            if spec is not None:
                X, sum1, sum2 = spec
                spec_cache.append(X)
                spec_sum1 += sum1; spec_sum2 += sum2
                n = X.shape[0]; del X
            # ====== mspec ====== #
            if mspec is not None:
                X, sum1, sum2 = mspec
                mspec_cache.append(X)
                mspec_sum1 += sum1; mspec_sum2 += sum2
                n = X.shape[0]; del X
            # ====== mfcc ====== #
            if mfcc is not None:
                X, sum1, sum2 = mfcc
                mfcc_cache.append(X)
                mfcc_sum1 += sum1; mfcc_sum2 += sum2
                n = X.shape[0]; del X
            # ====== pitch ====== #
            if pitch is not None:
                X, sum1, sum2 = pitch
                pitch_cache.append(X)
                pitch_sum1 += sum1; pitch_sum2 += sum2
                n = X.shape[0]; del X
            # ====== vad ====== #
            if vad is not None:
                assert vad.shape[0] == n,\
                    'VAD mismatch features shape: %d != %d' % (vad.shape[0], n)
                vad_cache.append(vad)
                del vad
            # ====== flush cache ====== #
            if count % 12 == 0:
                spec_cache = flush_feature('spec', spec_cache)
                mspec_cache = flush_feature('mspec', mspec_cache)
                mfcc_cache = flush_feature('mfcc', mfcc_cache)
                pitch_cache = flush_feature('pitch', pitch_cache)
                vad_cache = flush_feature('vad', vad_cache)
            # index
            indices.append([name, start, start + n])
            start += n
            # ====== update progress ====== #
            prog.title = name
            prog.add(1)
        # ====== end, flush the mean and std ====== #
        flush_feature('spec', spec_cache)
        flush_feature('mspec', mspec_cache)
        flush_feature('mfcc', mfcc_cache)
        flush_feature('pitch', pitch_cache)
        flush_feature('vad', vad_cache)
        dataset.flush()
        # ====== saving indices ====== #
        with open(os.path.join(dataset.path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d\n' % (name, start, end))

        # ====== save mean and std ====== #
        def save_mean_std(sum1, sum2, name, dataset):
            mean = sum1 / start
            std = np.sqrt(sum2 / start - mean**2)
            assert not np.any(np.isnan(mean)), 'Mean contains NaN'
            assert not np.any(np.isnan(std)), 'Std contains NaN'
            dataset[name + '_mean'] = mean
            dataset[name + '_std'] = std
        if spec is not None:
            save_mean_std(spec_sum1, spec_sum2, 'spec', dataset)
        if mspec is not None:
            save_mean_std(mspec_sum1, mspec_sum2, 'mspec', dataset)
        if mfcc is not None:
            save_mean_std(mfcc_sum1, mfcc_sum2, 'mfcc', dataset)
        if pitch is not None:
            save_mean_std(pitch_sum1, pitch_sum2, 'pitch', dataset)
        # ====== final flush() ====== #
        dataset.flush()
        dataset.close()


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


class VideoFeature(FeederRecipe):

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
            if 'frames' in dataset: dataset['frames'].append(X)
            else: dataset[('frames', datatype)] = X
            # update running statistics
            sum1 += s1; sum2 += s2; n = X.shape[0]
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
        dataset[name + '_mean'] = mean
        dataset[name + '_std'] = std
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
