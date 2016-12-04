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
import shutil
from numbers import Number
from multiprocessing import Pool, cpu_count, Process, Queue
from six import add_metaclass
from six.moves import zip, zip_longest, range
from abc import ABCMeta, abstractmethod, abstractproperty

from collections import defaultdict
import numpy as np

from odin.preprocessing import speech, video, image
from odin.utils import (queue, Progbar, segment_list, as_tuple,
                        get_all_files, get_tempdir)
from odin.utils.decorators import autoinit
from odin.utils.mpi import MPI
from .dataset import Dataset
from .recipes import FeederRecipe

try:
    import sidekit
except:
    warnings.warn('You need "sidekit" for audio signal processing.')

__all__ = [
    'speech_features_extraction',
    'SpeechProcessor'
]


# ===========================================================================
# Helper
# ===========================================================================
# ==================== For speech ==================== #
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
                               delta_order, energy, pitch_threshold,
                               get_spec, get_mspec, get_mfcc, get_pitch, get_vad):
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
    if get_vad:
        distribNb, nbTrainIt = 8, 12
        if isinstance(get_vad, (tuple, list)):
            distribNb, nbTrainIt = get_vad
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
    ret = []
    if spec is not None: ret.append(spec)
    if mspec is not None: ret.append(mspec)
    if mfcc is not None: ret.append(mfcc)
    if pitch is not None: ret.append(pitch)
    if vad_idx is not None: ret.append(vad_idx)
    return tuple(ret)


# ==================== general ==================== #
@add_metaclass(ABCMeta)
class FeatureProcessor(object):

    """ FeatureProcessor

    Example
    -------
    >>> feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', fs=8000,
    >>>                          win=0.025, shift=0.01, n_filters=40, n_ceps=13,
    >>>                          delta_order=2, energy=True, pitch_threshold=0.5,
    >>>                          get_spec=True, get_mspec=True, get_mfcc=True,
    >>>                          get_pitch=False, get_vad=True,
    >>>                          save_stats=True, substitute_nan=None,
    >>>                          dtype='float32', datatype='memmap', ncpu=4)
    >>> feat.run()
    """

    def __init__(self, output_path, datatype='memmap',
                 save_stats=True, substitute_nan=None,
                 ncpu=1):
        super(FeatureProcessor, self).__init__()
        if datatype not in ('memmap', 'hdf5'):
            raise ValueError('datatype must be "memmap", or "hdf5"')
        self.datatype = datatype
        if os.path.exists(output_path):
            warnings.warn('Remove exist dataset at path: "%s"' % output_path)
            shutil.rmtree(output_path)
        self.dataset = Dataset(output_path)
        self.save_stats = bool(save_stats)
        self.substitute_nan = substitute_nan
        self.ncpu = ncpu

    # ==================== Abstract properties ==================== #
    @abstractproperty
    def features_properties(self):
        """ Return list of features' properties
        (name, dtype, statistic-able)
        """
        pass

    @abstractmethod
    def map(self, job):
        pass

    def run(self):
        if not hasattr(self, 'jobs'):
            raise Exception('the Processor must has "jobs" attribute, which is '
                            'the list of all jobs.')
        prog = Progbar(target=len(self.jobs))
        dataset = self.dataset
        datatype = self.datatype
        if self.ncpu is None: # auto select number of CPU
            ncpu = min(len(self.jobs), int(1.2 * cpu_count()))
        else:
            ncpu = self.ncpu
        # ====== indices ====== #
        indices = []
        # ====== statistic ====== #
        sum1 = defaultdict(int)
        sum2 = defaultdict(int)
        # all data are cached for periodically flushed
        cache = defaultdict(list)
        cache_limit = max(2, int(0.12 * len(self.jobs)))
        ref_vars = {'start': 0, 'processed_count': 0}

        # ====== helper ====== #
        def flush_feature(name, cache_data):
            if len(cache_data) > 0:
                cache_data = np.concatenate(cache_data, 0)
                if name in dataset:
                    dataset[name].append(cache_data)
                else:
                    dataset[(name, datatype)] = cache_data

        def wrapped_reduce(result):
            name, data = result
            ref_vars['processed_count'] += 1
            # check data
            if not isinstance(data, (tuple, list)):
                data = (data,)
            length = [] # store length of all data for validation
            # processing
            for prop, d in zip(self.features_properties, data):
                length.append(len(d))
                n, t, s = prop # name, dtype, stats
                cache[n].append(d.astype(t))
                if s: # save stats
                    sum1[n] += np.sum(d, axis=0, dtype='float64')
                    sum2[n] += np.sum(np.power(d, 2), axis=0, dtype='float64')
                del d
            # check if lengths are matched
            if len(set(length)) != 1:
                raise Exception('length mismatch between all data: %s' % str(length))
            # ====== flush cache ====== #
            if ref_vars['processed_count'] % cache_limit == 0: # 12 + 8
                for i, j in cache.iteritems():
                    flush_feature(i, j)
                cache.clear()
            # index
            indices.append([name, ref_vars['start'], ref_vars['start'] + length[0]])
            ref_vars['start'] += length[0]
            # ====== update progress ====== #
            return name

        # ====== processing ====== #
        mpi = MPI(self.jobs, self.map, wrapped_reduce,
                  ncpu=ncpu, buffer_size=1, maximum_queue_size=ncpu * 3)
        for name in mpi:
            prog.title = '%-20s' % name
            prog.add(1)
        # ====== end, flush the mean and std ====== #
        for i, j in cache.iteritems():
            flush_feature(i, j)
        cache = None
        dataset.flush()
        # ====== saving indices ====== #
        with open(os.path.join(dataset.path, 'indices.csv'), 'w') as f:
            for name, start, end in indices:
                f.write('%s %d %d\n' % (name, start, end))

        # ====== save mean and std ====== #
        def save_mean_std(sum1, sum2, name, dataset):
            N = dataset[name].shape[0]
            mean = sum1 / N
            std = np.sqrt(sum2 / N - mean**2)
            if self.substitute_nan is not None:
                mean = np.where(np.isnan(mean), self.substitute_nan, mean)
                std = np.where(np.isnan(std), self.substitute_nan, std)
            else:
                assert not np.any(np.isnan(mean)), 'Mean contains NaN'
                assert not np.any(np.isnan(std)), 'Std contains NaN'
            dataset[name + '_sum1'] = sum1
            dataset[name + '_sum2'] = sum2
            dataset[name + '_mean'] = mean
            dataset[name + '_std'] = std
        # save all stats
        if self.save_stats:
            print('Saving statistics of each data ...')
            for n, d, s in self.features_properties:
                if s: # save stats
                    print(' * Name:', n)
                    s1, s2 = sum1[n], sum2[n]
                    save_mean_std(s1, s2, n, dataset)
        # ====== final flush() ====== #
        dataset.flush()
        dataset.close()


# ===========================================================================
# Speech features
# ===========================================================================
class SpeechProcessor(FeatureProcessor):

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

    Return
    ------
    spec, mspec, mfcc, pitch, vad_idx

    Example
    -------
    >>> feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav', fs=8000,
    >>>                          win=0.025, shift=0.01, n_filters=40, n_ceps=13,
    >>>                          delta_order=2, energy=True, pitch_threshold=0.5,
    >>>                          get_spec=True, get_mspec=True, get_mfcc=True,
    >>>                          get_pitch=False, get_vad=True,
    >>>                          save_stats=True, substitute_nan=None,
    >>>                          dtype='float32', datatype='memmap', ncpu=4)
    >>> feat.run()
    '''

    def __init__(self, segments, output_path,
                 audio_ext=None, fs=8000,
                 win=0.025, shift=0.01, n_filters=40, n_ceps=13,
                 delta_order=2, energy=True, pitch_threshold=0.5,
                 get_spec=False, get_mspec=True, get_mfcc=False,
                 get_pitch=False, get_vad=True,
                 save_stats=True, substitute_nan=None,
                 dtype='float32', datatype='memmap', ncpu=1):
        super(SpeechProcessor, self).__init__(output_path=output_path,
            datatype=datatype, save_stats=save_stats,
            substitute_nan=substitute_nan, ncpu=ncpu)
        audio_ext = as_tuple('' if audio_ext is None else audio_ext, t=str)
        # ====== load jobs ====== #
        # NOT loaded segments
        if isinstance(segments, str):
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
        elif isinstance(segments, (tuple, list)):
            # just a list of path to file
            if isinstance(segments[0], str):
                file_list = [(os.path.basename(i), os.path.abspath(i), 0.0, -1.0)
                             for i in segments]
            # list of all information
            elif isinstance(segments[0], (tuple, list)):
                if len(segments[0]) != 4 and len(segments[0]) != 5:
                    raise Exception('segments must contain information in following for:'
                                    '[name] [path] [start] [end]')
                file_list = segments
        # filter using support audio extension
        file_list = [f for f in file_list if any(ext == f[1][-len(ext):] for ext in audio_ext)]
        # if no channel is provided, append the channel
        file_list = [list(f) + [0] if len(f) == 4 else f for f in file_list]
        # convert into: audio_path -> segment(name, start, end, channel)
        self.jobs = defaultdict(list)
        for segment, file, start, end, channel in file_list:
            self.jobs[file].append((segment, float(start), float(end), int(channel)))
        self.jobs = sorted(self.jobs.items(), key=lambda x: x[0])
        # ====== which features to get ====== #
        if not get_spec and not get_mspec \
            and not get_mfcc and not get_pitch:
            raise Exception('You must specify which features you want: '
                            'spectrogram, filter-banks, MFCC, or pitch.')
        features_properties = []
        if get_spec: features_properties.append(('spec', dtype, True))
        if get_mspec: features_properties.append(('mspec', dtype, True))
        if get_mfcc: features_properties.append(('mfcc', dtype, True))
        if get_pitch: features_properties.append(('pitch', dtype, True))
        if get_vad: features_properties.append(('vad', 'uint8', False))
        self.__features_properties = features_properties

        self.get_spec = get_spec
        self.get_mspec = get_mspec
        self.get_mfcc = get_mfcc
        self.get_pitch = get_pitch
        self.get_vad = get_vad
        # ====== feature infor ====== #
        self.fs = fs
        self.win = win
        self.shift = shift
        self.n_filters = n_filters
        self.n_ceps = n_ceps
        self.delta_order = int(delta_order)
        self.energy = energy
        # constraint pitch threshold in 0-1
        self.pitch_threshold = min(max(pitch_threshold, 0.), 1.)

    # ==================== Abstract properties ==================== #
    @property
    def features_properties(self):
        """ Returnn all name of given features"""
        return self.__features_properties

    def map(self, job):
        '''
        Return
        ------
        [(name, spec, mspec, mfcc, pitch, vad), ...]
        '''

        audio_path, segments = job[0] if len(job) == 1 else job
        fs = self.fs
        try:
            # load audio data
            s, orig_fs = speech.read(audio_path)
            orig_fs = fs if orig_fs is None else orig_fs
            # check frequency for downsampling (if necessary)
            if fs is None:
                fs = orig_fs
            elif fs != orig_fs: # downsample or upsample
                import resampy
                s = resampy.resample(s, sr_orig=orig_fs, sr_new=fs, axis=0, filter='kaiser_best')
            N = len(s)
            # processing all segments
            ret = []
            for name, start, end, channel in segments:
                start = int(float(start) * fs)
                end = int(N if end < 0 else end * fs)
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                features = speech_features_extraction(data.ravel(), fs=fs,
                    n_filters=self.n_filters, n_ceps=self.n_ceps,
                    win=self.win, shift=self.shift, delta_order=self.delta_order,
                    energy=self.energy, pitch_threshold=self.pitch_threshold,
                    get_spec=self.get_spec, get_mspec=self.get_mspec,
                    get_mfcc=self.get_mfcc, get_pitch=self.get_pitch,
                    get_vad=self.get_vad)
                if features is not None:
                    ret.append((name, features))
                else:
                    msg = 'Ignore segments: %s, error: NaN values' % name
                    warnings.warn(msg)
            # return the results
            return (i for i in ret)
        except Exception, e:
            msg = 'Ignore file: %s, error: %s' % (audio_path, str(e))
            import traceback; traceback.print_exc()
            raise e


# ===========================================================================
# Images
# ===========================================================================
class ImageFeatures(FeederRecipe):
    """ ImageFeauters extractor
    This function take output from Feeder with SpeechFeatures recipe
    and update output dataset

    Parameters
    ----------
    image_ext: str, or list of str
        extensions of images
    grayscale: bool
        force to convert Image to grayscale or not
    crop: 4-tuple of int
         (left, upper, right, lower)
    target_size: 2-tuple of int
        desire size for image (image will be padded if the size
        mis-match)
    transpose: int, or list of int
        if a list of int is provided, will return a list of images
        <0: Do nothing
        0: PIL.Image.FLIP_LEFT_RIGHT
        1: PIL.Image.FLIP_TOP_BOTTOM
        2: PIL.Image.ROTATE_90
        3: PIL.Image.ROTATE_180
        4: PIL.Image.ROTATE_270
        5: PIL.Image.TRANSPOSE
    resample_mode: int
        0 = PIL.Image.NEAREST: use nearest neighbour
        1 = PIL.Image.LANCZOS: a high-quality downsampling filter
        2 = PIL.Image.BILINEAR: linear interpolation
        3 = PIL.Image.BICUBIC: cubic spline interpolation

    """

    def __init__(self, image_ext=None, grayscale=False,
                 crop=None, target_size=None,
                 transpose=None, resample_mode=2):
        super(ImageFeatures, self).__init__()
        self.image_ext = ('',) if image_ext is None else as_tuple(image_ext, t=str)
        self.crop = crop if crop is None else as_tuple(crop, 4, int)
        self.grayscale = bool(grayscale)
        self.target_size = target_size
        self.transpose = (-1,) if transpose is None else as_tuple(transpose, t=int)
        self.resample_mode = resample_mode

    def shape_transform(self, shape):
        """ Return the new shape that transformed by this Recipe """
        return (shape[0] * len(self.transpose),) + shape[1:]

    def preprocess_indices(self, indices):
        # filter using support audio extension
        file_list = np.array([f for f in indices
                              if any(ext in f for ext in self.image_ext)])
        return file_list

    def map(self, path):
        '''
        Return
        ------
        [(name, spec(x, sum1, sum2), # if available, otherwise None
                mspec(x, sum1, sum2), # if available, otherwise None
                mfcc(x, sum1, sum2), # if available, otherwise None
                pitch(x, sum1, sum2), # if available, otherwise None
                vad), ...]
        '''
        X = image.read(path, grayscale=self.grayscale,
                       crop=self.crop, scale=None,
                       target_size=self.target_size,
                       transpose=self.transpose,
                       resample_mode=self.resample_mode)
        if not isinstance(X, (tuple, list)):
            X = (X,)
        name = os.path.basename(path)
        ret = []
        for i, j in zip(self.transpose, X):
            ret.append(('%s,%d' % (name, i),
                        np.expand_dims(j, axis=0)))
        return ret

    def reduce(self, images):
        for img in images:
            for name, x in img:
                # contains different transpose of images
                yield (name, x)


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
