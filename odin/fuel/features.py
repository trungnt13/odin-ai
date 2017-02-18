# ===========================================================================
# Parallel features processing using multi-core CPU and multiprocessing
# Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import sys
import os
import types
import warnings
import shutil
from numbers import Number
from multiprocessing import Pool, cpu_count, Process, Queue
from six import add_metaclass, string_types
from six.moves import zip, zip_longest, range, cPickle
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

__all__ = [
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
                 pca=True, pca_whiten=False,
                 save_stats=True, substitute_nan=None,
                 ncache=0.12, ncpu=1):
        super(FeatureProcessor, self).__init__()
        if datatype not in ('memmap', 'hdf5'):
            raise ValueError('datatype must be "memmap", or "hdf5"')
        self.datatype = datatype
        if os.path.exists(output_path):
            warnings.warn('Remove exist dataset at path: "%s"' % output_path)
            shutil.rmtree(output_path)
        self.dataset = Dataset(output_path)
        # PCA
        self.pca = bool(pca)
        self.pca_whiten = bool(pca_whiten)
        # STATS
        self.save_stats = bool(save_stats)
        self.substitute_nan = substitute_nan
        self.ncpu = ncpu
        self.ncache = ncache

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
        if self.pca:
            from odin.ml import MiniBatchPCA
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
        statistic_able = {i[0]: i[-1] for i in self.features_properties}
        sum1 = defaultdict(int)
        sum2 = defaultdict(int)
        pca = defaultdict(lambda *args, **kwargs:
            MiniBatchPCA(n_components=None, whiten=self.pca_whiten,
                         copy=True, batch_size=None) if self.pca else None)
        # all data are cached for periodically flushed
        cache = defaultdict(list)
        if self.ncache <= 1:
            cache_limit = max(2, int(0.12 * len(self.jobs)))
        else:
            cache_limit = int(self.ncache)
        ref_vars = {'start': 0, 'processed_count': 0}

        # ====== helper ====== #
        def flush_feature(name, cache_data):
            if len(cache_data) > 0:
                cache_data = np.concatenate(cache_data, 0)
                # NOTE: if nb_samples < nb_features, fitting PCA
                # will course error
                if self.pca and statistic_able[name] and \
                cache_data.shape[0] > cache_data.shape[-1]:
                    pca[name].partial_fit(cache_data)
                # flush data
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
                if self.save_stats and s: # save stats
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
        def save_mean_std(sum1, sum2, pca, name, dataset):
            N = dataset[name].shape[0]
            mean = sum1 / N
            std = np.sqrt(sum2 / N - mean**2)
            if self.substitute_nan is not None:
                mean = np.where(np.isnan(mean), self.substitute_nan, mean)
                std = np.where(np.isnan(std), self.substitute_nan, std)
            else:
                assert not np.any(np.isnan(mean)), 'Mean contains NaN, %s' % name
                assert not np.any(np.isnan(std)), 'Std contains NaN, %s' % name
            dataset[name + '_sum1'] = sum1
            dataset[name + '_sum2'] = sum2
            dataset[name + '_mean'] = mean
            dataset[name + '_std'] = std
            dataset[name + '_pca'] = pca
        # save all stats
        if self.save_stats:
            print('Saving statistics of each data ...')
            for n, d, s in self.features_properties:
                if s: # save stats
                    print(' * Name:', n)
                    s1, s2, pca_ = sum1[n], sum2[n], pca[n]
                    save_mean_std(s1, s2, pca_, n, dataset)
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
    sr: int
        sample rate
    win: float
        window length in millisecond
    shift: float
        hop length between windows, in millisecond
    nb_melfilters: int
        number of Mel bands to generate
    nb_ceps: int
        number of MFCCs to return
    get_spec: bool
        if True, include the log-power spectrogram
    get_mspec: bool
        if True, include the log-power mel-spectrogram
    get_mfcc: bool
        if True, include the MFCCs features
    get_qspec: bool
        if True, return Q-transform coefficients
    get_phase: bool
        if True, return phase components of STFT
    get_pitch:
        if True, include the Pitch frequency (F0)
    get_vad: bool
        if True, include the indicators of voice activities detection
    get_energy: bool
        if True, include the log energy of each frames
    get_delta: bool or int
        if True and > 0, for each features append the delta with given order-th
        (e.g. delta=2 will include: delta1 and delta2)
    fmin : float > 0 [scalar]
        lower frequency cutoff.
    fmax : float > 0 [scalar]
        upper frequency cutoff.
    sr_new: int or None
        new sample rate
    preemphasis: float `(0, 1)`
        pre-emphasis coefficience
    pitch_threshold: float in `(0, 1)`
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`
    pitch_fmax: float
        maximum frequency of pitch
    smooth_vad: int, bool
        window length to smooth the vad indices.
        If True default window length is 3.
    cqt_bins : int > 0
        Number of frequency bins for constant Q-transform, starting at `fmin`
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
    center : bool
        If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        If `False`, then `D[:, t]` begins at `y[t * hop_length]`
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
    ncache: float or int
        number of samples are kept until flush to the disk.
    ncpu: int
        number of CPU used for this task.

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

    def __init__(self, segments, output_path, sr=None,
                win=0.02, shift=0.01, nb_melfilters=24, nb_ceps=12,
                get_spec=True, get_mspec=False, get_mfcc=False,
                get_qspec=False, get_phase=False, get_pitch=False,
                get_vad=True, get_energy=False, get_delta=False,
                fmin=64, fmax=None, sr_new=None, preemphasis=0.97,
                pitch_threshold=0.8, pitch_fmax=1200,
                smooth_vad=0, cqt_bins=96, pca=True, pca_whiten=False,
                center=True, audio_ext=None, save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap', ncache=0.12, ncpu=1):
        super(SpeechProcessor, self).__init__(output_path=output_path,
            datatype=datatype, pca=pca, pca_whiten=pca_whiten,
            save_stats=save_stats, substitute_nan=substitute_nan,
            ncache=ncache, ncpu=ncpu)
        audio_ext = as_tuple('' if audio_ext is None else audio_ext,
                             t=string_types)
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
        # check empty jobs
        if len(self.jobs) == 0:
            raise Exception('NO jobs found for processing.')
        # ====== which features to get ====== #
        if not get_spec and not get_mspec and not get_mfcc \
        and not get_pitch and not get_energy and not get_vad:
            raise Exception('You must specify which features you want: '
                            'spectrogram, filter-banks, MFCC, or pitch.')
        features_properties = []
        if get_mfcc: features_properties.append(('mfcc', dtype, True))
        if get_energy: features_properties.append(('energy', dtype, True))
        if get_spec: features_properties.append(('spec', dtype, True))
        if get_mspec: features_properties.append(('mspec', dtype, True))
        if get_qspec:
            features_properties.append(('qspec', dtype, True))
            if get_mspec: features_properties.append(('qmspec', dtype, True))
            if get_mfcc: features_properties.append(('qmfcc', dtype, True))
            if get_phase: features_properties.append(('qphase', dtype, True))
        if get_phase: features_properties.append(('phase', dtype, True))
        if get_pitch: features_properties.append(('pitch', dtype, True))
        if get_vad: features_properties.append(('vad', 'uint8', False))
        self.__features_properties = features_properties

        self.get_spec = get_spec
        self.get_mspec = get_mspec
        self.get_mfcc = get_mfcc
        self.get_pitch = get_pitch
        self.get_qspec = get_qspec
        self.get_phase = get_phase
        self.get_vad = get_vad
        self.get_energy = get_energy
        self.get_delta = int(get_delta)
        # ====== feature information ====== #
        self.sr = sr
        self.win = win
        self.shift = shift
        self.nb_melfilters = nb_melfilters
        self.nb_ceps = nb_ceps
        # constraint pitch threshold in 0-1
        self.pitch_threshold = min(max(pitch_threshold, 0.), 1.)
        self.pitch_fmax = pitch_fmax
        self.smooth_vad = smooth_vad
        self.cqt_bins = cqt_bins
        self.fmin = fmin
        self.fmax = fmax
        self.sr_new = sr_new
        self.preemphasis = preemphasis
        self.center = center

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
        try:
            # load audio data
            s, sr_orig = speech.read(audio_path)
            if sr_orig is not None and self.sr is not None and \
            sr_orig != self.sr:
                raise Exception('Given sample rate (%d Hz) is different from '
                                'audio file sample rate (%d Hz).' %
                                (self.sr, sr_orig))
            if sr_orig is None:
                sr_orig = self.sr
            N = len(s)
            # processing all segments
            ret = []
            for name, start, end, channel in segments:
                start = int(float(start) * sr_orig)
                end = int(N if end < 0 else end * sr_orig)
                data = s[start:end, channel] if s.ndim > 1 else s[start:end]
                features = speech.speech_features(data.ravel(), sr=sr_orig,
                    win=self.win, shift=self.shift,
                    nb_melfilters=self.nb_melfilters, nb_ceps=self.nb_ceps,
                    get_spec=self.get_spec, get_mspec=self.get_mspec,
                    get_mfcc=self.get_mfcc, get_qspec=self.get_qspec,
                    get_phase=self.get_phase, get_pitch=self.get_pitch,
                    get_vad=self.get_vad, get_energy=self.get_energy,
                    get_delta=self.get_delta,
                    pitch_threshold=self.pitch_threshold,
                    pitch_fmax=self.pitch_fmax, smooth_vad=self.smooth_vad,
                    cqt_bins=self.cqt_bins, fmin=self.fmin, fmax=self.fmax,
                    sr_new=self.sr_new, preemphasis=self.preemphasis,
                    center=self.center)
                if features is not None:
                    ret.append((name, [features[i[0]]
                                       for i in self.__features_properties]))
                else:
                    msg = 'Ignore segments: %s, error: NaN values' % name
                    warnings.warn(msg)
            # return the results as a generator
            return (i for i in ret)
        except Exception as e:
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
        self.image_ext = ('',) if image_ext is None else as_tuple(image_ext,
                                                                  t=string_types)
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
        assert not np.any(np.isnan(mean)), 'Mean contains NaN, name:' % os.path.basename(path)
        assert not np.any(np.isnan(std)), 'Std contains NaN, name:' % os.path.basename(path)
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
