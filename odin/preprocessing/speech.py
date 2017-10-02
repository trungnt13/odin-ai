# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram preprocess utilities is adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import six
import math
import copy
import warnings
from collections import OrderedDict, Mapping

import numpy as np
from scipy.signal import lfilter

from odin.utils import is_number, cache_memory, is_string, as_tuple
from .base import Extractor
from .signal import (pad_center, get_window, segment_axis, stft, istft,
                     compute_delta, smooth, pre_emphasis, spectra,
                     vad_energy, power2db, pitch_track, resample,
                     rastafilt, mvn, wmvn)


# ===========================================================================
# Predefined variables of speech datasets
# ===========================================================================
# ==================== Timit ==================== #
timit_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th',
    'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
timit_39 = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd',
    'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't',
    'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
timit_map = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
    'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil'}


# ===========================================================================
# Helper function
# ===========================================================================
def _extract_s_sr(s_sr):
    if isinstance(s_sr, Mapping):
        s_sr = (s_sr['raw'], s_sr['sr'])
    elif not isinstance(s_sr, (tuple, list)) or \
    not isinstance(s_sr[0], np.ndarray) or \
    not is_number(s_sr[1]):
        raise ValueError("Input to SpectraExtractor must be a tuple, or list "
                         "of raw signal (ndarray) and sample rate (int).")
    s, sr = s_sr
    return s, int(sr)


def _extract_frame_step_length(sr, frame_length, step_length):
    # ====== check frame length ====== #
    if frame_length < 1.:
        frame_length = int(sr * frame_length)
    else:
        frame_length = int(frame_length)
    # ====== check step length ====== #
    if step_length is None:
        step_length = frame_length // 4
    elif step_length < 1.:
        step_length = int(sr * step_length)
    else:
        step_length = int(step_length)
    return frame_length, step_length


@cache_memory
def _num_two_factors(x):
    """return number of times x is divideable for 2"""
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2
    return num_twos


@cache_memory
def _max_fft_bins(sr, n_fft, fmax):
    return [i + 1 for i, j in enumerate(np.linspace(0, float(sr) / 2, int(1 + n_fft // 2),
                                        endpoint=True)) if j >= fmax][0]


# ===========================================================================
# Audio feature extractor
# ===========================================================================
class AudioReader(Extractor):

    def __init__(self, sr=None, sr_new=None, best_resample=True,
                 remove_dc_n_dither=True, preemphasis=0.97,
                 dtype=None, saved=False):
        super(AudioReader, self).__init__()
        self.sr = sr
        self.sr_new = sr_new
        self.best_resample = best_resample
        self.remove_dc_n_dither = bool(remove_dc_n_dither)
        self.preemphasis = preemphasis
        self.dtype = dtype
        self.saved = bool(saved)

    def _transform(self, path_or_array):
        sr = None # by default, we don't know sample rate
        # ====== read audio from path or opened file ====== #
        if is_string(path_or_array) or isinstance(path_or_array, file):
            if isinstance(path_or_array, file):
                f = path_or_array
                path = f.name
            else:
                f = open(path_or_array, 'r')
                path = path_or_array
            # ====== process ====== #
            if '.pcm' in path.lower():
                s = np.memmap(f, dtype=np.int16, mode='r')
            else:
                import soundfile
                try:
                    s, sr = soundfile.read(f)
                except Exception as e:
                    if '.sph' in f.name.lower():
                        f.seek(0)
                        s = np.memmap(f, dtype=np.int16, mode='r')
                    else:
                        raise e
            # close file
            f.close()
        # ====== provided np.ndarray for normalization ====== #
        else:
            s = path_or_array
        # ====== valiate sample rate ====== #
        if self.sr is not None:
            if sr is not None and sr != self.sr:
                raise RuntimeError("Audio file at path: '%s' has sample rate: '%d'"
                                   ", but the provided sample rate is '%d'" %
                                   (path, sr, self.sr))
            sr = int(self.sr)
        # resampling if necessary
        if sr is not None and self.sr_new is not None:
            s = resample(s, sr, self.sr_new, best_algorithm=self.best_resample)
            sr = int(self.sr_new)
        # ====== normalizing ====== #
        np.random.seed(8)  # for repeatability
        # ====== remove DC offset and diterhing ====== #
        # Approached suggested by:
        # 'Omid Sadjadi': 'omid.sadjadi@nist.gov'
        if self.remove_dc_n_dither:
            # assuming 16-bit
            if max(abs(s)) <= 1.:
                s = s * 2**15
            # select alpha
            if sr == 16000:
                alpha = 0.99
            elif sr == 8000:
                alpha = 0.999
            else:
                raise ValueError('Sampling frequency %s not supported' % str(sr))
            slen = s.size
            s = lfilter([1, -1], [1, -alpha], s)
            dither = np.random.rand(slen) + np.random.rand(slen) - 1
            s_pow = max(s.std(), 1e-20)
            s = s + 1.e-6 * s_pow * dither
        else: # just remove DC offset
            s -= np.mean(s, 0)
        # ====== pre-emphasis ====== #
        if self.preemphasis is not None and 0. < self.preemphasis < 1.:
            s = pre_emphasis(s, coeff=float(self.preemphasis))
        # ====== dtype ====== #
        if self.dtype is not None:
            s = s.astype(self.dtype)
        if self.saved:
            return {'raw': s, 'sr': sr}
        return (s, sr)


class SpectraExtractor(Extractor):
    """AcousticExtractor

    Parameters
    ----------
    frame_length: int
        number of samples point for 1 frame
    step_length: int
        number of samples point for 1 step (when shifting the frames)
        If unspecified, defaults `win_length / 4`.
    nb_fft: int > 0 [scalar]
        FFT window size
        If not provided, uses the smallest power of 2 enclosing `frame_length`.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`
    """

    def __init__(self, frame_length, step_length=None, nfft=512, window='hann',
                 nmels=None, nceps=None, fmin=64, fmax=None,
                 power=2.0, log=True, padding=False):
        super(SpectraExtractor, self).__init__()
        # ====== STFT ====== #
        self.frame_length = frame_length
        self.step_length = step_length
        self.nfft = nfft
        self.window = window
        # ====== ceptral analysis ====== #
        self.nmels = nmels
        self.nceps = nceps
        self.fmin = fmin
        self.fmax = fmax
        # ====== power spectrum ====== #
        self.power = power
        self.log = log
        # ====== others ====== #
        self.padding = padding

    def _transform(self, s_sr):
        s, sr = _extract_s_sr(s_sr)
        frame_length, step_length = _extract_frame_step_length(
            sr, self.frame_length, self.step_length)
        # ====== extract spectra ====== #
        feat = spectra(sr=sr, frame_length=frame_length, y=s, S=None,
                       step_length=step_length, nfft=self.nfft,
                       window=self.window,
                       nmels=self.nmels, nceps=self.nceps,
                       fmin=self.fmin, fmax=self.fmax,
                       top_db=80., power=self.power, log=self.log,
                       padding=self.padding)
        feat['sr'] = sr
        return feat


class CQTExtractor(Extractor):
    """ CQTExtractor """

    def __init__(self, frame_length, step_length=None, nbins=96, window='hann',
                 nmels=None, nceps=None, fmin=64, fmax=None, padding=False):
        super(CQTExtractor, self).__init__()
        self.frame_length = frame_length
        self.step_length = step_length
        self.nbins = int(nbins)
        self.window = window
        self.nmels = nmels
        self.nceps = nceps
        self.fmin = fmin
        self.fmax = fmax
        self.padding = padding

    def _transform(self, s_sr):
        s, sr = _extract_s_sr(s_sr)
        frame_length, step_length = _extract_frame_step_length(
            sr, self.frame_length, self.step_length)
        # ====== extract CQT ====== #
        from librosa.core import constantq
        # auto adjust bins_per_octave to get maximum range of frequency
        bins_per_octave = np.ceil(float(self.nbins - 1) / np.log2(sr / 2. / self.fmin)) + 1
        # adjust the bins_per_octave to make acceptable hop_length
        # i.e. 2_factors(hop_length) < [ceil(cqt_bins / bins_per_octave) - 1]
        if _num_two_factors(step_length) < np.ceil(self.nbins / bins_per_octave) - 1:
            bins_per_octave = np.ceil(self.nbins / (_num_two_factors(step_length) + 1))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            qtrans = constantq.cqt(s, sr=sr, hop_length=step_length, n_bins=self.nbins,
                                   bins_per_octave=int(bins_per_octave),
                                   fmin=self.fmin, tuning=0.0, real=False, norm=1,
                                   filter_scale=1., sparsity=0.01).astype('complex64')
        # ====== ceptral analysis ====== #
        feat = spectra(sr, frame_length, y=None, S=qtrans.T,
                       step_length=step_length, nfft=None, window='hann',
                       nmels=self.nmels, nceps=self.nceps,
                       fmin=64, fmax=self.fmax,
                       top_db=80.0, power=2.0, log=True,
                       padding=self.padding)
        # ====== add 'q' prefix for CQT features ====== #
        feat = {'q' + name: X for name, X in feat.iteritems()}
        feat['sr'] = sr
        return feat


class PitchExtractor(Extractor):

    def __init__(self, frame_length, step_length=None,
                 threshold=0.2, fmin=20, fmax=260,
                 algo='swipe', f0=False):
        super(PitchExtractor, self).__init__()
        self.threshold = threshold
        self.fmin = int(fmin)
        self.fmax = int(fmax)
        self.algo = algo
        self.f0 = f0
        self.frame_length = frame_length
        self.step_length = step_length

    def _transform(self, s_sr):
        s, sr = _extract_s_sr(s_sr)
        frame_length, step_length = _extract_frame_step_length(
            sr, self.frame_length, self.step_length)
        # ====== extract pitch ====== #
        pitch_freq = pitch_track(s, sr, step_length, fmin=self.fmin,
            fmax=self.fmax, threshold=self.threshold, otype='pitch',
            algorithm=self.algo)
        if self.f0:
            f0_freq = pitch_track(s, sr, step_length, fmin=self.fmin,
                fmax=self.fmax, threshold=self.threshold, otype='f0',
                algorithm=self.algo)
            return {'pitch': pitch_freq, 'f0': f0_freq}
        return {'pitch': pitch_freq}


class VADextractor(Extractor):

    def __init__(self, nb_mixture=3, nb_train_it=24 + 1, smooth_window=3,
                 feat_type='energy'):
        super(VADextractor, self).__init__()
        self.nb_mixture = int(nb_mixture)
        self.nb_train_it = int(nb_train_it)
        self.smooth_window = int(smooth_window)
        self.feat_type = str(feat_type).lower()

    def _transform(self, feat):
        # ====== select features type ====== #
        features = feat[self.feat_type]
        # ====== calculate VAD ====== #
        vad, vad_threshold = vad_energy(log_energy=features.ravel(),
            distrib_nb=self.nb_mixture, nb_train_it=self.nb_train_it)
        if self.smooth_window > 0:
            # at least 2 voice frames
            threshold = (2. / self.smooth_window)
            vad = smooth(
                vad, win=self.smooth_window, window='flat') >= threshold
        # ====== vad is only 0 and 1 so 'uint8' is enough ====== #
        vad = vad.astype('uint8')
        return {'vad': vad, 'vad_threshold': float(vad_threshold)}


class AcousticNorm(Extractor):

    def __init__(self, mean_var_norm=True, window_mean_var_norm=True,
                 win_length=301, var_norm=True, rasta=True,
                 feat_type=('mspec', 'spec', 'mfcc',
                            'qspec', 'qmfcc', 'qmspec')):
        super(AcousticNorm, self).__init__()
        self.mean_var_norm = bool(mean_var_norm)
        self.window_mean_var_norm = bool(window_mean_var_norm)
        self.rasta = bool(rasta)
        self.var_norm = bool(var_norm)
        # ====== check win_length ====== #
        win_length = int(win_length)
        if win_length % 2 == 0:
            raise ValueError("win_length must be odd number")
        if win_length < 3:
            raise ValueError("win_length must >= 3")
        self.win_length = win_length
        # ====== check which features will be normalized ====== #
        self.feat_type = tuple([feat.lower()
            for feat in as_tuple(feat_type, t=str)])

    def _transform(self, feat):
        if not isinstance(feat, Mapping):
            raise ValueError("AcousticNorm only transform dictionary of: feature "
                             "name -> features matrices into normalized features,"
                             "the given input is not dictionary")
        feat_normalized = {}
        # all `features` is [t, f] shape
        for name, features in feat.iteritems():
            if name in self.feat_type:
                features = features.T
                if 'mfcc' in name and self.rasta:
                    features = rastafilt(features)
                if self.mean_var_norm:
                    features = mvn(features, varnorm=self.var_norm)
                if self.window_mean_var_norm:
                    features = wmvn(features, w=self.win_length,
                                    varnorm=self.var_norm)
                # transpose back to [t, f]
                features = features.T
            feat_normalized[name] = features
        return feat_normalized


# ===========================================================================
# Spectrogram manipulation
# ===========================================================================
def save(file_or_path, s, sr, subtype=None):
    '''
    Parameters
    ----------
    s : array_like
        The data to write.  Usually two-dimensional (channels x frames),
        but one-dimensional `data` can be used for mono files.
        Only the data types ``'float64'``, ``'float32'``, ``'int32'``
        and ``'int16'`` are supported.

        .. note:: The data type of `data` does **not** select the data
                  type of the written file. Audio data will be
                  converted to the given `subtype`. Writing int values
                  to a float file will *not* scale the values to
                  [-1.0, 1.0). If you write the value ``np.array([42],
                  dtype='int32')``, to a ``subtype='FLOAT'`` file, the
                  file will then contain ``np.array([42.],
                  dtype='float32')``.
    subtype: str
        'PCM_24': 'Signed 24 bit PCM'
        'PCM_16': 'Signed 16 bit PCM'
        'PCM_S8': 'Signed 8 bit PCM'

    Return
    ------
    waveform (ndarray), sample rate (int)
    '''
    from soundfile import write
    return write(file_or_path, s, sr, subtype=subtype)


def __to_separated_indices(idx, min_distance=1, min_length=8):
    """ For example:
    min_distance = 1
    [1,2,3,4,
     8,9,10,
     15,16,17,18] => [(1,5), (8, 11), (15, 19)]

    Paramaters
    ----------
    min_distance: int
        pass
    min_length: int
        pass
    """
    if len(idx) == 0: return idx
    segments = [[]]
    n = 0
    for i, j in zip(idx, idx[1:]):
        segments[n].append(i)
        # new segments
        if j - i > min_distance:
            segments.append([])
            n += 1
    segments[-1].append(idx[-1])
    return [(s[0], s[-1] + 1) for s in segments
            if len(s) >= min_length]
