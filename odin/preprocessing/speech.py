# -*- coding: utf-8 -*-
# ===========================================================================
# The waveform and spectrogram preprocess utilities is adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
import six
import math
import copy
import warnings
from collections import OrderedDict, Mapping, defaultdict

import numpy as np
from scipy.signal import lfilter

from odin.fuel import Dataset, MmapData, MmapDict
from odin.utils import (is_number, cache_memory, is_string, as_tuple,
                        get_all_files, is_pickleable)
from odin.utils.decorators import functionable
from .base import Extractor
from .signal import (smooth, pre_emphasis, spectra, vad_energy,
                     pitch_track, resample, rastafilt, mvn, wmvn,
                     shifted_deltas)
# import all OpenSMILE extractor
from ._opensmile import *


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


def read_pcm(path_or_file, encode=None):
    dtype = np.int16
    sr = None
    if encode is not None:
        if 'ulaw' in encode.lower():
            dtype = np.int8
            sr = 8000
        elif 'vast' in encode.lower():
            dtype = np.int16
            sr = 44000
    s = np.memmap(path_or_file, dtype=dtype, mode='r')
    return s, sr


# ===========================================================================
# Audio feature extractor
# ===========================================================================
class RawDSReader(Extractor):
    """ This reader, read data directly from raw waveform processed dataset
    The input is `name` of the raw audio in the dataset

    The given Dataset must contains:
     - 'raw': 1-D MmapData stored raw waveform
     - 'sr': MmapDict, mapping file name -> sample rate (integer value)
     - 'indices': MmapDict, mapping file name -> start, end index in 'raw'
     - 'path' (optional)
     - 'duration' (optional)
    """

    def __init__(self, path_or_ds):
        super(RawDSReader, self).__init__()
        # ====== check argument ====== #
        if is_string(path_or_ds):
            ds = Dataset(path_or_ds, read_only=True)
        elif isinstance(path_or_ds, Dataset):
            ds = path_or_ds
        else:
            raise ValueError("`path_or_ds` must be string path to a folder or "
                             "a loaded Dataset.")
        # ====== check the dataset ====== #
        if 'raw' not in ds or not isinstance(ds['raw'], MmapData):
            raise ValueError("Dataset at path:'%s' must contain 'raw' MmapData, "
                             "which stored the raw waveform." % ds.path)
        if 'sr' not in ds or not isinstance(ds['sr'], MmapDict):
            raise ValueError("Dataset at path:'%s' must contain 'sr' MmapDict, "
                             "which stored the sample rate (integer)." % ds.path)
        if 'indices' not in ds or not isinstance(ds['indices'], MmapDict):
            raise ValueError("Dataset at path:'%s' must contain 'indices' MmapDict, "
                             "which stored the mapping: name->(start, end)" % ds.path)
        self.ds = ds

    def _transform(self, name):
        start, end = self.ds['indices'][name]
        raw = self.ds['raw'][start:end]
        ret = {'raw': raw.astype('float32'),
               'sr': int(self.ds['sr'][name]),
               'name': name} # must include name here
        if 'duration' in self.ds:
            ret['duration'] = float(self.ds['duration'][name])
        if 'path' in self.ds:
            ret['path'] = str(self.ds['path'][name])
        return ret


class AudioReader(Extractor):

    """ Return a dictionary of
    {
        'raw': loaded_signal,
        'duration': in second,
        'sr': sample rate,
        'path': path_to_loaded_file
    }

    Parameters
    ----------
    sr: int or None
        provided sr for missing sr audio (i.e. pcm files)

    Input
    -----
    path_or_array: string, tuple, list, mapping
        - string for path
        - tuple or list for (path-or-raw, sr)
        - mapping for provding additional information include:
        sr, encode (ulaw, vast), 'raw' or 'path'

    Note
    ----
    Dithering introduces white noise when you save the raw array into
    audio file.
    For now only support one channel
    """

    def __init__(self, sr=None, sr_new=None, best_resample=True,
                 remove_dc_n_dither=True, preemphasis=0.97):
        super(AudioReader, self).__init__()
        self.sr = sr
        self.sr_new = sr_new
        self.best_resample = best_resample
        self.remove_dc_n_dither = bool(remove_dc_n_dither)
        self.preemphasis = preemphasis

    def _transform(self, path_or_array):
        sr = None # by default, we don't know sample rate
        path = None
        duration = None
        encode = None
        channel = 0
        # ====== check path_or_array ====== #
        if isinstance(path_or_array, (tuple, list)):
            if len(path_or_array) != 2:
                raise ValueError("`path_or_array` can be a tuple or list of "
                    "length 2, which contains: (string_path, sr) or "
                    "(sr, string_path) or (raw_array, sr) or (sr, raw_array).")
            if is_number(path_or_array[0]):
                sr, path_or_array = path_or_array
            else:
                path_or_array, sr = path_or_array
        elif isinstance(path_or_array, Mapping):
            if 'sr' in path_or_array:
                sr = path_or_array['sr']
            if 'encode' in path_or_array:
                encode = str(path_or_array['encode'])
            # get raw or path out of the Dictionary
            if 'raw' in path_or_array:
                path_or_array = path_or_array['raw']
            elif 'path' in path_or_array:
                path_or_array = path_or_array['path']
            else:
                raise ValueError('`path_or_array` can be a dictionary, contains '
                    'following key: sr, raw, path. One of the key `raw` for '
                    'raw array signal, or `path` for path to audio file must '
                    'be specified.')
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
                s, sr = read_pcm(path_or_file=f, encode=encode)
            else:
                import soundfile
                try:
                    s, sr = soundfile.read(f)
                except Exception as e:
                    if '.sph' in f.name.lower():
                        f.seek(0)
                        s, sr = read_pcm(path_or_file=f, encode=encode)
                    else:
                        raise e
            # close file
            f.close()
        # ====== provided np.ndarray for normalization ====== #
        else:
            s = path_or_array
        # ====== check channel ====== #
        if s.ndim == 1:
            pass
        elif s.ndim == 2:
            if s.shape[0] == 2:
                s = s[channel, :]
            elif s.shape[1] == 2:
                s = s[:, channel]
        else:
            raise ValueError("No support for %d-D signal from file: %s" %
                (s.ndim, str(path)))
        # ====== valiate sample rate ====== #
        if sr is None and self.sr is not None:
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
            s = s - np.mean(s, 0)
        # ====== pre-emphasis ====== #
        if self.preemphasis is not None and 0. < self.preemphasis < 1.:
            s = pre_emphasis(s, coeff=float(self.preemphasis))
        # ====== get duration if possible ====== #
        if sr is not None:
            duration = max(s.shape) / sr
        return {'raw': s, 'sr': sr,
                'duration': duration, # in second
                'path': os.path.abspath(path) if path is not None else None}


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
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy (or magnitude), 2 for power, etc.
    log: bool
        if True, convert all power spectrogram to DB
    padding : bool
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
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
        self.power = float(power)
        self.log = bool(log)
        # ====== others ====== #
        self.padding = bool(padding)

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
        return feat


class CQTExtractor(Extractor):
    """ CQTExtractor

    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
    """

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
        return feat


class PitchExtractor(Extractor):
    """
    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
    """

    def __init__(self, frame_length, step_length=None,
                 threshold=1., fmin=20, fmax=260,
                 algo='rapt', f0=False):
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
    """
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
    """

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
        if features.ndim > 1:
            features = features.sum(axis=-1)
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


class RASTAfilter(Extractor):
    """ RASTAfilter

    Specialized "Relative Spectral Transform" applying for MFCCs
    and PLP

     RASTA is a separate technique that applies a band-pass filter
     to the energy in each frequency subband in order to smooth over
     short-term noise variations and to remove any constant offset
     resulting from static spectral coloration in the speech channel
     e.g. from a telephone line

    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).

    References
    ----------
    [PLP and RASTA](http://www.ee.columbia.edu/ln/rosa/matlab/rastamat/)

    """

    def __init__(self, rasta=True, sdc=1):
        super(RASTAfilter, self).__init__()
        self.rasta = bool(rasta)
        self.sdc = int(sdc)

    def _transform(self, feat):
        if 'mfcc' in feat:
            mfcc = feat['mfcc']
            if self.rasta:
                mfcc = rastafilt(mfcc)
            if self.sdc >= 1:
                mfcc = np.hstack([
                    mfcc,
                    shifted_deltas(mfcc, N=7, d=self.sdc, P=3, k=7)
                ])
            feat['mfcc'] = mfcc
        return feat


class AcousticNorm(Extractor):
    """
    Parameters
    ----------
    sdc: int
        Lag size for delta feature computation for
        "Shifted Delta Coefficients", if `sdc` > 0, the
        shifted delta features will be append to MFCCs

    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
    """

    def __init__(self, mean_var_norm=True, window_mean_var_norm=True,
                 win_length=301, var_norm=True,
                 feat_type=('mspec', 'spec', 'mfcc',
                            'qspec', 'qmfcc', 'qmspec')):
        super(AcousticNorm, self).__init__()
        self.mean_var_norm = bool(mean_var_norm)
        self.window_mean_var_norm = bool(window_mean_var_norm)
        self.var_norm = bool(var_norm)
        # ====== check win_length ====== #
        win_length = int(win_length)
        if win_length % 2 == 0:
            raise ValueError("win_length must be odd number")
        if win_length < 3:
            raise ValueError("win_length must >= 3")
        self.win_length = win_length
        # ====== check which features will be normalized ====== #
        self.feat_type = as_tuple(feat_type, t=str)

    def _transform(self, feat):
        feat_normalized = {}
        # all `features` is [t, f] shape
        for name, features in feat.iteritems():
            if name in self.feat_type:
                if self.mean_var_norm:
                    features = mvn(features, varnorm=self.var_norm)
                if self.window_mean_var_norm:
                    features = wmvn(features, w=self.win_length,
                                    varnorm=False)
                # transpose back to [t, f]
            feat_normalized[name] = features
        return feat_normalized


class Read3ColSAD(Extractor):
    """ Read3ColSAD simple helper for applying 3 col
    SAD (name, start-in-second, end-in-second) to extracted acoustic features

    Parameters
    ----------
    path: str
        path to folder contain all SAD files
    name_converter: callable
        convert the 'path' element in features dictionary (provided to
        `transform`) to name for search in parsed SAD dictionary.
    file_regex: str
        regular expression for filtering the files name
    keep_unvoiced: bool
        if True, keep all the the features of utterances even though
        cannot file SAD for it. Otherwise, return None.
    feat_type: list of string
        all features type will be applied using given SAD

    Return
    ------
     - add 'sad': [(start, end), ...] to the features dictionary
     - each features specified in `feat_type` is cutted into segments of SAD,
     then concatenated into single matrix (hence, you can use features['sad']
     to get back the separated segments).

    Note
    ----
    The following order is recommended for extracting spectra:
    + AudioReader:
        - Loading raw audio
        - remove DC offeset and dithering
        - preemphasis
    + SpectraExtractor (or CQTExtractor):
        - Extracting the Spectra
    + VADextractor:
        - Extracting SAD (optional)
    + Rastafilt:
        - Rastafilt (optional for MFCC)
    + DeltaExtractor
        - Calculate Deltas (and shifted delta for MFCCs).
    + Read3ColSAD:
        - Applying SAD labels
    + AcousticNorm
        - Applying CMVN and WCMVN (This is important so the SAD frames
        are not affected by the nosie).
    """

    def __init__(self, path_or_map, step_length,
                 name_converter=None, ref_key='path', file_regex='.*',
                 keep_unvoiced=False, feat_type=('spec', 'mspec', 'mfcc',
                                                 'qspec', 'qmspec', 'qmfcc',
                                                 'pitch', 'f0', 'vad', 'energy')):
        super(Read3ColSAD, self).__init__()
        self.keep_unvoiced = bool(keep_unvoiced)
        self.feat_type = as_tuple(feat_type, t=str)
        self.step_length = float(step_length)
        # ====== file regex ====== #
        file_regex = re.compile(str(file_regex))
        # ====== name_converter ====== #
        if name_converter is not None:
            if not callable(name_converter):
                raise ValueError("`name_converter` must be callable.")
            name_converter = functionable(func=name_converter)
        self.name_converter = name_converter
        self.ref_key = as_tuple(ref_key, t=str)
        # ====== parse all file ====== #
        # read the SAD from file
        if is_string(path_or_map):
            sad = defaultdict(list)
            for f in get_all_files(path_or_map):
                if file_regex.search(os.path.basename(f)) is not None:
                    with open(f, 'r') as f:
                        for line in f:
                            name, start, end = line.strip().split(' ')
                            sad[name].append((float(start), float(end)))
        elif isinstance(path_or_map, Mapping):
            sad = path_or_map
        else:
            raise ValueError("`path` must be path to folder, or dictionary.")
        self.sad = sad

    def _transform(self, feats):
        # ====== get ref name ====== #
        name = None
        for k in self.ref_key:
            name = feats.get(k, None)
            if is_string(name):
                break
        # ====== start ====== #
        if is_string(name):
            if self.name_converter is not None:
                name = self.name_converter(name)
            # ====== convert step_length ====== #
            step_length = self.step_length
            if step_length >= 1: # step_length is number of frames
                step_length = step_length / feats['sr']
                # now step_length is in second
            # ====== found SAD ====== #
            if name in self.sad:
                sad = self.sad[name]
                if len(sad) > 0:
                    # SAD transformed features
                    feats_sad = defaultdict(list)
                    sad_indices = []
                    index = 0
                    for start_sec, end_sec in sad:
                        start_idx = int(start_sec / step_length)
                        end_idx = int(end_sec / step_length)
                        # ignore zero SAD
                        if end_idx - start_idx == 0:
                            continue
                        # cut SAD out from feats
                        for _, ftype in enumerate(self.feat_type):
                            X = feats[ftype]
                            X = X[start_idx:min(end_idx, X.shape[0])]
                            if X.shape[0] != 0:
                                feats_sad[ftype].append(X)
                                # store (start-frame_index, end-frame-index)
                                # of all SAD here
                                if _ == 0:
                                    sad_indices.append((index, index + X.shape[0]))
                                    index += X.shape[0]
                    # concatenate sad segments
                    feats_sad = {ftype: np.concatenate(y, axis=0)
                                 for ftype, y in feats_sad.iteritems()}
                    feats_sad['sad'] = sad_indices
                    return feats_sad
        # ====== return unvoiced or not ====== #
        if not self.keep_unvoiced:
            return None
        return feats


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
