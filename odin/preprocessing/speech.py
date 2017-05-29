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
from collections import OrderedDict

import numpy as np

from odin.utils import is_number, cache_memory, is_string
from .signal import (pad_center, get_window, segment_axis, stft, istft,
                     compute_delta, smooth, pre_emphasis, spectra,
                     vad_energy, power2db, pitch_track)

# ===========================================================================
# Predefined variables of speech datasets
# ===========================================================================
# ==================== Predefined datasets information ==================== #
nist15_cluster_lang = OrderedDict([
    ['ara', ['ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb']],
    ['zho', ['zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu']],
    ['eng', ['eng-gbr', 'eng-usg', 'eng-sas']],
    ['fre', ['fre-waf', 'fre-hat']],
    ['qsl', ['qsl-pol', 'qsl-rus']],
    ['spa', ['spa-car', 'spa-eur', 'spa-lac', 'por-brz']]
])
nist15_lang_list = np.asarray([
    # Egyptian, Iraqi, Levantine, Maghrebi, Modern Standard
    'ara-arz', 'ara-acm', 'ara-apc', 'ara-ary', 'ara-arb',
    # Cantonese, Mandarin, Min Dong, Wu
    'zho-yue', 'zho-cmn', 'zho-cdo', 'zho-wuu',
    # British, American, South Asian (Indian)
    'eng-gbr', 'eng-usg', 'eng-sas',
    # West african, Haitian
    'fre-waf', 'fre-hat',
    # Polish, Russian
    'qsl-pol', 'qsl-rus',
    # Caribbean, European, Latin American, Brazilian
    'spa-car', 'spa-eur', 'spa-lac', 'por-brz'
])


def nist15_label(label):
    '''
    Return
    ------
    lang_id : int
        idx in the list of 20 language, None if not found
    cluster_id : int
        idx in the list of 6 clusters, None if not found
    within_cluster_id : int
        idx in the list of each clusters, None if not found
    '''
    label = label.replace('spa-brz', 'por-brz')
    rval = [None, None, None]
    # lang_id
    if label not in nist15_lang_list:
        raise ValueError('Cannot found label:%s' % label)
    rval[0] = np.argmax(label == nist15_lang_list)
    # cluster_id
    for c, x in enumerate(nist15_cluster_lang.iteritems()):
        j = x[1]
        if label in j:
            rval[1] = c
            rval[2] = j.index(label)
    return rval

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


def timit_phonemes(phn, map39=False, blank=False):
    ''' Included blank '''
    if type(phn) not in (list, tuple, np.ndarray):
        phn = [phn]
    if map39:
        timit = timit_39
        timit_map = timit_map
        l = 39
    else:
        timit = timit_61
        timit_map = {}
        l = 61
    # ====== return phonemes ====== #
    rphn = []
    for p in phn:
        if p not in timit_map and p not in timit:
            if blank: rphn.append(l)
        else:
            rphn.append(timit.index(timit_map[p]) if p in timit_map else timit.index(p))
    return rphn


# ===========================================================================
# Audio helper
# ===========================================================================
def read(f, pcm=False, remove_dc_offset=True):
    '''
    Return
    ------
        waveform (ndarray: [samples;channel]), sample rate (int)
    '''
    if pcm or (isinstance(f, str) and
               any(i in f for i in ['pcm', 'PCM'])):
        s, fs = (np.memmap(f, dtype=np.int16, mode='r'), None)
    else:
        from soundfile import read
        s, fs = read(f)
    s = s.astype(np.float32)
    if remove_dc_offset:
        s -= np.mean(s, 0)
    return s, fs


def est_audio_length(fpath, sr=8000, bitdepth=16):
    """ Estimate audio length in second """
    if not os.path.exists(fpath):
        raise Exception('File at path:%s does not exist' % fpath)
    return os.path.getsize(fpath) / (bitdepth / 8) / sr


def resample(s, fs_orig, fs_new, axis=0, best_algorithm=True):
    '''
    '''
    fs_orig = int(fs_orig)
    fs_new = int(fs_new)
    if fs_orig != fs_new:
        import resampy
        s = resampy.resample(s, sr_orig=fs_orig, sr_new=fs_new, axis=axis,
                             filter='kaiser_best' if best_algorithm else 'kaiser_fast')
    return s


def save(file_or_path, s, fs, subtype=None):
    '''
    Return
    ------
    waveform (ndarray), sample rate (int)
    '''
    from soundfile import write
    return write(file_or_path, s, fs, subtype=subtype)


# ===========================================================================
# Spectrogram manipulation
# ===========================================================================
@cache_memory
def __num_two_factors(x):
    """return number of times x is divideable for 2"""
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2
    return num_twos


@cache_memory
def __max_fft_bins(sr, n_fft, fmax):
    return [i + 1 for i, j in enumerate(np.linspace(0, float(sr) / 2, int(1 + n_fft // 2),
                                        endpoint=True)) if j >= fmax][0]


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
    return [(s[0], s[-1] + 1) for s in segments if len(s) >= min_length]


def speech_features(s, sr=None,
                    win=0.02, hop=0.01, window='hann',
                    nb_melfilters=None, nb_ceps=None,
                    get_spec=True, get_qspec=False, get_phase=False,
                    get_pitch=False, get_f0=False,
                    get_vad=True, get_energy=False, get_delta=False,
                    fmin=64, fmax=None, sr_new=None,
                    pitch_threshold=0.3, pitch_fmax=260,
                    vad_smooth=3, vad_minlen=0.1,
                    cqt_bins=96, preemphasis=None,
                    power=2, log=True, backend='odin'):
    """ Automatically extract multiple acoustic representation of
    speech features

    Parameters
    ----------
    s: np.ndarray
        raw signal
    sr: int
        sample rate
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
        if True, include the indicators of voice activities detection
        if int, `get_vad` is the number of Gaussian mixture components for VAD
    get_energy: bool
        if True, include the log energy of each frames
    get_delta: bool or int
        if True and > 0, for each features append the delta with given order-th
        (e.g. delta=2 will include: delta1 and delta2)
    fmin : float > 0 [scalar]
        lower frequency cutoff (if you work with other voice than human speech,
        set `fmin` to 20Hz).
    fmax : float > 0 [scalar]
        upper frequency cutoff.
    sr_new: int or None
        new sample rate
    preemphasis: float `(0, 1)`, or None
        pre-emphasis coefficience
    pitch_threshold: float in `(0, 1)`
        Voice/unvoiced threshold for pitch tracking. (Default is 0.3)
    pitch_fmax: float
        maximum frequency of pitch. (Default is 260 Hz)
    vad_smooth: int, bool
        window length to smooth the vad indices.
        If True default window length is 3.
    vad_minlen: float (in second)
        the minimum length of audio segments that can be considered
        speech.
    cqt_bins : int > 0
        Number of frequency bins for constant Q-transform, starting at `fmin`
    center : bool
        If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        If `False`, then `D[:, t]` begins at `y[t * hop_length]`
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
    log: bool
        if True, convert all power spectrogram to DB
    backend: 'odin', 'sptk'
        support backend for calculating the spectra

    Return
    ------
    y = {
        'mfcc': np.ndarray (txd) - float32,
        'energy': np.ndarray (txd) - float32,
        'spec': np.ndarray (txd) - float32,
        'mspec': np.ndarray (txd) - float32,

        'qspec': np.ndarray (txd) - float32,
        'qmspec': np.ndarray (txd) - float32,
        'qmfcc': np.ndarray (txd) - float32,

        'phase': np.ndarray (txd) - float32,
        'pitch': np.ndarray (txd) - float32,
        'vad': np.ndarray (t,) - uint8
        'vadids': np.ndarray [(start, end), ...] - uint8
    }
    (txd): time x features
    """
    from odin.fuel import Data
    # file path
    if is_string(s):
        path = s
        s, sr_ = read(path)
        if sr_ is None and sr is None:
            raise ValueError("Cannot get sample rate from file: %s" % path)
        if sr_ is not None:
            sr = sr_
    # Data object
    elif isinstance(s, Data):
        s = s[:]
    # check valid 1-D numpy array
    if np.prod(s.shape) == np.max(s.shape):
        s = s.ravel()
    else:
        raise Exception('Speech Feature Extraction only accept 1-D signal')
    s = s.astype('float32')
    # resample if necessary
    if sr_new is not None and int(sr_new) != int(sr):
        s = resample(s, sr, sr_new, axis=0, best_algorithm=False)
        sr = sr_new
    # ====== check other info ====== #
    if fmax is None:
        fmax = sr // 2
    if fmin is None or fmin < 0 or fmin >= fmax:
        fmin = 0
    if pitch_fmax is None:
        pitch_fmax = fmax
    win_length = int(win * sr)
    # n_fft must be 2^x
    n_fft = 2 ** int(np.ceil(np.log2(win_length)))
    hop_length = int(hop * sr) # hop_length must be 2^x
    # nb_ceps += 1 # increase one so we can ignore the first MFCC
    # ====== 5: extract pitch features ====== #
    pitch_freq = None
    if get_pitch:
        pitch_freq = pitch_track(s, sr, hop_length, fmin=fmin,
            fmax=pitch_fmax, threshold=pitch_threshold, otype='pitch',
            algorithm='swipe').reshape(-1, 1)
    f0_freq = None
    if get_f0:
        f0_freq = pitch_track(s, sr, hop_length, fmin=fmin,
            fmax=pitch_fmax, threshold=pitch_threshold, otype='f0',
            algorithm='swipe').reshape(-1, 1)
    # ====== 0: extract Constant Q-transform ====== #
    q_melspectrogram = None
    q_mfcc = None
    qspec = None
    qphase = None
    # qspec requires librosa
    if get_qspec:
        from librosa.core import constantq
        # auto adjust bins_per_octave to get maximum range of frequency
        bins_per_octave = np.ceil(float(cqt_bins - 1) / np.log2(sr / 2. / fmin)) + 1
        # adjust the bins_per_octave to make acceptable hop_length
        # i.e. 2_factors(hop_length) < [ceil(cqt_bins / bins_per_octave) - 1]
        if __num_two_factors(hop_length) < np.ceil(cqt_bins / bins_per_octave) - 1:
            bins_per_octave = np.ceil(cqt_bins / (__num_two_factors(hop_length) + 1))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            qtrans = constantq.cqt(s, sr=sr, hop_length=hop_length, n_bins=cqt_bins,
                                   bins_per_octave=int(bins_per_octave),
                                   fmin=fmin, tuning=0.0, real=False, norm=1,
                                   filter_scale=1., sparsity=0.01).astype('complex64')
        # get log power Q-spectrogram
        Q = spectra(S=qtrans.T, sr=sr,
                    nb_melfilters=nb_melfilters, nb_ceps=nb_ceps,
                    fmin=fmin, fmax=fmax, power=2.0, log=True,
                    backend='odin')
        qspec = Q['spec']
        q_melspectrogram = Q['mspec'] if 'mspec' in Q else None
        q_mfcc = Q['mfcc'] if 'mfcc' in Q else None
        qphase = Q['phase']
    # ====== 1: extract STFT and Spectrogram ====== #
    # no padding for center
    feat = spectra(sr=sr, y=s, n_fft=n_fft, hop_length=hop_length,
                   window=window, nb_melfilters=nb_melfilters,
                   nb_ceps=nb_ceps, fmin=fmin, fmax=fmax,
                   power=power, log=log,
                   preemphasis=preemphasis, backend=backend)
    # ====== 4: extract spectrogram ====== #
    spec = feat['spec']
    mspec = feat['mspec'] if 'mspec' in feat else None
    mfcc = feat['mfcc'] if 'mfcc' in feat else None
    log_energy = feat['energy']
    # ====== 3: extract VAD ====== #
    vad = None
    vad_ids = None
    if get_vad:
        distribNb, nbTrainIt = 2, 24
        if is_number(get_vad) and get_vad >= 2:
            distribNb = int(get_vad)
        vad, vad_threshold = vad_energy(log_energy.ravel(), distrib_nb=distribNb,
                                        nb_train_it=nbTrainIt)
        vad = vad.astype('uint8')
        if vad_smooth:
            vad_smooth = 3 if int(vad_smooth) == 1 else vad_smooth
            # at least 2 voice frames
            vad = smooth(vad, win=vad_smooth, window='flat') >= 2. / vad_smooth
            vad = vad.astype('uint8')
        vad_ids = np.array(__to_separated_indices(vad.nonzero()[0],
                                                  min_distance=1,
                                                  min_length=int(vad_minlen / hop)),
                           dtype='int32')
    # ====== 7: compute delta ====== #
    if get_delta and get_delta > 0:
        get_delta = int(get_delta)
        if log_energy is not None:
            log_energy = log_energy[:, None]
            log_energy = np.concatenate(
                [log_energy] + compute_delta(log_energy, order=get_delta),
                axis=1)
        # STFT
        if mspec is not None:
            mspec = np.concatenate(
                [mspec] + compute_delta(mspec, order=get_delta),
                axis=1)
        if mfcc is not None:
            mfcc = np.concatenate(
                [mfcc] + compute_delta(mfcc, order=get_delta),
                axis=1)
        # Q-transform
        if q_melspectrogram is not None:
            q_melspectrogram = np.concatenate(
                [q_melspectrogram] + compute_delta(q_melspectrogram, order=get_delta),
                axis=1)
        if q_mfcc is not None:
            q_mfcc = np.concatenate(
                [q_mfcc] + compute_delta(q_mfcc, order=get_delta),
                axis=1)
    # ====== 8: make sure CQT give the same length with STFT ====== #
    if get_qspec and qspec.shape[1] > spec.shape[1]:
        n = qspec.shape[1] - spec.shape[1]
        qspec = qspec[:, n // 2:-int(np.ceil(n / 2))]
        if qphase is not None:
            qphase = qphase[:, n // 2:-int(np.ceil(n / 2))]
        if q_melspectrogram is not None:
            q_melspectrogram = q_melspectrogram[:, n // 2:-int(np.ceil(n / 2))]
        if q_mfcc is not None:
            q_mfcc = q_mfcc[:, n // 2:-int(np.ceil(n / 2))]
    return OrderedDict([
        ('spec', spec if get_spec else None),
        ('mspec', None if mspec is None else mspec),
        ('mfcc', None if mfcc is None else mfcc),
        ('energy', log_energy if get_energy else None),

        ('qspec', qspec if get_qspec else None),
        ('qmspec', None if q_melspectrogram is None else q_melspectrogram),
        ('qmfcc', None if q_mfcc is None else q_mfcc),

        ('phase', feat['phase'] if get_phase else None),
        ('qphase', qphase if get_phase and get_qspec else None),

        ('f0', f0_freq if get_f0 else None),
        ('pitch', pitch_freq if get_pitch else None),

        ('vad', vad if get_vad else None),
        ('vadids', vad_ids if get_vad else None),
    ])
