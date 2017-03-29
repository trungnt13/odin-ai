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
import scipy.fftpack as fft
import scipy.signal

from odin.utils import pad_center, framing, is_number, cache_memory

# Constrain STFT block sizes to 512 KB
MAX_MEM_BLOCK = 2**8 * 2**11
SMALL_FLOAT = 1e-20

VAD_MODE_STRICT = 1.2
VAD_MODE_STANDARD = 2.
VAD_MODE_SENSITIVE = 2.4
__current_vad_mode = VAD_MODE_STANDARD # alpha for vad energy


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
    'spa-car', 'spa-eur', 'spa-lac', 'por-brz'])


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


def pre_emphasis(s, coeff=0.97):
    """Pre-emphasis of an audio signal.
    Parameters
    ----------
    s: np.ndarray
        the input vector of signal to pre emphasize
    coeff: float (0, 1)
        coefficience that defines the pre-emphasis filter.
    """
    if s.ndim == 1:
        return np.append(s[0], s[1:] - coeff * s[:-1])
    else:
        return s - np.c_[s[:, :1], s[:, :-1]] * coeff


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


def save(f, s, fs, subtype=None):
    '''
    Return
    ------
    waveform (ndarray), sample rate (int)
    '''
    from soundfile import write
    return write(f, s, fs, subtype=subtype)


# ===========================================================================
# Spectrogram manipulation
# ===========================================================================
@cache_memory
def __get_window(window, Nx, fftbins=True):
    ''' Cached version of scipy.signal.get_window '''
    if six.callable(window):
        return window(Nx)
    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        return scipy.signal.get_window(window, Nx, fftbins=fftbins)
    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)
        raise ValueError('Window size mismatch: '
                         '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ValueError('Invalid window specification: {}'.format(window))


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
def __cqt_response_override(win_length):
    '''Compute the filter response with a target STFT hop.'''
    # Compute the STFT matrix and filter response energy
    return lambda y, n_fft, hop_length, fft_basis: fft_basis.dot(
        stft(np.pad(y, int(win_length // 2), mode='reflect'), n_fft=n_fft,
            win_length=win_length, hop_length=hop_length, window=np.ones))


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


def smooth(x, win=11, window='hanning'):
    """
    Paramaters
    ----------
    x: 1-D vector
        input signal.
    win: int
        length of window for smoothing, the longer the window, the more details
        are reduced for smoothing.
    window: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        window function, 'flat' for moving average.

    Return
    ------
    y: smoothed vector

    Example
    -------
    """
    if win < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.concatenate([2 * x[0] - x[win - 1::-1],
                        x,
                        2 * x[-1] - x[-1:-win:-1]], axis=0)
    # moving average
    if window == 'flat':
        w = np.ones(win, 'd')
    # windowing
    else:
        w = eval('np.' + window + '(win)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[win:-win + 1]


def set_vad_mode(mode):
    """
    Paramters
    ---------
    mode: float
        a number from 1.0 to 2.4, the higher the number, the more
        sensitive it is to any high-energy segments.
    """
    if is_number(mode):
        global __current_vad_mode
        mode = min(max(mode, 1.), 2.4)
        __current_vad_mode = float(mode)


def vad_energy(log_energy, distrib_nb=2, nb_train_it=24):
    from sklearn.mixture import GaussianMixture
    # center and normalize the energy
    log_energy = (log_energy - np.mean(log_energy)) / np.std(log_energy)
    if log_energy.ndim == 1:
        log_energy = log_energy[:, np.newaxis]
    # create mixture model: diag, spherical
    world = GaussianMixture(
        n_components=distrib_nb, covariance_type='diag',
        init_params='kmeans', max_iter=nb_train_it,
        weights_init=np.ones(distrib_nb) / distrib_nb,
        means_init=(-2 + 4.0 * np.arange(distrib_nb) / (distrib_nb - 1))[:, np.newaxis],
        precisions_init=np.ones((distrib_nb, 1)),
    )
    try:
        world.fit(log_energy)
    except (ValueError, IndexError): # index error because of float32 cumsum
        if distrib_nb - 1 >= 2:
            return vad_energy(log_energy,
                distrib_nb=distrib_nb - 1, nb_train_it=nb_train_it)
        return np.zeros(shape=(log_energy.shape[0],)), 0
    # Compute threshold
    threshold = world.means_.max() - \
        __current_vad_mode * np.sqrt(1.0 / world.precisions_[world.means_.argmax(), 0])
    # Apply frame selection with the current threshold
    label = log_energy.ravel() > threshold
    return label, threshold


def speech_enhancement(X, Gain, NN=2):
    """This program is only to process the single file seperated by the silence
    section if the silence section is detected, then a counter to number of
    buffer is set and pre-processing is required.

    Usage: SpeechENhance(wavefilename, Gain, Noise_floor)

    :param X: input audio signal
    :param Gain: default value is 0.9, suggestion range 0.6 to 1.4,
            higher value means more subtraction or noise redcution
    :param NN:

    :return: a 1-dimensional array of boolean that
        is True for high energy frames.

    Note
    ----
    I move this function here, so we don't have to import `sidekit`.
    You can check original version from `sidekit.frontend.vad`.
    Copyright 2014 Sun Han Wu and Anthony Larcher
    """
    if X.shape[0] < 512:  # creer une exception
        return X

    num1 = 40  # dsiable buffer number
    Alpha = 0.75  # original value is 0.9
    FrameSize = 32 * 2  # 256*2
    FrameShift = int(FrameSize / NN)  # FrameSize/2=128
    nfft = FrameSize  # = FrameSize
    Fmax = int(np.floor(nfft / 2) + 1)  # 128+1 = 129
    # arising hamming windows
    Hamm = 1.08 * (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FrameSize) / (FrameSize - 1)))
    y0 = np.zeros(FrameSize - FrameShift)  # 128 zeros

    Eabsn = np.zeros(Fmax)
    Eta1 = Eabsn

    ###################################################################
    # initial parameter for noise min
    mb = np.ones((1 + FrameSize // 2, 4)) * FrameSize / 2  # 129x4  set four buffer * FrameSize/2
    im = 0
    Beta1 = 0.9024  # seems that small value is better;
    pxn = np.zeros(1 + FrameSize // 2)  # 1+FrameSize/2=129 zeros vector

    ###################################################################
    old_absx = Eabsn
    x = np.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[
        np.arange(np.min((int(FrameShift), X.shape[0])))]  # fread(ifp, FrameSize, 'short')% read  FrameSize samples

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    ###################################################################
    # add the pre-noise estimates
    for i in range(200):
        Frame += 1
        fftn = fft.fft(x * Hamm)  # get its spectrum
        absn = np.abs(fftn[0:Fmax])  # get its amplitude

        # add the following part from noise estimation algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # Beta=0.9231 recursive pxn
        im = (im + 1) % 40  # noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = np.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn
            #  0-2  vector shifted to 1 to 3

        pn = 2 * np.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation
        # over_sub_noise= oversubtraction factor

        # end of noise detection algotihm
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]
        index1 = np.arange(FrameShift * Frame, np.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        if In_data.shape[0] < FrameShift:  # to check file is out
            EOF = 1
            break
        else:
            x[FrameSize - FrameShift:FrameSize] = In_data  # shift new 128 to position 129 to FrameSize location
            # end of for loop for noise estimation

    # end of prenoise estimation ************************
    x = np.zeros(FrameSize)
    x[FrameSize - FrameShift:FrameSize] = X[np.arange(np.min((int(FrameShift), X.shape[0])))]

    if x.shape[0] < FrameSize:
        EOF = 1
        return X

    EOF = 0
    Frame = 0

    X1 = np.zeros(X.shape)
    Frame = 0

    while EOF == 0:
        Frame += 1
        xwin = x * Hamm

        fftx = fft.fft(xwin, nfft)  # FrameSize FFT
        absx = np.abs(fftx[0:Fmax])  # Fmax=129,get amplitude of x
        argx = fftx[:Fmax] / (absx + np.spacing(1))  # normalize x spectrum phase

        absn = absx

        # add the following part from rainer algorithm
        pxn = Beta1 * pxn + (1 - Beta1) * absn  # s Beta=0.9231   recursive pxn

        im = int((im + 1) % (num1 * NN / 2))  # original =40 noise_memory=47;  im=0 (init) for noise level estimation

        if im:
            mb[:, 0] = np.minimum(mb[:, 0], pxn)  # 129 by 4 im<>0  update the first vector from PXN
        else:
            mb[:, 1:] = mb[:, :3]  # im==0 every 47 time shift pxn to first vector of mb
            mb[:, 0] = pxn

        pn = 2 * np.min(mb, axis=1)  # pn = 129x1po(9)=1.5 noise level estimate compensation

        Eabsn = pn
        Gaina = Gain

        temp1 = Eabsn * Gaina

        Eta1 = Alpha * old_absx + (1 - Alpha) * np.maximum(absx - temp1, 0)
        new_absx = (absx * Eta1) / (Eta1 + temp1)  # wiener filter
        old_absx = new_absx

        ffty = new_absx * argx  # multiply amplitude with its normalized spectrum

        y = np.real(np.fft.fftpack.ifft(np.concatenate((ffty, np.conj(ffty[np.arange(Fmax - 2, 0, -1)])))))

        y[:FrameSize - FrameShift] = y[:FrameSize - FrameShift] + y0
        y0 = y[FrameShift:FrameSize]  # keep 129 to FrameSize point samples
        x[:FrameSize - FrameShift] = x[FrameShift:FrameSize]

        index1 = np.arange(FrameShift * Frame, np.min((FrameShift * (Frame + 1), X.shape[0])))
        In_data = X[index1]  # fread(ifp, FrameShift, 'short');

        z = 2 / NN * y[:FrameShift]  # left channel is the original signal
        z /= 1.15
        z = np.minimum(z, 32767)
        z = np.maximum(z, -32768)
        index0 = np.arange(FrameShift * (Frame - 1), FrameShift * Frame)
        if not all(index0 < X1.shape[0]):
            idx = 0
            while (index0[idx] < X1.shape[0]) & (idx < index0.shape[0]):
                X1[index0[idx]] = z[idx]
                idx += 1
        else:
            X1[index0] = z

        if In_data.shape[0] == 0:
            EOF = 1
        else:
            x[np.arange(FrameSize - FrameShift, FrameSize + In_data.shape[0] - FrameShift)] = In_data

    X1 = X1[X1.shape[0] - X.shape[0]:]
    return X1


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann'):
    """ Modified version from `librosa`, since the `librosa` version create
    different windows from `kaldi` and `sidekit`, we make this version
    compatible to them.

    Short-time Fourier transform (STFT)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix

        a complex-valued matrix D such that:
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    See Also
    --------
    istft : Inverse STFT

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = __get_window(window, win_length, fftbins=True)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Window the time series.
    y_frames = framing(y, frame_length=win_length, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=np.complex64,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                     stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window * y_frames[:, bl_s:bl_t],
                                            n=n_fft,
                                            axis=0)[:stft_matrix.shape[0]].conj()
    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_.

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    See Also
    --------
    stft : Short-time Fourier Transform

    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        ifft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_sum = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
        ifft_window_sum[sample:(sample + n_fft)] += ifft_window_square

    # Normalize by sum of squared window
    approx_nonzero_indices = ifft_window_sum > SMALL_FLOAT
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y


def compute_delta(data, width=9, order=1, axis=-1, trim=True):
    r'''Compute delta features: local estimate of the derivative
    of the input data along the selected axis.

    Parameters
    ----------
    data      : np.ndarray
        the input data matrix (eg, spectrogram), shape=(d, t)
    width     : int >= 3, odd [scalar]
        Number of frames over which to compute the delta feature
    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.
    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).
    trim      : bool
        set to `True` to trim the output matrix to the original size.

    Returns
    -------
    delta_data   : list(np.ndarray) [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.
        return list of deltas

    Examples
    --------
    Compute MFCC deltas, delta-deltas
    >>> mfcc = mfcc(y=y, sr=sr)
    >>> mfcc_delta1, mfcc_delta2 = compute_delta(mfcc, 2)
    '''

    data = np.atleast_1d(data)

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        raise ValueError('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = np.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= np.sum(np.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = np.pad(data, padding, mode='edge')

    all_deltas = []
    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)
        all_deltas.append(delta_x)

    # Cut back to the original shape of the input data
    if trim:
        _ = []
        for delta_x in all_deltas:
            idx = [slice(None)] * delta_x.ndim
            idx[axis] = slice(- half_length - data.shape[axis], - half_length)
            delta_x = delta_x[idx]
            _.append(delta_x.astype('float32'))
        all_deltas = _

    return all_deltas


def speech_features(s, sr, win=0.02, shift=0.01, nb_melfilters=24, nb_ceps=12,
                    get_spec=True, get_mspec=False, get_mfcc=False,
                    get_qspec=False, get_phase=False, get_pitch=False,
                    get_vad=True, get_energy=False, get_delta=False,
                    fmin=64, fmax=None, sr_new=None, preemphasis=0.97,
                    pitch_threshold=0.8, pitch_fmax=1200,
                    vad_smooth=3, vad_minlen=0.1,
                    cqt_bins=96, center=True):
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
    preemphasis: float `(0, 1)`
        pre-emphasis coefficience
    pitch_threshold: float in `(0, 1)`
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`
    pitch_fmax: float
        maximum frequency of pitch
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
    if isinstance(s, Data):
        s = s[:]
    if np.prod(s.shape) == np.max(s.shape):
        s = s.ravel()
    elif s.ndim >= 2:
        raise Exception('Speech Feature Extraction only accept 1-D signal')
    s = s.astype('float32')
    import librosa
    from librosa.core import constantq
    # ====== resample if necessary ====== #
    if sr_new is not None and int(sr_new) != int(sr):
        s = resample(s, sr, sr_new, axis=0, best_algorithm=False)
        sr = sr_new
    if fmax is None:
        fmax = sr // 2
    if fmin is None or fmin < 0 or fmin >= fmax:
        fmin = 0
    if pitch_fmax is None:
        pitch_fmax = fmax
    win_length = int(win * sr)
    # n_fft must be 2^x
    n_fft = 2 ** int(np.ceil(np.log2(win_length)))
    shift_length = shift * sr
    # hop_length must be 2^x
    hop_length = int(shift_length)
    # preemphais
    s = pre_emphasis(s, coeff=preemphasis)
    nb_ceps += 1 # increase one so we can ignore the first MFCC
    # ====== 0: extract Constant Q-transform ====== #
    q_melspectrogram = None
    q_mfcc = None
    qspec = None
    qphase = None
    if get_qspec:
        constantq.__cqt_response = __cqt_response_override(win_length)
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
        qS = np.abs(qtrans)
        qS = qS**2
        if np.any(np.isnan(qS)):
            return None
        # power spectrum of Q-transform
        qspec = librosa.logamplitude(qS, amin=1e-10, top_db=80.0).astype('float32')
        # phase of Q-transform
        if get_phase:
            # GD: derivative along frequency axis
            qphase = compute_delta(np.angle(qtrans),
                width=9, axis=0, order=1)[-1].astype('float32')
        # perfom cepstral analysis for Q-transform
        if get_mspec or get_mfcc:
            q_melspectrogram = librosa.feature.melspectrogram(
                y=None, sr=sr, S=qS, n_fft=n_fft, hop_length=hop_length,
                n_mels=nb_melfilters, fmin=fmin, fmax=fmax, htk=False)
            q_melspectrogram = librosa.logamplitude(q_melspectrogram,
                amin=1e-10, top_db=80.0).astype('float32')
            if get_mfcc:
                q_mfcc = librosa.feature.mfcc(
                    y=None, sr=sr, S=q_melspectrogram, n_mfcc=nb_ceps).astype('float32')
                q_mfcc = q_mfcc[1:] # ignore the first coefficient
    # ====== 1: extract VAD and energy ====== #
    # centering the raw signal by padding
    if center:
        s = np.pad(s, int(win_length // 2), mode='reflect')
    log_energy = None
    vad = None
    vad_ids = None
    if get_energy or get_vad:
        frames = framing(s, frame_length=win_length, hop_length=hop_length)
        energy = (frames**2).sum(axis=0)
        energy = np.where(energy == 0., np.finfo(float).eps, energy)
        log_energy = np.log(energy).astype('float32')[None, :]
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
                                                      min_length=int(vad_minlen / shift)),
                               dtype='int32')
    # ====== 2: extract STFT and Spectrogram ====== #
    # no padding for center
    stft_ = stft(s, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    S = np.abs(stft_)
    if np.any(np.isnan(S)):
        return None
    # ====== 3: extract phase features ====== #
    phase = None
    if get_phase:
        # GD: derivative along frequency axis
        phase = compute_delta(np.angle(stft_),
            width=9, axis=0, order=1)[-1].astype('float32')
    # ====== 4: extract pitch features ====== #
    pitch_freq = None
    if get_pitch:
        # we don't care about pitch magnitude
        pitch_freq, _ = librosa.piptrack(
            y=None, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length,
            fmin=fmin, fmax=pitch_fmax, threshold=pitch_threshold)
        pitch_freq = pitch_freq.astype('float32')[:__max_fft_bins(sr, n_fft, pitch_fmax)]
        # normalize to 0-1
        _ = np.min(pitch_freq)
        pitch_freq = (pitch_freq - _) / (np.max(pitch_freq) - _)
        pitch_freq = compute_delta(pitch_freq, width=9, order=1, axis=-1)[-1]
    # ====== 5: extract power spectrogram ====== #
    S = S**2
    powerspectrogram = None
    if get_spec:
        powerspectrogram = librosa.logamplitude(S,
            amin=1e-10, top_db=80.0).astype('float32')
    # ====== 6: extract log-mel filter bank ====== #
    melspectrogram = None
    mfcc = None
    if get_mspec or get_mfcc:
        melspectrogram = librosa.feature.melspectrogram(
            y=None, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length,
            n_mels=nb_melfilters, fmin=fmin, fmax=fmax, htk=False)
        melspectrogram = librosa.logamplitude(melspectrogram,
            amin=1e-10, top_db=80.0).astype('float32')
        if get_mfcc:
            mfcc = librosa.feature.mfcc(
                y=None, sr=sr, S=melspectrogram, n_mfcc=nb_ceps).astype('float32')
            mfcc = mfcc[1:] # ignore the first coefficient
    # ====== 7: compute delta ====== #
    if get_delta and get_delta > 0:
        get_delta = int(get_delta)
        if log_energy is not None:
            log_energy = np.concatenate(
                [log_energy] + compute_delta(log_energy, order=get_delta),
                axis=0)
        # STFT
        if melspectrogram is not None:
            melspectrogram = np.concatenate(
                [melspectrogram] + compute_delta(melspectrogram, order=get_delta),
                axis=0)
        if mfcc is not None:
            mfcc = np.concatenate(
                [mfcc] + compute_delta(mfcc, order=get_delta),
                axis=0)
        # Q-transform
        if q_melspectrogram is not None:
            q_melspectrogram = np.concatenate(
                [q_melspectrogram] + compute_delta(q_melspectrogram, order=get_delta),
                axis=0)
        if q_mfcc is not None:
            q_mfcc = np.concatenate(
                [q_mfcc] + compute_delta(q_mfcc, order=get_delta),
                axis=0)
    # ====== 8: make sure CQT give the same length with STFT ====== #
    if get_qspec and qspec.shape[1] > powerspectrogram.shape[1]:
        n = qspec.shape[1] - powerspectrogram.shape[1]
        qspec = qspec[:, n // 2:-int(np.ceil(n / 2))]
        if qphase is not None:
            qphase = qphase[:, n // 2:-int(np.ceil(n / 2))]
        if q_melspectrogram is not None:
            q_melspectrogram = q_melspectrogram[:, n // 2:-int(np.ceil(n / 2))]
        if q_mfcc is not None:
            q_mfcc = q_mfcc[:, n // 2:-int(np.ceil(n / 2))]
    return OrderedDict([
        ('mfcc', None if mfcc is None else mfcc.T),
        ('energy', log_energy.T if get_energy else None),
        ('spec', None if powerspectrogram is None else powerspectrogram.T),
        ('mspec', None if melspectrogram is None else melspectrogram.T),

        ('qspec', None if qspec is None else qspec.T),
        ('qmspec', None if q_melspectrogram is None else q_melspectrogram.T),
        ('qmfcc', None if q_mfcc is None else q_mfcc.T),

        ('phase', phase.T if get_phase else None),
        ('qphase', qphase.T if get_phase and get_qspec else None),

        ('pitch', None if pitch_freq is None else pitch_freq.T),

        ('vad', vad),
        ('vadids', vad_ids),
    ])
