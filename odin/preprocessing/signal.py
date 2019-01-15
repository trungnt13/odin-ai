# -*- coding: utf-8 -*-
# ===========================================================================
# General toolkits for Signal processing in pure python, numpy and scipy
# The code is collected and modified from following library:
# * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
#       License: BSD 3-clause
#       Authors: Kyle Kastner
#       Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
#       http://ml.cs.yamanashi.ac.jp/world/english/
#       MGC code based on r9y9 (Ryusshichi Yamamoto) MelGeneralizedCepstrums.jl
#       Pieces also adapted from SPTK
# * https://github.com/librosa/librosa
#       License: https://github.com/librosa/librosa/blob/master/LICENSE.md
# * http://www-lium.univ-lemans.fr/sidekit/
#       Authors: Anthony Larcher & Kong Aik Lee & Sylvain Meignier
#       License: GNU Library or Lesser General Public License (LGPL)
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import six
import copy
import warnings
import subprocess
from io import BytesIO
from numbers import Number
from six import string_types

import numpy as np
import scipy as sp
from scipy import linalg, fftpack, signal
from numpy.lib.stride_tricks import as_strided
try:
  from odin.utils import cache_memory, cache_disk
except ImportError:
  def cache_memory(func):
    return func

  def cache_disk(func):
    return func

# Constrain STFT block sizes to 512 KB
MAX_MEM_BLOCK = 2**8 * 2**11
# ===========================================================================
# Helper
# ===========================================================================
def anything2wav(inpath, outpath=None,
                 channel=None, sample_rate=None, codec=None,
                 start=None, end=None,
                 dataset='unknown', return_data=True):
  """

  Parameters
  ----------
  inpath : string
    path to audio file

  outpath : {string, None}
    if None, return `BufferReader` contain the result of the conversion,
    call `read()` to get all the binary data of the .wav file.
    Otherwise, return the command for execute or the path of output file

  channel : {None, 0, 1}
    specific channel for the conversion
    0 for left channel, 1 for right channel

  sample_rate : {None, integer}
    sample rate for down-sampling

  dataset : {'unknown', 'voxceleb2'}
    predefine recipe for some datasets, supported datasets include:
     - voxceleb2
     - mx6 (mixer 6)

  codec : {None, string}
    the desire codec for output file
    for `sox`:
      'signed-integer' - pcm16
      'unsigned-integer' - pcm8
      'floating-point' -
      'a-law' - PCM A-law
      'u-law' -
      'mu-law' - PCM mu-law
      'oki-adpcm' -
    'ima-adpcm', 'ms-adpcm', 'gsm-full-rate'
    for `ffmpeg`:
      'alaw ' - PCM A-law
      'f32be' - PCM 32-bit floating-point big-endian
      'f32le' - PCM 32-bit floating-point little-endian
      'f64be' - PCM 64-bit floating-point big-endian
      'f64le' - PCM 64-bit floating-point little-endian
      'mulaw' - PCM mu-law
      's16be' - PCM signed 16-bit big-endian
      's16le' - PCM signed 16-bit little-endian (pcm 16)
      's24be' - PCM signed 24-bit big-endian
      's24le' - PCM signed 24-bit little-endian
      's32be' - PCM signed 32-bit big-endian
      's32le' - PCM signed 32-bit little-endian
      's8   ' - PCM signed 8-bit
      'u16be' - PCM unsigned 16-bit big-endian
      'u16le' - PCM unsigned 16-bit little-endian
      'u24be' - PCM unsigned 24-bit big-endian
      'u24le' - PCM unsigned 24-bit little-endian
      'u32be' - PCM unsigned 32-bit big-endian
      'u32le' - PCM unsigned 32-bit little-endian
      'u8   ' - PCM unsigned 8-bit (pcm8)

  start : {None, float}
    start time in second for trimming

  end : {None, float}
    end time in second (relative to the beginning) for trimming

  return_data : bool (default: True)
    if False, return the output path of .wav or the shell command in case
    `outpath=None`,
    otherwise, execute the command using `subprocess.Popen` and
    return the tuple of (data, sample_rate)

  Return
  ------
  if `return_data=True`: tuple(y, sr)
    y : numpy.ndarray (n_samples, n_channels)
    sr : integer (sample rate)
  else: string
    shell command for execute the conversion

  Note
  ----
  by default: we only accept channel of 0 (left) or 1 (right),
  and this value will be automatically converted to appropriate
  value for each toolkit:
    - for `sox` and `sph2pipe` the channel is 1 or 2
    - for `ffmpeg` the channel is 0 or 1

  """
  inpath = str(inpath)
  if dataset is None:
    dataset = 'unknown'
  dataset = str(dataset).lower()
  ext = os.path.splitext(inpath)[-1]
  is_given_outpath = True
  if outpath is None:
    outpath = '-'
    is_given_outpath = False
  # ====== unknown start and end time ====== #
  if start == '-':
    start = None
  if end == '-':
    end = None
  # ====== init ====== #
  tool = None
  cmd = None
  options = []
  # ====== mixer6 dataset ====== #
  if dataset.lower() == 'mx6':
    if '/ulaw_sphere/' in inpath:
      cmd = 'sph2pipe -f wav -p -c %d %s %s' % \
      (int(channel) + 1, inpath, '' if not is_given_outpath else outpath)
    else:
      # codec
      if codec is not None:
        codec = '-e signed-integer' if 'pcm16' in codec else '-e %s' % str(codec)
      else:
        codec = ''
      # main command
      cmd = 'sox -t flac %s %s -t wav %s %s trim %s =%s' % \
      (inpath,
       '' if sample_rate is None else '-r %d' % int(sample_rate),
       codec,
       outpath,
       start, end)
  # ====== other using SOX ====== #
  elif ext.lower() in ('.sph', '.wav', '.flac'):
    tool = 'sox'
    # options.append('-G')
    options.append('--no-dither')
    options.append('-t %s' % ext[1:])
    options.append(str(inpath))
    # downsample
    options.append('-r %d' % int(sample_rate)
                   if sample_rate is not None else '')
    options.append('-t wav')
    # codec
    if codec is not None:
      codec = 'signed-integer' if 'pcm16' in codec else str(codec)
    else:
      codec = ''
    if len(codec) > 0:
      options.append('-e %s' % codec)
    # output path
    options.append(str(outpath))
    # channel selection
    if channel is not None: # for sox channel is 1 or 2
      options.append('remix %d' % (int(channel) + 1))
    # trim audio
    if start is not None or end is not None:
      options.append('trim %s %s' %
                     ('%f' % start if start is not None else '',
                      '=%f' % end if end is not None else ''))
  # ====== other using ffmpeg ====== #
  elif ext.lower() in ('.m4a', '.mp3'):
    tool = 'ffmpeg'
    # force overwrite
    options.append('-y')
    # log level
    options.append('-v 8')
    options.append('-i %s' % inpath)
    # channel selection
    if channel is not None:
      options.append('-map_channel 0.0.%d' % int(channel))
    # downsample
    options.append('-ar %d' % int(sample_rate)
                   if sample_rate is not None else '')
    # output wav
    options.append('-f wav')
    # codec
    if dataset == 'voxceleb2':
      options.append('-acodec pcm_s16le')
    else:
      if codec is None:
        codec = ''
      else:
        codec = 'pcm_s16le' if 'pcm16' in codec else str(codec)
      if len(codec) > 0:
        options.append('-acodec %s' % codec)
    # output path
    options.append(str(outpath))
    # trim audio
    if start is not None:
      options.append('-ss %f' % start)
    if end is not None:
      options.append('-to %f' % end)
  else:
    raise ValueError("Nonsupport for file extension: %s" % inpath)
  # ====== merge everything ====== #
  if cmd is None:
    cmd = tool + ' ' + ' '.join([i for i in options
                                 if len(i) > 0])
  # ====== only return the shell command ====== #
  if not return_data:
    return cmd
  # ====== run the command and return the data ====== #
  import soundfile
  # read bytes data from PIPE output
  if outpath in ('-', '- |', '-|'):
    with subprocess.Popen(cmd, shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as task:
      data = BytesIO(task.stdout.read())
      try:
        data = soundfile.read(data) # y, sr
      except Exception as e:
        print("Error running command:", cmd)
        print("Stdout:", str(task.stdout.read(), 'utf-8'))
        print("Stderr:", str(task.stderr.read(), 'utf-8'))
        raise e
      return data
  # read audio file from outpath
  else:
    try:
      with subprocess.Popen(cmd, shell=True) as task:
        pass
    except Exception as e:
      print("Error running command:", cmd)
      print("Stdout:", str(task.stdout.read(), 'utf-8'))
      print("Stderr:", str(task.stderr.read(), 'utf-8'))
      raise e
    with open(outpath, 'rb') as f:
      return soundfile.read(f)

# ===========================================================================
# VAD
# ===========================================================================
VAD_MODE_STRICT = 1.2
VAD_MODE_STANDARD = 2.
VAD_MODE_SENSITIVE = 2.4
__current_vad_mode = VAD_MODE_STANDARD # alpha for vad energy

def set_vad_mode(mode):
  """
  Paramters
  ---------
  mode: float
      a number from 1.0 to 2.4, the higher the number, the more
      sensitive it is to any high-energy segments.
  """
  if isinstance(mode, Number):
    global __current_vad_mode
    mode = min(max(mode, 1.), 2.4)
    __current_vad_mode = float(mode)

def vad_energy(log_energy, distrib_nb=3, nb_train_it=25):
  """ Fitting Gaussian mixture model on the log-energy and the voice
  activity is the component with highest energy.

  Return
  ------
  vad: array of 0, 1
  threshold: scalar
  """
  from sklearn.exceptions import ConvergenceWarning
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
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ConvergenceWarning)
      world.fit(log_energy)
  except (ValueError, IndexError): # index error because of float32 cumsum
    if distrib_nb - 1 >= 2:
      return vad_energy(log_energy,
                        distrib_nb=distrib_nb - 1,
                        nb_train_it=nb_train_it)
    return np.zeros(shape=(log_energy.shape[0],)), 0
  # Compute threshold
  threshold = world.means_.max() - \
      __current_vad_mode * np.sqrt(1.0 / world.precisions_[world.means_.argmax(), 0])
  # Apply frame selection with the current threshold
  label = log_energy.ravel() > threshold
  return label, threshold

def vad_threshold(frames, threshold=35):
  """
  threshold : scalar (30,40)
  """
  energies = 20 * np.log10(np.std(frames, axis=0) + np.finfo(float).eps)
  max_energy = np.max(energies)
  return (energies > max_energy - threshold) & (energies > -55)

def vad_split_audio(s, sr, maximum_duration=30, minimum_duration=None,
                    frame_length=128, nb_mixtures=3, threshold=0.6,
                    return_vad=False, return_voices=False, return_cut=False):
  """ Splitting an audio based on VAD indicator.
  * The audio is segmented into multiple with length given by `frame_length`
  * Log-energy is calculated for each frames
  * Gaussian mixtures with `nb_mixtures` is fitted, and output vad indicator
    for each frames.
  * A flat window (ones-window) of `frame_length` is convolved with the
    vad indices.
  * All frames within the percentile >= `threshold` is treated as voiced.
  * The splitting process is greedy, frames is grouped until reaching the
    maximum duration

  Parameters
  ----------
  s: 1-D numpy.ndarray
      loaded audio array
  sr: int
      sample rate
  maximum_duration: float (second)
      maximum duration of each segments in seconds
  minimum_duration: None, or float (second)
      all segments below this length will be merged into longer segments,
      if None, any segments with half of the `maximum_duration`
      are considered.
  frame_length: int
      number of frames for windowing
  nb_mixtures: int
      number of Gaussian mixture for energy-based VAD (the higher
      the more overfitting).
  threshold: float (0. to 1.)
      The higher the values, the more frames are considered as voiced,
      this value is the lower percentile of voiced frames.
  return_vad: bool
      if True, return VAD confident values
  return_voices: bool
      if True, return the voices frames indices
  return_cut: bool
      if True, return the cut points of the audio.

  Return
  ------
  segments: list of audio arrays
  vad (optional): list of 0, 1 for VAD indices
  voices (optional): list of thresholded VAD for more precise voices frames.
  cut (optional): list of indicator 0, 1 (1 for the cut point)

  Note
  ----
  this function does not guarantee the output audio length is always
  smaller than `maximum_duration`, the higher the threshold, the better
  chance you get everything smaller than `maximum_duration`
  """
  frame_length = int(frame_length)
  maximum_duration = maximum_duration * sr
  results = []
  # ====== check if audio long enough ====== #
  if len(s) < maximum_duration:
    if return_cut or return_vad or return_voices:
      raise ValueError("Cannot return `cut` points, `vad` or `voices` since"
                  "the original audio is shorter than `maximum_duration`, "
                  "hence, no need for splitting.")
    return [s]
  maximum_duration /= frame_length
  if minimum_duration is None:
    minimum_duration = maximum_duration // 2
  else:
    minimum_duration = minimum_duration * sr / frame_length
    minimum_duration = np.clip(minimum_duration, 0., 0.99 * maximum_duration)
  # ====== start spliting ====== #
  frames = segment_axis(s, frame_length, frame_length,
                        axis=0, end='pad', pad_value=0.)
  energy = get_energy(frames, log=True)
  vad = vad_energy(energy, distrib_nb=nb_mixtures, nb_train_it=33)[0]
  vad = smooth(vad, win=frame_length, window='flat')
  # explicitly return VAD
  if return_vad:
    results.append(vad)
  # ====== get all possible sliences ====== #
  # all voice indices
  indices = np.where(vad >= np.percentile(vad, q=threshold * 100))[0].tolist()
  if len(vad) - 1 not in indices:
    indices.append(len(vad) - 1)
  # explicitly return voiced frames
  if return_voices:
    tmp = np.zeros(shape=(len(vad),))
    tmp[indices] = 1
    results.append(tmp)
  # ====== spliting the audio ====== #
  segments = []
  start = 0
  prev_end = 0
  # greedy adding new frames to reach desire maximum length
  for end in indices:
    # over-reach the maximum length
    if end - start > maximum_duration:
      segments.append((start, prev_end))
      start = prev_end
    # exact maximum length
    elif end - start == maximum_duration:
      segments.append((start, end))
      start = end
    prev_end = end
  # still no segments found
  if len(segments) == 0:
    segments = [(indices[0], indices[-1])]
  # add ending index if necessary
  if indices[-1] != segments[-1][-1]:
    segments.append((start, indices[-1]))
  # re-fining, short segments will be merged into bigger onces
  found_under_length = True
  while found_under_length and len(segments) > 1:
    new_segments = []
    found_under_length = False
    for (s1, e1), (s2, e2) in zip(segments, segments[1:]):
      if (e1 - s1) < minimum_duration or (e2 - s2) < minimum_duration:
        new_segments.append((s1, e2))
        found_under_length = True
      # keep both of the segments
      else:
        new_segments.append((s1, e1))
        new_segments.append((s2, e2))
    segments = new_segments
  # explicitly return cut points
  if return_cut:
    tmp = np.zeros(shape=(segments[-1][-1] + 1,))
    for i, j in segments:
      tmp[i] = 1; tmp[j] = 1
    results.append(tmp)
  # ====== convert everythng to raw signal index ====== #
  segments = [[i * frame_length, j * frame_length]
              for i, j in segments]
  segments[-1][-1] = s.shape[0]
  # cut segments out of raw audio array
  segments = [s[i:j] for i, j in segments]
  results = [segments] + results
  return results[0] if len(results) == 1 else results

# ===========================================================================
#
# ===========================================================================
def loudness2intensity(loudness):
  """ 60dB is consider the standard """
  if loudness.ndim == 2:
    loudness = loudness[:, 0]
  return loudness * 60

def hz2mel(frequencies):
  """Convert Hz to Mels
  Original code: librosa

  Examples
  --------
  >>> hz2mel(60)
  array([ 0.9])
  >>> hz2mel([110, 220, 440])
  array([ 1.65,  3.3 ,  6.6 ])

  Parameters
  ----------
  frequencies   : np.ndarray [shape=(n,)] , float
      scalar or array of frequencies

  Returns
  -------
  mels        : np.ndarray [shape=(n,)]
      input frequencies in Mels

  See Also
  --------
  mel_to_hz
  """
  frequencies = np.atleast_1d(frequencies)
  # Fill in the linear part
  f_min = 0.0
  f_sp = 200.0 / 3
  mels = (frequencies - f_min) / f_sp
  # Fill in the log-scale part
  min_log_hz = 1000.0                         # beginning of log region (Hz)
  min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
  logstep = np.log(6.4) / 27.0                # step size for log region

  log_t = (frequencies >= min_log_hz)
  mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
  return mels

def mel2hz(mels):
  """Convert mel bin numbers to frequencies
  Original code: librosa

  Examples
  --------
  >>> mel2hz(3)
  array([ 200.])
  >>> mel2hz([1,2,3,4,5])
  array([  66.667,  133.333,  200.   ,  266.667,  333.333])

  Parameters
  ----------
  mels          : np.ndarray [shape=(n,)], float
      mel bins to convert

  Returns
  -------
  frequencies   : np.ndarray [shape=(n,)]
      input mels in Hz

  See Also
  --------
  hz_to_mel
  """

  mels = np.atleast_1d(mels)

  # Fill in the linear scale
  f_min = 0.0
  f_sp = 200.0 / 3
  freqs = f_min + f_sp * mels

  # And now the nonlinear scale
  min_log_hz = 1000.0                         # beginning of log region (Hz)
  min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
  logstep = np.log(6.4) / 27.0                # step size for log region
  log_t = (mels >= min_log_mel)

  freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
  return freqs

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
  """Compute the center frequencies of mel bands.
  Original code: librosa

  Parameters
  ----------
  n_mels    : int > 0 [scalar]
      number of Mel bins
  fmin      : float >= 0 [scalar]
      minimum frequency (Hz)
  fmax      : float >= 0 [scalar]
      maximum frequency (Hz)

  Returns
  -------
  bin_frequencies : ndarray [shape=(n_mels,)]
      vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
      axis.

  Examples
  --------
  >>> librosa.mel_frequencies(n_mels=40)
  array([     0.   ,     85.317,    170.635,    255.952,
            341.269,    426.586,    511.904,    597.221,
            682.538,    767.855,    853.173,    938.49 ,
           1024.856,   1119.114,   1222.042,   1334.436,
           1457.167,   1591.187,   1737.532,   1897.337,
           2071.84 ,   2262.393,   2470.47 ,   2697.686,
           2945.799,   3216.731,   3512.582,   3835.643,
           4188.417,   4573.636,   4994.285,   5453.621,
           5955.205,   6502.92 ,   7101.009,   7754.107,
           8467.272,   9246.028,  10096.408,  11025.   ])

  """
  min_mel = hz2mel(fmin)
  max_mel = hz2mel(fmax)
  mels = np.linspace(min_mel, max_mel, n_mels)
  return mel2hz(mels)

def db2power(S_db, ref=1.0):
  '''Convert a dB-scale spectrogram to a power spectrogram.

  This effectively inverts `power_to_db`:

      `db_to_power(S_db) ~= ref * 10.0**(S_db / 10)`

  Original code: librosa

  Parameters
  ----------
  S_db : np.ndarray
      dB-scaled spectrogram
  ref : number > 0
      Reference power: output will be scaled by this value

  Returns
  -------
  S : np.ndarray
      Power spectrogram

  Notes
  -----
  This function caches at level 30.
  '''
  return ref * np.power(10.0, 0.1 * S_db)

def power2db(S, ref=1.0, amin=1e-10, top_db=80.0):
  """Convert a power spectrogram (amplitude/magnitude squared)
  to decibel (dB) units (using logarithm)

  This computes the scaling ``10 * log10(S / ref)`` in a numerically
  stable way.

  Original code: librosa

  Parameters
  ----------
  S : np.ndarray
      input power
  ref : scalar or call-able
      If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
      `10 * log10(S / ref)`.
      Zeros in the output correspond to positions where `S == ref`.
      If call-able, the reference value is computed as `ref(S)`.
  amin : float > 0 [scalar]
      minimum threshold for `abs(S)` and `ref`
  top_db : float >= 0 [scalar]
      threshold the output at `top_db` below the peak:
      ``max(10 * log10(S)) - top_db``

  Returns
  -------
  S_db   : np.ndarray
      ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
  """
  if amin <= 0:
    raise ValueError('amin must be strictly positive')
  magnitude = np.abs(S)
  if hasattr(ref, '__call__'):
    # User supplied a function to calculate reference power
    ref_value = ref(magnitude)
  else:
    ref_value = np.abs(ref)
  log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
  log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
  # clip top db
  if top_db is not None:
    if top_db < 0:
      raise ValueError('top_db must be non-negative')
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)
  return log_spec

@cache_memory
def dct_filters(n_filters, n_input):
  """Discrete cosine transform (DCT type-III) basis.

  .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

  Original code: librosa

  Parameters
  ----------
  n_filters : int > 0 [scalar]
      number of output components (DCT filters)

  n_input : int > 0 [scalar]
      number of input components (frequency bins)

  Returns
  -------
  dct_basis: np.ndarray [shape=(n_filters, n_input)]
      DCT (type-III) basis vectors [1]_

  Notes
  -----
  This function caches at level 10.

  Examples
  --------
  >>> n_fft = 2048
  >>> dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
  >>> dct_filters
  array([[ 0.031,  0.031, ...,  0.031,  0.031],
         [ 0.044,  0.044, ..., -0.044, -0.044],
         ...,
         [ 0.044,  0.044, ..., -0.044, -0.044],
         [ 0.044,  0.044, ...,  0.044,  0.044]])

  >>> import matplotlib.pyplot as plt
  >>> plt.figure()
  >>> librosa.display.specshow(dct_filters, x_axis='linear')
  >>> plt.ylabel('DCT function')
  >>> plt.title('DCT filter bank')
  >>> plt.colorbar()
  >>> plt.tight_layout()
  """
  basis = np.empty((n_filters, n_input))
  basis[0, :] = 1.0 / np.sqrt(n_input)

  samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

  for i in range(1, n_filters):
    basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)
  return basis

@cache_memory
def mel_filters(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
  """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins
  Original code: librosa

  Parameters
  ----------
  sr        : number > 0 [scalar]
      sampling rate of the incoming signal

  n_fft     : int > 0 [scalar]
      number of FFT components

  n_mels    : int > 0 [scalar]
      number of Mel bands to generate

  fmin      : float >= 0 [scalar]
      lowest frequency (in Hz)

  fmax      : float >= 0 [scalar]
      highest frequency (in Hz).
      If `None`, use `fmax = sr / 2.0`

  Returns
  -------
  M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
      Mel transform matrix

  Examples
  --------
  >>> melfb = mel_filters(22050, 2048)
  >>> melfb
  array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
         [ 0.   ,  0.   , ...,  0.   ,  0.   ],
         ...,
         [ 0.   ,  0.   , ...,  0.   ,  0.   ],
         [ 0.   ,  0.   , ...,  0.   ,  0.   ]])
  """
  if fmax is None:
    fmax = float(sr) / 2
  # Initialize the weights
  n_mels = int(n_mels)
  weights = np.zeros((n_mels, int(1 + n_fft // 2)))

  # Center freqs of each FFT bin
  fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2),
                         endpoint=True)

  # 'Center freqs' of mel bands - uniformly spaced between limits
  min_mel = hz2mel(fmin)
  max_mel = hz2mel(fmax)
  mel_f = mel2hz(mels=np.linspace(min_mel, max_mel, n_mels + 2))

  fdiff = np.diff(mel_f)
  ramps = np.subtract.outer(mel_f, fftfreqs)

  for i in range(n_mels):
    # lower and upper slopes for all bins
    lower = -ramps[i] / fdiff[i]
    upper = ramps[i + 2] / fdiff[i + 1]

    # .. then intersect them with each other and zero
    weights[i] = np.maximum(0, np.minimum(lower, upper))

  # Slaney-style mel is scaled to be approx constant energy per channel
  enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
  weights *= enorm[:, np.newaxis]

  # Only check weights if f_mel[0] is positive
  if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    # This means we have an empty channel somewhere
    print('[WARNING] Empty filters detected in mel frequency basis. '
          'Some channels will produce empty responses. '
          'Try increasing your sampling rate (and fmax) or '
          'reducing n_mels.')
  return weights

@cache_memory
def get_window(window, frame_length, periodic=True):
  ''' Cached version of scipy.signal.get_window '''
  # Funtion
  if hasattr(window, '__call__'):
    return window(frame_length)
  # Window name or scalar
  elif (isinstance(window, (six.string_types, tuple)) or
        np.isscalar(window)):
    return signal.get_window(window, frame_length, fftbins=periodic)
  # Predefined-array
  elif isinstance(window, (np.ndarray, list)):
    if len(window) == frame_length:
      return np.asarray(window)
    raise ValueError('Window size mismatch: '
                     '{:d} != {:d}'.format(len(window), frame_length))
  # Unknown
  else:
    raise ValueError('Invalid window specification: %s' % str(window))

# ===========================================================================
# Array utils
# ===========================================================================
def resample(y, sr_orig, sr_new, axis=0, best_algorithm=True):
  '''
  '''
  sr_orig = int(sr_orig)
  sr_new = int(sr_new)
  if sr_new > sr_orig:
    raise ValueError("Do not support upsampling audio from %d(Hz) to %d(Hz)."
        % (sr_orig, sr_new))
  elif sr_orig != sr_new:
    import resampy
    y = resampy.resample(y, sr_orig=sr_orig, sr_new=sr_new, axis=axis,
            filter='kaiser_best' if best_algorithm else 'kaiser_fast')
  return y

_fnorm1 = lambda x, x_stat, keepdims: x - x_stat.mean(axis=0, keepdims=keepdims)
_fnorm2 = lambda x, x_stat, keepdims: ((x - x_stat.mean(axis=0, keepdims=keepdims)) /
                                       (x_stat.std(axis=0, keepdims=keepdims) + 1e-18))

def mvn(x, varnorm=True, indices=None):
  """ Mean and Variance Normalization
  Normalization is applied on time-axis

  Parameters
  ----------
  x: [t, f]
      [time, frequency]
  varnorm : bool
    if True, normalized by standard deviation
  indices : numpy.ndarray [time,]
    `numpy.bool` array, the speech activities boolean indices,
    which frames will be taken into account for calculating
    the `mean` and `std`

  Note
  ----
  Just standard normalization, not a big deal for its name
  """
  x_stat = x[indices] if indices is not None else x
  if varnorm:
    return _fnorm2(x, x_stat, True)
  else:
    return _fnorm1(x, x_stat, True)

def wmvn(x, w=301, varnorm=True, indices=None):
  """ Windowed - Mean and Variance Normalization
  Normalization is applied on time-axis

  Parameters
  ----------
  x : [t, f]
      [time, frequency]
  w : int
    width of normalization window.
  varnorm : bool
    if True, normalized by standard deviation
  indices : numpy.ndarray [time,]
    `numpy.bool` array, the speech activities boolean indices,
    which frames will be taken into account for calculating
    the `mean` and `std`

  """
  if w < 3 or (w & 1) != 1:
    raise ValueError('Window length should be an odd integer >= 3')
  nobs, ndim = x.shape
  if nobs < w:
    return mvn(x, varnorm=varnorm, indices=indices)
  fnorm = _fnorm2 if varnorm else _fnorm1
  # ====== init ====== #
  hlen = int((w - 1) / 2)
  y = np.zeros((nobs, ndim), dtype=x.dtype)
  # ====== first window ====== #
  # applying indices
  x_stat = x[:w] if indices is None else x[:w][indices[:w]]
  # normalize
  y[:hlen] = fnorm(x[:hlen], x_stat, True)
  # ====== iterate ====== #
  for ix in range(hlen, nobs - hlen):
    # applying indices
    if indices is None:
      x_stat = x[ix - hlen:ix + hlen + 1]
    else:
      sad = indices[ix - hlen:ix + hlen + 1]
      x_stat = x[ix - hlen:ix + hlen + 1][sad]
    # normalize
    y[ix] = fnorm(x[ix], x_stat, False)
  # ====== last window ====== #
  # applying indices
  x_stat = x[nobs - w:] if indices is None else x[nobs - w:][indices[nobs - w:]]
  y[nobs - hlen:nobs] = fnorm(x[nobs - hlen:nobs], x_stat, True)
  return y

def rastafilt(x):
  """ Based on rastafile.m by Dan Ellis
     rows of x = critical bands, cols of x = frame
     same for y but after filtering
     default filter is single pole at 0.94

  The filter is applied on frequency axis

  Parameters
  ----------
  x: [t, f]
      time x frequency
  """
  x = x.T # lazy style to reuse the code from [f, t] libraries
  ndim, nobs = x.shape
  numer = np.arange(-2, 3)
  # careful with division here (float point suggested by Ville Vestman)
  numer = -numer / np.sum(numer * numer)
  denom = [1, -0.94]
  y = np.zeros((ndim, 4))
  z = np.zeros((ndim, 4))
  zi = [0., 0., 0., 0.]
  for ix in range(ndim):
    y[ix, :], z[ix, :] = signal.lfilter(numer, 1, x[ix, :4], zi=zi, axis=-1)
  y = np.zeros((ndim, nobs))
  for ix in range(ndim):
    y[ix, 4:] = signal.lfilter(numer, denom, x[ix, 4:], zi=z[ix, :], axis=-1)[0]
  return y.T

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
    w = np.ones(win, dtype='d')
  # windowing
  else:
    w = eval('np.' + window + '(win)')
  y = np.convolve(w / w.sum(), s, mode='same')
  return y[win:-win + 1]

def delta(data, width=9, order=1, axis=0):
  r'''Compute delta features: local estimate of the derivative
  of the input data along the selected axis.
  Original implementation: librosa

  Parameters
  ----------
  data      : np.ndarray
      the input data matrix (e.g. for spectrogram, delta should be applied
      on time-axis).
  width     : int >= 3, odd [scalar]
      Number of frames over which to compute the delta feature
  order     : int > 0 [scalar]
      the order of the difference operator.
      1 for first derivative, 2 for second, etc.
  axis      : int [scalar]
      the axis along which to compute deltas.
      Default is -1 (columns).

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

  order = int(order)
  if order <= 0:
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

  # ====== compute deltas ====== #
  all_deltas = []
  for _ in range(order):
    delta_x = signal.lfilter(window, 1, delta_x, axis=axis)
    all_deltas.append(delta_x)
  # ====== Cut back to the original shape of the input data ====== #
  trim_deltas = []
  for delta_x in all_deltas:
    idx = [slice(None)] * delta_x.ndim
    idx[axis] = slice(- half_length - data.shape[axis], - half_length)
    delta_x = delta_x[idx]
    trim_deltas.append(delta_x.astype('float32'))
  return trim_deltas[0] if order == 1 else trim_deltas

def shifted_deltas(x, N=7, d=1, P=3, k=7):
  """ Calculate Shifted Delta Coefficients
  (suggested applying on time-dimension) for MFCCs

  Parameters
  ----------
  x: [t, f]
      time x frequency
  """
  x = x.T
  if d < 1:
    raise ValueError('d should be an integer >= 1')
  nobs = x.shape[1]
  x = x[:N]
  w = 2 * d + 1
  dx = delta(x, w, order=1, axis=-1)
  sdc = np.empty((k * N, nobs))
  sdc[:] = np.tile(dx[:, -1], k).reshape(k * N, 1)
  for ix in range(k):
    if ix * P > nobs:
      break
    sdc[ix * N:(ix + 1) * N, :nobs - ix * P] = dx[:, ix * P:nobs]
  return sdc.T

@cache_memory('__strict__')
def pad_center(data, size, axis=-1, **kwargs):
  '''Wrapper for numpy.pad to automatically center an array prior to padding.
  This is analogous to `str.center()`

  Parameters
  ----------
  data : numpy.ndarray
      Vector to be padded and centered

  size : int >= len(data) [scalar]
      Length to pad `data`

  axis : int
      Axis along which to pad and center the data

  kwargs : additional keyword arguments
    arguments passed to `numpy.pad()`

  Returns
  -------
  data_padded : numpy.ndarray
      `data` centered and padded to length `size` along the
      specified axis

  Raises
  ------
  ParameterError
      If `size < data.shape[axis]`

  See Also
  --------
  numpy.pad
  '''
  kwargs.setdefault('mode', 'constant')

  n = data.shape[axis]

  lpad = int((size - n) // 2)

  lengths = [(0, 0)] * data.ndim
  lengths[axis] = (lpad, int(size - n - lpad))

  if lpad < 0:
    raise ValueError(('Target size ({:d}) must be '
                      'at least input size ({:d})').format(size, n))
  return np.pad(data, lengths, **kwargs)

def one_hot(y, nb_classes=None, dtype='float32'):
  '''Convert class vector (integers from 0 to nb_classes)
  to binary class matrix, for use with categorical_crossentropy

  Note
  ----
  if any class index in y is smaller than 0, then all of its one-hot
  values is 0.
  '''
  if 'int' not in str(y.dtype):
    y = y.astype('int32')
  if nb_classes is None:
    nb_classes = np.max(y) + 1
  else:
    nb_classes = int(nb_classes)
  return np.eye(nb_classes, dtype=dtype)[y]

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.,
                  transformer=None):
  """Pads each sequence to the same length:
  the length of the longest sequence.

  If maxlen is provided, any sequence longer
  than maxlen is truncated to maxlen.
  Truncation happens off either the beginning (default) or
  the end of the sequence.

  Supports post-padding and pre-padding (default).

  Parameters
  ----------
  sequences: list
      a list that contains a list of object
  maxlen: int
      maximum length of each individual sequence
  dtype: np.dtype
      desire data type of output array
  padding: 'pre' or 'post'
      pad either before or after each sequence.
  truncating: 'pre' or 'post'
      remove values from sequences larger than maxlen either
      in the beginning or in the end of the sequence
  value: object
      padding object
  transformer: call-able
      a function transform each element in sequence into desire value
      (e.g. a dictionary)

  Returns
  -------
  numpy array with dimensions (number_of_sequences, maxlen)
  """
  # ====== check valid input ====== #
  if truncating not in ('pre', 'post'):
    raise ValueError('truncating must be "pre" or "post", given value is %s'
                     % truncating)
  if padding not in ('pre', 'post'):
    raise ValueError('padding must be "pre" or "post", given value is %s'
                     % padding)
  if transformer is None:
    transformer = lambda x: x
  if not hasattr(transformer, '__call__'):
    raise ValueError('transformer must be call-able, but given value is %s' %
                     type(transformer))
  # ====== processing ====== #
  if maxlen is None:
    maxlen = int(max(len(s) for s in sequences))
  nb_samples = len(sequences)
  value = np.cast[dtype](value)
  X = np.full(shape=(nb_samples, maxlen), fill_value=value, dtype=dtype)
  for idx, s in enumerate(sequences):
    s = [transformer(_) for _ in s]
    if len(s) == 0: continue # empty list
    # check truncating
    if len(s) >= maxlen:
      slice_ = slice(None, None)
      s = s[-maxlen:] if truncating == 'pre' else s[:maxlen]
    # check padding
    elif len(s) < maxlen:
      slice_ = slice(-len(s), None) if padding == 'pre' else slice(None, len(s))
    # assign value
    X[idx, slice_] = np.asarray(s, dtype=dtype)
  return X

def stack_frames(X, frame_length, step_length=None,
                 keep_length=False, make_contigous=False):
  """ Stack consecutive frames into single vector, each operation
  is shifted by `step_length`

  Parameters
  ----------
  X: numpy.ndarray
      2D arrray
  frame_length: int
      number of frames will be stacked into 1 sample.
  step_length: {int, None} (default: None)
      number of shifted frame after each stacking operation,
      if None, its value equals to `frame_length // 2`
  keep_length: bool
      if True, padding zeros to begin and end of `X` to
      make the output array has the same length as original
      array.
  make_contigous: bool
      if True, use `numpy.ascontiguousarray` to ensure input `X`
      is contiguous.

  Example
  -------
  >>> X = [[ 0  1]
  ...      [ 2  3]
  ...      [ 4  5]
  ...      [ 6  7]
  ...      [ 8  9]
  ...      [10 11]
  ...      [12 13]
  ...      [14 15]
  ...      [16 17]
  ...      [18 19]]
  >>> frame_length = 5
  >>> step_length = 2
  >>> stack_frames(X, frame_length, step_length)
  >>> [[ 0  1  2  3  4  5  6  7  8  9]
  ...  [ 4  5  6  7  8  9 10 11 12 13]
  ...  [ 8  9 10 11 12 13 14 15 16 17]]
  """
  if frame_length > len(X) and keep_length is False:
    raise ValueError("`frame_length=%d` is greater than the length of input matrix %s;"
                     "`keep_length` must be set to True to allow padding." %
                     (frame_length, str(X.shape)))
  if keep_length:
    if step_length != 1:
      raise ValueError("`keep_length` is only supported when `step_length` = 1.")
    add_frames = (int(np.ceil(frame_length / 2)) - 1) * 2 + \
        (1 if frame_length % 2 == 0 else 0)
    right = add_frames // 2
    left = add_frames - right
    X = np.pad(X,
               pad_width=((left, right),) + ((0, 0),) * (X.ndim - 1),
               mode='constant')
  # ====== check input ====== #
  assert X.ndim == 2, "Only support 2D matrix for stacking frames."
  if not X.flags['C_CONTIGUOUS']:
    if make_contigous:
      X = np.ascontiguousarray(X)
    else:
      raise ValueError('Input buffer must be contiguous.')
  # ====== stacking ====== #
  frame_length = int(frame_length)
  if step_length is None:
    step_length = frame_length // 2
  shape = (1 + (X.shape[0] - frame_length) // step_length,
           frame_length * X.shape[1])
  strides = (X.strides[0] * step_length, X.strides[1])
  return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

def segment_axis(a, frame_length=2048, step_length=512, axis=0,
                 end='cut', pad_value=0, pad_mode='post'):
  """Generate a new array that chops the given array along the given axis
  into overlapping frames.

  This method has been implemented by Anne Archibald,
  as part of the talk box toolkit
  example::

      segment_axis(arange(10), 4, 2)
      array([[0, 1, 2, 3],
         ( [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

  Parameters
  ----------
  a: numpy.ndarray
      the array to segment
  frame_length: int
      the length of each frame
  step_length: int
      the number of array elements by which the frames should overlap
  axis: int, None
      the axis to operate on; if None, act on the flattened array
  end: 'cut', 'wrap', 'pad'
      what to do with the last frame, if the array is not evenly
          divisible into pieces. Options are:
          - 'cut'   Simply discard the extra values
          - 'wrap'  Copy values from the beginning of the array
          - 'pad'   Pad with a constant value
  pad_value: int
      the value to use for end='pad'
  pad_mode: 'pre', 'post'
      if "pre", padding or wrapping at the beginning of the array.
      if "post", padding or wrapping at the ending of the array.

  Return
  ------
  a ndarray

  The array is not copied unless necessary (either because it is unevenly
  strided and being flattened or because end is set to 'pad' or 'wrap').

  Note
  ----
  Modified work and error fixing Copyright (c) TrungNT

  """
  if axis is None:
    a = np.ravel(a) # may copy
    axis = 0

  length = a.shape[axis]
  overlap = frame_length - step_length

  if overlap >= frame_length:
    raise ValueError("frames cannot overlap by more than 100%")
  if overlap < 0 or frame_length <= 0:
    raise ValueError("overlap must be nonnegative and length must" +
                     "be positive")

  if length < frame_length or (length - frame_length) % (frame_length - overlap):
    if length > frame_length:
      roundup = frame_length + (1 + (length - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
      rounddown = frame_length + ((length - frame_length) // (frame_length - overlap)) * (frame_length - overlap)
    else:
      roundup = frame_length
      rounddown = 0
    assert rounddown < length < roundup
    assert roundup == rounddown + (frame_length - overlap) \
    or (roundup == frame_length and rounddown == 0)
    a = a.swapaxes(-1, axis)

    if end == 'cut':
      a = a[..., :rounddown]
    elif end in ['pad', 'wrap']: # copying will be necessary
      s = list(a.shape)
      s[-1] = roundup
      b = np.empty(s, dtype=a.dtype)
      # pre-padding
      if pad_mode == 'post':
        b[..., :length] = a
        if end == 'pad':
          b[..., length:] = pad_value
        elif end == 'wrap':
          b[..., length:] = a[..., :roundup - length]
      # post-padding
      elif pad_mode == 'pre':
        b[..., -length:] = a
        if end == 'pad':
          b[..., :(roundup - length)] = pad_value
        elif end == 'wrap':
          b[..., :(roundup - length)] = a[..., :roundup - length]
      # error
      else:
        raise RuntimeError("No support for pad mode: %s" % pad_mode)
      a = b
    a = a.swapaxes(-1, axis)
    length = a.shape[0] # update length

  if length == 0:
    raise ValueError("Not enough data points to segment array " +
            "in 'cut' mode; try 'pad' or 'wrap'")
  assert length >= frame_length
  assert (length - frame_length) % (frame_length - overlap) == 0
  n = 1 + (length - frame_length) // (frame_length - overlap)
  s = a.strides[axis]
  newshape = a.shape[:axis] + (n, frame_length) + a.shape[axis + 1:]
  newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) + a.strides[axis + 1:]

  try:
    return np.ndarray.__new__(np.ndarray, strides=newstrides,
                              shape=newshape, buffer=a, dtype=a.dtype)
  except TypeError:
    a = a.copy()
    # Shape doesn't change but strides does
    newstrides = a.strides[:axis] + ((frame_length - overlap) * s, s) \
    + a.strides[axis + 1:]
    return np.ndarray.__new__(np.ndarray, strides=newstrides,
                              shape=newshape, buffer=a, dtype=a.dtype)

# ===========================================================================
# Fourier transform
# ===========================================================================
def get_energy(frames, log=True):
  """ Calculate frame-wise energy

  Parameters
  ----------
  frames: ndarray
      framed signal with shape (nb_frames x window_length)
  log: bool
      if True, return log energy of each frames

  Return
  ------
  E : ndarray [shape=(nb_frames,), dtype=float32]
  """
  log_energy = (frames**2).sum(axis=1)
  log_energy = np.where(log_energy == 0., np.finfo(np.float32).eps,
                        log_energy)
  if log:
    log_energy = np.log(log_energy)
  return np.expand_dims(log_energy.astype('float32'), -1)

def stft(y,
         frame_length=None, step_length=None, n_fft=None,
         window='hann', scale=None,
         padding=False, energy=False):
  """Short-time Fourier transform (STFT)

  Returns a complex-valued matrix D such that
      `np.abs(D[f, t])` is the magnitude of frequency bin `f`
      at frame `t`

      `np.angle(D[f, t])` is the phase of frequency bin `f`
      at frame `t`

  Parameters
  ----------
  y : numpy.ndarray [shape=(n_samples,)] or [shape=(n_frames, frame_length)]
      the input signal (audio time series) or framed signal in case
      of 2-D matrix with `shape[1] > 2`

  frame_length: int
      number of samples point for 1 frame

  step_length: int
      number of samples point for 1 step (when shifting the frames)
      If unspecified, defaults `frame_length / 4`.

  n_fft: int > 0 [scalar]
      FFT window size
      If not provided, uses the smallest power of 2 enclosing `frame_length`.

  window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a vector or array of length `n_fft`

  scale : {None, float}
      re-scale the STFT matrix after windowing, it is important factor
      for reconstruct original signal using iSTFT

  padding: boolean
      - If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * step_length]`.
      - If `False`, then `D[:, t]` begins at `y[t * step_length]`

  energy: bool
      if True, return log-frame-wise energy

  Returns
  -------
  D : np.ndarray [shape=(t, 1 + n_fft/2), dtype=complex64]
      STFT matrix
  log_energy : ndarray [shape=(t,), dtype=float32]
      (log) energy of each frame

  Note
  ----
  This implementation have been tested to achieve slightly better speed
  than scipy implementation
  """
  # ====== check input signal ====== #
  if y.ndim == 2 and y.shape[1] > 2:
    y, frames = None, y
  else:
    y, frames = y, None
  # ====== check others ====== #
  # check frame_length
  if frame_length is None:
    if frames is not None:
      frame_length = frames.shape[1]
    else:
      raise ValueError("When `frame_length` is None, `frames` must be provided")
  else:
    frame_length = int(frame_length)
  # check step_length
  if step_length is None:
    step_length = frame_length // 4
  else:
    step_length = int(step_length)
  # check n_fft
  if n_fft is None:
    n_fft = int(2**np.ceil(np.log(frame_length) / np.log(2.0)))
  elif n_fft < frame_length:
    raise ValueError('n_fft must be greater than or equal to `frame_length`.')
  # ====== doing framing on raw signal ====== #
  if frames is None:
    # check if padding zeros
    if padding:
      y = np.pad(y, int(frame_length // 2), mode='constant')
    # framing the signal
    shape = y.shape[:-1] + (y.shape[-1] - frame_length + 1, frame_length)
    strides = y.strides + (y.strides[-1],)
    y_frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    if y_frames.ndim > 2:
      y_frames = np.rollaxis(y_frames, 1)
    # [n_frames, frame_length]
    y_frames = y_frames[::step_length]
  else:
    y_frames = frames
  # ====== prepare the window function ====== #
  if window is not None:
    fft_window = get_window(window, frame_length, periodic=True
        ).reshape(1, -1)
    y_frames = fft_window * y_frames
    # scaling factor
    scale = np.sqrt(1.0 / fft_window.sum()**2) if scale is None else float(scale)
  else:
    scale = np.sqrt(1.0 / frame_length**2) if scale is None else float(scale)
  # ====== calculate frames energy ====== #
  if energy:
    log_energy = get_energy(y_frames, log=True).astype('float32')
  # ====== STFT matrix ====== #
  # norm='ortho' ?
  S = np.fft.rfft(a=y_frames, n=n_fft, axis=-1)
  # this scale is important for iSTFT reconstruct original signal
  if scale is not None:
    S *= scale
  # return in form (t, d)
  if energy:
    return S, log_energy
  return S


def istft(stft_matrix, frame_length, step_length=None,
          window='hann', padding=False):
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
  frame_length: int
      number of samples point for 1 frame
  step_length: int
      number of samples point for 1 step (when shifting the frames)
      If unspecified, defaults `frame_length / 4`.
  window      : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a user-specified window vector of length `n_fft`
  padding: boolean
      - If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * step_length]`.
      - If `False`, then `D[:, t]` begins at `y[t * step_length]`

  Returns
  -------
  y : np.ndarray [shape=(n,), dtype=float32]
      time domain signal reconstructed from `stft_matrix`
  """
  # ====== check arguments ====== #
  frame_length = int(frame_length)
  if step_length is None:
    step_length = frame_length // 4
  else:
    step_length = int(step_length)
  n_fft = 2 * (stft_matrix.shape[1] - 1)
  # ====== use scipy here ====== #
  try:
    from scipy.signal.spectral import istft as _istft
  except ImportError:
    raise RuntimeError("`istft` requires scipy version >= 0.19")
  return _istft(stft_matrix, fs=1.0, window=window,
                nperseg=frame_length, noverlap=frame_length - step_length, nfft=n_fft,
                input_onesided=True, boundary=padding,
                time_axis=0, freq_axis=-1)[-1]

def power_spectrogram(S, power=2.0):
  """ Extracting power spectrum from a complex-STFT array

  Parameters
  ----------
  S : array [nb_samples, n_fft]
    complex type or real type
  power : float
    factor for converting spectrogram to power spectrum

  Note
  ----
  To convert power spectrum to Decibel (db)
  >>> signal.power2db(power_spectrogram(S, power=2.0), top_db=80.0)

  """
  power = int(power)
  # ====== extract the magnitude spectrogram ====== #
  if 'complex' in str(S.dtype): # get magnitude from STFT
    spec = np.abs(S)
  else:
    spec = S
  # ====== power ====== #
  if power > 1:
    spec = np.power(spec, power)
  return spec

def mels_spectrogram(spec, sr, n_mels,
                     fmin=64, fmax=None, top_db=80.0):
  """ Extracting mel-filter bands from power spectrum
  (i.e. the output from function
  `odin.preprocessing.signal.power_spectrogram`)

  Parameters
  ----------
  spec : array [nb_samples, n_fft]
    power spectrum array
  sr : int
    sample rate
  fmin: int
      min frequency for mel-filter bands
  fmax: int, or None
      maximum frequency for mel-filter bands.
      If None, usng `sr / 2` as fmax
  top_db: int
      maximum deciben

  """
  # ====== check arguments ====== #
  n_fft = int(2 * (spec.shape[1] - 1))
  # check fmax
  if sr is None and fmax is None:
    fmax = 4000
  else:
    fmax = sr // 2 if fmax is None else int(fmax)
  # check fmin
  fmin = int(fmin)
  if fmin >= fmax:
    raise ValueError("fmin must < fmax, but fmin=%d and fmax=%d" %
                     (fmin, fmax))
  # ====== mel transform ====== #
  mel_basis = mel_filters(sr,
      n_fft=n_fft, n_mels=24 if n_mels is None else int(n_mels),
      fmin=fmin, fmax=fmax)
  # transpose to (nb_samples; nb_mels)
  mel_spec = np.dot(mel_basis, spec.T)
  mel_spec = mel_spec.T
  mel_spec = power2db(mel_spec, top_db=top_db)
  return mel_spec

def ceps_spectrogram(mspec, n_ceps, remove_first_coef=True):
  """ Compute the MFCCs coefficients (cepstrum analysis)
  from extracted mel-filter bands spectrogram
  (i.e. output from `odin.preprocessing.signal.mels_spectrogram`)

  Parameters
  ----------
  mspec : array [nb_samples, n_fft]
    mels-spectrogram array
  n_ceps : int
    number of ceptrum for cepstral analysis
  remove_first_coef : bool
    if True remove the first coefficient of the extracted MFCCs

  """
  if remove_first_coef:
    n_ceps = int(n_ceps) + 1
    dct_basis = dct_filters(n_ceps, mspec.shape[1])
    mfcc = np.dot(dct_basis, mspec.T)[1:, :].T
  else:
    n_ceps = int(n_ceps)
    dct_basis = dct_filters(n_ceps, mspec.shape[1])
    mfcc = np.dot(dct_basis, mspec.T).T
  return mfcc

def spectra(sr, frame_length, y=None, S=None,
            step_length=None, n_fft=512, window='hann',
            n_mels=None, n_ceps=None,
            fmin=64, fmax=None,
            top_db=80.0, power=2.0, log=True,
            padding=False):
  """Compute spectra information from STFT matrix or a power spectrogram,
  The extracted spectra include:
  * log-power spectrogram
  * mel-scaled spectrogram.
  * MFCC (cepstrum analysis)

  If a spectrogram input `S` is provided, then it is mapped directly onto
  the mel basis `mel_f` by `mel_f.dot(S)`.

  If a time-series input `y, sr` is provided, then its magnitude spectrogram
  `S` is first computed, and then mapped onto the mel scale by
  `mel_f.dot(S**power)`.  By default, `power=2` operates on a power spectrum.

  Parameters
  ----------
  sr : number > 0 [scalar]
      sampling rate of `y`
  frame_length: int
      number of samples point for 1 frame
  y : np.ndarray [shape=(n,)] or None
      audio time-series
  S : np.ndarray [shape=(d, t)]
      spectrogram or complex STFT
  step_length: int
      number of samples point for 1 step (when shifting the frames)
      If unspecified, defaults `frame_length / 4`.
  n_fft : int > 0 [scalar]
      length of the FFT window
  window      : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a user-specified window vector of length `n_fft`
  n_mels: int, or None
      number of mel-filter bands
  n_ceps: int, or None
      number of ceptrum for cepstral analysis
  fmin: int
      min frequency for mel-filter bands
  fmax: int, or None
      maximum frequency for mel-filter bands.
      If None, usng `sr / 2` as fmax
  top_db: int
      maximum deciben
  power : float > 0 [scalar]
      Exponent for the magnitude spectrogram.
      e.g., 1 for energy (or magnitude), 2 for power, etc.
  log: bool
      if True, convert all power spectrogram to DB
  padding : bool
      - If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * step_length]`.
      - If `False`, then `D[:, t]` begins at `y[t * step_length]`

  Returns
  -------
  S : np.ndarray [shape=(t, nb_melfilters)]
      Mel spectrogram

  Note
  ----
  `log` and `power` don't work for `sptk` backend
  This method is optimized for speed and memory when extracting
  a sequence of speech features at once.

  """
  mel_spec = None
  mfcc = None
  log_energy = None
  # ====== compute STFT if needed ====== #
  if S is None:
    S, log_energy = stft(y, frame_length=frame_length,
                         step_length=step_length, n_fft=n_fft,
                         window=window, padding=padding, energy=True)
  n_fft = int(2 * (S.shape[1] - 1))
  # ====== check arguments ====== #
  power = int(power)
  # check fmax
  if sr is None and fmax is None:
    fmax = 4000
  else:
    fmax = sr // 2 if fmax is None else int(fmax)
  # check fmin
  fmin = int(fmin)
  if fmin >= fmax:
    raise ValueError("fmin must < fmax, but fmin=%d and fmax=%d" %
                     (fmin, fmax))
  # ====== extract the basic spectrogram ====== #
  if 'complex' in str(S.dtype): # get magnitude from STFT
    spec = np.abs(S)
  if power > 1:
    spec = np.power(spec, power)
  # ====== extrct mel-filter-bands features ====== #
  if n_mels is not None or n_ceps is not None:
    mel_spec = mels_spectrogram(spec, sr, n_mels)
  # ====== extract cepstrum features ====== #
  # extract MFCC
  if n_ceps is not None:
    mfcc = ceps_spectrogram(mel_spec, n_ceps)
  # applying log to convert to db
  if log:
    spec = power2db(spec, top_db=top_db)
  # ====== return result ====== #
  results = {}
  results['spec'] = spec.astype('float32')
  results['energy'] = log_energy
  results['mspec'] = None if mel_spec is None else mel_spec.astype('float32')
  results['mfcc'] = None if mfcc is None else mfcc.astype('float32')
  return results


# ===========================================================================
# invert spectrogram
# ===========================================================================
def ispec(spec, frame_length, step_length=None, window="hann",
          nb_iter=48, normalize=True,
          db=False, padding=False,
          de_preemphasis=0.97):
  """ Using Griffin-Lim algorithm for Inverting
  (power) spectrogram back to raw waveform

  Parameters
  ----------
  spec : np.ndarray [shape=(t, n_fft / 2 + 1)]
      magnitude, power, or DB spectrogram of STFT
  frame_length: int
      number of samples point for 1 frame
  step_length: int
      number of samples point for 1 step (when shifting the frames)
      If unspecified, defaults `frame_length / 4`.
  window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a vector or array of length `n_fft`
  nb_iter: int
      number of iteration, the higher the better audio quality
  db: bool
      if the given spectrogram is in decibel (dB) units (used logarithm)
  normalize: bool
      normalize output raw signal to have mean=0., and std=1.

  """
  spec = spec.astype('float64')
  # ====== check arguments ====== #
  frame_length = int(frame_length)
  if step_length is None:
    step_length = frame_length // 4
  else:
    step_length = int(step_length)
  if frame_length < step_length:
    raise ValueError('frame_length=%d < step_length=%d' %
      (frame_length, step_length))
  n_fft = (spec.shape[1] - 1) * 2
  # ====== convert to power spectrogram ====== #
  if db:
    spec = db2power(spec)
  # ====== iterative estmate best phase ====== #
  X_best = copy.deepcopy(spec)
  for i in range(nb_iter):
    X_t = istft(X_best, frame_length=frame_length, step_length=step_length,
                window=window, padding=padding)
    est = stft(X_t, frame_length=frame_length, step_length=step_length,
               n_fft=n_fft, window=window, padding=padding, energy=False)
    phase = est / np.maximum(1e-8, np.abs(est))
    X_best = spec * phase
  X_t = istft(X_best, frame_length=frame_length, step_length=step_length,
              window=window, padding=padding)
  # ====== de-preemphasis ====== #
  if isinstance(de_preemphasis, Number) and 0. < de_preemphasis < 1.:
    X_t = signal.lfilter([1], [1, -de_preemphasis], X_t)
  y = np.real(X_t)
  if normalize:
    y = y[1000:-1000]
    y = (y - y.mean()) / y.std()
  return y

# ===========================================================================
# F0 analysis
# ===========================================================================
def pitch_track(y, sr, step_length, fmin=60.0, fmax=260.0, threshold=0.3,
                otype="pitch", algorithm='swipe'):
  """

  Parameters
  ----------
  y : array
      A whole audio signal
  sr : int
      Sampling frequency.
  step_length : int
      Hop length.
  fmin : float, optional
      Minimum fundamental frequency. Default is 60.0
  fmax : float, optional
      Maximum fundamental frequency. Default is 260.0
  threshold : float, optional
      Voice/unvoiced threshold. Default is 0.3 (as suggested for SWIPE)
      Threshold >= 1.0 is suggested for RAPT
  otype : str or int, optional
      Output format
          (0) pitch
          (1) f0
          (2) log(f0)
      Default is f0.
  algorithm: 'swipe', 'rapt', 'avg'
      SWIPE - A Saw-tooth Waveform Inspired Pitch Estimation.
      RAPT - a robust algorithm for pitch tracking.
      avg - apply swipe and rapt at the same time, then take average.
      Default is 'SWIPE'

  Note
  ----
  by default, the pitch tracking algorithm always center the signal before
  processing
  """
  try:
    import pysptk
  except ImportError:
    raise RuntimeError("Pitch tracking requires pysptk library.")
  # ====== check arguments ====== #
  algorithm = str(algorithm).lower()
  if algorithm not in ('rapt', 'swipe', 'avg'):
    raise ValueError("'algorithm' argument must be: 'rapt', 'swipe' or 'avg'")
  otype = str(otype)
  if otype not in ('f0', 'pitch'):
    raise ValueError("Support 'otype' include: 'f0', 'pitch'")
  # ====== Do it ====== #
  sr = int(sr)
  if algorithm == 'avg':
    y1 = pysptk.swipe(y.astype(np.float64), fs=sr, hopsize=step_length,
                      threshold=threshold, min=fmin, max=fmax, otype=otype)
    y2 = pysptk.rapt(y.astype(np.float32), fs=sr, hopsize=step_length,
                     voice_bias=threshold, min=fmin, max=fmax, otype=otype)
    y = (y1 + y2) / 2
  elif algorithm == 'swipe':
    y = pysptk.swipe(y.astype(np.float64), fs=sr, hopsize=step_length,
                     threshold=threshold, min=fmin, max=fmax, otype=otype)
  else:
    y = pysptk.rapt(y.astype(np.float32), fs=sr, hopsize=step_length,
                    voice_bias=threshold, min=fmin, max=fmax, otype=otype)
  return y.astype('float32')
