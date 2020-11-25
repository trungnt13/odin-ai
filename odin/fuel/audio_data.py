from __future__ import absolute_import, division, print_function

import os
import tarfile
from urllib.request import urlretrieve
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from odin.utils import as_tuple, get_all_files, get_datasetpath
from odin.utils.crypto import md5_checksum


class AudioFeatureLoader():
  r"""
  Arguments:
    frame_length: An integer scalar Tensor. The window length in samples.
    frame_step: An integer scalar Tensor. The number of samples to step.
    fft_length: An integer scalar Tensor. The size of the FFT to apply.
      If not provided, uses the smallest power of 2 enclosing frame_length.
    sample_rate: An integer or float Tensor. Samples per second of the input
      signal used to create the spectrogram. Used to figure out the frequencies
      corresponding to each spectrogram bin, which dictates how they are mapped
      into the mel scale.
    power: An integer. Scale of input tensor ('power' or 'magnitude'). The
        power being the elementwise square of the magnitude.
        `power=1.0` for energy or magnitude, `power=2.0` for power
    top_db (float, optional): minimum negative cut-off in decibels.
        A reasonable number is 80. (Default: ``80``)
    window_fn: A callable that takes a window length and a dtype keyword
      argument and returns a [window_length] Tensor of samples in the
      provided datatype.
      If set to None, no windowing is used.
    pad_end: Whether to pad the end of signals with zeros when the provided
      frame length and step produces a frame that lies partially past its end.
    num_mel_bins: Python int. How many bands in the resulting mel spectrum.
    num_cepstral: Python int. Number of Mel-frequency Cepstral Coefficients
      (i.e. MFCCs)
    log_mels: Python bool. Whether to use log-mel spectrograms instead of
      db-scaled
    lower_edge_hertz: Python float. Lower bound on the frequencies to be
      included in the mel spectrum. This corresponds to the lower edge of the
      lowest triangular band.
    upper_edge_hertz: Python float. The desired top edge of the highest
      frequency band.
    dtype: The DType of the result matrix. Must be a floating point type.
  """

  def __init__(self,
               frame_length=256,
               frame_step=80,
               fft_length=None,
               sample_rate=8000,
               power=2.0,
               top_DB=80.0,
               window_fn=tf.signal.hann_window,
               pad_end=False,
               num_mel_bins=20,
               num_cepstral=None,
               log_mels=False,
               lower_edge_hertz=125.0,
               upper_edge_hertz=3800.0,
               dtype=tf.float32,
               save_path=None,
               seed=8):
    ### preprocessing the arguments
    if save_path is None:
      save_path = get_datasetpath(name='audio_datasets', override=False)
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    if fft_length is None:
      fft_length = frame_length
    fft_length = 2**int(np.ceil(np.log2(fft_length)))
    ### store
    self.save_path = save_path
    self.frame_length = int(frame_length)
    self.frame_step = int(frame_step)
    self.fft_length = int(fft_length)
    self.log_mels = bool(log_mels)
    self.power = power
    self.top_DB = top_DB
    self.window_fn = window_fn
    self.pad_end = pad_end
    self.num_mel_bins = num_mel_bins
    self.num_cepstral = num_cepstral
    self.sample_rate = int(sample_rate)
    self.lower_edge_hertz = lower_edge_hertz
    self.upper_edge_hertz = upper_edge_hertz
    self.dtype = dtype
    self.seed = seed
    ### mel-frequency
    self.mel_weight = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.num_mel_bins,
        num_spectrogram_bins=self.fft_length // 2 + 1,
        sample_rate=self.sample_rate,
        lower_edge_hertz=self.lower_edge_hertz,
        upper_edge_hertz=self.upper_edge_hertz,
        dtype=self.dtype)

  def create_dataset(self,
                     file_list,
                     feature='mels',
                     channels=0,
                     batch_size=32,
                     drop_remainder=False,
                     shuffle=None,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     max_length=300,
                     random_clipping=True,
                     return_path=False):
    r""" Create a tensorflow dataset extracting spectrogram from a list of
      audio file

    Arguments:
      file_list: List of path to audio files
      feature: {'spec', 'mels', 'mfcc'}
      channels: {0, 1, [0, 1]}, which channel(s) will be selected for
        processing
      batch_size: A tf.int64 scalar tf.Tensor, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: A tf.bool scalar tf.Tensor, representing whether the
        last batch should be dropped in the case it has fewer than batch_size
        elements; the default behavior is not to drop the smaller batch.
      shuffle: A tf.int64 scalar tf.Tensor, representing the number of elements
        from this dataset from which the new dataset will sample.
        If `None` or smaller or equal 0, turn off shuffling
      prefetch:  A tf.int64 scalar tf.Tensor, representing the maximum number
        of elements that will be buffered when prefetching.
      cache: A tf.string scalar tf.Tensor, representing the name of a directory
        on the filesystem to use for caching elements in this Dataset. If a
        filename is not provided, the dataset will be cached in memory.
        If `None`, turn off caching
      parallel: A tf.int32 scalar tf.Tensor, representing the number elements
        to process asynchronously in parallel. If not specified, elements will
        be processed sequentially. If the value `tf.data.experimental.AUTOTUNE`
        is used, then the number of parallel calls is set dynamically based
        on available CPU.
      max_length: A Integer. Maximum length (number of frames) for an
        utterance.
      random_clipping: A Boolean. Perform random clipping within the provided
        maximum length.
      return_path: A Boolean. Return the audio path together with the features.
    """
    feature = str(feature).lower()
    assert feature in ('mels', 'spec', 'mfcc'), \
      "feature must be one of: 'mels', 'spec', 'mfcc'"
    channels = tf.nest.flatten(channels)
    ds_org = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle is not None and shuffle != 0:
      ds_org = ds_org.shuffle(shuffle,
                              seed=self.seed,
                              reshuffle_each_iteration=True)
    ### read audio
    ds = ds_org.map(self.read_wav)
    if len(channels) == 2:
      ds1 = ds.map(self.channel0)
      ds2 = ds.map(self.channel1)
      ds = ds1.concatenate(ds2)
      ds_org = ds_org.concatenate(ds_org)
    else:
      ds = ds.map(self.channel0 if channels[0] == 0 else self.channel1)
    ### extract features
    def map_spec(y):
      s = self.stft(y)
      if feature == 'spec':
        s = self.amplitude_to_DB(self.magnitude(s))
      elif feature == 'mels':
        s = self.melspec(s)
      elif feature == 'mfcc':
        s = self.mfccs(s)
      return s

    ds = ds.map(map_spec, num_parallel_calls=parallel)
    if cache is not None:
      ds = ds.cache(cache)
    ### batching
    def map_batch(s):
      if random_clipping:
        start = tf.maximum(tf.shape(s)[0] - max_length, 0)
        start = tf.random.uniform((), 0, start + 1, dtype=tf.int32)
      else:
        start = 0
      s = s[start:(start + max_length)]
      return s

    ds = ds.map(map_batch, num_parallel_calls=None)
    dim = tf.data.experimental.get_structure(ds).shape[-1]
    ds = ds.padded_batch(batch_size,
                         padded_shapes=[max_length, dim],
                         drop_remainder=drop_remainder)
    ### post-processing
    if return_path:
      ds = tf.data.Dataset.zip(
          (ds, ds_org.batch(batch_size, drop_remainder=drop_remainder)))
    ### prefetch
    if prefetch is not None and prefetch != 0:
      ds = ds.prefetch(prefetch)
    return ds

  def stft(self, y):
    return tf.signal.stft(y,
                          frame_length=self.frame_length,
                          frame_step=self.frame_step,
                          fft_length=self.fft_length,
                          window_fn=self.window_fn,
                          pad_end=self.pad_end)

  def melspec(self, s):
    r""" linear spectrogram scale to [log-magnitude] mel-scale
    Log scale of mel-spectrogram is more stabilized
    """
    if s.dtype in (tf.complex64, tf.complex128):
      s = self.magnitude(s)
    mel = tf.matmul(s, self.mel_weight)
    if self.log_mels:
      mel = tf.math.log(mel + 1e-6)
    else:
      mel = self.amplitude_to_DB(mel)
    return mel

  def mfccs(self, s):
    r""" Mel-frequency cepstral coefficients """
    if s.dtype in (tf.complex64, tf.complex128):
      s = self.melspec(s)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(s)
    if self.num_cepstral is not None:
      mfcc = mfcc[:, int(self.num_cepstral)]
    return mfcc

  def read_wav(self, path):
    binary = tf.io.read_file(path)
    y, sr = tf.audio.decode_wav(binary)
    tf.assert_equal(sr, self.sample_rate)
    return y

  def channel0(self, y):
    return y[:, 0]

  def channel1(self, y):
    return y[:, 1]

  def magnitude(self, s):
    r""" Magnitude of complex stft transform """
    assert s.dtype in (tf.complex64, tf.complex128)
    mag = tf.math.abs(s)
    if self.power > 1.0:
      mag = tf.math.pow(mag, self.power)
    return mag

  def amplitude_to_DB(self, s):
    r""" Convert the magnitude spectrogram to Deciben (DB) """
    multiplier = 10.0 if self.power == 2.0 else 20.0
    amin = 1e-10
    ref_value = 1.0
    loge10 = tf.math.log(tf.constant(10., dtype=s.dtype))
    s_db = multiplier * (tf.math.log(tf.maximum(s, amin)) / loge10)
    s_db -= multiplier * (tf.math.log(max(amin, ref_value)) / loge10)
    if self.top_DB is not None:
      s_db = tf.maximum(s_db, tf.reduce_max(s_db) - self.top_DB)
    return s_db

  # ===========================================================================
  # Load dataset
  # ===========================================================================
  def load_fsdd(self):
    r""" Free Spoken Digit Dataset
      A simple audio/speech dataset consisting of recordings of spoken digits
      in wav files at 8kHz. The recordings are trimmed so that they have near
      minimal silence at the beginnings and ends.

    Sample rate: 8,000

    Reference:
      Link: https://github.com/Jakobovski/free-spoken-digit-dataset
    """
    LINK = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip"
    MD5 = "471b0df71a914629e2993300c1ccf33f"
    save_path = os.path.join(self.save_path, 'FSDD')
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    # ====== download zip dataset ====== #
    if md5_checksum(''.join(sorted(os.listdir(save_path)))) != MD5:
      zip_path = os.path.join(save_path, 'FSDD.zip')
      urlretrieve(url=LINK, filename=zip_path)
      try:
        with ZipFile(zip_path, mode='r', compression=ZIP_DEFLATED) as zf:
          wav_files = [name for name in zf.namelist() if '.wav' == name[-4:]]
          for name in wav_files:
            data = zf.read(name)
            name = os.path.basename(name)
            with open(os.path.join(save_path, name), 'wb') as f:
              f.write(data)
      finally:
        os.remove(zip_path)
    # ====== get all records ====== #
    all_name = os.listdir(save_path)
    all_files = sorted([os.path.join(save_path, name) for name in all_name])
    all_speakers = list(set(i.split('_')[1] for i in all_name))
    # ====== splitting train, test ====== #
    rand = np.random.RandomState(seed=self.seed)
    rand.shuffle(all_speakers)
    train_spk = all_speakers[:-1]
    test_spk = all_speakers[-1:]
    train_files = [
        i for i in all_files if os.path.basename(i).split('_')[1] in train_spk
    ]
    test_files = [
        i for i in all_files if os.path.basename(i).split('_')[1] in test_spk
    ]
    rand.shuffle(train_files)
    rand.shuffle(test_files)
    return train_files, test_files

  def load_command(self):
    r""" Warden P. Speech Commands: A public dataset for single-word speech
      recognition, 2017. Available from
      http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

    Sample rate: 16,000

    Example:
      ds = AudioFeatureLoader(sample_rate=16000,
                              frame_length=int(0.025 * 16000),
                              frame_step=int(0.005 * 16000))
      train, valid, test = ds.load_command()
      train = ds.create_dataset(train, max_length=40, return_path=True)
    """
    LINK = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    MD5 = "a08eb256cea8cbb427c6c0035fffd881"
    save_path = os.path.join(self.save_path, 'speech_commands')
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    audio_path = os.path.join(save_path, 'audio')
    audio_files = sorted(
        get_all_files(audio_path, filter_func=lambda x: '.wav' == x[-4:]))
    md5 = md5_checksum(''.join([os.path.basename(i) for i in audio_files]))
    # ====== Download and extract the data ====== #
    if md5 != MD5:
      zip_path = get_file(fname='speech_commands_v0.01.tar.gz',
                          origin=LINK,
                          outdir=save_path,
                          verbose=True)
      with tarfile.open(zip_path, 'r:gz') as tar:
        tar.extractall(audio_path)
    # ====== processing the audio file list ====== #
    audio_files = [i for i in audio_files if '_background_noise_' not in i]
    with open(os.path.join(audio_path, 'validation_list.txt'), 'r') as f:
      valid_list = {i.strip(): 1 for i in f}
    with open(os.path.join(audio_path, 'testing_list.txt'), 'r') as f:
      test_list = {i.strip(): 1 for i in f}
    train_files = []
    valid_files = []
    test_files = []
    for f in audio_files:
      name = '/'.join(f.split('/')[-2:])
      if name in valid_list:
        valid_files.append(f)
      elif name in test_list:
        test_files.append(f)
      else:
        train_files.append(f)
    return train_files, valid_files, test_files
