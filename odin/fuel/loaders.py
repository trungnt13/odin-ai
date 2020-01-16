from __future__ import print_function

import base64
import inspect
import os
import shutil
import sys
from abc import ABCMeta, abstractproperty
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf
from six import add_metaclass

from odin.fuel.dataset import Dataset
from odin.utils import (as_tuple, ctext, get_datasetpath, get_file,
                        get_script_path, string_normalize)
from odin.utils.crypto import md5_checksum, unzip_aes


# ===========================================================================
# Helper
# ===========================================================================
def parse_dataset(dataset_name):
  fn_norm = lambda x: string_normalize(
      x, lower=True, remove_whitespace='', remove_non_alphanumeric=True)
  dataset_name = fn_norm(dataset_name)
  # ====== get all datasets ====== #
  all_datasets = []
  for name, member in inspect.getmembers(sys.modules[__name__]):
    if (isinstance(member, type) and issubclass(member, DataLoader)) \
    or 'load_' == name[:5]:
      name = name.replace('load_', '')
      name = fn_norm(name)
      all_datasets.append((name, member))
  # ====== search for best match ====== #
  match_dataset = [name for name, _ in all_datasets if name == dataset_name]
  if len(match_dataset) == 0:
    match_dataset = [
        name for name, _ in all_datasets
        if dataset_name == name[:len(dataset_name)]
    ]
  if len(match_dataset) == 0:
    raise RuntimeError("Cannot find dataset with name '%s', "
                       "all available datasets are: %s" %
                       (dataset_name, ', '.join([i[0] for i in all_datasets])))
  if len(match_dataset) > 1:
    raise RuntimeError("Found multiple dataset for name: '%s', "
                       "all the candidates are: %s" %
                       (dataset_name, ', '.join([i for i in match_dataset])))
  # ====== extract the found dataset ====== #
  match_dataset = match_dataset[0]
  dataset = [ds for name, ds in all_datasets if name == match_dataset][0]
  if 'load_' in match_dataset:
    return match_dataset()
  return dataset.load()


def unzip_folder(zip_path, out_path, remove_zip=True):
  if '.zip' not in zip_path:
    raise ValueError(".zip extension must be in the zip_path.")
  if not os.path.exists(zip_path):
    raise ValueError("Cannot find zip file at path: %s" % zip_path)
  try:
    zf = ZipFile(zip_path, mode='r', compression=ZIP_DEFLATED)
    zf.extractall(path=out_path)
    zf.close()
  except Exception:
    shutil.rmtree(out_path)
    import traceback
    traceback.print_exc()
  finally:
    if remove_zip:
      os.remove(zip_path)


@add_metaclass(ABCMeta)
class DataLoader(object):
  ORIGIN = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzLw==\n'
  BASE_DIR = get_datasetpath(root='~')

  def __init__(self):
    super(DataLoader, self).__init__()

  @classmethod
  def md5(clazz, ext=''):
    return None

  @classmethod
  def get_name(clazz, ext=''):
    name = clazz.__name__
    name = name if ext is None or len(ext) == 0 \
        else '_'.join([name, ext])
    return name

  @classmethod
  def get_zip_path(clazz, ext=''):
    return os.path.join(DataLoader.BASE_DIR, clazz.get_name(ext) + '.zip')

  @classmethod
  def get_ds_path(clazz, ext=''):
    return os.path.join(DataLoader.BASE_DIR, clazz.get_name(ext))

  @classmethod
  def get_link(clazz, ext=''):
    name = clazz.get_name(ext) + '.zip'
    path = base64.decodebytes(DataLoader.ORIGIN).decode() + name
    return path

  @classmethod
  def load(clazz, ext='', override=False):
    return clazz.get_dataset(ext=ext, override=override)

  @classmethod
  def get_dataset(clazz, ext='', override=False):
    # ====== all path ====== #
    name = clazz.get_name(ext) + '.zip'
    path = base64.decodebytes(DataLoader.ORIGIN).decode() + name
    zip_path = clazz.get_zip_path(ext)
    out_path = clazz.get_ds_path(ext)
    # ====== check out_path ====== #
    if os.path.isfile(out_path):
      raise RuntimeError("Found a file at path: %s, we need a folder "
                         "to unzip downloaded files." % out_path)
    elif os.path.isdir(out_path):
      if override or len(os.listdir(out_path)) == 0:
        shutil.rmtree(out_path)
      else:
        return Dataset(out_path, read_only=True)
    # ====== download the file ====== #
    if os.path.exists(zip_path) and override:
      os.remove(zip_path)
    if not os.path.exists(zip_path):
      get_file(name, path, DataLoader.BASE_DIR)
    # ====== upzip dataset ====== #
    unzip_folder(zip_path, out_path, remove_zip=True)
    ds = Dataset(out_path, read_only=True)
    md5_checksum = clazz.md5(ext=ext)
    if md5_checksum is not None:
      assert ds.md5 == md5_checksum, "MD5 checksum mismatch for dataset: %s" % ds.path
    return ds


# ===========================================================================
# Images dataset
# ===========================================================================
class MNIST(DataLoader):
  pass


class MNIST_original(DataLoader):
  pass


class FMNIST_original(DataLoader):
  pass


class MNIST_dropout(DataLoader):
  pass


class FMNIST_dropout(DataLoader):
  pass


class CIFAR10(DataLoader):
  pass


class CIFAR100(DataLoader):
  pass


# ===========================================================================
# AUdio dataset
# ===========================================================================
class TIDIGITS(DataLoader):
  """ Audio digits dataset
  Encrypted and required password for access
  """

  @classmethod
  def md5(clazz, ext=''):
    return '82e2d8df7c376341a1f6deb35acdd1b0c569f0af82fb3f0173' + \
    'd11af2b74780a6b27611fa1ad0aaf16f0d4f52bddc9be1640ac3881f1' + \
    'ad074ce34f59650702632f919301507166bc190620ef168d5ef89b983' + \
    'a428e38814f17af05a4bc51b9a913d1300d54239a6cfa7dd5d75bfdc1' + \
    '7d2d958e5c3b50b3da499cf5c5eab9ee574'

  @classmethod
  def get_dataset(clazz, ext='', override=False):
    # ====== all path ====== #
    name = clazz.get_name(ext) + '.zip'
    path = base64.decodebytes(DataLoader.ORIGIN).decode() + name
    zip_path = clazz.get_zip_path(ext)
    out_path = clazz.get_ds_path(ext)
    # ====== check out_path ====== #
    if os.path.isfile(out_path):
      raise RuntimeError("Found a file at path: %s, we need a folder "
                         "to unzip downloaded files." % out_path)
    elif os.path.isdir(out_path):
      if override or len(os.listdir(out_path)) == 0:
        shutil.rmtree(out_path)
      else:
        return Dataset(out_path, read_only=True)
    # ====== download the file ====== #
    if os.path.exists(zip_path) and override:
      os.remove(zip_path)
    if not os.path.exists(zip_path):
      get_file(name, path, DataLoader.BASE_DIR)
    # ====== upzip dataset ====== #
    unzip_aes(in_path=zip_path, out_path=out_path)
    ds = Dataset(out_path, read_only=True)
    if ds.md5 != clazz.md5():
      ds.close()
      shutil.rmtree(out_path)
      raise RuntimeError("Incorrect password for loading DIGITS dataset")
    else:
      os.remove(zip_path)
    return ds


class SPEECH_SAMPLES(DataLoader):

  @classmethod
  def get_dataset(clazz, ext='', override=False):
    # ====== all path ====== #
    name = clazz.get_name(ext) + '.zip'
    path = base64.decodebytes(DataLoader.ORIGIN).decode() + name
    zip_path = clazz.get_zip_path(ext)
    out_path = clazz.get_ds_path(ext)
    # ====== check out_path ====== #
    if os.path.isfile(out_path):
      raise RuntimeError("Found a file at path: %s, we need a folder "
                         "to unzip downloaded files." % out_path)
    elif os.path.isdir(out_path):
      if override or len(os.listdir(out_path)) == 0:
        shutil.rmtree(out_path)
      else:
        return Dataset(out_path, read_only=True)
    # ====== download the file ====== #
    if os.path.exists(zip_path) and override:
      os.remove(zip_path)
    if not os.path.exists(zip_path):
      get_file(name, path, DataLoader.BASE_DIR)
    # ====== upzip dataset ====== #
    unzip_aes(in_path=zip_path, out_path=out_path)
    ds = Dataset(out_path, read_only=True)
    if os.path.exists(zip_path):
      os.remove(zip_path)
    return ds


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
    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
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
                     batch_size=128,
                     drop_remainder=False,
                     shuffle=None,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=tf.data.experimental.AUTOTUNE,
                     max_length=300,
                     random_clipping=True,
                     return_path=False):
    r"""
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

  def load_fsdd(self):
    r""" Free Spoken Digit Dataset
      A simple audio/speech dataset consisting of recordings of spoken digits
      in wav files at 8kHz. The recordings are trimmed so that they have near
      minimal silence at the beginnings and ends.

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
      zip_path = get_file(fname='FSDD.zip',
                          origin=LINK,
                          outdir=save_path,
                          verbose=True)
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
    pass

  def load_yesno(self):
    LINK = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"


# ===========================================================================
# More experimental dataset
# ===========================================================================
class IRIS(DataLoader):
  pass


# ===========================================================================
# Speech synthesis
# ===========================================================================
class CMUarctic(DataLoader):
  pass


# ===========================================================================
# Others
# ===========================================================================
class MUSAN(DataLoader):
  pass


class openSMILEsad(DataLoader):
  """ This dataset contains 2 files:
  * lstmvad_rplp18d_12.net
  * rplp18d_norm.dat
  """
  pass


# ===========================================================================
# Others
# ===========================================================================
def load_glove(ndim=100):
  """ Automaticall load a MmapDict which contains the mapping
      (word -> [vector])
  where vector is the embedding vector with given `ndim`.
  """
  ndim = int(ndim)
  if ndim not in (50, 100, 200, 300):
    raise ValueError('Only support 50, 100, 200, 300 dimensions.')
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2dsb3ZlLjZCLiVkZA==\n'
  link = str(base64.decodebytes(link) % ndim, 'utf-8')
  fname = os.path.basename(link)
  embedding = get_file(fname, link, outdir=get_datasetpath(root='~'))
  return MmapDict(embedding, read_only=True)


def load_lre_sad():
  """
  key: 'LDC2017E23/data/eval/lre17_lqoyrygc.sph'
  value: [(1.99, 3.38), (8.78, 16.41)] (in second)
  """
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2xyZV9zYWQ=\n'
  link = str(base64.decodebytes(link), 'utf-8')
  path = get_file(fname=os.path.basename(link),
                  origin=link,
                  outdir=get_datasetpath(root='~'))
  return MmapDict(path=path, read_only=True)


def load_lre_list():
  """ The header include following column:
  * name: LDC2017E22/data/ara-acm/ar-20031215-034005_0-a.sph
  * lre: {'train17', 'eval15', 'train15', 'dev17', 'eval17'}
  * language: {'ara-arb', 'ara-ary', 'ara-apc', 'ara-arz', 'ara-acm',
               'eng-gbr', 'eng-usg', 'eng-sas',
               'fre-hat', 'fre-waf'
               'zho-wuu', 'zho-cdo', 'zho-cmn', 'zho-yue', 'zho-nan',
               'spa-lac', 'spa-eur', 'spa-car',
               'qsl-pol', 'qsl-rus',
               'por-brz'}
  * corpus: {'pcm', 'alaw', 'babel', 'ulaw', 'vast', 'mls14'}
  * duration: {'3', '30', '5', '15', '10', '20', '1000', '25'}

  Note
  ----
  Suggested namming scheme:
    `lre/lang/corpus/dur/base_name`
  """
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL2xyZV9saXN0LnR4dA==\n'
  link = str(base64.decodebytes(link), 'utf-8')
  path = get_file(fname=os.path.basename(link),
                  origin=link,
                  outdir=get_datasetpath(root='~'))
  return np.genfromtxt(fname=path, dtype=str, delimiter=' ', skip_header=1)


def load_sre_list():
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL1NSRV9GSUxFUy56aXA=\n'
  link = str(base64.decodebytes(link), 'utf-8')
  ds_path = get_datasetpath(name='SRE_FILES',
                            root='~',
                            is_folder=False,
                            override=False)
  if os.path.exists(ds_path) and len(os.listdir(ds_path)) != 24:
    shutil.rmtree(ds_path)
  if not os.path.exists(ds_path):
    path = get_file(fname=os.path.basename(link),
                    origin=link,
                    outdir=get_datasetpath(root='~'))
    unzip_folder(zip_path=path, out_path=ds_path, remove_zip=True)
  return Dataset(ds_path, read_only=True)


def load_voxceleb_list():
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL3ZveGNlbGViX2xpc3RzLnppcA==\n'
  link = str(base64.decodebytes(link), 'utf-8')
  ds_path = get_datasetpath(name='voxceleb_lists',
                            root='~',
                            is_folder=False,
                            override=False)
  if not os.path.exists(ds_path):
    path = get_file(fname=os.path.basename(link),
                    origin=link,
                    outdir=get_datasetpath(root='~'))
    unzip_folder(zip_path=path, out_path=os.path.dirname(path), remove_zip=True)
  return Dataset(ds_path, read_only=True)
