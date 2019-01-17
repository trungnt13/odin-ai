from __future__ import print_function
import os
import sys
import base64
import shutil
import inspect
from six import add_metaclass
from zipfile import ZipFile, ZIP_DEFLATED
from abc import ABCMeta, abstractproperty

import numpy as np

from odin.utils.crypto import unzip_aes
from odin.utils import (get_file, get_script_path, ctext, get_datasetpath,
                        string_normalize)
from odin.fuel.dataset import Dataset

# ===========================================================================
# Helper
# ===========================================================================
def parse_dataset(dataset_name):
  fn_norm = lambda x: string_normalize(x, lower=True,
                                       remove_whitespace='',
                                       remove_non_alphanumeric=True)
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
  match_dataset = [name for name, _ in all_datasets
                   if name == dataset_name]
  if len(match_dataset) == 0:
    match_dataset = [name for name, _ in all_datasets
                     if dataset_name == name[:len(dataset_name)]]
  if len(match_dataset) == 0:
    raise RuntimeError("Cannot find dataset with name '%s', "
      "all available datasets are: %s" % (
          dataset_name,
          ', '.join([i[0] for i in all_datasets])))
  if len(match_dataset) > 1:
    raise RuntimeError("Found multiple dataset for name: '%s', "
      "all the candidates are: %s" % (
          dataset_name,
          ', '.join([i for i in match_dataset])))
  # ====== extract the found dataset ====== #
  match_dataset = match_dataset[0]
  dataset = [ds for name, ds in all_datasets
             if name == match_dataset][0]
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
    import traceback; traceback.print_exc()
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
    return os.path.join(DataLoader.BASE_DIR,
                        clazz.get_name(ext) + '.zip')

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

class FSDD(object):
  """ Free Spoken Digit Dataset
  A simple audio/speech dataset consisting of recordings of
  spoken digits in wav files at 8kHz. The recordings are
  trimmed so that they have near minimal silence at the
  beginnings and ends.

  Return
  ------
  list : contain path to all .wav files of the dataset
  numpy.ndarray : the numpy array of loaded `metadata.csv`

  Reference
  ---------
  Link: https://github.com/Jakobovski/free-spoken-digit-dataset
  """
  LINK = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.7.zip'

  @classmethod
  def load(clazz):
    """ Return
    records: list of all path to recorded audio files
    metadata: numpy.ndarray
    """
    dat_path = get_datasetpath(name='FSDD', override=False)
    tmp_path = dat_path + '_tmp'
    zip_path = dat_path + '.zip'
    # ====== download zip dataset ====== #
    if not os.path.exists(dat_path) or \
    len(os.listdir(dat_path)) != 1501:
      if not os.path.exists(zip_path):
        get_file(fname='FSDD.zip', origin=FSDD.LINK, outdir=get_datasetpath())
      if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
      unzip_folder(zip_path=zip_path, out_path=tmp_path, remove_zip=True)
      tmp_path = os.path.join(tmp_path, os.listdir(tmp_path)[0])
      # ====== get all records ====== #
      record_path = os.path.join(tmp_path, 'recordings')
      all_records = [os.path.join(record_path, i)
                     for i in os.listdir(record_path)]
      for f in all_records:
        name = os.path.basename(f)
        shutil.copy2(src=f, dst=os.path.join(dat_path, name))
      # ====== copy the metadata ====== #
      meta_path = os.path.join(tmp_path, 'metadata.py')
      import imp
      meta = imp.load_source('metadata', meta_path).metadata
      assert len(set(len(i) for i in meta.values())) == 1, "Invalid metadata"
      rows = []
      for name, info in meta.items():
        info = sorted(info.items(), key=lambda x: x[0])
        header = ['name'] + [i[0] for i in info]
        rows.append([name] + [i[1] for i in info])
      with open(os.path.join(dat_path, 'metadata.csv'), 'w') as f:
        for r in [header] + rows:
          f.write(','.join(r) + '\n')
    # ====== clean ====== #
    if os.path.exists(tmp_path):
      shutil.rmtree(tmp_path)
    # ====== return dataset ====== #
    all_files = [os.path.join(dat_path, i)
                 for i in os.listdir(dat_path)
                 if '.wav' in i]
    meta = np.genfromtxt(os.path.join(dat_path, 'metadata.csv'),
                         dtype=str, delimiter=',')
    return all_files, meta

  @classmethod
  def get_dataset(clazz):
    """ Return
    records: list of all path to recorded audio files
    metadata: numpy.ndarray
    """
    return clazz.load()

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
  return np.genfromtxt(fname=path, dtype=str, delimiter=' ',
                       skip_header=1)

def load_sre_list():
  link = b'aHR0cHM6Ly9zMy5hbWF6b25hd3MuY29tL2FpLWRhdGFzZXRzL1NSRV9GSUxFUy56aXA=\n'
  link = str(base64.decodebytes(link), 'utf-8')
  ds_path = get_datasetpath(name='SRE_FILES', root='~',
                            is_folder=False, override=False)
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
  ds_path = get_datasetpath(name='voxceleb_lists', root='~',
                            is_folder=False, override=False)
  if not os.path.exists(ds_path):
    path = get_file(fname=os.path.basename(link),
                    origin=link,
                    outdir=get_datasetpath(root='~'))
    unzip_folder(zip_path=path, out_path=os.path.dirname(path),
                 remove_zip=True)
  return Dataset(ds_path, read_only=True)
