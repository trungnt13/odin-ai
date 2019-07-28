from __future__ import print_function, division, absolute_import

import os
import shutil
import pickle
from collections import OrderedDict, Mapping
from typing import Text, Any
from six import string_types
from six.moves import zip, range, cPickle

import numpy as np

from odin.utils import (get_file, Progbar, is_string,
                        ctext, as_tuple, eprint, wprint,
                        is_callable, flatten_list, UnitTimer)
from odin.fuel.databases import MmapDict, SQLiteDict
from odin.fuel.mmap_array import (
  read_mmaparray_header, MmapArray, MmapArrayWriter)

__all__ = [
    'Dataset',
]

# ===========================================================================
# Helper
# ===========================================================================
def _infer_separator(path):
  all_sep = ('\t', ' ', ';', ',')
  with open(path, 'r') as f:
    line = f.readline()
    line = line.strip()
    orig_length = len(line)
    for s in all_sep:
      if s not in line:
        continue
      if 1 < len(line.split(s)) < orig_length:
        return s
    raise RuntimeError("CSV file with the first line: `%s`, "
                       "cannot match separator in known list: `%s`"
                       % (line, str(all_sep)))

_audio_ext = ('.3gp', '.aa', '.aac', '.aax', '.act', '.aiff',
              '.amr', '.ape', '.au', '.awb', '.dct', '.dss',
              '.dvf', '.flac', '.gsm', '.ivs', '.m4a', '.m4b',
              '.m4p', '.mmf', '.mp3', '.mpc', '.msv', '.nsf',
              '.ogg,', '.opus', '.raw', '.sln', '.tta', '.vox',
              '.wav', '.wma', '.wv', '.webm', '.sph', '.pcm')

_image_ext = ('.tif', '.tiff', '.gif', '.jpeg', '.jpg', '.jif',
              '.jfif', '.jp2', '.jpx', '.j2k', '.j2c', '.fpx',
              '.pcd', '.png', '.pdf')

_ignore_files = ('.DS_Store',)

def _parse_data_descriptor(path, read_only):
  """ Return mapping: name -> (dtype, shape, Data, path) """
  if not os.path.isfile(path):
    return None
  file_ext = os.path.splitext(path)[-1].lower()
  file_name = os.path.basename(path)
  # ====== ignore ====== #
  if os.path.basename(path) in _ignore_files:
    return None
  # ====== audio file ====== #
  if file_ext in _audio_ext:
    return [(file_name, ('audio', 'unknown', None, path))]
  # ====== image file ====== #
  if file_ext in _image_ext:
    return [(file_name, ('image', 'unknown', None, path))]
  # ====== text file .txt ====== #
  if file_ext in ('.txt',):
    return [(file_name, ('txt', 'unknown', None, path))]
  # ====== check if is csv file ====== #
  if file_ext in ('.csv', '.tsv'):
    sep = _infer_separator(path)
    data = []
    # read by manually open file much faster than numpy.genfromtxt
    with open(path, 'r') as f:
      for line in f:
        line = line.strip()
        data.append(line.split(sep))
      data = np.array(data, dtype=str)
    return [('.'.join(file_name.split('.')[:-1]),
             ('csv', data.shape, data, path))]
  # ====== check if a file is Data ====== #
  try:
    dtype, shape = read_mmaparray_header(path)
    data = MmapArray(path)
    assert np.dtype(dtype) == data.dtype and shape == data.shape, \
      "Metadata mismatch for MmapArray"
    return [(file_name, (data.dtype, data.shape, data, path))]
  except Exception: # cannot read the header of MmapArray
    pass
  # ====== try to load pickle file if possible ====== #
  try: # try with unpickling
    with open(path, 'rb') as f:
      data = cPickle.load(f)
      shape_info = 0
      if hasattr(data, 'shape'):
        shape_info = data.shape
      elif hasattr(data, '__len__'):
        shape_info = len(data)
      return [(file_name, (str(data.dtype) if hasattr(data, 'dtype') else
                           type(data).__name__,
                           shape_info, data, path))]
  except cPickle.UnpicklingError:
    try: # try again with numpy load
      with open(path, 'rb') as f:
        data = np.load(f)
        return [(file_name,
        (str(data.dtype) if hasattr(data, 'dtype') else type(data).__name__,
         len(data) if hasattr(data, '__len__') else 0, data, path))]
    except Exception:
      pass
  # ====== load memmap dict ====== #
  try:
    data = MmapDict(path, read_only=read_only)
    return [(file_name, ('memdict', len(data), data, path))]
  except Exception as e:
    pass
  # ====== load SQLiteDict ====== #
  if '.db' in os.path.splitext(path)[1]:
    try:
      db = SQLiteDict(path, read_only=read_only)
      name = os.path.basename(path).replace('.db', '')
      return [(tab if tab != SQLiteDict._DEFAULT_TABLE else name,
               ('sqlite', len(db.set_table(tab)), db.as_table(tab), path))
              for tab in db.get_all_tables()]
    except Exception as e:
      pass
  # ====== unknown datatype ====== #
  return [(file_name, ('unknown', 'unknown', None, path))]


# ===========================================================================
# Datasets
# ===========================================================================
class Dataset(object):
  """ This Dataset can automatically parse memmap (created by MmapData),
  MmapDict, pickled dictionary and hdf5 files and keep tracking the changes.

  Any file name with "readme" prefix will be parsed as text and showed as
  readme.

  Support data type:
   - .txt or .csv files:
   -

  Note
  ----
  for developer: _data_map contains: name -> (dtype, shape, Data or pathtoData)
  readme included with the dataset should contain license information
  All the file with `.db` extension will be treat as SQLite data
  """

  __INSTANCES = {}

  def __new__(cls, *args, **kwargs):
    path = kwargs.get('path', None)
    if path is None:
      path = args[0]
    if not is_string(path):
      raise ValueError("`path` for Dataset must be string, but given "
                       "object with type: %s" % type(path))
    path = os.path.abspath(path)
    # Found old instance
    if path in Dataset.__INSTANCES:
      return Dataset.__INSTANCES[path]
    # new Dataset
    new_instance = super(Dataset, cls).__new__(cls)
    Dataset.__INSTANCES[path] = new_instance
    return new_instance

  def __init__(self, path, read_only=False, override=False):
    path = os.path.abspath(path)
    self.read_only = read_only
    self._readme_info = [ctext('README:', 'yellow'),
                         '------',
                         '  No information!']
    self._readme_path = None
    # flag to check cPickle called with protocol 2
    self._new_args_called = False
    # parse all data from path
    if path is not None:
      if override and os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print('Overrided old dataset at path:', path)
      if os.path.isfile(path) and '.zip' in os.path.basename(path):
        self._load_archive(path,
            extract_path=path.replace(os.path.basename(path), ''),
            read_only=read_only)
      else:
        self._set_path(path, self.read_only)
    else:
      raise ValueError('Invalid path for Dataset: %s' % path)

  def _set_path(self, path, read_only):
    MAXIMUM_README_LINE = 25
    # all files are opened with default_mode=r+
    self._data_map = OrderedDict()
    self._path = os.path.abspath(path)
    self._default_hdf5 = os.path.basename(self._path) + '_default.h5'
    # svaed feeder info
    self._saved_indices = {}
    self._saved_recipes = {}
    # just make new dir
    if not os.path.exists(path):
      os.mkdir(path)
      os.mkdir(self.recipe_path)
      os.mkdir(self.index_path)
      return # not thing to do more
    elif not os.path.isdir(path):
      raise ValueError('Dataset path must be a folder.')
    # ====== Load all Data ====== #
    files = os.listdir(path)
    for fname in files:
      # found README
      if 'readme' == fname[:6].lower():
        readme_path = os.path.join(path, fname)
        with open(readme_path, 'r') as readme_file:
          readme = readme_file.readlines()[:MAXIMUM_README_LINE]
          readme = ['  ' + i[:-1] for i in readme if len(i) > 0 and i != '\n']
          readme.append(' => For more information: ' + readme_path)
          self._readme_info = [ctext('README:', 'yellow'),
                               '------'] + readme
          self._readme_path = readme_path
      # parse data
      data = _parse_data_descriptor(os.path.join(path, fname),
                                    read_only)
      if data is None: continue
      for key, d in data:
        if key in self._data_map:
          raise ValueError('Found duplicated data with follow info: '
                           '{}'.format(key))
        else:
          self._data_map[key] = d

  # ==================== Pickle ==================== #
  def __getstate__(self):
    if not self._new_args_called:
      raise RuntimeError(
          "You must use argument `protocol=cPickle.HIGHEST_PROTOCOL` "
          "when using `pickle` or `cPickle` to be able pickling Dataset.")
    self._new_args_called = False
    return self.path, self.read_only

  def __setstate__(self, states):
    path, read_only = states
    self._new_args_called = False
    self._set_path(path, read_only)

  def __getnewargs__(self):
    self._new_args_called = True
    return (self.path,)

  # ==================== archive loading ==================== #
  def _load_archive(self, path, extract_path, read_only):
    from zipfile import ZipFile, ZIP_DEFLATED
    try:
      zfile = ZipFile(path, mode='r', compression=ZIP_DEFLATED)
      allfile = zfile.namelist()
      # validate extract_path
      if not os.path.isdir(extract_path):
        raise ValueError('Extract path must be path folder, but path'
                         '={} is a file'.format(extract_path))
      extract_path = os.path.join(extract_path,
                                  os.path.basename(path).replace('.zip', ''))
      # found the extracted dir, use it
      if os.path.isdir(extract_path) and \
         set(os.listdir(extract_path)) == set(allfile):
        self._set_path(extract_path, read_only=read_only)
        return
      # decompress everything
      if not os.path.exists(extract_path):
        os.mkdir(extract_path)
      maxlen = max([len(i) for i in allfile])
      pb = Progbar(target=len(allfile), name="[Dataset] Loading Archive",
                   print_summary=True, print_report=True)
      for i, f in enumerate(allfile):
        zfile.extract(f, path=extract_path)
        pb['File'] = ('Unarchiving: %-' + str(maxlen) + 's') % f
        pb.add(1)
      # ====== finally set path ====== #
      self._set_path(extract_path, read_only=read_only)
    except IOError as e:
      raise IOError('Error loading archived dataset, path:{}, error:{}'
                    '.'.format(path, e))
    return None

  # ==================== properties ==================== #
  @property
  def basename(self):
    return os.path.basename(self.path)

  @property
  def path(self):
    return self._path

  @property
  def recipe_path(self):
    return os.path.join(self.path, 'recipe')

  @property
  def index_path(self):
    return os.path.join(self.path, 'index')

  @property
  def archive_path(self):
    """Return default archive path, which is:
        ../[dataset_name].zip
    """
    name = os.path.basename(self._path)
    return os.path.join(self._path, '..', name + '.zip')

  @property
  def md5_checksum(self):
    return self.get_md5_checksum()

  @property
  def size(self):
    """ return size in MegaByte"""
    size_bytes = 0
    for name, (dtype, shape, data, path) in self._data_map.items():
      try:
        size_bytes += os.path.getsize(path) # in bytes
      except Exception as e:
        eprint("Cannot acquire file size information, file: %s; error: %s"
               % (str(name), str(e)))
    return size_bytes / 1024. / 1024.

  def __len__(self):
    """ Return total number of data """
    return len(self._data_map)

  def __iter__(self):
    return self.items()

  def items(self):
    for name in self._data_map.keys():
      yield name, self.__getitem__(name)

  def iterinfo(self):
    """Return iteration of: (dtype, shape, loaded_data, path)"""
    for name, (dtype, shape, data, path) in self._data_map.items():
      yield (dtype, shape, self.__getitem__(name), path)

  def keys(self):
    """
    Return
    ------
    name of all Data
    """
    return self._data_map.keys()

  def values(self):
    """
    Return
    ------
    (dtype, shape, data, path) of Data
    """
    for k in self._data_map.keys():
      yield self.__getitem__(k)

  def archive(self):
    from zipfile import ZipFile, ZIP_DEFLATED
    path = self.archive_path
    zfile = ZipFile(path, mode='w', compression=ZIP_DEFLATED)

    files = set([_[-1] for _ in self._data_map.values()])

    prog = Progbar(target=len(files), name="[Dataset] Archiving",
                   print_report=True, print_summary=True)
    maxlen = max([len(os.path.basename(i)) for i in files])
    for i, f in enumerate(files):
      zfile.write(f, os.path.basename(f))
      prog['Data'] = ('Archiving: %-' + str(maxlen) + 's') \
          % os.path.basename(f)
      prog.add(1)
    zfile.close()
    return path

  # ==================== Data management ==================== #
  def copy(self, destination,
           indices_filter=None, data_filter=None,
           override=False):
    """ Copy the dataset to a new folder and closed
    the old dataset

    """
    from distutils.dir_util import copy_tree
    read_only = self.read_only
    raise NotImplementedError

  def flush(self):
    for dtype, shape, data, path in self._data_map.values():
      if hasattr(data, 'flush'):
        data.flush()
      elif data is not None: # Flush pickling data
        with open(path, 'wb') as f:
          cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)

  def close(self, name=None):
    # ====== close all Data ====== #
    if name is None: # close all files
      for name, (dtype, shape, data, path) in list(self._data_map.items()):
        if hasattr(data, 'close'):
          data.close()
        del data
        del self._data_map[name]
      # close all external indices and recipes
      for name, ids in self._saved_indices.items():
        ids.close()
      self._saved_indices.clear()
      for name, rcp in list(self._saved_recipes.items()):
        del rcp
      self._saved_recipes.clear()
      # Check if exist global instance
      if self.path in Dataset.__INSTANCES:
        del Dataset.__INSTANCES[self.path]
    # ====== close a particular file ====== #
    elif name in self._data_map:
      (dtype, shape, data, path) = self._data_map[name]
      if dtype == 'sqlite':
        data.sqlite.close()
      elif hasattr(data, 'close'):
        data.close()
      del data
      del self._data_map[name]

  # ==================== Some info ==================== #
  def __contains__(self, key):
    return key in self._data_map

  def find_prefix(self, feat_name, prefix):
    """ Specialized method for searching for Data or NoSQL
    with prefix, for example `prefix='indices'`:
      - `indices_%s_%s` % (feat1_name, feat2, ...)
    if no indices found, return the default indices with
    name 'indices'
    """
    indices = self[prefix] if prefix in self else None
    for key in self.keys():
      if prefix == key[:len(prefix)] and '_' + feat_name in key:
        indices = self[key]
    if indices is None:
      raise RuntimeError("Cannot find prefix: '%s' for feature with name: '%s', "
                         "all available name with given prefix are: %s" %
                         (prefix, feat_name, ','.join([k for k in self.keys()
                                                       if prefix == k[:len(k)]])
                         ))
    return indices

  def __getitem__(self, key):
    if is_string(key):
      if key not in self._data_map:
        raise KeyError('%s not found in this dataset' % key)
      dtype, shape, data, path = self._data_map[key]
      return path if data is None else data
    raise ValueError('Only accept key type is string.')

  def get(self, key, default=None):
    if key in self._data_map:
      return self.__getitem__(key)
    return default

  def __setitem__(self, name: Text, value: Any):
    """
    Parameters
    ----------
    key : str or tuple
        if tuple is specified, it contain the key and the datatype
        which must be "memmap", "hdf5"
        for example: ds[('X', 'hdf5')] = numpy.ones((8, 12))
    """
    assert isinstance(name, string_types), \
      "name must be given as string types."
    path = os.path.join(self.path, name)
    with open(path, 'wb') as f:
      pickle.dump(value, f)
    self._data_map[name] = (
      value.dtype if hasattr(value, 'dtype') else str(type(value)),
      value.shape if hasattr(value, 'shape') else 'unknown',
      value, path)


  def get_md5_checksum(self, excluded_name=[]):
    from odin.utils.crypto import md5_checksum
    md5_text = ''
    all_data_items = {i: j
                      for i, j in self._data_map.items()
                      if i not in excluded_name}
    for name, (dtype, shape, data, path) in sorted(all_data_items.items(),
                                                   key=lambda x: x[0]):
      md5_text += md5_checksum(path)
    return md5_text

  def __str__(self):
    padding = '  '
    # NOTE: each element in the list is one line
    s = ['==========  ' +
         ctext('Dataset:%s Total:%d Size:%.2f(MB)', 'magenta') %
         (self.path, len(self._data_map), self.size) +
         '  ==========']
    s += self._readme_info
    s += [ctext('DATA:', 'yellow'),
          '----']
    # ====== Find longest string ====== #
    longest_name = 0
    longest_shape = 0
    longest_dtype = 0
    longest_file = 0
    print_info = []
    for name, (dtype, shape, data, path) in sorted(self._data_map.items()):
      shape = data.shape if hasattr(data, 'shape') else shape
      longest_name = max(len(name), longest_name)
      longest_dtype = max(len(str(dtype)), longest_dtype)
      longest_shape = max(len(str(shape)), longest_shape)
      longest_file = max(len(str(path)), longest_file)
      print_info.append([name, dtype, shape, path])
    # ====== return print string ====== #
    format_str = (padding + '%-' + str(longest_name + 2) + 's  '
                  '%-' + str(longest_dtype) + 's' + ctext(':', 'yellow') +
                  '%-' + str(longest_shape) + 's  '
                  'path:%-' + str(longest_file) + 's')
    for name, dtype, shape, path in print_info:
      s.append(format_str % ('"%s"' % name, dtype, shape, path))
    # ====== add recipes info ====== #
    for name, recipe in self._saved_recipes.items():
      s.append(ctext('(Recipe) ', 'yellow') + '"%s"' % name)
      for rcp in recipe:
        rcp = str(rcp)
        s.append('\n'.join([padding + line
                            for line in rcp.split('\n')]))
    # ====== add indices info ====== #
    for name, index in self._saved_indices.items():
      s.append(ctext('(Index) ', 'yellow') + '"%s"' % name)
      s.append(padding + str(index))
      name, (start, end) = next(index.items())
      s.append(padding + 'Sample: "%s %d-%d"' % (name, start, end))
    return '\n'.join(s)

  @property
  def readme(self):
    """ return text string of README of this dataset """
    if self._readme_path is not None:
      with open(self._readme_path, 'r') as f:
        readme = f.read()
    else:
      readme = self._readme_info[-1]
    return readme
