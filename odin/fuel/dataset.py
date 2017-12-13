from __future__ import print_function, division, absolute_import

import os
import shutil
from collections import OrderedDict, Mapping
from six.moves import zip, range, cPickle

import numpy as np

from .data import (MmapData, Hdf5Data, open_hdf5, get_all_hdf_dataset,
                   MAX_OPEN_MMAP, Data, as_data)
from .utils import MmapDict, SQLiteDict, NoSQL

from odin.utils import (get_file, Progbar, is_string,
                        ctext, as_tuple, eprint, wprint)
from .recipe_base import FeederRecipe, RecipeList


__all__ = [
    'Dataset',
]


# ===========================================================================
# dataset
# ===========================================================================
def _parse_data_descriptor(path, read_only):
  """ Return mapping: name -> (dtype, shape, Data, path) """
  if not os.path.isfile(path):
    return None
  # ====== check if a file is Data ====== #
  try:
    dtype, shape = MmapData.read_header(path, read_only=True,
                                        return_file=False)
    # shape[1:], because first dimension can be resize afterward
    return [(os.path.basename(path), (dtype, shape, None, path))]
  except Exception as e: # cannot read the header of MmapData, maybe Hdf5
    try:
      f = open_hdf5(path, read_only=read_only)
      ds = get_all_hdf_dataset(f)
      data = [Hdf5Data(dataset=i, hdf=f) for i in ds]
      return [(str(i.name), (str(i.dtype), i.shape, i, i.path)) for i in data]
    except Exception as e:
      pass
  # ====== try to load pickle file if possible ====== #
  name = os.path.basename(path)
  try:
    with open(path, 'rb') as f:
      data = cPickle.load(f)
      return [(name,
      (type(data).__name__, len(data) if hasattr(data, '__len__') else 0, data, path))]
  except Exception as e:
    pass
  # ====== load memmap dict ====== #
  try:
    data = MmapDict(path, read_only=read_only)
    return [(name, ('memdict', len(data), data, path))]
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
  return [(name, ('unknown', 'unknown', None, path))]


class Dataset(object):
  """ This Dataset can automatically parse memmap (created by MmapData),
  MmapDict, pickled dictionary and hdf5 files and keep tracking the changes.

  Any file name with "readme" prefix will be parsed as text and showed as
  readme.

  Note
  ----
  for developer: _data_map contains: name -> (dtype, shape, Data or pathtoData)
  readme included with the dataset should contain license information
  All the file with `.db` extension will be treat as SQLite data
  """

  __INSTANCES = {}

  def __new__(clazz, *args, **kwargs):
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
    new_instance = super(Dataset, clazz).__new__(clazz)
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
            extract_path=path.replace(os.path.basename(path), ''))
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
          readme.append(' * For more information: ' + readme_path)
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
    # ====== Load stored recipes ====== #
    # check recipes path
    if os.path.exists(self.recipe_path) and os.path.isfile(self.recipe_path):
      raise RuntimeError("Found a file at path: '%s', which supposed to "
                         "be a folder used for saving `recipe`"
                         % self.recipe_path)
    if not os.path.exists(self.recipe_path):
      os.mkdir(self.recipe_path)
    # all recipes is pickle-able
    for recipe_name in os.listdir(self.recipe_path):
      with open(os.path.join(self.recipe_path, recipe_name), 'rb') as f:
        recipe = cPickle.load(f)
        self._saved_recipes[recipe_name] = recipe
    # ====== load stored indices ====== #
    # check indices path
    if os.path.exists(self.recipe_path) and os.path.isfile(self.recipe_path):
      raise RuntimeError("Found a file at path: '%s', which supposed to "
                         "be a folder used for saving `index`"
                         % self.recipe_path)
    if not os.path.exists(self.index_path):
      os.mkdir(self.index_path)
    # load all saved indices
    for index_name in os.listdir(self.index_path):
      path = os.path.join(self.index_path, index_name)
      index = MmapDict(path=path, read_only=True)
      # remove extension from the name
      self._saved_indices[index_name] = index

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
  def _load_archive(self, path, extract_path):
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
        self._set_path(extract_path)
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
      self._set_path(extract_path)
    except IOError as e:
      raise IOError('Error loading archived dataset, path:{}, error:{}'
                    '.'.format(path, e))
    return None

  # ==================== properties ==================== #
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

  def recipes(self):
    return {name: rcp
            for name, rcp in self._saved_recipes.items()}

  def indices(self):
    return {name: ids
            for name, ids in self._saved_indices.items()}

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

  # ==================== Feeder management ==================== #
  def add_indices(self, indices, name, override=False):
    # ====== validate name ====== #
    if not is_string(name):
      raise ValueError("`name` must be string, but given: %s" % str(type(name)))
    if name in self._saved_indices and not override:
      raise ValueError("Cannot override pre-defined INDEX with name: '%s'"
                      % name)
    # ====== validate indices ====== #
    path = os.path.join(self.index_path, name)
    ids = MmapDict(path)
    # predefined mapping, save or copy everything to a
    # MmapDict
    if isinstance(indices, Mapping):
      for name, (start, end) in indices.items():
        ids[name] = (start, end)
    # list of name, or (name, (start, end))
    elif isinstance(indices, (tuple, list, np.ndarray)):
      for i in indices:
        if is_string(i): # only name
          ids[i] = self['indices'][i]
        elif len(i) == 2: # name, (start, end)
          name, (start, end) = i
          ids[name] = (int(start), int(end))
        elif len(i) == 3: # name, start, end
          name, start, end = i
          ids[name] = (int(start), int(end))
        else:
          raise ValueError("Unsupport index parsing (name, start, end)"
                           "for: %s" % str(i))
    # flush everything to disk
    ids.flush(save_all=True)
    ids.close()
    # ====== assign new index ====== #
    self._saved_indices[name] = MmapDict(path, read_only=True)
    return self

  def remove_indices(self, name):
    if name is None: # remove all recipes
      for name in self._saved_indices.keys():
        self.remove_indices(name)
    # remove only a selected
    elif name in self._saved_indices:
      self._saved_indices[name].close()
      del self._saved_indices[name]
      os.remove(os.path.join(self.index_path, name))
    return self

  def add_recipes(self, recipes, name, override=False):
    """
    Parameters
    ----------
    """
    # ====== validate arguments ====== #
    if not is_string(name):
      raise ValueError("`name` must be string, but given: %s" % str(type(name)))
    if name in self._saved_recipes and not override:
      raise ValueError("Cannot override pre-defined RECIPE with name: '%s'"
                      % name)
    # ====== validate recipes list ====== #
    if isinstance(recipes, RecipeList):
      recipes = tuple(recipes._recipes)
    else:
      tmp = []
      for rcp in as_tuple(recipes, t=FeederRecipe):
        if isinstance(rcp, RecipeList):
          tmp += list(rcp._recipes)
        else:
          tmp.append(rcp)
      recipes = tuple(tmp)
    # ====== store the recipes to disk ====== #
    path = os.path.join(self.recipe_path, name)
    with open(path, 'wb') as f:
      cPickle.dump(recipes, f, protocol=cPickle.HIGHEST_PROTOCOL)
    # ====== update local recipes list ====== #
    self._saved_recipes[name] = recipes
    return self

  def remove_recipes(self, name):
    if name is None: # remove all recipes
      for name in self._saved_recipes.keys():
        self.remove_recipes(name)
    # remove only a selected
    elif name in self._saved_recipes:
      del self._saved_recipes[name]
      os.remove(os.path.join(self.recipe_path, name))
    return self

  def create_feeder(self, data, recipes, indices=None,
                    batch_filter=None, batch_mode='batch',
                    name=None, override=False):
    """
    Parameters
    ----------
    data: list of str
        list of name for all data used, the order of this
        list is the order of returned data.
    recipes: list or single odin.fuel.FeederRecipe
        the list of recipes defining the rule of transforming
        the data
    indices: None, string, dict, list
        list of (name, (start, end)) for iterating over files in Feeder
    batch_filter: call-able
        must be a function has take a list of np.ndarray as first arguments
        ([X]) or ([X, y]), you can return None to ignore given batch, return the
        data for accepting the batch
    batch_mode: 'batch' or 'file' (string type)
        'batch' mode return shuffling and return everything in small batches
        'file' mode return [(file_name, order_index, data...), ...]
    name: None, or string
        if name is provided, the feeder information will be saved,
        which include the `indices`, `recipes`

    Note
    ----
    by defaults, the Feeder is created using only 1 CPU with `buffer_size=1`
    using the method `set_multiprocessing(ncpu=None, buffer_size=None,
    maximum_queue_size=None)` for changing this information.
    """
    from .feeder import Feeder, DataDescriptor
    # check data
    data = [self.__getitem__(dat) if is_string(dat) else
            as_data(dat)
            for dat in as_tuple(data)]
    # check recipes
    if is_string(recipes):
      recipes = self._saved_recipes[recipes]
    else:
      recipes = as_tuple(recipes, t=FeederRecipe)
    # check indices
    if indices is None:
      indices = self.__getitem__('indices')
    elif is_string(indices):
      indices = self._saved_indices[indices]
    elif isinstance(indices, (Mapping, tuple, list, np.ndarray)):
      pass
    # ====== saving recipes and indices, if name is not None ====== #
    if is_string(name):
      if name not in self._saved_indices or override:
        self.add_indices(indices, name, override=True)
      if name not in self._saved_recipes or override:
        self.add_recipes(recipes, name, override=True)
    # ====== create Feeder ====== #
    feeder = Feeder(DataDescriptor(data=data, indices=indices),
                    batch_filter=batch_filter, batch_mode=batch_mode,
                    ncpu=1, buffer_size=1)
    return feeder.set_recipes(recipes)

  # ==================== Data management ==================== #
  def copy(self, destination):
    """ Copy the dataset to a new folder and closed
    the old dataset

    Note
    ----
    if the dataset size >= 1GB, need confirmation for copying
    """
    read_only = self.read_only
    if self.size > 1024:
      confirmation = raw_input(
          'The size of the dataset at path: "%s" is %d (MB)\n'
          'Do you want to copy everything to "%s" (Y - yes):')
      if 'y' not in confirmation.lower():
        return self
    destination = os.path.abspath(str(destination))
    if not os.path.exists(destination):
      os.mkdir(destination)
    elif not os.path.isdir(destination):
      raise ValueError('path at "%s" must be a folder' % destination)
    else:
      ds = Dataset(destination, read_only=read_only)
      if ds.size != self.size:
        raise RuntimeError("Found another Dataset at path: '%s' has "
                           "size %d(MB), which is different from size "
                           "of this Dataset %d(MB)." %
                           (destination, ds.size, self.size))
      return ds
    from distutils.dir_util import copy_tree
    copy_tree(self.path, destination)
    # ====== open new dataset ====== #
    self.close()
    return Dataset(destination, read_only=read_only)

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
  def _validate_memmap_max_open(self, name):
    # ====== check if MmapData excess limit, close 1 files ====== #
    if len(MmapData._INSTANCES) + 1 >= MAX_OPEN_MMAP:
      for i, (_dtype, _shape, _data, _path) in self._data_map.items():
        if isinstance(_data, MmapData) and i != name:
          self.close(name=i)
          self._data_map[i] = (_dtype, _shape, _path)
          break

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
      # return type is just a descriptor, create MmapData for it
      if data is None and \
      dtype is not 'unknown' and shape is not 'unknown':
        data = MmapData(path, read_only=self.read_only)
        self._data_map[key] = (data.dtype, data.shape, data, path)
        self._validate_memmap_max_open(key)
      return path if data is None else data
    raise ValueError('Only accept key type is string.')

  def __setitem__(self, key, value):
    """
    Parameters
    ----------
    key : str or tuple
        if tuple is specified, it contain the key and the datatype
        which must be "memmap", "hdf5"
        for example: ds[('X', 'hdf5')] = numpy.ones((8, 12))
    """
    if not is_string(key) and not isinstance(key, (tuple, list)):
      raise ValueError('"key" is the name for Data and must be String or '
                       'tuple specified the name and datatype (memmap, hdf5).')
    # ====== check datatype ====== #
    datatype = 'memmap' # default datatype
    if isinstance(key, (tuple, list)):
      key, datatype = key
      datatype = str(datatype).lower()
      if datatype not in ('memmap', 'hdf5'):
        raise ValueError('datatype can only be "memmap" or "hdf5", but '
                         'the given data type is "%s"' % datatype)
    # ====== do nothing ====== #
    if key in self._data_map:
      return
    # ====== dict ====== #
    path = os.path.join(self.path, key)
    if isinstance(value, Mapping):
      if os.path.exists(path):
        raise Exception('File with path=%s already exist.' % path)
      d = MmapDict(path)
      for i, j in value.items():
        d[i] = j
      d.flush()
      # store new dict
      self._data_map[key] = (type(d).__name__, len(d), d, path)
    # ====== ndarray ====== #
    elif isinstance(value, np.ndarray):
      dtype, shape = value.dtype, value.shape
      if datatype == 'memmap':
        data = MmapData(path, dtype=dtype, shape=shape)
      else:
        path = os.path.join(self.path, self._default_hdf5)
        f = open_hdf5(path)
        data = Hdf5Data(key, hdf=f, dtype=dtype, shape=shape)
      # store new key
      self._data_map[key] = (data.dtype, data.shape, data, path)
      data[:shape[0]] = value
      # check maximum opened memmap
      self._validate_memmap_max_open(key)
    # ====== other types ====== #
    else:
      if os.path.exists(path):
        raise Exception('File with path=%s already exist.' % path)
      with open(path, 'wb') as f:
        cPickle.dump(value, f, protocol=cPickle.HIGHEST_PROTOCOL)
      # store new dict
      self._data_map[key] = (type(value).__name__,
                             len(value) if hasattr(value, '__len__') else 0,
                             value, path)

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
    for name, (dtype, shape, data, path) in self._data_map.items():
      shape = data.shape if hasattr(data, 'shape') else shape
      longest_name = max(len(name), longest_name)
      longest_dtype = max(len(str(dtype)), longest_dtype)
      longest_shape = max(len(str(shape)), longest_shape)
      longest_file = max(len(str(path)), longest_file)
      print_info.append([name, dtype, shape, path])
    # ====== return print string ====== #
    format_str = (padding + 'Name:%-' + str(longest_name) + 's  '
                  'dtype:%-' + str(longest_dtype) + 's '
                  'shape:%-' + str(longest_shape) + 's  '
                  'file:%-' + str(longest_file) + 's')
    for name, dtype, shape, path in print_info:
      s.append(format_str % (name, dtype, shape, path))
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
