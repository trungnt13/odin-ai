from __future__ import print_function, division, absolute_import

import inspect
from six import add_metaclass
from collections import Mapping
from abc import ABCMeta, abstractmethod

import numpy as np

from odin.utils import (flatten_list, is_string, as_tuple,
                        ctext, is_number, is_primitives)

# ===========================================================================
# Recipes
# ===========================================================================
@add_metaclass(ABCMeta)
class FeederRecipe(object):
  """ All method of this function a called in following order
  preprocess_indices(indices): return new_indices
  init(ntasks, batch_size, seed): right before create the iter

  [multi-process] process(*x): x->(name, data)
                               return (if iterator, it will be iterated to
                                       get a list of results)
  [multi-process] group(x): x->(object from group(x))
                            return iterator

  Note
  ----
  This class should not store big amount of data, or the data
  will be replicated to all processes
  """

  def __init__(self):
    super(FeederRecipe, self).__init__()
    self._nb_desc = 0

  # ==================== basic properties ==================== #
  def set_feeder_info(self, nb_desc=None):
    if nb_desc is not None:
      self._nb_desc = int(nb_desc)
    return self

  @property
  def nb_desc(self):
    return self._nb_desc

  # ==================== abstract ==================== #
  def shape_transform(self, shapes):
    """
    Parameters
    ----------
    shapes: list of [(shape0, indices0), (shape1, indices1), ...]
        list of data shape tuple and indices, the indices is list
        of tuple (name, length)

    Return
    ------
    new shape that transformed by this Recipe
    new indices
    """
    return shapes

  @abstractmethod
  def process(self, name, X):
    """
    Parameters
    ----------
    name: string
        the name of file in indices
    X: list of data
        list of all features given in IndexedData(s)
    """
    raise NotImplementedError

  def __str__(self):
    # ====== get all attrs ====== #
    all_attrs = dir(self)
    print_attrs = {}
    for name in all_attrs:
      if '_' != name[0] and (len(name) >= 2 and '__' != name[:2]) and\
      name not in ('nb_desc'):
        attr = getattr(self, name)
        if name == 'data_idx':
          print_attrs[name] = str(attr)
        elif isinstance(attr, slice):
          print_attrs[name] = str(attr)
        elif inspect.isfunction(attr):
          print_attrs[name] = "(f)" + attr.__name__
        elif isinstance(attr, np.ndarray):
          print_attrs[name] = ("(%s)" % str(attr.dtype)) + \
              str(attr.shape)
        elif isinstance(attr, (tuple, list)):
          print_attrs[name] = "(list)" + str(len(attr))
        elif isinstance(attr, Mapping):
          print_attrs[name] = "(map)" + str(len(attr))
        elif is_primitives(attr):
          print_attrs[name] = str(attr)
    print_attrs = sorted(print_attrs.items(), key=lambda x: x[0])
    print_attrs = [('#desc', self.nb_desc)] + print_attrs
    print_attrs = ' '.join(["%s:%s" % (ctext(key, 'yellow'), val)
                            for key, val in print_attrs])
    # ====== format the output ====== #
    s = '<%s %s>' % (ctext(self.__class__.__name__, 'cyan'), print_attrs)
    return s

  def __repr__(self):
    return self.__str__()


class RecipeList(FeederRecipe):

  def __init__(self, *recipes):
    super(RecipeList, self).__init__()
    self._recipes = recipes

  # ==================== List methods ==================== #
  def __iter__(self):
    return self._recipes.__iter__()

  def __len__(self):
    return len(self._recipes)

  # ==================== Override ==================== #
  def set_recipes(self, *recipes):
    # filter out None value
    recipes = flatten_list(as_tuple(recipes))
    recipes = [rcp for rcp in recipes
               if rcp is not None and isinstance(rcp, FeederRecipe)]
    # ====== set the new recipes ====== #
    if len(recipes) > 0:
      self._recipes = recipes
      for rcp in self._recipes:
        rcp.set_feeder_info(self.nb_desc)
    return self

  def set_feeder_info(self, nb_desc=None):
    super(RecipeList, self).set_feeder_info(nb_desc)
    for rcp in self._recipes:
      rcp.set_feeder_info(nb_desc)
    return self

  def process(self, name, X, **kwargs):
    for i, f in enumerate(self._recipes):
      # return iterator (iterate over all of them)
      args = f.process(name, X)
      # break the chain if one of the recipes get error,
      # and return None
      if args is None:
        return None
      if not isinstance(args, (tuple, list)) or \
      len(args) != 2 or \
      not is_string(args[0]) or \
      not isinstance(args[1], (tuple, list)):
        raise ValueError("The returned from `process` must be tuple or "
            "list, of length 2 which contains (name, [x1, x2, x3,...])."
            "`name` must string type, and [x1, x2, ...] is tuple or list.")
      name, X = args
    return name, X

  def shape_transform(self, shapes):
    """
    Parameters
    ----------
    shapes: list of [(shape0, indices0), (shape1, indices1), ...]
        list of data shape tuple and indices, the indices is list
        of tuple (name, length)

    Return
    ------
    new shape that transformed by this Recipe
    new indices
    """
    for i in self._recipes:
      shapes = i.shape_transform(shapes)
      # ====== check returned ====== #
      if not all((isinstance(shp, (tuple, list)) and
                  all(is_number(s) for s in shp) and
                  is_string(ids[0][0]) and is_number(ids[0][1]))
                 for shp, ids in shapes):
        raise RuntimeError("Returned `shapes` must be the list of pair "
                           "`(shape, indices)`, where `indices` is the "
                           "list of (name, length(int)).")
    return shapes
