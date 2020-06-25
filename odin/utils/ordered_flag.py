from enum import Enum, auto
from numbers import Number


class OrderedFlag(str, Enum):
  r""" This class operate similar to `enum.Flag`, however,

   - This is string Enum.
   - The representation of an instance is `string.split(cls._sep())`
   - The order is preserved by performing bitwise operator of ordered set.

  The separator could be changed by override class method `_sep`

  Note:
    - During comparison, the order of elements won't be taken into account,
      i.e. `[1, 2] == [2, 1]`
    - The `.name` attribute could be different from `.value` attribute,
      `OrderedFlag.name` will return an unique identity regardless the order,
      i.e. "1_2" for both `[1, 2]` and `[2, 1]`
  """

  @classmethod
  def parse(cls, value, raise_not_found=True):
    r""" Return the matched OrderedFlag, in case of not found:

      - raise `ValueError` if `raise_not_found=True`
      - return False otherwise.

    """
    if not isinstance(value, cls):
      value = str(value)
      for val in cls:
        if value in val.name:
          return val
    else:
      return value
    if raise_not_found:
      raise ValueError(f"Invalid value={value} for OrderedFlag {cls}, "
                       f"all support values are: {list(cls)}")
    return False

  @classmethod
  def _sep(cls):
    return '_'

  @classmethod
  def _missing_(cls, value):
    r""" Create a composite member iff value contains only members. """
    sep = cls._sep()
    pseudo_member = cls._value2member_map_.get(value, None)
    if pseudo_member is None:
      flags = [i for i in value.split(sep) if len(i) > 0]
      if len(flags) == 0:
        value = ''
      else:
        if not all(f in cls._value2member_map_ for f in flags if sep not in f):
          raise ValueError("Invalid value: %s for %s" % (value, cls.__name__))
      # check for duplication
      value = []
      for f in flags:
        if f not in value:
          value.append(f)
      value = sep.join(value)
      # construct a singleton enum pseudo-member
      pseudo_member = str.__new__(cls)
      pseudo_member._name_ = sep.join(sorted(value.split(sep)))
      pseudo_member._value_ = value
      # use setdefault in case another thread already created a composite
      # with this value
      pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
    return pseudo_member

  def __contains__(self, other):
    cls = self.__class__
    other = cls.parse(other, raise_not_found=True)
    return other._value_ in self._value_.split(cls._sep())

  def __invert__(self):
    cls = self.__class__
    sep = cls._sep()
    # original members
    members = [k for k in cls._value2member_map_.keys() if sep not in k]
    values = self._value_.split(sep)
    return self.__class__(sep.join([i for i in members if i not in values]))

  def __or__(self, other):
    cls = self.__class__
    other = cls.parse(other, raise_not_found=True)
    return self.__class__(cls._sep().join([self._value_, other._value_]))

  def __and__(self, other):
    cls = self.__class__
    other = cls.parse(other, raise_not_found=True)
    sep = cls._sep()
    other = other._value_.split(sep)
    values = [i for i in self._value_.split(sep) if i in other]
    return self.__class__(sep.join(values))

  def __xor__(self, other):
    cls = self.__class__
    other = cls.parse(other, raise_not_found=True)
    sep = cls._sep()
    values = self._value_.split(sep)
    others = other._value_.split(sep)
    values = [i for i in values if i not in others] + \
      [i for i in others if i not in values]
    return self.__class__(sep.join(values))

  def __iter__(self):
    for i in self._value_.split(self.__class__._sep()):
      yield self.__class__._value2member_map_[i]

  def index(self, element):
    element = self.__class__.parse(element, raise_not_found=True)
    return list(self).index(element)

  def __getitem__(self, key):
    if isinstance(key, Number):
      return list(self)[key]
    raise ValueError("No support for OrderedFlag indexing with key='%s'" %
                     str(key))

  def __ne__(self, other):
    cls = self.__class__
    if not isinstance(other, cls):
      return True
    sep = cls._sep()
    s1 = set(self._value_.split(sep))
    s2 = set(other._value_.split(sep))
    return s1 != s2

  def __eq__(self, other):
    cls = self.__class__
    if not isinstance(other, cls):
      return False
    sep = cls._sep()
    s1 = set(self._value_.split(sep))
    s2 = set(other._value_.split(sep))
    return s1 == s2

  def __hash__(self):
    return id(self)
