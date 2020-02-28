from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np

from odin.utils.ordered_flag import OrderedFlag, auto

np.random.seed(8)


class Enum1(OrderedFlag):
  T1 = auto()
  T2 = auto()
  T3 = auto()
  T4 = auto()


class Enum2(OrderedFlag):
  T1 = 1
  T2 = 2
  T3 = 3
  T4 = 4


class OrderedFlagTest(unittest.TestCase):

  def test_contain(self):
    t1 = Enum1.T2 | Enum1.T1 | Enum1.T3
    self.assertTrue(Enum1.T1 in t1)
    self.assertTrue(Enum1.T4 not in t1)

  def test_and(self):
    t1 = Enum1.T1 | Enum1.T4
    t2 = Enum1.T2 | Enum1.T1 | Enum1.T3
    self.assertTrue((t1 & t2) == Enum1.T1)

  def test_or(self):
    t1 = Enum1.T1 | Enum1.T2
    self.assertTrue(t1.value == '1_2')

  def test_xor(self):
    t1 = Enum1.T1 | Enum1.T2
    t2 = Enum1.T2 | Enum1.T1 | Enum1.T3
    self.assertTrue(t1 ^ t2 == Enum1.T3)

  def test_not(self):
    t1 = Enum1.T1 | Enum1.T2
    self.assertTrue(~t1 == (Enum1.T3 | Enum1.T4))

  def test_iter(self):
    t1 = Enum1.T1 | Enum1.T2
    for i, j in zip(t1, [Enum1.T1, Enum1.T2]):
      self.assertTrue(isinstance(i, Enum1))
      self.assertTrue(i == j)

  def test_base(self):
    for i, j in zip(Enum1, Enum2):
      self.assertTrue(i != j)
      self.assertFalse(i == j)
      self.assertTrue(i == i)
      self.assertTrue(j == j)
      self.assertFalse(i != i)
      self.assertFalse(j != j)

    t1 = Enum1.T1 | Enum1.T2
    t2 = Enum2.T1 | Enum2.T2
    t3 = Enum1.T2 | Enum1.T1
    t4 = Enum1.T1 | Enum1.T2
    self.assertTrue(t1 != t2)
    self.assertTrue(t1 == t3)
    self.assertFalse(t1 != t4)
    self.assertFalse(t3 != t4)

  def test_members(self):
    t1 = Enum1.T1 | Enum1.T2
    t2 = Enum1.T2 | Enum1.T1 | Enum1.T3
    self.assertEqual(len(Enum1), 4)
    self.assertEqual(len(list(Enum1)), 4)

if __name__ == '__main__':
  unittest.main()
