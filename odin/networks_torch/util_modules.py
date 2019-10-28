from __future__ import absolute_import, division, print_function

from itertools import chain

import numpy as np
import torch

from odin.utils import as_tuple


class Sequential(torch.nn.Sequential):

  def __init__(self, *args):
    args = list(chain(*[as_tuple(a) for a in args]))
    super().__init__(*args)


class Parallel(Sequential):

  def forward(self, input):
    outputs = []
    for module in self._modules.values():
      outputs.append(module(input))
    return outputs
