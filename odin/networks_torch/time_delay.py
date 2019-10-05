from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from torch import nn

from odin.networks_torch.keras_torch import Layer


class TimeDelay(Layer):

  def __init__(self,
               fn_layer_creator,
               delay_context=(-2, -1, 0, 1, 2),
               pooling='sum',
               name=None,
               **kwargs):
    super().__init__()
    self.activation = get_activation_function(settings)
    self.delay_nets = nn.ModuleList()
    self.delays = delays

    self.max_shift_front = -min(0, min(delays))
    self.max_shift_end = max(0, max(delays))

    bias = True
    for _ in range(len(delays)):
      self.delay_nets.append(nn.Linear(D_in, D_out, bias=bias))
      bias = False
    self.norm = get_normalizer(D_out, settings)

  def call(self, x):
    y = self.delay_nets[0](
        x[:, self.max_shift_front + self.delays[0]:x.size()[1] -
          self.max_shift_end + self.delays[0], :])
    for delay, net in zip(self.delays[1:], self.delay_nets[1:]):
      y += net(x[:, self.max_shift_front + delay:x.size()[1] -
                 self.max_shift_end + delay, :])
    return self.norm(self.activation(y))
