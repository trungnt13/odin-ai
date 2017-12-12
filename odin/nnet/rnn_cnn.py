from __future__ import print_function, division, absolute_import

from .rnn import BaseRNN


class LSTMcnn(BaseRNN):
  """ LSTMcnn """

  def __init__(self, arg):
    super(LSTMcnn, self).__init__()
    self.arg = arg
