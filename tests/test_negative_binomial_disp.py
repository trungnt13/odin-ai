from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import torch

from odin.bay.distributions import NegativeBinomialDisp, ZeroInflated
from odin.stats import describe
from scvi.models.log_likelihood import log_nb_positive, log_zinb_positive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.random.set_seed(8)
np.random.seed(8)
torch.manual_seed(8)


def torch_nb(mean, disp):
  px_rate = torch.Tensor(mean)
  px_r = torch.Tensor(disp)

  p = px_rate / (px_rate + px_r)
  r = px_r
  l_train = torch.distributions.Gamma(concentration=r,
                                      rate=(1 - p) / p).sample()
  l_train = torch.clamp(l_train, max=1e18)
  X = torch.distributions.Poisson(l_train).sample()
  return X


shape = (12000, 800)
x = np.random.randint(1, 20, size=shape).astype('float32')
mean = np.random.randint(1, 20, size=shape).astype('float32')
disp = np.random.randint(1, 20, size=shape).astype('float32')
disp_col = np.random.randint(1, 20, size=shape[1]).astype('float32')
disp_row = np.random.randint(1, 20, size=shape[0]).astype('float32')
pi = np.random.rand(*shape).astype('float32')

# constant dispersion (only for tensorflow)
nb = NegativeBinomialDisp(loc=mean, disp=2)
llk1 = tf.reduce_sum(nb.log_prob(x), axis=1).numpy()
print(llk1)

# broadcast disp in column
nb = NegativeBinomialDisp(loc=mean, disp=disp_col)
llk1 = tf.reduce_sum(nb.log_prob(x), axis=1).numpy()
llk2 = log_nb_positive(x=torch.Tensor(x),
                       mu=torch.Tensor(mean),
                       theta=torch.Tensor(disp_col)).numpy()
print(np.all(np.isclose(llk1, llk2)))

# broadcast disp in row
try:
  nb = NegativeBinomialDisp(loc=mean, disp=disp_row)
  llk1 = tf.reduce_sum(nb.log_prob(x), axis=1).numpy()
  llk2 = log_nb_positive(x=torch.Tensor(x),
                         mu=torch.Tensor(mean),
                         theta=torch.Tensor(disp_row)).numpy()
  print(np.all(np.isclose(llk1, llk2)))
except:
  print("NOT POSSIBLE TO BROADCAST the first dimension")

# all disp available
nb = NegativeBinomialDisp(loc=mean, disp=disp)
llk1 = tf.reduce_sum(nb.log_prob(x), axis=1).numpy()
llk2 = log_nb_positive(x=torch.Tensor(x),
                       mu=torch.Tensor(mean),
                       theta=torch.Tensor(disp)).numpy()
print(np.all(np.isclose(llk1, llk2)))

s1 = nb.sample().numpy()
s2 = torch_nb(mean, disp).numpy()
print(describe(s1))
print(describe(s2))

zinb = ZeroInflated(nb, probs=pi)
llk1 = tf.reduce_sum(zinb.log_prob(x), axis=1).numpy()
llk2 = log_zinb_positive(x=torch.Tensor(x),
                         mu=torch.Tensor(mean),
                         theta=torch.Tensor(disp),
                         pi=torch.Tensor(pi)).numpy()
print(llk1)
print(llk2)
