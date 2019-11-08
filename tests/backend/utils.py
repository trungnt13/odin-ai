import os

import numpy as np
import tensorflow as tf
import torch

np.random.seed(8)
torch.manual_seed(8)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x = np.random.rand(12, 25, 8).astype('float32')
y = torch.Tensor(x)
z = tf.convert_to_tensor(x)


def assert_equal(self, info, a: np.ndarray, b: torch.Tensor, c: tf.Tensor):
  assert all(
      int(i) == int(j) == int(k) for i, j, k in zip(a.shape, b.shape, c.shape)),\
        "Input shape: %s, info: %s, output shapes mismatch: %s, %s and %s" % \
          (str(x.shape), str(info), str(a.shape), str(b.shape), str(c.shape))
  self.assertTrue(np.all(
      np.logical_and(np.allclose(a, b.numpy()), np.allclose(a, c.numpy()))),
                  msg="info: %s, output value mismatch, \n%s\n%s\n%s" %
                  (info, str(a), str(b.numpy()), str(c.numpy())))
