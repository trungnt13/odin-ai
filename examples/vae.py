from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("Agg")

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import cPickle

import numpy as np
from scipy.misc import imsave

from odin.config import get_rng
from odin import nnet as N, backend as K, fuel as F, training
from odin.utils import Progbar, get_modelpath, get_logpath


# ====== const ====== #
# DATA. MNIST batches are fed at training time.
from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = "/tmp/data/mnist"
IMG_DIR = "/tmp/img"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
M = 128  # batch size during training
n_epoch = 3
n_iter_per_epoch = 1000

# ====== load dataset ====== #
ds = F.load_mnist()
print(ds)

X_train = ds['X_train'].transform(
    lambda x: np.random.binomial(1, np.clip(x, 0., 1.)))
X_valid = ds['X_valid'].transform(
    lambda x: np.random.binomial(1, np.clip(x, 0., 1.)))

# ====== get model ====== #
X = K.placeholder(shape=(M,) + ds['X_train'].shape[1:], dtype='int32', name='X')
print("Input shape:", X)
model = N.get_model_descriptor('dense_vae')
K.set_training(True); (y1, z1, qz1) = model(X)
K.set_training(False); (y2, z2, qz2) = model(X)
print([i.name for i in K.ComputationGraph(y1).variables])
print()
print([i.name for i in K.ComputationGraph(y2).variables])
exit()
# import tensorflow as tf
# writer = tf.summary.FileWriter('/tmp/testlog',
#     graph=y_score.graph)
# ====== create optimizer ====== #
# opt = K.optimizers.Adam(lr=0.01)

# ====== other trianing ====== #
for epoch in range(n_epoch):
    avg_loss = 0.0

    pbar = Progbar(n_iter_per_epoch)
    for t in range(1, n_iter_per_epoch + 1):
        pbar.update(t)
        x_train, _ = mnist.train.next_batch(M)
        x_train = np.random.binomial(1, x_train)
        avg_loss += f_train(x_train.reshape(-1, 28, 28))

    # Print a lower bound to the average marginal likelihood for an
    # image.
    avg_loss = avg_loss / n_iter_per_epoch
    avg_loss = avg_loss / M
    print("log p(x) >= {:0.3f}".format(avg_loss))

    # Visualize hidden representations.
    imgs = y_score.eval()
    for m in range(M):
        imsave(os.path.join(IMG_DIR, '%d.png') % m, imgs[m])

# ====== trainng ====== #
# trainer, hist = training.standard_trainer(
#     train_data=X_train, valid_data=X_valid, test_data=None,
#     cost_train=train_loss, cost_score=None, cost_regu=None,
#     optimizer=opt, parameters=parameters,
#     batch_size=128, nb_epoch=2, valid_freq=1.,
#     save_path=get_modelpath(name='cnn_vae.ai', override=True), save_obj=model,
#     report_path=get_logpath(name='cnn_vae.pdf', override=True),
#     stop_callback=opt.get_lr_callback())
# trainer.run()
