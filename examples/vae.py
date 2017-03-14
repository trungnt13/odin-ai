from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("Agg")

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import cPickle

import numpy as np
import tensorflow as tf

from odin.config import get_rng
from odin import nnet as N, backend as K, fuel as F, training
from odin.utils import Progbar
import edward as ed
# ====== load dataset ====== #
ds = F.load_mnist()
print(ds)
input_shape = (None,) + ds['X_train'].shape[1:]
print("Input shape:", input_shape)

# ====== get model ====== #
model = N.get_model_descriptor('convolutional_vae')
K.set_training(True); (z, qz, x) = model(
    N.InputDescriptor(shape=input_shape, dtype='float32', name='X'))
# K.set_training(False); y_score = model()
X = model.placeholder
parameters = model.parameters
print("Parameters:", [p.name for p in parameters])

# Bind p(x, z) and q(z | x) to the same placeholder for x.
inference = ed.KLqp(latent_vars={z: qz}, data={x: model.placeholder})
optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer, var_list=parameters)

# ====== initialize everything ====== #
print("Initialize all necessary variables")
init = tf.global_variables_initializer()
init.run()
print(inference.train)
print(inference.loss)
print()
exit()
# ====== Training ====== #
X_train = (ds['X_train'][:] == 0).astype("int32")
for nb_epoch in range(12):
    x_train = X_train[get_rng().permutation(X_train.shape[0])]
    prog = Progbar(target=x_train.shape[0])
    cost_train = []
    for i in range(0, x_train.shape[0], 256):
        x = x_train[i:i + 256]
        cost = inference.update(feed_dict={X: x})
        prog.title = cost['loss']
        prog.add(x.shape[0])
        cost_train.append(cost['loss'])
    print("Epoch %d:" % (nb_epoch + 1), np.mean(cost_train))


# # ====== create trainer ====== #
# opt = K.optimizers.RMSProp(lr=0.0001)
# trainer, hist = training.standard_trainer(
#     train_data=ds['X_train'], valid_data=ds['X_valid'], test_data=ds['X_test'],
#     X=model.placeholder, y_train=y_train[-1], y_score=y_score[-1],
#     y_target=model.placeholder, parameters=parameters,
#     optimizer=opt,
#     confusion_matrix=False, gradient_norm=True,
#     cost_train=K.squared_error, cost_score=K.squared_error,
#     batch_size=64, nb_epoch=3, valid_freq=0.6,
#     report_path='/tmp/tmp.pdf')
# trainer.run()
