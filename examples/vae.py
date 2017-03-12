from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("Agg")

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'
import cPickle

import numpy as np
from odin import nnet as N, backend as K, fuel as F, training

# ====== load dataset ====== #
ds = F.load_mnist()
print(ds)
input_shape = (None,) + ds['X_train'].shape[1:]
print("Input shape:", input_shape)

# ====== get model ====== #
model = N.get_model_descriptor('convolutional_vae')
K.set_training(True); y_train = model(input_shape)
K.set_training(False); y_score = model()
parameters = model.parameters
print("Parameters:", [p.name for p in parameters])

# ====== create trainer ====== #
opt = K.optimizers.RMSProp(lr=0.0001)
trainer, hist = training.standard_trainer(
    train_data=ds['X_train'], valid_data=ds['X_valid'], test_data=ds['X_test'],
    X=model.placeholder, y_train=y_train[-1], y_score=y_score[-1],
    y_target=model.placeholder, parameters=parameters,
    optimizer=opt,
    confusion_matrix=False, gradient_norm=True,
    cost_train=K.squared_error, cost_score=K.squared_error,
    batch_size=64, nb_epoch=3, valid_freq=0.6,
    report_path='/tmp/tmp.pdf')
trainer.run()
